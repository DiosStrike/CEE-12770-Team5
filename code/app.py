import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import sys
import gradio as gr
import torch
import torch.nn as nn
import librosa
import numpy as np

# =========================================================
# Config
# =========================================================
TARGET_SR = 16000
N_MELS = 64
FRAMES = 5
N_FFT = 1024
HOP_LENGTH = 512
POWER = 2.0

# Live streaming settings
WINDOW_SECONDS = 10          # rolling buffer length
MIN_ANALYZE_SECONDS = 2      # start producing results after this much audio
STREAM_EVERY = 1.0           # UI refresh interval in seconds

# =========================================================
# 1. Feature Extraction
# =========================================================
def file_to_vector_array(
    file_name,
    n_mels=N_MELS,
    frames=FRAMES,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    power=POWER,
):
    y, sr = librosa.load(file_name, sr=TARGET_SR, mono=True)
    return waveform_to_vector_array(
        y=y,
        sr=sr,
        n_mels=n_mels,
        frames=frames,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
    )


def waveform_to_vector_array(
    y,
    sr,
    n_mels=N_MELS,
    frames=FRAMES,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    power=POWER,
):
    """
    Convert waveform array directly to the same stacked log-mel vector array
    used by the baseline model.
    """
    if y is None or len(y) == 0:
        return np.empty((0, n_mels * frames), float)

    y = np.asarray(y)

    # stereo -> mono
    if y.ndim == 2:
        y = np.mean(y, axis=1)

    # convert to float32
    y = y.astype(np.float32)

    # if data looks like int16 range, normalize it
    max_abs = np.max(np.abs(y)) if len(y) > 0 else 0.0
    if max_abs > 1.5:
        y = y / 32768.0

    # resample to match training pipeline
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )
    log_mel_spectrogram = 20.0 / power * np.log10(
        mel_spectrogram + sys.float_info.epsilon
    )

    vectorarray_size = log_mel_spectrogram.shape[1] - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, n_mels * frames), float)

    dims = n_mels * frames
    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[
            :, t : t + vectorarray_size
        ].T

    return vectorarray


# =========================================================
# 2. Autoencoder Model
# =========================================================
class MIMII_Baseline_AE(nn.Module):
    def __init__(self, input_dim=320):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# =========================================================
# 3. Load Models
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_ids = ["00", "02", "04", "06"]
models = {}

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "..", "model", "baseline")

for mid in model_ids:
    model = MIMII_Baseline_AE(input_dim=320).to(device)
    pth_path = os.path.join(model_dir, f"baseline_fan_id_{mid}.pth")
    try:
        checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models[mid] = model
        print(f"Loaded Model ID_{mid}")
    except Exception as e:
        print(f"Failed to load Model ID_{mid}: {str(e)}")

THRESHOLDS = {
    "00": 6.9,
    "02": 6.0,
    "04": 5.3,
    "06": 7.0,
}


# =========================================================
# 4. Rendering
# =========================================================
def render_result_cards(scores):
    """
    scores: dict like {"00": 1.23, "02": 4.56, ...}
    """
    html_output = "<div style='display:flex; flex-direction:column; gap:12px;'>"

    for mid in model_ids:
        if mid not in models:
            html_output += """
            <div style='border:2px solid #c62828; background:#ffebee; padding:12px; border-radius:8px;'>
                <h4 style='margin:0 0 6px 0; color:#c62828;'>Model missing</h4>
                <div style='color:#111111 !important; font-size:14px;'>
                    Model weights for this machine ID were not found.
                </div>
            </div>
            """
            continue

        score = scores[mid]
        threshold = THRESHOLDS[mid]

        if score <= threshold:
            status_color = "#2e7d32"
            status_text = "Normal"
            bg_color = "#e8f5e9"
        else:
            status_color = "#c62828"
            status_text = "Anomaly"
            bg_color = "#ffebee"

        html_output += f"""
        <div style='border:2px solid {status_color}; background:{bg_color}; padding:14px; border-radius:8px;'>
            <h4 style='margin:0 0 8px 0; color:{status_color}; font-weight:700;'>
                {status_text} (Model ID_{mid})
            </h4>
            <div style='font-family:monospace; font-size:16px; color:#111111 !important;'>
                <span style='color:#111111 !important;'>MSE:</span>
                <span style='font-weight:700; color:#111111 !important;'>{score:.4f}</span>
                <span style='color:#111111 !important;'> | Threshold: </span>
                <span style='color:#111111 !important;'>{threshold:.1f}</span>
            </div>
        </div>
        """

    html_output += "</div>"
    return html_output


# =========================================================
# 5. Core Inference
# =========================================================
def scores_from_vector_array(vector_array):
    if vector_array.shape[0] == 0:
        return None

    data_tensor = torch.FloatTensor(vector_array).to(device)
    scores = {}

    for mid in model_ids:
        if mid not in models:
            continue

        model = models[mid]
        with torch.no_grad():
            reconstructed = model(data_tensor)
            mse_per_frame = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
            file_error = torch.mean(mse_per_frame).cpu().item()

        scores[mid] = file_error

    return scores


def predict_all_models(audio_path):
    if audio_path is None:
        return "<h3 style='color:#111111;'>Please upload an audio file.</h3>"

    try:
        vector_array = file_to_vector_array(audio_path)
        if vector_array.shape[0] == 0:
            return "<h3 style='color:#111111;'>Audio is too short.</h3>"

        scores = scores_from_vector_array(vector_array)
        if scores is None:
            return "<h3 style='color:#111111;'>Audio is too short.</h3>"

        return render_result_cards(scores)

    except Exception as e:
        return f"<div style='color:#c62828;'>Error: {str(e)}</div>"


def predict_all_models_from_waveform(y, sr):
    try:
        vector_array = waveform_to_vector_array(y, sr)
        if vector_array.shape[0] == 0:
            return "<h3 style='color:#111111;'>Not enough audio yet.</h3>"

        scores = scores_from_vector_array(vector_array)
        if scores is None:
            return "<h3 style='color:#111111;'>Not enough audio yet.</h3>"

        return render_result_cards(scores)

    except Exception as e:
        return f"<div style='color:#c62828;'>Error: {str(e)}</div>"


# =========================================================
# 6. Live Streaming State
# =========================================================
def reset_live_state():
    return {"sr": None, "buffer": np.array([], dtype=np.float32)}


def stream_detect(audio_chunk, state):
    """
    audio_chunk is expected from gr.Audio(type='numpy', streaming=True):
    typically (sample_rate, np.ndarray)
    """
    if state is None:
        state = reset_live_state()

    if audio_chunk is None:
        return state, "<h3 style='color:#111111;'>Waiting for microphone input...</h3>"

    try:
        sr, chunk = audio_chunk
    except Exception:
        return state, "<h3 style='color:#111111;'>Waiting for microphone input...</h3>"

    if chunk is None or len(chunk) == 0:
        return state, "<h3 style='color:#111111;'>Waiting for microphone input...</h3>"

    chunk = np.asarray(chunk)

    # stereo -> mono
    if chunk.ndim == 2:
        chunk = np.mean(chunk, axis=1)

    chunk = chunk.astype(np.float32)

    # normalize if int16-like
    max_abs = np.max(np.abs(chunk)) if len(chunk) > 0 else 0.0
    if max_abs > 1.5:
        chunk = chunk / 32768.0

    # initialize or reset on sample rate change
    if state["sr"] is None or state["sr"] != sr:
        state = {"sr": sr, "buffer": np.array([], dtype=np.float32)}

    # append chunk
    state["buffer"] = np.concatenate([state["buffer"], chunk])

    # keep only last WINDOW_SECONDS
    max_len = int(WINDOW_SECONDS * sr)
    if len(state["buffer"]) > max_len:
        state["buffer"] = state["buffer"][-max_len:]

    # wait until there is enough signal for analysis
    if len(state["buffer"]) < int(MIN_ANALYZE_SECONDS * sr):
        current_sec = len(state["buffer"]) / sr
        return state, (
            f"<h3 style='color:#111111;'>Collecting audio... "
            f"{current_sec:.1f}s / {WINDOW_SECONDS}s rolling window</h3>"
        )

    result_html = predict_all_models_from_waveform(state["buffer"], sr)
    return state, result_html


# =========================================================
# 7. Gradio UI
# =========================================================
with gr.Blocks() as demo:
    gr.Markdown("# HVAC Motor Anomaly Detection")
    gr.Markdown(
        "This demo supports both uploaded audio analysis and live browser microphone streaming."
    )

    with gr.Tab("Upload Audio"):
        upload_audio = gr.Audio(
            sources=["upload"],
            type="filepath",
            label="Input Audio File"
        )
        upload_button = gr.Button("Analyze Uploaded Audio")
        upload_result = gr.HTML(label="Diagnostic Report")

        upload_button.click(
            fn=predict_all_models,
            inputs=upload_audio,
            outputs=upload_result,
        )

    with gr.Tab("Live Microphone"):
        gr.Markdown(
            f"""
            **How to use:** click the microphone record button below and speak / play fan sound near the mic.  
            The app keeps a rolling **{WINDOW_SECONDS}-second** buffer and updates the model results continuously.
            """
        )

        live_state = gr.State(value=reset_live_state())

        live_audio = gr.Audio(
            sources=["microphone"],
            type="numpy",
            streaming=True,
            label="Live Microphone Input"
        )

        with gr.Row():
            reset_button = gr.Button("Reset Live Buffer")

        live_result = gr.HTML(label="Live Diagnostic Report")

        reset_button.click(
            fn=reset_live_state,
            inputs=None,
            outputs=live_state,
        )

        live_audio.stream(
            fn=stream_detect,
            inputs=[live_audio, live_state],
            outputs=[live_state, live_result],
            stream_every=STREAM_EVERY,
        )

if __name__ == "__main__":
    demo.launch(share=False)