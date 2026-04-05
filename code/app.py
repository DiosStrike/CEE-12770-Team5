import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gradio as gr
import torch
import torch.nn as nn
import librosa
import numpy as np
import sys

# ==========================================
# 1. Feature Extraction (Aligned with Baseline)
# ==========================================
def file_to_vector_array(file_name, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0):
    y, sr = librosa.load(file_name, sr=16000, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
    )
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    
    vectorarray_size = log_mel_spectrogram.shape[1] - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, n_mels * frames), float)
    
    dims = n_mels * frames
    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t+1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T
        
    return vectorarray

# ==========================================
# 2. Autoencoder Model Architecture
# ==========================================
class MIMII_Baseline_AE(nn.Module):
    def __init__(self, input_dim=320):
        super(MIMII_Baseline_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim) 
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# ==========================================
# 3. Load Models
# ==========================================
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
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models[mid] = model
        print(f"Loaded Model ID_{mid}")
    except Exception as e:
        print(f"Failed to load Model ID_{mid}: {str(e)}")

# 更新为官方计算出的最优阈值
THRESHOLDS = {
    "00": 6.9, 
    "02": 6.0, 
    "04": 5.3, 
    "06": 7.0
}

# ==========================================
# 4. Inference & UI Rendering
# ==========================================
def predict_all_models(audio_path):
    if audio_path is None:
        return "<h3 style='color: white;'>Please upload an audio file.</h3>"
        
    try:
        vector_array = file_to_vector_array(audio_path)
        if vector_array.shape[0] == 0:
            return "<h3 style='color: white;'>Audio is too short.</h3>"
            
        data_tensor = torch.FloatTensor(vector_array).to(device)
        
        html_output = "<div style='display: flex; flex-direction: column; gap: 10px;'>"
        
        for mid in model_ids:
            if mid not in models:
                html_output += f"<div style='border: 1px solid red; padding: 10px; color: white;'>Model {mid} weights missing.</div>"
                continue
                
            model = models[mid]
            threshold = THRESHOLDS[mid]
            
            with torch.no_grad():
                reconstructed = model(data_tensor)
                mse_per_frame = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
                file_error = torch.mean(mse_per_frame).cpu().item()
            
            if file_error <= threshold:
                status_color = "#2e7d32" 
                status_text = "✅ Normal"
                bg_color = "#e8f5e9"
            else:
                status_color = "#c62828" 
                status_text = "🚨 Anomaly"
                bg_color = "#ffebee"
                
            html_output += f"""
            <div style='border: 2px solid {status_color}; background-color: {bg_color}; padding: 12px; border-radius: 6px;'>
                <h4 style='margin: 0 0 5px 0; color: {status_color};'>{status_text} (Model ID_{mid})</h4>
                <div style='font-family: monospace; font-size: 14px; color: #000000;'>
                    MSE: <span style='font-weight: bold; color: #000000;'>{file_error:.4f}</span> | Threshold: <span style='color: #000000;'>{threshold:.1f}</span>
                </div>
            </div>
            """
            
        html_output += "</div>"
        return html_output
        
    except Exception as e:
        return f"<div style='color: red;'>Error: {str(e)}</div>"

# ==========================================
# 5. Gradio Interface
# ==========================================
demo = gr.Interface(
    fn=predict_all_models,
    inputs=gr.Audio(type="filepath", label="Input Audio"),
    outputs=gr.HTML(label="Diagnostic Report"),
    title="HVAC Motor Anomaly Detection",
    description="Upload a test audio clip to run parallel anomaly detection across four autoencoder models.",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(share=False)