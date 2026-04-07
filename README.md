# HVAC Fan Anomaly Detection

## Project Website

Project progress updates are available at:

[https://diosstrike.github.io/CEE-12770-Team5/](https://diosstrike.github.io/CEE-12770-Team5/)

## Hardware & Materials

The following hardware and materials are used or planned for this project:

| Item | Purpose | Quantity | Status | Purchased by | Notes |
|------|--------|----------|--------|--------------|-------|
| ESP32 Development Board | Embedded audio acquisition | 1 | Purchased | Team | Used with I2S microphone |
| INMP441 Microphone | Sound sensing (embedded setup) | 1 | Purchased | Team | Connected via I2S |
| USB Microphone (Razer) | Real-time audio input for demo | 1 | Purchased | Zexi | Used for live inference |
| Breadboard & Jumper Wires | Circuit prototyping | 1 set | Purchased | Team | For ESP32 wiring |
| MicroSD Card Module | Local data storage (optional) | 1 | Purchased | Team | SPI interface |
| MicroSD Card | Audio data storage | 1 | Purchased | Team | For recording experiments |
| Fan (test setup) | Simulated HVAC system | 1 | Purchased | Team | Used for controlled experiments |
| Laptop / Workstation | Model inference and visualization | 1 | Purchased | Team | Runs demo and processing |

We use a hybrid sensing setup combining both embedded (ESP32 + INMP441) and external USB microphone pipelines.The embedded system allows exploration of low-cost, deployable sensing solutions, while the USB microphone pipeline enables stable real-time data acquisition for prototyping and demonstration.This dual setup helps us balance system realism and development efficiency.

## Introduction

Heating, Ventilation, and Air Conditioning (HVAC) systems are critical components in modern buildings, and their reliable operation is essential for maintaining indoor comfort, energy efficiency, and system safety. Early detection of abnormal operating conditions in HVAC fans can help prevent system failures and reduce maintenance costs. Traditional monitoring approaches often rely on vibration or temperature sensors. While effective, these methods typically require physical installation on equipment, leading to higher deployment and maintenance costs and making large-scale deployment more challenging. In contrast, acoustic sensing provides a non-invasive and potentially lower-cost alternative. However, most existing work focuses primarily on model performance using pre-recorded datasets and does not fully address challenges related to real-world deployment, such as data acquisition, noise robustness, and real-time inference.

To address these limitations, this project adopts an end-to-end system perspective for HVAC fan anomaly detection, integrating three key components: (1) acoustic data acquisition, (2) machine learning-based anomaly detection, and (3) real-time interactive deployment. On the sensing side, we explore both embedded (ESP32 with I2S microphone) and external USB microphone setups to collect real-world audio under controlled fan conditions. On the modeling side, we implement reconstruction-based anomaly detection using both a baseline Dense Autoencoder (DenseAE) and an enhanced Residual Autoencoder (ResidualAE), trained on the MIMII dataset and adapted to our own collected data. On the deployment side, we develop an interactive Gradio-based interface that supports both offline audio analysis and real-time microphone streaming with window-based inference.

Unlike prior work that focuses primarily on either sensing or modeling in isolation, our approach emphasizes the integration of low-cost sensing, machine learning, and real-time system deployment. This enables us to not only evaluate model performance but also investigate practical challenges such as signal stability, noise interference, and system responsiveness in realistic environments.

### Comparison with Existing Approaches

| Approach Type | Typical Methods | Limitations | Our Approach |
|--------------|----------------|------------|--------------|
| Traditional sensor-based monitoring | Vibration sensors, temperature sensors | Expensive, invasive, difficult to scale | Low-cost, non-invasive acoustic sensing |
| Offline acoustic anomaly detection | Autoencoder-based models on pre-recorded data (e.g., MIMII) | No real-time capability, limited deployment relevance | Supports both offline and real-time inference |
| Our system | Acoustic sensing + ML models + live demo | Further alignment and calibration are required | End-to-end, low-cost, real-time anomaly detection system |

In summary, this project explores the feasibility of using low-cost acoustic sensing combined with machine learning for real-time HVAC fan anomaly detection, with a particular focus on bridging the gap between model development and practical system deployment. In addition, differences between embedded sensing (ESP32-based) and USB microphone setups may introduce variations in signal quality, which could affect model performance and require further calibration.

## Project Objectives

The main objectives of this project are:

1. To develop a reconstruction-based acoustic anomaly detection model capable of identifying abnormal HVAC fan behavior.

2. To build an end-to-end pipeline that integrates data acquisition, model inference, and real-time processing.

3. To implement an interactive demo system that supports both offline audio analysis and real-time microphone-based anomaly detection.

4. To evaluate the system under controlled experimental conditions and assess its feasibility for real-world deployment.

## Team Members

- Tanghao Chen — tanghaoc@andrew.cmu.edu
- Yizhen Xu — yizhenxu@andrew.cmu.edu
- Zexi Yin — zexiyin@andrew.cmu.edu

## Project Roadmap

The project is currently organized into two stages.

### Stage 1: Baseline System

The current repository focuses on a baseline anomaly detection workflow:

- audio preprocessing and log-mel feature extraction
- reconstruction-based anomaly scoring
- threshold-based decision logic
- multi-model inference through a Gradio demo
- live microphone streaming for real-time analysis
- real-world audio recording through a microphone utility

This stage establishes the end-to-end pipeline and provides a deployable demonstration environment for testing uploaded recordings and live audio input.

### Stage 2: Fine-Tuning for the Real Fan Setup

The next stage of the project is to adapt the anomaly detector to our own experimental fan hardware.

The planned fine-tuning workflow is:

1. Collect normal audio recordings from our small fan under real operating conditions.
2. Use these normal recordings to continue training the existing reconstruction model for a small number of epochs.
3. Preserve previously learned generic acoustic structure while improving adaptation to the target device and environment.
4. Continue using reconstruction error as the anomaly score during deployment.

The purpose of this stage is to reduce the mismatch between the baseline training domain and the actual sound characteristics of our real fan setup.

## Methodology

### Audio Preprocessing and Feature Extraction

The input audio is first loaded as a mono waveform and resampled to 16 kHz. This standardizes the input format and reduces unnecessary bandwidth while preserving the dominant frequency content relevant to fan anomaly detection.

Short-Time Fourier Transform (STFT) is then applied using:

- `n_fft = 1024`
- `hop_length = 512`

This converts the waveform into a time-frequency representation with overlapping analysis windows, allowing the model to capture local spectral structure over time.

The linear-frequency spectrum is then mapped into a 64-band mel representation. This step reduces the feature dimension and produces a compact spectral description suitable for downstream learning.

Next, the mel spectrogram is converted into log-mel energy:

`LogMel = (20 / power) * log10(Mel + epsilon)`

The logarithmic transform compresses the dynamic range of the signal and makes the feature scale more suitable for neural-network-based learning.

### Frame Stacking

Rather than using a single spectral frame in isolation, the system stacks 5 consecutive frames to form one input sample. With 64 mel coefficients per frame, this produces a 320-dimensional input vector:

- 64 mel coefficients × 5 frames = 320 dimensions

This allows the model to use short-term temporal context instead of relying only on an instantaneous spectral snapshot.

### Vector Sequence Generation

For each audio clip, the stacked-frame procedure is applied in a sliding-window manner across the full signal. As a result, a single audio file is transformed into a sequence of 320-dimensional feature vectors.

During inference, reconstruction error is computed for each vector, and the frame-level errors are aggregated into a file-level anomaly score. This improves robustness by reducing the effect of isolated local fluctuations.

## Current Baseline Model

The current implementation uses a fully connected autoencoder. The encoder compresses the 320-dimensional input into a low-dimensional latent representation, and the decoder reconstructs the original feature vector.

Current architecture:

- Input: 320
- Encoder: 320 -> 64 -> 64 -> 8
- Decoder: 8 -> 64 -> 64 -> 320

ReLU is used in the hidden layers to provide nonlinear representation capacity, while the output layer remains linear so that the reconstructed vector can match the continuous numerical range of the log-mel input.

The bottleneck layer contains only 8 dimensions. This narrow latent space forces the model to retain only the most salient structure of normal acoustic patterns. As a result, abnormal sounds are generally harder to reconstruct accurately.

The current training data for the baseline model are based on the **fan** audio subset provided in the MIMII baseline dataset repository:

- https://github.com/MIMII-hitachi/mimii_baseline/blob/master/dataset/7z.sh

The baseline implementation is referenced from:

- DOI: `10.5281/zenodo.7551260`

## Fine-Tuning Plan

In the next phase, we plan to fine-tune the reconstruction model using normal audio collected from our own fan setup.

The key idea is to adapt the model from the baseline domain to the target deployment domain while preserving the reconstruction-based anomaly detection principle.

The planned fine-tuning strategy includes:

- collecting only normal audio from the real small fan
- continuing training for a limited number of epochs
- preserving lower-level learned acoustic structure as much as possible
- using reconstruction error as the final anomaly indicator

This design is motivated by the fact that anomaly detection in our setting is based primarily on modeling normal behavior. By refining the model on real normal recordings, we expect the system to better reconstruct target-domain normal sounds and become more sensitive to deviations caused by abnormal conditions.

## Anomaly Score and Thresholding

The current system uses mean squared error (MSE) between the input feature vector and the reconstructed output as the anomaly score. This serves two purposes:

1. As the reconstruction loss during training.
2. As the anomaly score during inference.

A lower reconstruction error indicates that the input is more consistent with the learned normal pattern, while a higher reconstruction error suggests abnormality.

For a given audio file, the system first computes frame-level MSE values and then averages them to obtain a file-level anomaly score. The final decision is made by comparing this score with a model-specific threshold.

The current demo uses fixed thresholds for the four machine IDs:

- ID 00: 6.9
- ID 02: 6.0
- ID 04: 5.3
- ID 06: 7.0

If the file-level MSE is less than or equal to the threshold, the sample is classified as normal. Otherwise, it is classified as anomalous.

In the fine-tuning stage, these thresholds may be re-estimated based on the score distribution of the adapted model on validation recordings from the target setup.

## Experimental Design

We design a series of controlled and real-world experiments to evaluate the effectiveness of our audio-based anomaly detection system for HVAC fan monitoring.

### Data Sources
- Public dataset: MIMII fan dataset for baseline model training
- Real-world data:
- ESP32 + INMP441 microphone recordings
- Smartphone-recorded audio under different operating conditions

### Experimental Conditions
We consider multiple operating scenarios to reflect real-world variability:
- Quiet environment (background noise only)
- Fan under constant voltage (stable operation)
- Fan under varying voltage (dynamic behavior)
- Physically disturbed condition (partially blocked airflow)

### Variables
The experiments vary across:
- Operating conditions (voltage, noise, obstruction)
- Model architecture (Dense Autoencoder vs Residual Autoencoder)
- Signal processing strategy (raw vs window-based features)

### Experimental Procedure
- Audio signals are processed into log-mel spectrogram features
- Frame stacking is applied to capture temporal context
- Reconstruction-based models are trained on normal data
- Anomaly scores are computed using reconstruction error (MSE)

### Baselines
- MIMII baseline autoencoder model
- Comparison between DenseAE and ResidualAE architectures
- Offline audio inference vs real-time streaming inference

### Evaluation Strategy
- Statistical analysis (mean, variance, distribution)
- Time-series behavior (rolling mean trends)
- Qualitative separability between operating conditions
- Real-time inference stability and responsiveness

---

For detailed experimental procedures, weekly progress, and real-world deployment setups, please refer to our project website:
https://diosstrike.github.io/CEE-12770-Team5/

## Demo Application

This repository includes a Gradio-based demo for interactive anomaly diagnosis.

### Demo Functions

The current demo supports two analysis modes:

#### 1. Uploaded Audio Analysis

- Accepts an uploaded audio clip as input.
- Applies the same preprocessing pipeline used by the baseline model.
- Loads trained checkpoints for four machine IDs.
- Computes reconstruction error for each model.
- Compares each score with its corresponding threshold.
- Displays the final normal or anomaly decision together with the MSE value and threshold.

#### 2. Live Microphone Analysis

- Accepts live microphone input directly from the browser.
- Continuously receives streaming audio chunks during recording.
- Maintains a rolling buffer containing the most recent 10 seconds of audio.
- Starts producing predictions once at least 2 seconds of audio have been accumulated.
- Updates the model results every 1 second during live recording.
- Applies the same feature extraction and four-model inference pipeline used in offline analysis.

In live mode, the backend does not analyze each tiny chunk independently. Instead, it continuously aggregates incoming audio into a rolling window and repeatedly runs anomaly detection on the buffered waveform. This design provides a more stable and interpretable form of real-time detection.

The current interface provides a simple and interpretable way to test the anomaly detection pipeline in a browser environment.

In future updates, the demo may also be extended to support inference with the fine-tuned model adapted to the real fan setup.

## Real-World Microphone Workflow

In addition to uploaded audio analysis, the current system now supports direct live microphone streaming through the Gradio interface.

### Live Microphone Workflow

1. Open the **Live Microphone** tab in the demo.
2. Start recording through the browser microphone input.
3. Let the system accumulate audio in a rolling 10-second buffer.
4. After at least 2 seconds of signal have been collected, the backend begins inference.
5. The buffered waveform is converted into log-mel stacked features and passed through the four trained autoencoder models.
6. The predictions are refreshed every 1 second while recording continues.

This allows the system to perform rolling-window real-time anomaly detection instead of relying only on file-based interaction.

### Standalone Recording Utility

The repository also includes a standalone microphone recording script for collecting real-world audio signals.

#### Recording Utility Functions

- Records audio from a selected microphone input device.
- Uses a sampling rate of 16 kHz and a single audio channel.
- Visualizes the waveform in real time during recording.
- Saves the final recording as a `.wav` file.
- Displays the complete waveform after recording.
- Displays the spectrogram for qualitative inspection.

This standalone utility remains useful for debugging, inspecting audio quality, and collecting real-world recordings that may later be used for fine-tuning or offline testing.

## Repository Structure

## Repository Structure

```text
.
├── code/                       # Source code for the project
│   ├── .gradio/flagged/        # Gradio flagged data/logs
│   ├── app.py                  # Gradio web demo application for real-time and offline analysis
│   ├── baseline_runner.ipynb   # Jupyter notebook for running and evaluating baseline models
│   └── data/                   # Directory for storing audio datasets and samples
├── docs/                       # Project documentation and Assignment deliverables
│   ├── images/                 # Image assets (e.g., hardware receipts, figures)
│   ├── experimental_plan.md    # Task 5: Detailed experimental validation plan
│   ├── future_weekly_plan.md   # Task 6: Week-by-week timeline to Demo Day
│   ├── hardware.md             # Task 2: Hardware list and reimbursement statements
│   └── index.md                # GitHub Pages entry point (contains Task 7 learnings summary)
├── report/                     # Task 3 & 4: Final project report templates and placeholders
├── .gitignore                  # Git ignore rules
├── .python-version             # Python version specification
├── README.md                   # Project overview and instructions
├── pyproject.toml              # Task 1: uv project configuration and dependencies
└── uv.lock                     # Task 1: uv lockfile for exact dependency versions
