# HVAC Motor Anomaly Detection

## Overview

This repository contains the implementation of our course project on HVAC motor anomaly detection using audio-based machine learning. The project aims to identify abnormal operating conditions from motor sound recordings by combining a reconstruction-based anomaly detection pipeline with an interactive Gradio demo and a real-time microphone recording utility.

At the current stage, the repository includes a baseline autoencoder inference system for uploaded audio clips and a standalone microphone-based recording script for collecting real-world sound data. In the next stage of the project, we plan to extend this baseline through fine-tuning on normal recordings collected from our own small fan setup, allowing the model to better adapt to the target acoustic environment.

## Project Objectives

The main objectives of this project are:

1. To detect anomalous HVAC motor behavior from acoustic signals.
2. To implement a baseline reconstruction-based anomaly detection pipeline.
3. To support inference across multiple machine IDs using separately trained models.
4. To provide an interactive web-based interface for testing uploaded audio clips.
5. To support real-world audio collection through a microphone recording utility.
6. To adapt the baseline model to our own physical fan setup through a fine-tuning stage based on normal operating sounds.

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
- real-world audio recording through a microphone utility

This stage establishes the end-to-end pipeline and provides a deployable demonstration environment for testing uploaded recordings.

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

The input audio is first loaded as a mono waveform and resampled to 16 kHz. This standardizes the input format and reduces unnecessary bandwidth while preserving the dominant frequency content relevant to motor anomaly detection.

Short-Time Fourier Transform (STFT) is then applied using:

- `n_fft = 1024`
- `hop_length = 512`

This converts the waveform into a time-frequency representation with overlapping analysis windows, allowing the model to capture local spectral structure over time.

The linear-frequency spectrum is then mapped into a 64-band mel representation. This step reduces the feature dimension and produces a compact spectral description suitable for downstream learning.

Next, the mel spectrogram is converted into log-mel energy:

$\text{LogMel} = \frac{20}{power} \cdot \log_{10}(\text{Mel} + \epsilon)$

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

## Demo Application

This repository includes a Gradio-based demo for interactive anomaly diagnosis.

### Demo Functions

- Accepts an uploaded audio clip as input.
- Applies the same preprocessing pipeline used by the baseline model.
- Loads trained checkpoints for four machine IDs.
- Computes reconstruction error for each model.
- Compares each score with its corresponding threshold.
- Displays the final normal or anomaly decision together with the MSE value and threshold.

The current interface provides a simple and interpretable way to test the anomaly detection pipeline in a browser environment.

In future updates, the demo may also be extended to support inference with the fine-tuned model adapted to the real fan setup.

## Real-World Microphone Workflow

In addition to the Gradio demo, the repository includes a standalone microphone recording script for collecting real-world audio signals.

### Recording Utility Functions

- Records audio from a selected microphone input device.
- Uses a sampling rate of 16 kHz and a single audio channel.
- Visualizes the waveform in real time during recording.
- Saves the final recording as a `.wav` file.
- Displays the complete waveform after recording.
- Displays the spectrogram for qualitative inspection.

### How the Microphone Connects to the Demo

At the current stage, the microphone utility and the Gradio demo are connected through file-based interaction rather than direct live streaming.

The workflow is as follows:

1. Record real-world motor audio using the microphone script.
2. Save the output audio as a `.wav` file.
3. Upload the recorded file into the Gradio demo.
4. Run anomaly detection across the trained models.

This modular design allows us to inspect recordings before sending them into the inference interface.

In the fine-tuning stage, these real-world microphone recordings will also serve as the primary source of target-domain normal data.

## Repository Structure

```text
CEE-12770-Team5/
├── code/                 # Source code for training, inference, and utilities
├── data/                 # Audio data, test samples, or dataset-related files
├── pyproject.toml        # Project metadata and dependency configuration
├── uv.lock               # Locked dependency file for uv
├── .gitignore
└── README.md