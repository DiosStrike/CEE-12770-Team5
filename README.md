# HVAC Fan Anomaly Detection
## Week 5 Progress Update (2026-04-16)
- **Task 3 & 4 deliverable (ACM format)**: [`report/12770ProjectProgressReport_Task3_Task4/final_report_v2.tex`](report/12770ProjectProgressReport_Task3_Task4/final_report_v2.tex)
## Week 5 Progress Update (2026-04-13)
This week focused on two major efforts: (1) completing the experimental evaluation pipeline and (2) refining the project's problem statement around **domain shift in acoustic anomaly detection**.
### What We Did
**Cross-Model Transfer Evaluation** — Evaluated all 4 pretrained MIMII baseline autoencoders (id_00, id_02, id_04, id_06) on our 180-sample target dataset. Compared AUC, F1, threshold behavior, and MSE distributions across conditions, voltages, and noise environments.
**Latent Feature Analysis & Multi-Class Classification** — Extracted 8-dim bottleneck features from the best model (id_04), performed t-SNE visualization and per-dimension analysis, and ran 3-class classification (normal / blocked / imbalance) using SVM, Random Forest, and KNN with 5-fold cross-validation.
**Fine-Tuning with Architecture Modification** — Replaced the bottleneck ReLU with LeakyReLU to address dead neurons, and fine-tuned the autoencoder on our 60 normal-condition samples. Re-evaluated all metrics and compared with the frozen baseline.
### Key Results
| Metric | Baseline (frozen) | Fine-tuned | Change |
|--------|-------------------|------------|--------|
| AUC (binary) | 0.769 | **0.997** | +0.228 |
| Best F1 (binary) | 0.846 | **0.984** | +0.138 |
| Optimal threshold | 38.18 | **7.56** | Back to source-domain scale |
| Active latent dims | 2/8 | **4/8** | +2 dims activated |
| 3-class accuracy (RF, 5-fold CV) | 95.0% | 95.0% | Maintained |
### Key Conclusions
1. **Domain shift is extreme but recoverable**: Threshold shifts 5-10x between source and target domain, but fine-tuning restores near-perfect detection.
2. **Source-domain best ≠ transfer best**: id_06 (best on source, F1=0.910) transferred poorly; id_04 (mediocre on source, F1=0.731) transferred best.
3. **Only 2/8 latent dimensions are active** in the pretrained model due to ReLU dead neurons; LeakyReLU activates 2 more.
4. **Multi-class fault diagnosis is feasible** from pretrained latent features without any supervised training (95% CV accuracy).
5. **Latent space encodes hierarchical structure**: condition at the coarse level, voltage at the fine level; noise-robust.
### Where to Find Details
- Full experimental and evaluation report: [`report/12770_final_report/Experimental_And_Evelation_Report.tex`](report/12770_final_report/Experimental_And_Evelation_Report.tex)

- Detailed weekly progress and figures: [Project Website — Week 5](https://diosstrike.github.io/CEE-12770-Team5/)
---

## Repository Structure
```
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
```

## Project Website

Project progress updates are available at:

[https://diosstrike.github.io/CEE-12770-Team5/](https://diosstrike.github.io/CEE-12770-Team5/)


## Hardware and Materials

Our system uses a hybrid sensing setup combining embedded hardware and external audio input devices.

- **Embedded sensing**: ESP32 + INMP441 I2S microphone for low-cost, deployable audio acquisition
- **External microphone**: USB microphone (Razer Seiren Mini) for stable real-time demo input
- **Auxiliary components**: breadboard, jumper wires, fan setup, and optional SD card storage

This hybrid design allows us to balance system realism (embedded sensing) and development stability (USB-based recording for demo and evaluation).

For a detailed list of hardware, including quantities, purposes, and costs, please refer to:

[Hardware and Materials List](docs/hardware.md)

## Introduction

Traditional HVAC condition monitoring relies on vibration or temperature sensors, which require physical installation and per-unit instrumentation that is expensive to scale. Acoustic monitoring offers a non-invasive alternative: a single microphone can capture fan operational sound and detect deviations from normal behavior. The MIMII dataset and the annual DCASE challenges have established autoencoder-based reconstruction error as a baseline for unsupervised anomalous sound detection (ASD).

### The Core Problem: Domain Shift

The central challenge for practical ASD deployment is **domain shift** — when the acoustic characteristics of the deployment environment differ from the training environment, detection performance degrades significantly. While the DCASE community has studied this problem since 2021, existing benchmarks only address **controlled** domain shifts (e.g., changing microphone position or noise level within the same recording setup).

**Our study addresses a far more extreme scenario**: deploying a model trained in one laboratory onto a completely different fan, microphone, and environment. The table below highlights this difference:

| | MIMII Source | DCASE Target | Our Target |
|---|---|---|---|
| Fan | Industrial fan | Same type, different unit | Wathai 120mm AC axial fan |
| Microphone | TAMAGO-03 array, 8-ch | Same | Razer Seiren V3 Mini, 1-ch |
| Mic grade | Research-grade | Same | Consumer USB |
| Noise | Post-hoc mix at exact SNR | Same method, different SNR | Natural ambient (uncontrolled) |
| Environment | Hitachi lab, Japan | Same facility | Different facility |
| **Factors changed** | — | 1–2 | **All simultaneously** |
| **Threshold shift** | — | ~1.5–3x | **~5–10x** |

This cross-hardware, cross-environment domain shift is qualitatively different from — and substantially more severe than — controlled DCASE benchmarks, yet it represents the scenario most likely encountered in practice.

Our long-term goal is to push even further toward edge deployment by replacing the USB microphone with an ESP32 + INMP441 I2S MEMS module, introducing an additional sensor-grade shift on top of the already extreme mismatch studied here.

For the full literature review and domain shift analysis, see the [project report](report/Experimental_And_Evaluation_Report.tex) and the [project website](https://diosstrike.github.io/CEE-12770-Team5/).

### Comparison with Existing Approaches

| Approach Type | Typical Methods | Limitations | Our Approach |
|--------------|----------------|------------|--------------|
| Traditional sensor-based monitoring | Vibration sensors, temperature sensors | Expensive, invasive, difficult to scale | Low-cost, non-invasive acoustic sensing |
| Offline acoustic anomaly detection | Autoencoder-based models on pre-recorded data (e.g., MIMII) | No real-time capability, limited deployment relevance | Supports both offline and real-time inference |
| Our system | Acoustic sensing + ML models + live demo | Further alignment and calibration are required | End-to-end, low-cost, real-time anomaly detection system |

In summary, this project explores the feasibility of using low-cost acoustic sensing combined with machine learning for real-time HVAC fan anomaly detection, with a particular focus on bridging the gap between model development and practical system deployment. In addition, differences between embedded sensing (ESP32-based) and USB microphone setups may introduce variations in signal quality, which could affect model performance and require further calibration.

## Project Objectives

The main objectives of this project are:

1. To evaluate and quantify the impact of extreme cross-hardware domain shift on pretrained acoustic anomaly detection models.

2. To develop domain adaptation strategies (fine-tuning, architecture modification) that recover detection performance under severe domain mismatch.
  
3. To explore multi-class fault diagnosis (normal / blocked / imbalance) using latent representations from the autoencoder bottleneck.

4. To develop a reconstruction-based acoustic anomaly detection model capable of identifying abnormal HVAC fan behavior.

5. To build an end-to-end pipeline that integrates data acquisition, model inference, and real-time processing.

6. To implement an interactive demo system that supports both offline audio analysis and real-time microphone-based anomaly detection.

7. To evaluate the system under controlled experimental conditions and assess its feasibility for real-world deployment.

## Team Members

- Tanghao Chen — tanghaoc@andrew.cmu.edu
- Yizhen Xu — yizhenxu@andrew.cmu.edu
- Zexi Yin — zexiyin@andrew.cmu.edu

## Project Roadmap

The project is organized into three stages.

### Stage 1: Baseline System (Completed)

The current repository includes a baseline anomaly detection workflow:

- audio preprocessing and log-mel feature extraction
- reconstruction-based anomaly scoring
- threshold-based decision logic
- multi-model inference through a Gradio demo
- live microphone streaming for real-time analysis
- real-world audio recording through a microphone utility

### Stage 2: Transfer Evaluation & Fine-Tuning (Current — Week 5)

This stage evaluates and adapts the pretrained models to our target hardware:

- Cross-model transfer evaluation across 4 MIMII pretrained autoencoders
- Source-domain vs. target-domain performance comparison
- Latent feature analysis (t-SNE, per-dimension activation, multi-class classification)
- Architecture modification (ReLU → LeakyReLU at bottleneck)
- Fine-tuning on target-domain normal data (60 samples)
- Comprehensive before/after comparison (AUC: 0.769 → 0.997)

### Stage 3: Multi-Task Learning & Edge Deployment (Planned)

The next stage extends the system toward supervised fault diagnosis and edge deployment:

1. Attach a classification head to the bottleneck layer for end-to-end multi-class training.
2. Joint reconstruction + classification loss to enable dual-purpose anomaly detection and fault diagnosis.
3. Migrate from USB microphone to ESP32 + INMP441 I2S MEMS module for edge deployment.
4. Evaluate the additional sensor-grade domain shift introduced by embedded hardware.

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

**Architecture finding (Week 5)**: Latent analysis revealed that only 2 of 8 bottleneck dimensions are active in the pretrained model due to ReLU-induced dead neurons. Replacing the bottleneck ReLU with LeakyReLU (negative slope = 0.01) and fine-tuning on target-domain data activated 2 additional dimensions, improving anomaly detection AUC from 0.769 to 0.997.

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
**Domain shift finding (Week 5)**: These source-domain thresholds are completely invalid on our target data — all target-domain MSE values far exceed the source thresholds (35–73 vs. 5–7), causing 100% false positive rate. After fine-tuning, the optimal threshold returned to 7.56, matching the source-domain scale and confirming successful domain adaptation.


## Experimental Design

We design a series of controlled and real-world experiments to evaluate the effectiveness of our audio-based anomaly detection system for HVAC fan monitoring.

### Data Sources
- Public dataset: MIMII fan dataset for baseline model training
- Real-world data: 180 recordings collected with USB microphone under controlled conditions
  - 3 conditions (normal, blocked, imbalance) × 3 voltages (4V, 8V, 12V) × 2 noise environments × 10 runs
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
- Source-domain vs. target-domain performance comparison
- Cross-model transfer evaluation (4 pretrained models)
- Threshold-free AUC as primary separability metric
- F1-maximizing threshold search on target data
- Latent feature analysis (t-SNE, per-dimension activation)
- Multi-class classification with cross-validation (SVM, RF, KNN)
- Before/after fine-tuning comparison

---

For a detailed experimental design, including condition setup, number of runs, evaluation metrics, and planned experiments, please refer to:

[Experimental Plan](docs/experimental_plan.md)

For full experimental logs, weekly progress, and real-world deployment details, see:

[Project Website](https://diosstrike.github.io/CEE-12770-Team5/)

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



