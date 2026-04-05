# Experimental Plan

## Overview

This experimental plan describes how we will validate the value of our acoustic anomaly detection system for small fan and HVAC-like conditions. Our goal is to test whether the current reconstruction-based model can distinguish normal and abnormal fan behavior under controlled real-world recording settings, and whether fine-tuning on target-domain normal audio improves performance.

The experiments focus on fan sound recordings collected under a fixed microphone geometry and multiple operating conditions. We will evaluate both offline uploaded-audio inference and the live demo workflow.

## Recording Setup

All controlled fan recording experiments will use the same basic setup:

- microphone distance: **15 cm**
- microphone angle: **45 degrees relative to the fan outlet**
- sampling condition: real-world recording environment
- voltage settings: **4 V, 8 V, 12 V**

This fixed geometry is intended to reduce unnecessary variation across recordings and make the comparison among conditions more consistent.

## Experimental Conditions

We plan to collect recordings under the following conditions.

### 1. Normal Fan Operation

The fan runs normally under each voltage condition:

- 4 V
- 8 V
- 12 V

These recordings will serve as the reference normal operating condition.

### 2. Blocked-Airflow Condition

A board higher than the fan will be placed in front of the airflow outlet to partially obstruct the fan during operation. This condition is intended to simulate disturbed airflow and abnormal acoustic behavior.

This condition will also be tested under:

- 4 V
- 8 V
- 12 V

### 3. Blade-Imbalance Condition

To simulate a faulty or damaged fan, tape will be attached to a single fan blade so that the rotating mass becomes imbalanced during operation. This is intended to create a physically meaningful abnormal condition that produces irregular sound patterns.

This condition will also be tested under:

- 4 V
- 8 V
- 12 V

## Data Collection Plan

For each voltage level and operating condition, we will collect multiple recordings in order to reduce the effect of random noise and improve the reliability of evaluation.

### Planned Recording Groups

The full condition matrix is:

- normal fan at 4 V
- normal fan at 8 V
- normal fan at 12 V
- blocked fan at 4 V
- blocked fan at 8 V
- blocked fan at 12 V
- blade-imbalance fan at 4 V
- blade-imbalance fan at 8 V
- blade-imbalance fan at 12 V

### Planned Number of Runs

For each condition, we plan to collect **at least 5 recordings**.

This gives a minimum total of:

- 9 conditions × 5 recordings = **45 recordings**

If time allows, we will increase the number of runs to improve robustness.

### Recording Length

Each recording will be long enough to support stable model inference and comparison across methods. Since the live system uses a rolling time window, recordings should be sufficiently long to include multiple seconds of stable fan sound.

## Experiments

## Experiment 1: Condition Separability Under Real Fan Recordings

### Goal

To test whether the acoustic characteristics of normal and abnormal fan operation are distinguishable under controlled recording geometry.

### Input Data

Real-world fan recordings collected under:

- three voltage settings
- normal condition
- blocked-airflow condition
- blade-imbalance condition

### Evaluation Idea

We will compare the anomaly scores produced by the model across these conditions and check whether abnormal conditions consistently produce higher scores than normal operation.

## Experiment 2: Voltage Sensitivity

### Goal

To determine whether the system remains stable across different voltage levels and whether the model can still distinguish abnormality when fan intensity changes.

### Input Data

Recordings collected at:

- 4 V
- 8 V
- 12 V

for each operating condition.

### Evaluation Idea

We will compare how anomaly scores vary with voltage and check whether the distinction between normal and abnormal states is preserved across all three voltage settings.

## Experiment 3: Blocked vs. Blade-Imbalance Comparison

### Goal

To compare two different types of abnormal conditions:

- airflow obstruction
- mechanical imbalance caused by tape on one blade

### Evaluation Idea

We will examine whether the model responds differently to these two abnormal patterns and whether both can be separated from normal operation.

## Experiment 4: Baseline Model vs. Fine-Tuned Model

### Goal

To determine whether fine-tuning the baseline model on target-domain normal fan recordings improves real-world performance.

### Models to Compare

- **Baseline model**: the original reconstruction-based model trained from the MIMII fan dataset
- **Fine-tuned model**: the baseline model further adapted using normal recordings collected from our own fan setup

### Evaluation Idea

We will compare the score distributions and classification performance of the two models on the same set of real-world test recordings.

The expectation is that the fine-tuned model will better reconstruct target-domain normal sounds and produce clearer separation between normal and abnormal conditions.

## Experiment 5: Offline Inference vs. Live Demo Consistency

### Goal

To evaluate whether the live microphone demo produces results that are broadly consistent with offline uploaded-audio inference.

### Evaluation Idea

We will record fan sounds and test them in two ways:

- offline uploaded-audio analysis
- live microphone inference through the demo

We will compare whether both workflows produce similar model decisions and score trends under the same physical condition.

## Baselines for Comparison

The main comparison in this project will be between:

1. **Original baseline model**
2. **Fine-tuned target-domain model**

In addition, we will use the following practical reference comparisons:

- normal vs. blocked condition
- normal vs. blade-imbalance condition
- offline uploaded-audio inference vs. live microphone inference

## Metrics

We will use the following metrics to assess experimental value.

### 1. Reconstruction MSE

The primary anomaly score of the system is the mean squared reconstruction error (MSE). This will be used to compare score distributions across conditions.

### 2. Threshold-Based Classification Result

Using the selected thresholds, we will examine whether each recording is classified as normal or anomalous.

### 3. Score Separation Across Conditions

We will compare whether abnormal conditions consistently produce higher scores than normal conditions.

### 4. Qualitative Consistency

We will examine whether:

- similar physical conditions produce similar score patterns
- the live demo agrees with offline analysis
- the fine-tuned model behaves more consistently on our own fan data

## Expected Outcome

We expect the following trends:

- normal fan recordings should produce lower reconstruction errors
- blocked-airflow recordings should produce higher anomaly scores than normal recordings
- blade-imbalance recordings should also produce higher anomaly scores than normal recordings
- the fine-tuned model should improve target-domain normal reconstruction and lead to better separation between normal and abnormal conditions
- the live demo should show broadly consistent behavior with offline uploaded-audio analysis

## Final Objective

The overall objective of these experiments is to demonstrate that the proposed acoustic anomaly detection pipeline is meaningful in real-world fan conditions, that it can capture multiple forms of abnormal behavior, and that model adaptation through fine-tuning can improve performance on our target setup.
