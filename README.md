# Soil Invertebrate Vibroacoustic Classification (CNN)

Machine-learning pipeline for classifying soil invertebrate vibroacoustic
recordings — **ants, termites, and beetle larvae** — from an ex-situ controlled
chamber study, using convolutional neural networks on mel-spectrograms.

This repository contains the code and documentation for the machine-learning
component of a PhD research project on soil invertebrate bioacoustics
(collaboration with the University of Liverpool and Natural State, Kenya). It
covers the full pipeline from raw audio to a trained classifier, and a
cross-season generalisation study across two field campaigns.

> **Status:** research in progress; results are preliminary and part of an
> unpublished study. Please do not redistribute data or results without the
> collaborators' consent.

---

## Overview

The goal is to identify which invertebrate taxon is present in a recording of
substrate vibrations. Recordings were collected in a controlled chamber setup
over two field seasons (2024 and 2025). The pipeline:

1. **Inventory & cleaning** — organise recordings, remove non-representative
   files, verify data integrity.
2. **Preprocessing** — bandpass filtering, windowing, and conversion to
   mel-spectrograms.
3. **Modelling** — a compact VGG-style CNN classifies 5-second windows;
   window predictions are aggregated to a recording-level label.
4. **Evaluation** — recording-level accuracy and macro-F1, with per-class
   analysis.
5. **Cross-season study** — testing how models trained on one field campaign
   generalise to another, and a merged multi-season model.

A full audit trail of every methodological decision and finding is kept in
[`docs/methodology_log.docx`](docs/methodology_log.docx).

---

## Repository structure

```
Soil-invertebrate-vibroacoustic-cnn/
├── README.md
├── .gitignore
├── notebooks/
│   ├── 01_season1_pipeline_v1_v2.ipynb      # inventory → preprocessing → CNN (v1) → edge-trim (v2)
│   ├── 02_notebook_v2_edgetrim.ipynb        # v2 edge-trimming pipeline and diagnostics
│   └── 03_notebook_v3_generalization.ipynb  # season 2, cross-season test, merged model (v3)
├── docs/
│   └── methodology_log.docx                 # full decision/finding log across v1–v3
└── results/                                 # key metrics (JSON/CSV) — no raw data or models
```

> **Note on data and models.** Raw audio, extracted features, embeddings, and
> trained model weights are **not** included in this repository — they are large
> and are managed separately by the project. The `.gitignore` excludes them.
> The notebooks read data from a project Google Drive location; paths near the
> top of each notebook indicate the expected layout.

---

## Pipeline summary

### Data
- Three target taxa (ants, termites, beetle larvae) plus a control condition,
  recorded at two organism densities each.
- Two field seasons collected under identical experimental conditions but with
  different recorder units/firmware — the basis of the generalisation study.

### Preprocessing
- Native sampling rate 48 kHz.
- 4th-order Butterworth bandpass filter, 200 Hz – 12 kHz.
- 5-second analysis windows (50% overlap for training, non-overlapping for
  evaluation).
- Mel-spectrograms: 2048-point FFT, hop 1024, 128 mel bands, absolute-dB scale.
- Global normalisation by default; per-recording normalisation used for the
  cross-season model.

### Model
- Compact VGG-style CNN: four convolutional blocks (32→64→128→256 channels),
  batch normalisation, ReLU, max-pooling; global average pooling; dropout (0.5);
  linear classification head. ~1.17M parameters.
- Trained with class-weighted cross-entropy, Adam, learning-rate scheduling and
  early stopping. Fixed random seed for reproducibility.

### Evaluation
- **Session-level** train/validation/test splits (70/15/15) to prevent leakage.
- Window predictions aggregated to **recording level** by mean class probability.
- Reported with accuracy and **macro-averaged F1** (primary metric, given class
  imbalance), plus per-class metrics and confusion matrices.

---

## Experiments (high level)

| Version | Description |
|---|---|
| **v1** | Baseline pipeline and CNN on the first field season. |
| **v2** | Revised session-edge trimming following domain-expert review; retrained CNN. |
| **v3** | Second field season added; cross-season generalisation test and a merged multi-season model. |

Additional analyses include a comparison against pretrained audio-model
embeddings (transfer learning), an activity/silence-filtering experiment, and a
normalisation comparison. Detailed results and their interpretation are recorded
in the methodology log. (Headline results are withheld here pending publication.)

---

## Reproducing the workflow

The notebooks were developed in Google Colab with GPU acceleration and a
Google Drive–mounted dataset.

1. Open the notebooks in order (`01` → `02` → `03`).
2. Mount Google Drive and set the dataset paths at the top of each notebook.
3. Run cells sequentially. Long-running steps (feature extraction, training)
   write checkpoints to Drive so they can resume after a disconnect.

Key libraries: `numpy`, `pandas`, `librosa`, `soundfile`, `scikit-learn`,
`torch`, `matplotlib`. Training benefits from a high-RAM GPU runtime.

---

## Collaborators

- **Samuel Babalola** — machine-learning development (this repository)
- **Jonathan Timperley** (University of Liverpool) — PhD researcher; study design,
  field data collection, and domain expertise
- **Natural State (Kenya)** — parallel bioacoustic classification workstream
  (BirdNET), for comparison

---

## Citation and use

This is unpublished research forming part of a PhD project. If you wish to use
or reference this work, please contact the collaborators first. A formal
citation will be added upon publication.
