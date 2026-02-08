# EEG Motor Imagery Classification (BCI)

This project implements a complete EEG motor imagery pipeline using the EEGBCI dataset and MNE-Python.

## Dataset
- EEGBCI Dataset (PhysioNet)
- Subject 1
- Left vs Right hand motor imagery

## Methods
- Band-pass filtering (8–12 Hz, mu rhythm)
- Epoching and baseline correction
- Time-frequency analysis (Morlet wavelets)
- Common Spatial Patterns (CSP)
- Linear Discriminant Analysis (LDA)
- Cross-validation and sliding window classification

## Requirements
See `requirements.txt`

## How to run
```bash
pip install -r requirements.txt
python main.py
