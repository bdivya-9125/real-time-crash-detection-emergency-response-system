# real-time-crash-detection-emergency-response-system
Real-Time Crash Detection and Emergency Response System

This project implements a complete AI-driven accident detection and emergency alert system using CCTV footage. A ResNet50 deep learning model identifies accident frames with high accuracy, while a MATLAB-based communication system guarantees reliable transmission of emergency alerts using adaptive modulation and Unequal Error Protection (UEP).
A WhatsApp-based alert mechanism automatically notifies the nearest hospitals and includes a fail-safe escalation logic to ensure timely medical response.

🚀 Overview

Detects accidents in CCTV images using ResNet50 (Transfer Learning)

Achieves 94.9% validation accuracy

Simulates communication reliability using MATLAB

Uses Reed–Solomon coding and 16-QAM for robust alert transmission

Sends real-time WhatsApp alerts using PyWhatKit

Intelligent hospital confirmation + escalation workflow

Designed for smart-city accident monitoring and emergency response

📂 Project Structure
├── model_training/
│   ├── resnet50_training.ipynb
│   ├── preprocessing.py
│   └── sample_predictions/
│
├── alert_system/
│   ├── matlab_simulation.m
│   ├── whatsapp_alert.py
│   ├── hospitals.geojson
│
├── results/
│   ├── confusion_matrix.png
│   ├── snr_ber_plot.png
│   ├── delivery_probability_plot.png
│   └── alert_screenshots/
│
├── report/
│   └── dcs_final_report.pdf
│
└── README.md

🧠 Accident Detection Model

Model: ResNet50 (ImageNet pretrained, fine-tuned)
Dataset: Accident Detection from CCTV Footage
Preprocessing: 224×224 resizing, normalization, augmentation

Training Setup

Epochs: 35

Batch Size: 16

Optimizer: Adam (LR = 5×10⁻⁵)

Loss: Cross-Entropy

Scheduler: ReduceLROnPlateau

✔ Performance
Metric	Score
Accuracy	94.9%
Precision	93%
Recall	93%
F1 Score	93%

Confusion matrix and outputs are available in the results/ folder.

📡 MATLAB-Based Emergency Alert Transmission

Once an accident is detected, MATLAB simulates the alert transmission through noisy channels to ensure message reliability.

✔ Techniques Used

16-QAM Modulation

AWGN Channel Simulation

Unequal Error Protection (RS Coding)

High Priority: RS(255,223)

Medium Priority: RS(127,111)

Low Priority: RS(63,55)

✔ Key Results

BER decreases from 2.6×10⁻¹ → 1×10⁻⁴ as SNR increases from 0 → 20 dB

High-priority alerts achieve near-perfect delivery for SNR > 18 dB

Ensures robust emergency communication even under noisy conditions

Plots are included in the results/ folder.

📲 WhatsApp Emergency Alert System

Uses PyWhatKit to instantly notify nearby hospitals with:

Accident coordinates

Distance to the accident location

Accident detection time

Alert category

✔ Escalation Logic

Alerts are sent to the three nearest hospitals

System waits for confirmation

If one hospital confirms:

Other hospitals receive a cancellation message

If none confirm:

Alert automatically escalates to Backup Hospital 4

Screenshots of all alert stages are located in results/alert_screenshots/.

🔄 End-to-End Pipeline

Extract frames from CCTV footage

ResNet50 detects accident frames

MATLAB simulates communication channel + UEP transmission

Python script sends WhatsApp alerts

Confirmation or escalation ensures guaranteed medical dispatch

🛠️ Technologies Used

Python: TensorFlow, Keras, OpenCV, PyWhatKit, Geopy

MATLAB: UEP, Reed–Solomon codes, 16-QAM modulation

Google Colab: GPU-accelerated training

GitHub: Documentation & version control

▶️ How to Run
1. Accident Detection Model
cd model_training
jupyter notebook resnet50_training.ipynb

2. MATLAB Communication Simulation

Open MATLAB and run:

matlab_simulation.m

3. WhatsApp Alert Module
python whatsapp_alert.py

📄 Full Report

The detailed project report is available at:

📘 report/dcs_final_report.pdf

👥 Contributors

Aishwarya S

Bojja Divya

Swetha CA

Dr. Nirmala Paramanantham (Supervisor)

📜 License

This project is licensed under the MIT License.
