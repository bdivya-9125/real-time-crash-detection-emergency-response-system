Real-Time Crash Detection and Emergency Response System

This project implements a complete AI-driven accident detection and emergency alert system using CCTV footage. A ResNet50 deep learning model identifies accident frames with high accuracy, while a MATLAB-based communication system ensures reliable transmission of emergency alerts using adaptive modulation and Unequal Error Protection (UEP).
A WhatsApp-based alert mechanism automatically notifies the nearest hospitals and includes a fail-safe escalation logic to guarantee timely medical response.

🚀 Overview

Detects accidents in CCTV images using ResNet50 (Transfer Learning)

Achieves 94.9% validation accuracy

MATLAB-based simulation for reliable alert transmission

Uses Reed–Solomon coding and 16-QAM modulation

Sends real-time WhatsApp alerts using PyWhatKit

Intelligent hospital confirmation + escalation workflow

Suitable for smart-city traffic management and emergency response

🧠 Accident Detection Model

Model: ResNet50 (ImageNet pretrained, fine-tuned)
Dataset: Accident Detection from CCTV Footage
Preprocessing: resizing to 224×224, normalization, augmentation

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

Confusion matrix and testing outputs are available in the results/ folder.

📡 MATLAB-Based Emergency Alert Transmission

After detecting an accident, MATLAB simulates the reliability of the communication channel used to transmit emergency alerts.

✔ Techniques Used

16-QAM modulation

AWGN channel

Reed–Solomon UEP coding:

High Priority → RS(255,223)

Medium Priority → RS(127,111)

Low Priority → RS(63,55)

✔ Key Results

BER decreases from 2.6×10⁻¹ → 1×10⁻⁴ as SNR increases (0–20 dB)

High-priority alerts deliver nearly 100% accuracy for SNR > 18 dB

Ensures robust emergency communication in noisy environments

Plots are included in the results/ folder.

📲 WhatsApp Emergency Alert System

Real-time emergency alerts are sent to hospitals via PyWhatKit with:

Accident coordinates

Detection timestamp

Distance to each hospital

Alert category

✔ Escalation Logic

Alerts are first sent to the three nearest hospitals

System waits for confirmation

If one hospital confirms →

Remaining hospitals receive a cancellation message

If none confirm →

Alert automatically escalates to a backup hospital

Screenshots of alert messages are included in results/alert_screenshots/.

🔄 End-to-End Pipeline

CCTV frames are extracted

ResNet50 model detects accident frames

MATLAB simulates alert transmission under noise

Python script sends WhatsApp alerts

Confirmation or escalation ensures guaranteed emergency dispatch

🛠️ Technologies Used

Python: TensorFlow, Keras, OpenCV, PyWhatKit, Geopy

MATLAB: Reed–Solomon coding, UEP, 16-QAM, AWGN

Google Colab: GPU-accelerated training

GitHub: Documentation & version control

▶️ How to Run
1. Accident Detection Model
cd model_training
jupyter notebook resnet50_training.ipynb

2. MATLAB Communication Simulation
matlab_simulation.m

3. WhatsApp Alert Module
python whatsapp_alert.py

📄 Full Report

📘 report/dcs_final_report.pdf

👥 Contributors

Aishwarya S

Bojja Divya

Swetha CA

Dr. Nirmala Paramanantham (Supervisor)

📜 License

This project is licensed under the MIT License.
