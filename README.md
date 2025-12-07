# Real-Time Crash Detection and Emergency Response System

This project implements an AI-driven accident detection and automated emergency alert system using CCTV footage.  
A ResNet50 deep learning model identifies accident frames with high accuracy, while a MATLAB-based communication system ensures reliable delivery of alerts using adaptive modulation and Unequal Error Protection (UEP).  
A WhatsApp-based alert mechanism automatically notifies the nearest hospitals with an escalation logic that guarantees timely medical response.

---

## Overview

- Accident detection using ResNet50 (Transfer Learning)
- 94.9% validation accuracy
- MATLAB simulation for channel reliability analysis
- Reed–Solomon coding + 16-QAM modulation
- Automated WhatsApp emergency alerts
- Hospital confirmation + escalation workflow
- Designed for smart-city accident monitoring

---

## Accident Detection Model

**Model:** ResNet50 (ImageNet pretrained, fine-tuned)  
**Dataset:** Accident Detection from CCTV Footage  
**Preprocessing:** 224×224 resizing, normalization, augmentation  

### Training Setup
- Epochs: 35  
- Batch Size: 16  
- Optimizer: Adam (LR = 5e-5)  
- Loss: Cross-Entropy  
- Scheduler: ReduceLROnPlateau  

### Performance
| Metric     | Score |
|------------|--------|
| Accuracy   | 94.9% |
| Precision  | 93%   |
| Recall     | 93%   |
| F1 Score   | 93%   |

Testing outputs and confusion matrix are available in the results folder.

---

## MATLAB-Based Emergency Alert Transmission

MATLAB simulates the reliability of alert transmission under noisy channel conditions.

### Techniques Used
- 16-QAM modulation  
- AWGN channel  
- Reed–Solomon UEP Coding:  
  - High Priority: RS(255,223)  
  - Medium Priority: RS(127,111)  
  - Low Priority: RS(63,55)

### Key Results
- BER decreases from 2.6×10⁻¹ to 1×10⁻⁴ as SNR increases (0–20 dB)
- High-priority alerts deliver near-perfect reliability for SNR > 18 dB
- Robust communication even under high noise

Related plots are included in the results folder.

---

## WhatsApp Emergency Alert System

Emergency alerts include:
- Accident coordinates  
- Distance to each hospital  
- Time of detection  
- Emergency category  

### Escalation Workflow
1. Alerts sent to the three nearest hospitals  
2. System waits for confirmation  
3. If a hospital accepts the case:
   - Remaining hospitals receive a cancellation message  
4. If no hospital confirms:
   - Alert escalates to a backup (fourth) hospital  

Screenshots of alert stages are included in the results folder.

---

## End-to-End Pipeline

1. Extract frames from CCTV footage  
2. ResNet50 detects accident frames  
3. MATLAB simulates alert transmission  
4. Python script sends WhatsApp alerts  
5. Confirmation or escalation ensures guaranteed medical dispatch  

---

## How to Run

### Accident Detection Model
```bash
cd model_training
jupyter notebook resnet50_training.ipynb
