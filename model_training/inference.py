# ===============================
# Imports
# ===============================
from google.colab import files
import cv2
from PIL import Image
import torch
from torchvision import transforms, models
import matplotlib.pyplot as plt

# ===============================
# Device setup
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# Validation transforms
# ===============================
val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===============================
# Load model and classes
# ===============================
classes = ['Accident', 'Non-Accident']

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model = model.to(device)
model.load_state_dict(torch.load('best_accident_model.pth', map_location=device))
model.eval()

# ===============================
# Upload video manually
# ===============================
uploaded = files.upload()  # Choose your video file
for filename in uploaded.keys():
    video_path = filename
    print(f"Uploaded video: {video_path}")

# ===============================
# Open video and process frames
# ===============================
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL for model
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_tensor = val_transform(frame_pil).unsqueeze(0).to(device)

    # Predict class
    with torch.no_grad():
        output = model(frame_tensor)
        _, predicted = torch.max(output, 1)
        class_name = classes[predicted.item()]

    # Display frame with predicted class
    plt.figure(figsize=(6,4))
    plt.imshow(frame_pil)
    plt.title(f"Frame {frame_idx} → Predicted: {class_name}", fontsize=14)
    plt.axis('off')
    plt.show()

    frame_idx += 1

cap.release()
print("✅ All frames processed and displayed with predicted classes.")
