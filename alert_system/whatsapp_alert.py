import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic
import pywhatkit as kit
import time
from tkinter import Tk, filedialog

# ================= CONFIG =================
MODEL_PATH = r"C:\Users\divya\Downloads\best_accident_model.pth"
HOSPITALS_FILE = r"C:\Users\divya\Downloads\export.geojson"
ACCIDENT_LOCATION = (17.385044, 78.486671)  # Default location

# Map top 4 hospitals to their dedicated phone numbers
HOSPITAL_PHONE_MAPPING = {
    0: "+916302877035",  # Hospital 1
    1: "+917702610993",  # Hospital 2
    2: "+919150301075",  # Hospital 3
    3: "+918015263854"   # Hospital 4 (backup)
}

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ================= MODEL =================
class AccidentModel(nn.Module):
    def __init__(self):
        super(AccidentModel, self).__init__()
        self.model = models.resnet50(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, x):
        return self.model(x)

# Load model
model = AccidentModel().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)
model.eval()
print("[INFO] ✅ Model loaded successfully!")

# ================= LOAD HOSPITAL DATA =================
def load_hospitals(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".geojson":
        return gpd.read_file(file_path)
    else:
        return None

hospitals_df = load_hospitals(HOSPITALS_FILE)

def find_nearest_hospitals(accident_coords, hospital_data, top_n=3):
    hospitals = []
    if isinstance(hospital_data, gpd.GeoDataFrame):
        for _, row in hospital_data.iterrows():
            lat, lon = row.geometry.y, row.geometry.x
            dist = geodesic(accident_coords, (lat, lon)).km
            hospitals.append({"name": row.get("name", "Unknown"), "distance_km": dist})
    else:
        for _, row in hospital_data.iterrows():
            lat, lon = row.get("latitude"), row.get("longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue
            dist = geodesic(accident_coords, (lat, lon)).km
            hospitals.append({"name": row.get("name", "Unknown"), "distance_km": dist})
    hospitals.sort(key=lambda x: x["distance_km"])
    return hospitals[:top_n]

# ================= IMAGE UPLOAD =================
def upload_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Accident Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    return file_path

# ================= ACCIDENT DETECTION & ALERT =================
def detect_accident_and_alert(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("[ERROR] Could not read the image")
        return

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        print("Raw logits:", output)
        pred_index = int(torch.argmax(output, dim=1).item())
        print("Predicted class index:", pred_index)

    class_labels = {0: "No Accident", 1: "Accident"}  # Adjust if needed
    predicted_label = class_labels[pred_index]
    print(f"[INFO] Predicted class: {predicted_label}")

    if predicted_label == "Accident":
        print("[ALERT] 🚨 Accident detected!")
        nearest = find_nearest_hospitals(ACCIDENT_LOCATION, hospitals_df, top_n=3)

        # Step 1: Send accident alert to top 3 hospitals
        for idx, h in enumerate(nearest):
            number = HOSPITAL_PHONE_MAPPING[idx]
            message = (f"⚠ Accident Detected!\n"
                       f"Location: {ACCIDENT_LOCATION}\n"
                       f"Nearest Hospital: {h['name']}\n"
                       f"Distance: {h['distance_km']:.2f} km\n"
                       f"Immediate medical help required!")
            kit.sendwhatmsg_instantly(number, message, wait_time=10, tab_close=True)
            print(f"[SENT] Accident alert sent to {h['name']} ({number})")
            time.sleep(5)

        # Step 2: Wait for hospital confirmations
        service_confirmed = False
        for idx, h in enumerate(nearest):
            response = input(f"Did {h['name']} confirm medical service? (Yes/No): ").strip().lower()
            if response == "yes":
                service_confirmed = True
                print(f"[INFO] Medical service confirmed by {h['name']}.")
                # Notify the other hospitals
                for other_idx, other_h in enumerate(nearest):
                    if other_idx != idx:
                        other_number = HOSPITAL_PHONE_MAPPING[other_idx]
                        info_message = (f"ℹ Medical service already confirmed from another hospital.\n"
                                        f"No need to respond.")
                        kit.sendwhatmsg_instantly(other_number, info_message, wait_time=10, tab_close=True)
                        print(f"[SENT] Info sent to {other_h['name']} ({other_number})")
                break

        # Step 3: If none of the first 3 hospitals confirm, notify 4th hospital
        if not service_confirmed:
            fourth_hospital_number = HOSPITAL_PHONE_MAPPING[3]
            message_4 = (f"⚠ Accident Detected!\n"
                         f"Location: {ACCIDENT_LOCATION}\n"
                         f"Nearest hospitals did not confirm service.\n"
                         f"Immediate medical help required!")
            kit.sendwhatmsg_instantly(fourth_hospital_number, message_4, wait_time=10, tab_close=True)
            print(f"[SENT] Alert sent to 4th hospital ({fourth_hospital_number}) as backup.")
    else:
        print("[INFO] No accident detected.")

# ================= MAIN =================
if __name__== "__main__":
    selected_image = upload_image()
    if selected_image:
        detect_accident_and_alert(selected_image)
    else:
        print("[INFO] No image selected.")




