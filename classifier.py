# classifier.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import os
import requests
from pathlib import Path

# === CLASS NAMES ===
class_names = [
    'Atopic Dermatitis',
    'Basal Cell Carcinoma',
    'Benign Keratosis-like Lesions',
    'Eczema',
    'Melanocytic Nevi',
    'Melanoma',
    'Psoriasis',
    'Seborrheic Keratoses',
    'Tinea Ringworm Candidiasis',
    'Warts Molluscum'
]

# === DEVICE & TRANSFORM ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === DOWNLOAD MODEL ===
# https://drive.google.com/file/d/12o3b6TCXBnL0dESJOfMl4ntXHvmKEJjL
MODEL_URL = "https://drive.google.com/uc?id=12o3b6TCXBnL0dESJOfMl4ntXHvmKEJjL"  # Replace with real ID
MODEL_LOCAL_PATH = "model.pth"

def download_model():
    if not os.path.exists(MODEL_LOCAL_PATH):
        print("⏬ Downloading model...")
        r = requests.get(MODEL_URL, allow_redirects=True)
        if r.status_code == 200:
            with open(MODEL_LOCAL_PATH, "wb") as f:
                f.write(r.content)
            print("✅ Model downloaded.")
        else:
            raise Exception("Failed to download model.")
    else:
        print("✅ Model already downloaded.")

# === LOAD MODEL ===
def load_model():
    download_model()
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_LOCAL_PATH, map_location=device))
    model.eval()
    model.to(device)
    return model

model = load_model()

# === GradCAM ===
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(
            self.target_layer.register_backward_hook(backward_hook))

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[0, class_idx]
        loss.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()
        cam_np = cv2.resize(cam_np, (224, 224))

        return cam_np

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# === Overlay Heatmap on Image ===
def apply_heatmap_on_image(img_pil, cam_mask, alpha=0.5):
    img_np = np.array(img_pil.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(heatmap, alpha, img_np, 1 - alpha, 0)
    return Image.fromarray(overlayed)

# === Main Prediction Function ===
def predict_with_gradcam(image_pil, save_path="gradcam_result.png"):
    try:
        input_tensor = transform(image_pil).unsqueeze(0).to(device)
        cam = GradCAM(model, target_layer=model.layer4[-1])

        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred_class = torch.max(probs, 0)

        cam_mask = cam.generate_cam(input_tensor, class_idx=pred_class.item())
        cam.clear_hooks()

        heatmap_img = apply_heatmap_on_image(image_pil, cam_mask)
        heatmap_img.save(save_path)

        return class_names[pred_class], confidence.item(), save_path

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return "Error", 0.0, None
