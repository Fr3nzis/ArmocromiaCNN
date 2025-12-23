import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from src.Network.model import CNNMultiTask
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "modello_migliore.pth")
IMAGE_PATH = os.path.join(SCRIPT_DIR, "vincenzo1.jpg")
OUT_PATH = os.path.join(SCRIPT_DIR, "grad_cam_output_vin.jpg")


class SeasonModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.features = base_model.features

    def forward(self, x):
        logits_stagione, _ = self.base(x)
        return logits_stagione


base_model = CNNMultiTask()
base_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
base_model.eval()

model = SeasonModel(base_model)
model.eval()

target_layer = model.features[2]
cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

img_bgr_original = cv2.imread(IMAGE_PATH)
if img_bgr_original is None:
    raise FileNotFoundError(f"Immagine non trovata: {IMAGE_PATH}")

orig_h, orig_w = img_bgr_original.shape[:2]

img_rgb_original = cv2.cvtColor(img_bgr_original, cv2.COLOR_BGR2RGB)
img_rgb_norm_original = img_rgb_original.astype(np.float32) / 255.0

img_resized = cv2.resize(img_rgb_original, (128, 128), interpolation=cv2.INTER_LINEAR)
img_pil = Image.fromarray(img_resized)

transform = transforms.ToTensor()
x = transform(img_pil).unsqueeze(0)

with torch.no_grad():
    logits = model(x)

pred_class = int(torch.argmax(logits, dim=1).item())
targets = [ClassifierOutputTarget(pred_class)]

grayscale_cam = cam(input_tensor=x, targets=targets)[0]

grayscale_cam = cv2.resize(grayscale_cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

heatmap_img = show_cam_on_image(img_rgb_norm_original, grayscale_cam, use_rgb=True)
heatmap_bgr = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR)

cv2.imwrite(OUT_PATH, heatmap_bgr)

print(f"Salvato: {OUT_PATH}")
