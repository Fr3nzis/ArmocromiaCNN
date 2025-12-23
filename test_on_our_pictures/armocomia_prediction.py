import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

import torch
from PIL import Image

from src.Network.model import CNNMultiTask
from src.train import get_device
from dataset.labeled_dataset import (
    idx_to_season,
    idx_to_subtype,
    get_default_transforms
)
from dataset.preprocess import process_single_image


BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "our_pics")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_pic")
MODEL_PATH = os.path.join(PROJECT_ROOT, "modello_migliore.pth")
OUTPUT_TXT = os.path.join(BASE_DIR, "predictions.txt")



def preprocess_our_pictures(in_dir=INPUT_DIR, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(in_dir):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        process_single_image(
            input_dir=in_dir,
            output_dir=out_dir,
            filename=file
        )


def load_model(model_path):
    device = get_device()
    model = CNNMultiTask()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def predict_image_with_model(image_path, model, device):
    img = Image.open(image_path).convert("RGB")

    transform = get_default_transforms(train=False)
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_season, out_subtype = model(img_t)

        season_idx = out_season.argmax(dim=1).item()
        subtype_idx = out_subtype.argmax(dim=1).item()

    season = idx_to_season[season_idx]
    subtype = idx_to_subtype[subtype_idx]

    return season, subtype


def predict_and_save(
    images_dir=OUTPUT_DIR,
    output_txt_path=OUTPUT_TXT,
    model_path=MODEL_PATH
):
    model, device = load_model(model_path)
    results = []

    for file in sorted(os.listdir(images_dir)):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(images_dir, file)

        season, subtype = predict_image_with_model(
            img_path, model, device
        )
        results.append((file, season, subtype))
        print(f"{file}: {season}, {subtype}")

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("filename,season,subtype\n")
        for file, season, subtype in results:
            f.write(f"{file},{season},{subtype}\n")

    print(f"\nRisultati salvati in: {output_txt_path}")


if __name__ == "__main__":
    preprocess_our_pictures()
    predict_and_save()