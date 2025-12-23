import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

TARGET_BG = (100, 100, 100)
FEATHER = 15

session = new_session("u2net")


def segment_and_clean(image_bgr):
    h, w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)

    cutout = remove(pil_img, session=session)
    cutout = np.array(cutout)

    if cutout.shape[2] < 4:
        return None

    rgb = cutout[:, :, :3]
    alpha = cutout[:, :, 3].astype(np.float32) / 255.0

    if np.max(alpha) < 0.05:
        return None

    alpha = cv2.GaussianBlur(alpha, (FEATHER, FEATHER), 0)
    alpha_3 = np.stack([alpha] * 3, axis=-1)

    bg = np.full((h, w, 3), TARGET_BG, dtype=np.uint8)

    out_rgb = (rgb * alpha_3 + bg * (1 - alpha_3)).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    return out_bgr


def process_dataset(src_root, dst_root):
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            src_path = os.path.join(root, file)
            rel = os.path.relpath(root, src_root)
            out_dir = os.path.join(dst_root, rel)
            os.makedirs(out_dir, exist_ok=True)
            dst_path = os.path.join(out_dir, file)

            img = cv2.imread(src_path)
            if img is None:
                continue

            clean = segment_and_clean(img)
            if clean is None:
                continue

            cv2.imwrite(dst_path, clean)


def process_single_image(input_dir, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)

    src_path = os.path.join(input_dir, filename)
    dst_path = os.path.join(output_dir, filename)

    img = cv2.imread(src_path)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata: {src_path}")

    clean = segment_and_clean(img)
    if clean is None:
        raise RuntimeError("Segmentazione fallita")

    cv2.imwrite(dst_path, clean)
    print(f"Salvata: {dst_path}")


if __name__ == "__main__":

    process_dataset(
        src_root="data_raw/train",
        dst_root="data_blurred/train"
    )

    process_dataset(
        src_root="data_raw/val",
        dst_root="data_blurred/val"
    )

    process_dataset(
        src_root="data_raw/test",
        dst_root="data_blurred/test"
    )

    print("\nFINITO! Il dataset preprocessato si trova in data_blurred/")