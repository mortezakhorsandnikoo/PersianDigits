"""Run the released Persian digit CNN on your own image(s).

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/folder_of_images/

Loads model/persian_digit_cnn.pt and model/preprocessing.json, applies the same
preprocessing used in training, and prints the predicted digit and confidence.
"""
import sys, os, json, glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import DigitCNN

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")


def load_model():
    with open(os.path.join(MODEL_DIR, "preprocessing.json")) as f:
        meta = json.load(f)
    model = DigitCNN(image_size=meta["image_size"],
                     n_classes=meta["n_classes"],
                     dropout=meta.get("dropout", 0.3))
    state = torch.load(os.path.join(MODEL_DIR, "persian_digit_cnn.pt"),
                       map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta


def preprocess(path, meta):
    size = meta["image_size"]
    img = Image.open(path).convert("L").resize((size, size))   # grayscale, single channel
    arr = np.asarray(img, dtype=np.float32) / 255.0            # [0,1]
    arr = (arr - meta["norm_mean"]) / meta["norm_std"]         # dataset-specific standardization
    return torch.from_numpy(arr)[None, None, :, :].float()     # (1,1,H,W)


def predict_path(model, meta, path):
    with torch.no_grad():
        logits = model(preprocess(path, meta))
        prob = F.softmax(logits, dim=1)[0]
        pred = int(prob.argmax())
    return pred, float(prob[pred])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    model, meta = load_model()
    target = sys.argv[1]
    if os.path.isdir(target):
        files = sorted(sum([glob.glob(os.path.join(target, e))
                            for e in ("*.jpg", "*.jpeg", "*.png", "*.bmp")], []))
    else:
        files = [target]
    for f in files:
        pred, conf = predict_path(model, meta, f)
        print(f"{os.path.basename(f):30s} -> digit {pred}  (confidence {conf*100:.1f}%)")
