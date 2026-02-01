import json
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import importlib.util
import sys

# Load app.py module
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
spec = importlib.util.spec_from_file_location("app", str(APP_PATH))
app = importlib.util.module_from_spec(spec)
sys.modules["app"] = app
spec.loader.exec_module(app)

tf_model = app.tf_model
CLASS_NAMES = app.CLASS_NAMES

IMG_DIR = Path("tests/images")
TARGETS = ["OIP.jpg", "OIP (3).jpg"]
OUT_JSON = Path("tests/ablation_results.json")

# prepare resample
try:
    resample = Image.Resampling.LANCZOS
except Exception:
    if hasattr(Image, 'LANCZOS'):
        resample = Image.LANCZOS
    elif hasattr(Image, 'ANTIALIAS'):
        resample = Image.ANTIALIAS
    else:
        resample = Image.BICUBIC

scales = {
    'scale_0_1': lambda arr: arr / 255.0,
    'scale_127': lambda arr: (arr / 127.5) - 1.0,
    'scale_255': lambda arr: arr,  # raw
}

variants = []
# variant generators: name, function that returns numpy batch (1,32,32,3)

def fit_32(img):
    im = img.convert('RGB')
    return np.array(ImageOps.fit(im, (32,32), method=resample)).astype('float32')

def resize_32(img):
    im = img.convert('RGB').resize((32,32), resample)
    return np.array(im).astype('float32')

for scale_name in scales.keys():
    variants.append(('fit', scale_name, False))
    variants.append(('resize', scale_name, False))
    variants.append(('fit', scale_name, True))
    variants.append(('resize', scale_name, True))

results = {}
for fname in TARGETS:
    p = IMG_DIR / fname
    if not p.exists():
        print(f"Skipping {fname}: file not found in {IMG_DIR}")
        continue
    img = Image.open(p).convert('RGB')
    results[fname] = {}
    for kind, scale_name, do_flip in variants:
        if kind == 'fit':
            arr = fit_32(img)
        else:
            arr = resize_32(img)
        if do_flip:
            arr = np.array(Image.fromarray(arr.astype('uint8')).transpose(Image.FLIP_LEFT_RIGHT)).astype('float32')
        scaled = scales[scale_name](arr)
        batch = np.expand_dims(scaled, axis=0)
        # run prediction
        probs = tf_model.model.predict(batch, verbose=0)
        probs = np.asarray(probs).squeeze()
        # if logits-like
        s = float(np.sum(probs))
        if not np.isfinite(s) or s <= 0 or abs(s - 1.0) > 1e-3:
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.sum(exp)
        top_idx = int(np.argmax(probs))
        top_label = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx])

        key = f"{kind}|{scale_name}|{'flip' if do_flip else 'noflip'}"
        results[fname][key] = {
            'top_label': top_label,
            'top_conf': top_conf,
            'probs': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        }

# Save results
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

# Print concise summary
for fname, data in results.items():
    print(f"\nImage: {fname}")
    for key, v in data.items():
        print(f"  {key} -> {v['top_label']} ({v['top_conf']:.3f})")

print(f"\nSaved ablation results to {OUT_JSON}")
