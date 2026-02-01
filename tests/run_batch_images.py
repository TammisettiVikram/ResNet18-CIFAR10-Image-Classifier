import os
import json
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# Import models from app (they load on import)
import importlib.util
import sys
from pathlib import Path

# Load app.py as a module to access model wrappers and globals
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
spec = importlib.util.spec_from_file_location("app", str(APP_PATH))
app = importlib.util.module_from_spec(spec)
sys.modules["app"] = app
spec.loader.exec_module(app)

pytorch_model = app.pytorch_model
tf_model = app.tf_model
CLASS_NAMES = app.CLASS_NAMES

OUT_DIR = Path("tests/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# List of test images (public URLs) and expected CIFAR-10 label (if known)
TEST_IMAGES = [
    # (url, expected_label)
    ("https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg", "cat"),
    ("https://upload.wikimedia.org/wikipedia/commons/3/3c/Golden_Retriever_medium-to-light_coat.jpg", "dog"),
    ("https://upload.wikimedia.org/wikipedia/commons/7/7b/Aeroplane_Airbus_A320.jpg", "airplane"),
    ("https://upload.wikimedia.org/wikipedia/commons/6/6b/Truck-tractor-01.jpg", "truck"),
    ("https://upload.wikimedia.org/wikipedia/commons/4/42/Hippopotamus_in_water.jpg", "deer"),
    ("https://upload.wikimedia.org/wikipedia/commons/5/5f/Frog_in_pond.jpg", "frog"),
    ("https://upload.wikimedia.org/wikipedia/commons/1/12/Horse_portrait.jpg", "horse"),
    ("https://upload.wikimedia.org/wikipedia/commons/9/9a/USS_Arizona_%28BB-39%29_sinking.jpg", "ship"),
    ("https://upload.wikimedia.org/wikipedia/commons/7/71/1999df_1981_Volkswagen_Golf_%28front%29.jpg", "automobile"),
    ("https://upload.wikimedia.org/wikipedia/commons/3/32/House_finch_male.jpg", "bird"),
]

results = []

for i, (url, expected) in enumerate(TEST_IMAGES):
    name = f"img_{i}.jpg"
    path = OUT_DIR / name
    print(f"Downloading {url} -> {path} ...")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(path)
    except Exception as e:
        print(f"  failed to download: {e}")
        continue

    # Run predictions
    try:
        pt_probs = pytorch_model.predict_probs(img)
    except Exception as e:
        pt_probs = None
        print("PyTorch prediction failed:", e)

    try:
        tf_probs = tf_model.predict_probs(img, ensemble=True)
    except Exception as e:
        tf_probs = None
        print("TensorFlow prediction failed:", e)

    def top1(probs):
        if probs is None:
            return None, None
        import numpy as np
        idx = int(np.argmax(probs))
        return CLASS_NAMES[idx], float(probs[idx])

    pt_label, pt_conf = top1(pt_probs)
    tf_label, tf_conf = top1(tf_probs)

    ok_pt = (pt_label == expected) if pt_label is not None else False
    ok_tf = (tf_label == expected) if tf_label is not None else False

    results.append({
        "file": str(path),
        "expected": expected,
        "pytorch": {"label": pt_label, "conf": pt_conf, "correct": ok_pt},
        "tensorflow": {"label": tf_label, "conf": tf_conf, "correct": ok_tf},
    })

# Save results
out_json = Path("tests/batch_results.json")
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

# Print concise table
print("\nResults:")
for r in results:
    print(f"{Path(r['file']).name}: expected={r['expected']}")
    print(f"  PyTorch -> {r['pytorch']['label']} ({r['pytorch']['conf']:.3f}) correct={r['pytorch']['correct']}")
    print(f"  TensorFlow -> {r['tensorflow']['label']} ({r['tensorflow']['conf']:.3f}) correct={r['tensorflow']['correct']}")

print(f"\nSaved detailed results to {out_json}")
