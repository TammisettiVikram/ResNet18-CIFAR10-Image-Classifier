import json
from pathlib import Path
from PIL import Image
import importlib.util
import sys

# Load app.py module
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
spec = importlib.util.spec_from_file_location("app", str(APP_PATH))
app = importlib.util.module_from_spec(spec)
sys.modules["app"] = app
spec.loader.exec_module(app)

pytorch_model = app.pytorch_model
tf_model = app.tf_model
CLASS_NAMES = app.CLASS_NAMES

IMG_DIR = Path("tests/images")
OUT_JSON = Path("tests/batch_results_local.json")

results = []

for img_path in sorted(IMG_DIR.iterdir()):
    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".gif"):
        continue
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open {img_path}: {e}")
        continue

    try:
        pt_probs = pytorch_model.predict_probs(img)
    except Exception as e:
        pt_probs = None
        print(f"PyTorch prediction failed for {img_path}: {e}")

    try:
        tf_probs = tf_model.predict_probs(img, ensemble=True)
    except Exception as e:
        tf_probs = None
        print(f"TensorFlow prediction failed for {img_path}: {e}")

    def top1(probs):
        if probs is None:
            return None, None
        import numpy as np
        idx = int(np.argmax(probs))
        return CLASS_NAMES[idx], float(probs[idx])

    pt_label, pt_conf = top1(pt_probs)
    tf_label, tf_conf = top1(tf_probs)

    results.append({
        "file": str(img_path),
        "pytorch": {"label": pt_label, "conf": pt_conf},
        "tensorflow": {"label": tf_label, "conf": tf_conf},
    })

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("Results:")
for r in results:
    print(f"{Path(r['file']).name}: PyTorch -> {r['pytorch']['label']} ({r['pytorch']['conf']}) | TensorFlow -> {r['tensorflow']['label']} ({r['tensorflow']['conf']})")

print(f"Saved detailed results to {OUT_JSON}")
