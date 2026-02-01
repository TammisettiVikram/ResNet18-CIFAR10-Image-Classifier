import os
import gradio as gr
import torch
import torchvision.transforms as T
import torchvision.models as models
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# -------------------------
# Constants & Paths
# -------------------------
PYTORCH_MODEL_PATH = "models/pytorch/resnet18_cifar10.pt"
TF_MODEL_PATH = "models/tensorflow/cifar10_cnn.keras"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# -------------------------
# PyTorch Model Wrapper
# -------------------------
class PyTorchResNetModel:
    def __init__(self):
        self.model = None
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self):
        try:
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 10)

            state_dict = torch.load(PYTORCH_MODEL_PATH, map_location="cpu")
            model.load_state_dict(state_dict)

            model.eval()
            self.model = model
            print("✅ PyTorch model loaded successfully")

        except Exception as e:
            print("❌ PyTorch model load failed:", e)
            self.model = None

    def predict_top5(self, image: Image.Image):
        if self.model is None:
            raise RuntimeError("PyTorch model not loaded")

        x = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        top5_probs, top5_idx = torch.topk(probs, 5)

        return {
            CLASS_NAMES[top5_idx[i].item()]: float(top5_probs[i].item())
            for i in range(5)
        }
    
    def predict_probs(self, image: Image.Image):
        """Return full-class probabilities as a numpy array (shape: (10,))."""
        if self.model is None:
            raise RuntimeError("PyTorch model not loaded")

        x = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        return probs

# -------------------------
# TensorFlow Model Wrapper
# -------------------------
class TensorFlowCIFAR10Model:
    def __init__(self):
        self.model = None

    def preprocess(self, image: Image.Image, size: int = 32):
        """Preprocess image for CIFAR-10 TF model.
        Uses ImageOps.fit to preserve aspect ratio while resizing and center-cropping.
        Returns a batch array shape (1, size, size, 3) scaled to [0,1].
        Handles Pillow API differences where `ANTIALIAS` was moved/renamed.
        """
        img = image.convert("RGB")
        # Choose a resampling/filter constant compatible across Pillow versions
        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            # Pillow < 9.1 may expose LANCZOS or ANTIALIAS at module level
            if hasattr(Image, "LANCZOS"):
                resample = Image.LANCZOS
            elif hasattr(Image, "ANTIALIAS"):
                resample = Image.ANTIALIAS
            else:
                resample = Image.BICUBIC

        # ImageOps.fit accepts `method` for the resampling filter
        img = ImageOps.fit(img, (size, size), method=resample)
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr

    def load(self):
        try:
            self.model = tf.keras.models.load_model(
                "models/tensorflow/cifar10_cnn.keras"
            )
            print("✅ TensorFlow model loaded successfully")
        except Exception as e:
            print("❌ TensorFlow model load failed:", e)
            self.model = None

    def predict_top5(self, image: Image.Image):
        if self.model is None:
            raise RuntimeError("TensorFlow model not loaded")

        try:
            probs = self.predict_probs(image, ensemble=True)

            # Convert to 1D numpy array
            probs = np.asarray(probs).squeeze()

            # If output doesn't look like probabilities (sum far from 1), apply softmax
            s = float(np.sum(probs))
            if not np.isfinite(s) or s <= 0 or abs(s - 1.0) > 1e-3:
                # treat as logits -> apply stable softmax
                exp = np.exp(probs - np.max(probs))
                probs = exp / np.sum(exp)

            # Debug info
            try:
                top_debug_idx = np.argsort(probs)[-5:][::-1]
                debug_str = ", ".join([f"{CLASS_NAMES[i]}:{probs[i]:.3f}" for i in top_debug_idx])
                print(f"🔎 TensorFlow prediction debug — sum={probs.sum():.6f}, top5={debug_str}")
            except Exception:
                print(f"🔎 TensorFlow prediction debug — sum={probs.sum():.6f}")

            top5_idx = np.argsort(probs)[-5:][::-1]

            return {
                CLASS_NAMES[idx]: float(probs[idx])
                for idx in top5_idx
            }
        except Exception as e:
            print(f"❌ TensorFlow prediction error: {e}")
            raise gr.Error(f"TensorFlow prediction failed: {str(e)}")

    def predict_probs(self, image: Image.Image, ensemble: bool = True):
        """Return full-class probabilities as a numpy array.
        If `ensemble` is True, create multiple resized/cropped variants (and flips),
        predict all at once and average probabilities to improve robustness on real photos.
        """
        if self.model is None:
            raise RuntimeError("TensorFlow model not loaded")

        # choose resample constant (same logic as preprocess)
        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            if hasattr(Image, "LANCZOS"):
                resample = Image.LANCZOS
            elif hasattr(Image, "ANTIALIAS"):
                resample = Image.ANTIALIAS
            else:
                resample = Image.BICUBIC

        variants = []

        # primary: aspect-preserving fit
        variants.append(self.preprocess(image, size=32))

        if ensemble:
            # simple resize (may stretch) - sometimes useful for real photos
            img_resize = image.convert("RGB").resize((32, 32), resample)
            arr_resize = np.expand_dims(np.array(img_resize).astype("float32") / 255.0, axis=0)
            variants.append(arr_resize)

            # flipped versions
            img_flip = image.convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
            variants.append(self.preprocess(img_flip, size=32))
            img_flip_resize = img_flip.resize((32, 32), resample)
            variants.append(np.expand_dims(np.array(img_flip_resize).astype("float32") / 255.0, axis=0))

        # stack batch
        batch = np.vstack(variants)

        probs_batch = self.model.predict(batch, verbose=0)
        probs_avg = np.mean(np.asarray(probs_batch), axis=0)

        # If output looks like logits, apply softmax
        s = float(np.sum(probs_avg))
        if not np.isfinite(s) or s <= 0 or abs(s - 1.0) > 1e-3:
            exp = np.exp(probs_avg - np.max(probs_avg))
            probs_avg = exp / np.sum(exp)

        return probs_avg

# -------------------------
# Load Models (Global)
# -------------------------
pytorch_model = PyTorchResNetModel()
pytorch_model.load()

tf_model = TensorFlowCIFAR10Model()
tf_model.load()

# -------------------------
# Prediction Function
# -------------------------
def predict_image(image, backend, show_raw=False):
    if image is None:
        raise gr.Error("Please upload an image")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # compute full probabilities and top5 mapping for consistent UI outputs
    if backend == "PyTorch":
        probs = pytorch_model.predict_probs(image)

    elif backend == "TensorFlow":
        # tf_model.predict_top5 previously returned top5 only; use predict_top5 internals
        # but here call the TF model predict to get full probs array
        try:
            arr = tf_model.preprocess(image, size=32)
            probs = tf_model.model.predict(arr, verbose=0)
            probs = np.asarray(probs).squeeze()
            # softmax fallback if needed (mirrors TF wrapper logic)
            s = float(np.sum(probs))
            if not np.isfinite(s) or s <= 0 or abs(s - 1.0) > 1e-3:
                exp = np.exp(probs - np.max(probs))
                probs = exp / np.sum(exp)
        except Exception as e:
            print(f"❌ TensorFlow prediction error (UI flow): {e}")
            raise gr.Error(f"TensorFlow prediction failed: {str(e)}")

    elif backend == "Blend":
        # Weighted soft-vote between PyTorch and TensorFlow probabilities
        try:
            pt_probs = pytorch_model.predict_probs(image)
        except Exception as e:
            print("❌ PyTorch prediction failed for Blend:", e)
            raise gr.Error(f"PyTorch prediction failed: {e}")

        try:
            tf_probs = tf_model.predict_probs(image, ensemble=True)
        except Exception as e:
            print("❌ TensorFlow prediction failed for Blend:", e)
            raise gr.Error(f"TensorFlow prediction failed: {e}")

        pt_probs = np.asarray(pt_probs)
        tf_probs = np.asarray(tf_probs)
        w_pt, w_tf = 0.6, 0.4
        probs = (w_pt * pt_probs + w_tf * tf_probs)
        s = float(np.sum(probs))
        if s > 0 and np.isfinite(s):
            probs = probs / s

    else:
        raise gr.Error("Invalid backend selected")

    # build full dict and top5 mapping
    full_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5 = {CLASS_NAMES[i]: float(probs[i]) for i in top5_idx}

    # If the UI asked to show raw probabilities, return JSON visible update
    if show_raw:
        return top5, gr.update(value=full_probs, visible=True)
    else:
        return top5, gr.update(value=full_probs, visible=False)


# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="CIFAR-10 Dual Backend Classifier") as demo:
    gr.Markdown("# 🖼️ CIFAR-10 Dual Backend Classifier")
    gr.Markdown(
        "Upload an image and choose a backend (**PyTorch ResNet18** or "
        "**TensorFlow CNN**) to classify it into one of the 10 CIFAR-10 classes."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Image",
                type="pil"
            )
            gr.Markdown("**Note:** The TensorFlow backend center-crops/resizes images to 32x32 and rescales pixel values to [0, 1] before inference.")
            backend_input = gr.Radio(
                ["PyTorch", "TensorFlow", "Blend"],
                value="PyTorch",
                label="Backend"
            )
            show_raw = gr.Checkbox(value=False, label="Show raw probabilities (debug)")
            classify_btn = gr.Button("Classify")

        with gr.Column():
            output = gr.Label(num_top_classes=5, label="Predictions")
            raw_json = gr.JSON(value={}, visible=False, label="All class probabilities")

    classify_btn.click(
        fn=predict_image,
        inputs=[image_input, backend_input, show_raw],
        outputs=[output, raw_json],
        api_name="predict_image"
    )

# -------------------------
# Launch
# -------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
