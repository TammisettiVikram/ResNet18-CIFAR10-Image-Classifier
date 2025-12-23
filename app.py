import os
import gradio as gr
import torch
import torchvision.transforms as T
import torchvision.models as models
import tensorflow as tf
import numpy as np
from PIL import Image

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

# -------------------------
# TensorFlow Model Wrapper
# -------------------------
class TensorFlowCIFAR10Model:
    def __init__(self):
        self.model = None

    def load(self):
        try:
            self.model = tf.keras.models.load_model(TF_MODEL_PATH)
            print("✅ TensorFlow model loaded successfully")

        except Exception as e:
            print("❌ TensorFlow model load failed:", e)
            self.model = None

    def predict_top5(self, image: Image.Image):
        if self.model is None:
            raise RuntimeError("TensorFlow model not loaded")

        img = image.convert("RGB").resize((32, 32))
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)

        probs = self.model.predict(arr, verbose=0)[0]
        top5_idx = np.argsort(probs)[-5:][::-1]

        return {
            CLASS_NAMES[idx]: float(probs[idx])
            for idx in top5_idx
        }

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
def predict_image(image, backend):
    if image is None:
        return {"Error": 1.0}

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    try:
        if backend == "PyTorch":
            return pytorch_model.predict_top5(image)
        else:
            return tf_model.predict_top5(image)

    except Exception as e:
        print("❌ Prediction error:", e)
        return {"Error": 1.0}

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
            backend_input = gr.Radio(
                ["PyTorch", "TensorFlow"],
                value="PyTorch",
                label="Backend"
            )
            classify_btn = gr.Button("Classify")

        with gr.Column():
            output = gr.Label(label="Predictions")

    classify_btn.click(
        fn=predict_image,
        inputs=[image_input, backend_input],
        outputs=output,
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
