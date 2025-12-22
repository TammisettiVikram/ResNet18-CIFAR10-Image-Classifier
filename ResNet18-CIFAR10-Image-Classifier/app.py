import gradio as gr
import torch
import torchvision.transforms as T
import torchvision.models as models
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Model classes (adapted from your app)
class PyTorchResNetModel:
    def __init__(self):
        # Load ResNet18
        self.model = models.resnet18()
        # Replace final layer for CIFAR-10
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, 10)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    def load(self):
        self.model.load_state_dict(torch.load("models/pytorch/resnet18_cifar10.pt", map_location="cpu"))

    def predict_top5(self, img_pil):
        x = self.transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top5_prob, top5_idx = torch.topk(probs, 5)
        results = []
        for i in range(5):
            results.append((self.class_names[top5_idx[i].item()], float(top5_prob[i].item())))
        return results

class TensorFlowCIFAR10Model:
    def __init__(self):
        self.model = None
        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    def load(self):
        self.model = tf.keras.models.load_model("models/tensorflow/cifar10_cnn.keras")

    def predict_top5(self, pil_image):
        img = pil_image.resize((32, 32))
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        probs = self.model.predict(arr, verbose=0)[0]
        top5_idx = np.argsort(probs)[-5:][::-1]
        results = []
        for idx in top5_idx:
            results.append((self.classes[idx], float(probs[idx])))
        return results

# Load models globally
pytorch_model = PyTorchResNetModel()
pytorch_model.load()

tf_model = TensorFlowCIFAR10Model()
tf_model.load()

def predict_image(image, backend):
    if image is None:
        return "Please upload an image."

    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if backend == "PyTorch":
        results = pytorch_model.predict_top5(image)
    else:
        results = tf_model.predict_top5(image)

    # Format output
    output = f"**Top Prediction:** {results[0][0]} ({results[0][1]*100:.1f}%)\n\n"
    output += "**Top 5 Predictions:**\n"
    for i, (cls, conf) in enumerate(results):
        bar = "█" * int(conf * 20)  # Simple bar
        output += f"{i+1}. {cls}: {conf*100:.1f}% {bar}\n"

    return output

# Gradio Interface
with gr.Blocks(title="CIFAR-10 Image Classifier") as demo:
    gr.Markdown("# 🖼️ CIFAR-10 Dual Backend Classifier")
    gr.Markdown("""
    Upload an image and choose a backend (PyTorch or TensorFlow) to classify it into one of the 10 CIFAR-10 classes.
    This app uses ResNet18 (PyTorch) and a CNN (TensorFlow) trained on CIFAR-10.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="pil")
            backend_input = gr.Radio(["PyTorch", "TensorFlow"], label="Backend", value="PyTorch")
            submit_btn = gr.Button("Classify")

        with gr.Column():
            output = gr.Markdown(label="Predictions")

    submit_btn.click(predict_image, inputs=[image_input, backend_input], outputs=output)

if __name__ == "__main__":
    demo.launch()