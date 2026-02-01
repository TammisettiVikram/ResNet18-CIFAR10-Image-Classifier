import sys
import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.models as models
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(__file__))
PYTORCH_MODEL_PATH = os.path.join(ROOT, "models", "pytorch", "resnet18_cifar10.pt")
TF_MODEL_PATH = os.path.join(ROOT, "models", "tensorflow", "cifar10_cnn.keras")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_pytorch():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    state = torch.load(PYTORCH_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def load_tf():
    model = tf.keras.models.load_model(TF_MODEL_PATH)
    return model


def topk_from_probs(probs, k=5):
    idx = np.argsort(probs)[-k:][::-1]
    return [(CLASS_NAMES[i], float(probs[i])) for i in idx]


def predict_single(image_path):
    img = Image.open(image_path).convert("RGB")

    # PyTorch
    pt_model, pt_transform = load_pytorch()
    x = pt_transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = pt_model(x)
        pt_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    print("\n--- PyTorch (ResNet18) ---")
    print("probs sum:", float(pt_probs.sum()))
    print("top5:", topk_from_probs(pt_probs, 5))

    # TensorFlow
    tf_model = load_tf()
    img32 = img.resize((32, 32))
    arr = np.array(img32).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    tf_out = tf_model.predict(arr, verbose=0)
    tf_out = np.asarray(tf_out).squeeze()

    s = float(np.sum(tf_out))
    applied_softmax = False
    if not np.isfinite(s) or s <= 0 or abs(s - 1.0) > 1e-3:
        # treat as logits
        exp = np.exp(tf_out - np.max(tf_out))
        tf_probs = exp / np.sum(exp)
        applied_softmax = True
    else:
        tf_probs = tf_out

    print("\n--- TensorFlow (CNN) ---")
    print("raw sum:", s)
    print("applied softmax?:", applied_softmax)
    print("top5:", topk_from_probs(tf_probs, 5))

    # side-by-side comparison
    print("\n--- Side-by-side top1 ---")
    pt_top1 = CLASS_NAMES[int(np.argmax(pt_probs))]
    tf_top1 = CLASS_NAMES[int(np.argmax(tf_probs))]
    print("PyTorch top1:", pt_top1)
    print("TensorFlow top1:", tf_top1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python tests/predict_single.py path/to/image.jpg")
        sys.exit(1)
    predict_single(sys.argv[1])
