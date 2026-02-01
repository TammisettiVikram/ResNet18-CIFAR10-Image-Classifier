import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR10

import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(__file__))
PYTORCH_MODEL_PATH = os.path.join(ROOT, "models", "pytorch", "resnet18_cifar10.pt")
TF_MODEL_PATH = os.path.join(ROOT, "models", "tensorflow", "cifar10_cnn.keras")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_pytorch_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    state_dict = torch.load(PYTORCH_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def load_tf_model():
    model = tf.keras.models.load_model(TF_MODEL_PATH)
    return model


def eval_models(n_samples=200):
    pt_model, pt_transform = load_pytorch_model()
    tf_model = load_tf_model()

    dataset = CIFAR10(root="./data", train=False, download=True)

    pt_correct = 0
    tf_correct = 0
    total = 0
    mismatches = []

    for idx in range(min(n_samples, len(dataset))):
        img, label = dataset[idx]

        # PyTorch prediction
        x = pt_transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = pt_model(x)
            pt_pred = int(torch.argmax(logits, dim=1).item())

        # TensorFlow prediction (resize to 32x32 and scale)
        img32 = img.convert("RGB").resize((32, 32))
        arr = np.array(img32).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        probs = tf_model.predict(arr, verbose=0)
        probs = np.asarray(probs).squeeze()
        # softmax fallback
        s = float(np.sum(probs))
        if not np.isfinite(s) or s <= 0 or abs(s - 1.0) > 1e-3:
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.sum(exp)
        tf_pred = int(np.argmax(probs))

        total += 1
        if pt_pred == label:
            pt_correct += 1
        if tf_pred == label:
            tf_correct += 1

        if pt_pred != tf_pred:
            mismatches.append((idx, label, pt_pred, tf_pred, probs.tolist()))

    print(f"Checked {total} samples")
    print(f"PyTorch accuracy: {pt_correct}/{total} = {pt_correct/total:.3f}")
    print(f"TensorFlow accuracy: {tf_correct}/{total} = {tf_correct/total:.3f}")

    if mismatches:
        print("\nExamples where PyTorch and TensorFlow disagree (index, true, pt, tf, tf_probs_top5):")
        for m in mismatches[:10]:
            idx, true, ptp, tfp, probs = m
            top5_idx = np.argsort(probs)[-5:][::-1]
            top5 = [(CLASS_NAMES[i], float(probs[i])) for i in top5_idx]
            print(idx, true, ptp, tfp, top5)


if __name__ == '__main__':
    eval_models(n_samples=200)
