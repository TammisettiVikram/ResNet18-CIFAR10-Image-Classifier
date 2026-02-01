import tensorflow as tf
import numpy as np
import sys

MODEL_PATH = "models/tensorflow/cifar10_cnn.keras"

print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded. Input shape:", model.input_shape)
print("Model dtype:", model.input.dtype)

# Print layer types and look for Rescaling/Normalization/Lambda
for i, layer in enumerate(model.layers[:20]):
    print(i, layer.name, type(layer).__name__)
    if isinstance(layer, tf.keras.layers.Rescaling):
        try:
            print("  -> Rescaling scale:", layer.scale)
        except Exception:
            pass
    if isinstance(layer, tf.keras.layers.Normalization):
        print("  -> Normalization layer detected")
    if isinstance(layer, tf.keras.layers.Lambda):
        print("  -> Lambda layer detected; config:", layer.get_config())

# Try to inspect first layer config
try:
    cfg = model.layers[0].get_config()
    print("First layer config keys:", list(cfg.keys()))
except Exception as e:
    print("Could not get first layer config:", e)

# Test predictions with inputs scaled 0-1 vs 0-255 vs rescaled by 1/127.5-1
x_rand = np.random.rand(1, 32, 32, 3).astype(np.float32)
try:
    out_01 = model.predict(x_rand, verbose=0)
    out_255 = model.predict(x_rand * 255.0, verbose=0)
    # Also test common preprocessing: (x/127.5)-1
    out_127 = model.predict((x_rand * 255.0) / 127.5 - 1.0, verbose=0)

    def summarize(o):
        o = np.asarray(o).squeeze()
        return {"sum": float(np.sum(o)), "max": float(np.max(o)), "argmax": int(np.argmax(o))}

    print("Random input test summaries:")
    print("  scaled 0-1:", summarize(out_01))
    print("  scaled 0-255:", summarize(out_255))
    print("  (x/127.5)-1:", summarize(out_127))

    print("Max abs diff 0-1 vs 0-255:", float(np.max(np.abs(out_01 - out_255))))
    print("Max abs diff 0-1 vs (x/127.5)-1:", float(np.max(np.abs(out_01 - out_127))))
except Exception as e:
    print("Prediction tests failed:", e)

print("Done.")
