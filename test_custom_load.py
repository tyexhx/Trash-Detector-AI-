import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_DIR = Path(r"C:\Users\TugraUysal\.gemini\antigravity\scratch\illegal_dumping_detector\model")
h5_path = str(MODEL_DIR / "keras_model.h5")

class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        if "groups" in kwargs:
            kwargs.pop("groups")
        super().__init__(**kwargs)

try:
    print("Loading with CustomDepthwiseConv2D...")
    model = tf.keras.models.load_model(h5_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}, compile=False)
    print("Success loading model!")

    dummy_input = np.ones((1, 224, 224, 3), dtype=np.float32)
    preds = model.predict(dummy_input)
    print("Dummy Predictions (ones):", preds)

    np.random.seed(42)
    dummy_noise = np.random.uniform(-1, 1, size=(1, 224, 224, 3)).astype(np.float32)
    preds_noise = model.predict(dummy_noise)
    print("Dummy Predictions (noise):", preds_noise)
except Exception as e:
    print("Error:", e)
