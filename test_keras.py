import sys
import numpy as np
from pathlib import Path

MODEL_DIR = Path(r"C:\Users\TugraUysal\.gemini\antigravity\scratch\illegal_dumping_detector\model")
MODEL_PATH_H5 = MODEL_DIR / "keras_model.h5"

try:
    import tensorflow as tf
    print(f"TF Version: {tf.__version__}")
except ImportError:
    print("TF not installed.")
    sys.exit(1)

def test_native(dummy_input):
    print("\n--- Native Load ---")
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH_H5), compile=False)
        preds = model.predict(dummy_input, verbose=0)
        print("Native Predictions:", preds)
    except Exception as e:
        print("Native failed:", e)

def test_manual(dummy_input):
    print("\n--- Manual Load ---")
    try:
        import h5py
        base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            alpha=0.35,
            include_top=False,
            weights=None
        )
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(100, activation="relu")(x)
        outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        with h5py.File(str(MODEL_PATH_H5), "r") as hf:
            if "model_weights" in hf:
                mw = hf["model_weights"]
                seq1 = mw["sequential_1"]
                seq3 = mw["sequential_3"]
                
                for layer in base.layers:
                    name = layer.name
                    if name not in seq1:
                        continue
                    grp = seq1[name]
                    wnames = sorted(grp.keys())
                    weights = [grp[w][()] for w in wnames]
                    if len(weights) == len(layer.weights):
                        layer.set_weights(weights)

                dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]

                d1_grp = seq3["dense_Dense1"]
                d1_weights = [d1_grp["kernel:0"][()], d1_grp["bias:0"][()]]
                dense_layers[0].set_weights(d1_weights)

                d2_grp = seq3["dense_Dense2"]
                d2_weights = [d2_grp["kernel:0"][()]]
                dense_layers[1].set_weights(d2_weights)
                
                print("Manual load successful!")
                preds = model.predict(dummy_input, verbose=0)
                print("Manual Predictions:", preds)
            else:
                print("No model_weights in h5")
    except Exception as e:
        print("Manual failed:", e)

if __name__ == "__main__":
    dummy_input = np.ones((1, 224, 224, 3), dtype=np.float32)
    # also test some random noise
    np.random.seed(42)
    dummy_noise = np.random.uniform(-1, 1, size=(1, 224, 224, 3)).astype(np.float32)

    print("Dummy Ones:")
    test_native(dummy_input)
    test_manual(dummy_input)

    print("\nDummy Noise:")
    test_native(dummy_noise)
    test_manual(dummy_noise)
