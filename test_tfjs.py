import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"C:\Users\TugraUysal\.gemini\antigravity\scratch\illegal_dumping_detector")))
from app import load_model_and_labels
import numpy as np

model, labels, error = load_model_and_labels()
if error:
    print(f"Error loading model: {error}")
    sys.exit(1)

print("Model loaded successfully!")
print("Labels:", labels)

dummy = np.ones((1, 224, 224, 3), dtype=np.float32)
preds = model.predict(dummy, verbose=0)
print("Dummy predictions:", preds)
