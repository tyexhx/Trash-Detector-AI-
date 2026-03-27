# Trash-Detector-AI-
AI made trash detector that uses your webcam

## Illegal Dumping Detector (Streamlit)

A Streamlit app aligned with **UN SDG 11 (Sustainable Cities and Communities)** that uses a **Teachable Machine TensorFlow SavedModel** to detect illegal dumping activity from a webcam feed, with a Thomas More University–styled UI.

### AI disclosure

This project was **made completely with AI**.

### Project structure

- `app.py`: Streamlit application
- `model/`
  - `model.savedmodel/`: TensorFlow SavedModel directory (contains `saved_model.pb` + `variables/`)
  - `labels.txt`: class labels used for display
- `requirements.txt`: Python dependencies

### Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run

From this folder:

```bash
streamlit run app.py
```

### Notes

- The app expects a Teachable Machine **SavedModel** to exist under `model/` (it searches for `saved_model.pb`).
- If you plan to publish this repo, avoid committing large, duplicate artifacts (e.g. exported ZIPs) unless you really need them.

