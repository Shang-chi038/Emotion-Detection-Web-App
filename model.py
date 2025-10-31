def predict_emotions(image_path: str):
    """Stub for model prediction — replace later."""
    emotions = {
    'happy': 45.0,
    'sad': 20.0,
    'neutral': 25.0,
    'angry': 10.0
    }
    dominant = max(emotions, key=emotions.get)
    return dominant, emotions