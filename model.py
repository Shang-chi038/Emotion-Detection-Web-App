from deepface import DeepFace
import cv2

# Define the 7 emotions we want to detect
VALID_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def predict_emotion(image_path):
    """
    Predict emotion from an image using DeepFace
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Predicted emotion as string
    """
    try:
        # Analyze the image
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        
        # DeepFace returns a list if multiple faces, or dict if single face
        if isinstance(result, list):
            result = result[0]
        
        # Get the dominant emotion
        emotions = result['emotion']
        dominant_emotion = result['dominant_emotion']
        
        # Map 'angry' to 'anger' and handle 'contempt' (not in DeepFace default)
        emotion_mapping = {
            'angry': 'anger',
            'happy': 'happiness',
            'sad': 'sadness'
        }
        
        dominant_emotion = emotion_mapping.get(dominant_emotion, dominant_emotion)
        
        # Note: DeepFace doesn't have 'contempt' by default
        # You might need to train a custom model or use neutral as fallback
        
        return dominant_emotion
        
    except Exception as e:
        print(f"Error in emotion prediction: {str(e)}")
        return "error"