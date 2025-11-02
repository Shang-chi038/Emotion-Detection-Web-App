import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from deepface import DeepFace
import cv2
import warnings
warnings.filterwarnings('ignore')

# Define the 7 emotions we want to detect
VALID_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_model():
    """
    Initialize the DeepFace model.
    The model is loaded automatically when DeepFace.analyze() is first called.
    This function documents the model loading process.
    """
    print("Using DeepFace pretrained emotion detection model")
    print(f"Supported emotions: {VALID_EMOTIONS}")
    return True

def predict_emotion(image_path):
    """
    Predict emotion from an image using DeepFace pretrained model
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Predicted emotion as string
    """
    try:
        # Analyze the image using DeepFace's pretrained model
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,  # Don't fail if no face detected
            detector_backend='opencv'  # Use OpenCV for face detection
        )
        
        # DeepFace returns a list if multiple faces, or dict if single face
        if isinstance(result, list):
            result = result[0]
        
        # Get the dominant emotion
        emotions = result['emotion']
        dominant_emotion = result['dominant_emotion']
        
        # Map DeepFace emotions to required format
        emotion_mapping = {
            'angry': 'anger',
            'happy': 'happiness',
            'sad': 'sadness'
        }
        
        dominant_emotion = emotion_mapping.get(dominant_emotion, dominant_emotion)
        
        # Note: 'contempt' is not in DeepFace's default emotions
        # Using 'neutral' as closest alternative
        
        return dominant_emotion
        
    except Exception as e:
        print(f"Error in emotion prediction: {str(e)}")
        return "error"

# Initialize model on module import
if __name__ == "__main__":
    load_model()
    print("Model ready for predictions")