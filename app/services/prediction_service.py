import pickle
import numpy as np
import cv2
import mediapipe as mp
import os # For path manipulation

# --- Configuration: Define paths to your saved model assets ---
# IMPORTANT: These paths should be relative to where your 'app' directory is located
# or absolute paths on your server.
# For local testing, ensure these files are in sign_language_api/app/data/models/
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'sign_language_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'scaler.pkl')
PCA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'pca.pkl')

# Global variables to hold the loaded model, scaler, and PCA
# These will be loaded once when the service is initialized
loaded_model = None
loaded_scaler = None
loaded_pca = None
hands_detector = None # MediaPipe Hands object

def load_model_assets():
    """
    Loads the trained model, StandardScaler, and PCA objects from disk.
    This function should be called once when the FastAPI application starts.
    """
    global loaded_model, loaded_scaler, loaded_pca, hands_detector
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")
        if not os.path.exists(PCA_PATH):
            raise FileNotFoundError(f"PCA file not found at: {PCA_PATH}")

        with open(MODEL_PATH, 'rb') as f:
            loaded_model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            loaded_scaler = pickle.load(f)
        with open(PCA_PATH, 'rb') as f:
            loaded_pca = pickle.load(f)

        # Initialize MediaPipe Hands detector
        mp_hands = mp.solutions.hands
        hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

        print("All model assets and MediaPipe detector loaded successfully!")
        return True
    except FileNotFoundError as e:
        print(f"ERROR: Missing model asset. Please ensure all .pkl files are in the 'app/data/models/' directory. {e}")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during model asset loading: {e}")
        return False

def preprocess_image_and_predict(image_bytes):
    """
    Processes an image (as bytes), extracts hand landmarks, preprocesses them,
    and returns a prediction from the loaded model.
    """
    if loaded_model is None or loaded_scaler is None or loaded_pca is None or hands_detector is None:
        return {"error": "Model assets or MediaPipe detector not loaded. Server not ready."}

    try:
        # 1. Decode image bytes to OpenCV image format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Could not decode image from provided bytes."}

        # 2. Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. Extract Hand Landmarks
        results = hands_detector.process(image_rgb)

        landmark_data_for_prediction = None

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            all_hand_landmarks = []

            # Determine if it's a left or right hand (MediaPipe provides this)
            # This is important for consistent ordering if you train with specific hand order
            # For simplicity, we'll just concatenate, assuming order doesn't strictly matter
            # or that your training data handled this implicitly.
            # A more robust solution might sort hands by handedness if available in results.
            # For now, we'll just collect them.
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_coords = []
                for landmark in hand_landmarks.landmark:
                    hand_coords.extend([landmark.x, landmark.y, landmark.z])
                all_hand_landmarks.append(np.array(hand_coords).flatten())

            # Ensure the feature vector always has 126 elements
            if num_hands == 1:
                single_hand_data = all_hand_landmarks[0]
                if single_hand_data.shape[0] == 63:
                    # Pad with zeros for the missing hand (e.g., right hand if only left is detected)
                    # This MUST match how you handled single hands in your training data!
                    # Assuming your training data had left hand landmarks first, then right.
                    # If MediaPipe detects a right hand only, you might need to pad the beginning.
                    # A robust solution would identify handedness and place data accordingly.
                    # For now, we concatenate, assuming it's always the first 63 or last 63.
                    # The simplest approach is to pad to 126 if one hand is detected.
                    # This assumes your model can generalize from this padding.
                    padded_data = np.concatenate([single_hand_data, np.zeros(63)])
                    landmark_data_for_prediction = padded_data.reshape(1, -1)
                else:
                    return {"error": "Unexpected number of landmarks for one hand (expected 63)."}

            elif num_hands == 2:
                if len(all_hand_landmarks) == 2 and all_hand_landmarks[0].shape[0] == 63 and all_hand_landmarks[1].shape[0] == 63:
                    # Concatenate landmarks from both hands.
                    # Ensure the order (e.g., left then right) is consistent with training data.
                    # MediaPipe's `multi_handedness` can help here if needed.
                    two_hand_data = np.concatenate(all_hand_landmarks)
                    landmark_data_for_prediction = two_hand_data.reshape(1, -1)
                else:
                    return {"error": "Unexpected number of landmarks for two hands (expected 63 each)."}
            else:
                return {"error": "Detected more than two hands. Current logic only supports 1 or 2 hands."}
        else:
            return {"error": "No hands detected in the image."}

        # 4. Preprocess Landmarks (Scale and PCA)
        if landmark_data_for_prediction is not None:
            # Ensure the input to scaler has the expected 126 features
            if landmark_data_for_prediction.shape[1] != 126:
                return {"error": f"Processed landmark data has {landmark_data_for_prediction.shape[1]} features, but model expects 126."}

            landmark_data_scaled = loaded_scaler.transform(landmark_data_for_prediction)
            landmark_data_pca = loaded_pca.transform(landmark_data_scaled)

            # 5. Make Prediction
            prediction_label = loaded_model.predict(landmark_data_pca)[0]
            return {"prediction": str(prediction_label)}
        else:
            return {"error": "Could not prepare landmark data for prediction."}

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": f"An internal server error occurred during prediction: {e}"}

# --- Service for Model Info ---
def get_model_info():
    """
    Returns information about the loaded model.
    """
    if loaded_model is None or loaded_pca is None:
        return {"status": "Model not loaded"}

    return {
        "status": "Ready",
        "model_type": type(loaded_model).__name__,
        "pca_components": loaded_pca.n_components,
        "num_classes": len(loaded_model.classes_) if hasattr(loaded_model, 'classes_') else "N/A"
    }

# --- Service for Health Check ---
def get_health_status():
    """
    Returns the health status of the server.
    """
    if loaded_model and loaded_scaler and loaded_pca and hands_detector:
        return {"status": "OK", "message": "All components are loaded and ready."}
    else:
        return {"status": "ERROR", "message": "Some components failed to load."}

# --- Initial Model Loading ---
# This will be called when the prediction_service.py module is imported
# in main.py, ensuring models are loaded only once.
load_model_assets()