from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from app.services import prediction_service
import base64

router = APIRouter()

# Pydantic model for the request body
class ImageRequest(BaseModel):
    image: str # Base64 encoded image string

@router.post("/predict")
async def predict_sign(request: ImageRequest):
    """
    Receives a Base64 encoded image, extracts hand landmarks,
    and returns a sign language prediction.
    """
    try:
        # Extract the Base64 string from the request
        base64_image_string = request.image

        # Decode the Base64 string to bytes
        # Remove the "data:image/jpeg;base64," prefix if present from the frontend
        if "base64," in base64_image_string:
            base64_image_string = base64_image_string.split("base64,")[1]

        image_bytes = base64.b64decode(base64_image_string)

        # Call the prediction service
        prediction_result = prediction_service.preprocess_image_and_predict(image_bytes)

        if "error" in prediction_result:
            raise HTTPException(status_code=400, detail=prediction_result["error"])

        return {"prediction": prediction_result["prediction"]}

    except HTTPException as e:
        raise e # Re-raise FastAPI HTTP exceptions
    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")