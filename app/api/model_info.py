from fastapi import APIRouter
from app.services import prediction_service

router = APIRouter()

@router.get("/model_info")
async def get_model_details():
    """
    Returns information about the loaded machine learning model.
    """
    info = prediction_service.get_model_info()
    return info