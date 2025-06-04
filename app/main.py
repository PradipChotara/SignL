from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health, model_info, predict
from app.services import prediction_service # Import to trigger model loading

app = FastAPI(
    title="Sign Language Recognition API",
    description="API for real-time sign language prediction using hand landmarks.",
    version="1.0.0"
)

# --- CORS Configuration ---
# This is crucial for allowing your frontend (running on a different origin)
# to make requests to your backend.
origins = [
    "http://localhost",
    "http://localhost:8000", # Your frontend might run on this port
    "http://127.0.0.1:8000",
    # Add the URL of your deployed frontend here if you deploy it
    # "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Include API Routers ---
app.include_router(health.router, tags=["Health Check"])
app.include_router(model_info.router, tags=["Model Info"])
app.include_router(predict.router, tags=["Prediction"])

# --- Startup Event ---
# This ensures that the model assets are loaded when the FastAPI application starts up.
@app.on_event("startup")
async def startup_event():
    print("FastAPI application starting up...")
    success = prediction_service.load_model_assets()
    if not success:
        print("Failed to load all model assets. Prediction service might not function correctly.")
    else:
        print("FastAPI startup complete. Models are ready.")

# --- Root Endpoint (Optional) ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Sign Language Recognition API! Visit /docs for API documentation."}