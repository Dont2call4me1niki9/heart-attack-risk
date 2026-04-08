from fastapi import FastAPI, HTTPException

from api.schemas import PredictRequest, PredictResponse
from src.config import AppConfig
from src.predictor import HeartRiskPredictor

app = FastAPI(title="Heart Attack Risk API")

MODEL_PATH = "models/model.cbm"
METADATA_PATH = "models/metadata.json"


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Heart Attack Risk API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        predictor = HeartRiskPredictor(config=AppConfig())
        predictor.load_model(MODEL_PATH, METADATA_PATH)
        result = predictor.predict_from_csv(
            csv_path=request.csv_path,
            output_path=request.output_path,
        )
        return PredictResponse(
            status="ok",
            rows=result["rows"],
            output_path=result["output_path"],
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
