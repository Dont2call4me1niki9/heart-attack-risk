from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    csv_path: str = Field(..., description="Path to input CSV file.")
    output_path: str = Field(..., description="Path to output predictions CSV file.")


class PredictResponse(BaseModel):
    status: str
    rows: int
    output_path: str
