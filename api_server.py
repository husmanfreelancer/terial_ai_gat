from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_with_new_entry

app = FastAPI()


# Define input schema
class FeatureInput(BaseModel):
    num_workers: int
    material_cost: float
    area_sqft: float
    roof_pitch: float


@app.post("/predict/")
def predict_endpoint(input_data: FeatureInput):
    # Convert to the expected dict structure
    record = {
        "features": {
            "num_workers": input_data.num_workers,
            "material_cost": input_data.material_cost,
            "area_sqft": input_data.area_sqft,
            "roof_pitch": input_data.roof_pitch
        }
    }

    result = predict_with_new_entry(record)
    return result
