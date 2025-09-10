from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import os

# Define the input data schema
class PredictionInput(BaseModel):
    features: list[float]

app = FastAPI()

# Load the model at startup
@app.on_event("startup")
def load_model():
    global model
    try:
        # Path to the local MLflow tracking directory
        mlflow_tracking_dir = os.path.join(os.getcwd(), "mlruns")
        mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
        
        # Get the latest run from the experiment
        experiment_name = "SVM_Classification_POC"
        df = mlflow.search_runs(experiment_names=[experiment_name], order_by=["start_time DESC"])
        if df.empty:
            raise HTTPException(status_code=404, detail="No MLflow runs found for the experiment.")
        
        # Get the latest run ID for the RBF model
        rbf_runs = df[df['tags.mlflow.runName'] == 'RBF_SVM']
        if rbf_runs.empty:
            raise HTTPException(status_code=404, detail="No RBF_SVM model runs found.")
        
        latest_run_id = rbf_runs.iloc[0]['run_id']
        model_uri = f"runs:/{latest_run_id}/RBF_SVM"
        
        # Load the model
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully from MLflow.")

    except Exception as e:
        print(f"Error loading model: {e}")
        model = None # Set model to None if loading fails

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Check server logs for errors.")

    try:
        # Create a DataFrame from the input features
        # The model expects 10 features
        if len(input_data.features) != 10:
            raise HTTPException(status_code=400, detail=f"Expected 10 features, but got {len(input_data.features)}.")

        df = pd.DataFrame([input_data.features], columns=[f'feature_{i}' for i in range(10)])
        
        # Make prediction
        prediction = model.predict(df)
        
        return {'prediction': int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/model_info')
def model_info():
    if model is None:
        return {"model_info": "Model not loaded"}
    return {"model_info": str(model)}

# To run this app, use the command:
# uvicorn serve_fastapi:app --reload