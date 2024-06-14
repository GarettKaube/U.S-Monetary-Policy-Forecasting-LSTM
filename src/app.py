import logging
from fastapi import FastAPI
import mlflow
from pydantic import BaseModel
from src.score import score
from contextlib import asynccontextmanager

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s", level=logging.INFO)

tracking_uri = "http://127.0.0.1:8080/"
seq_length = 3

models = {}
scalers = {}

class MLflowInstance:
    def __init__(self, uri:str) -> None:
        mlflow.set_tracking_uri(uri)
        self.client = mlflow.MlflowClient(tracking_uri=uri)
        self.model_name = "LSTMProduction@production"
        self.scaler_name = "standardscaler@production"


    def check_server_health(self):
        try:
            experiments = self.client.search_experiments()
            return "Mlflow ok"
        except:
            return "MLflow connection error"
        

    def get_pytorch_model(self):
        model = mlflow.pytorch.load_model(
            model_uri = f"models:/{self.model_name}"
        )
        return model
    

    def get_scaler(self):

        scaler = mlflow.sklearn.load_model(
            model_uri = f"models:/{self.scaler_name}"
        )
        return scaler


@asynccontextmanager
async def setup(app: FastAPI):
    global mlflow_inst
    mlflow_inst = MLflowInstance(tracking_uri)
    logging.info(f"Connected to MLflow with URI: {tracking_uri}")
    yield
    models.clear()
    scalers.clear()


app = FastAPI(lifespan=setup)


class Request(BaseModel):
    n_periods: int | None = 1


def forecast_period(n_periods: str = None):
    if n_periods == None:
        n_periods = 1
    return n_periods


@app.get("/health/", status_code=200)
async def health_check():
    logging.info("Checking health")
    return {
        "MLFlow health": mlflow_inst.check_server_health()
    }


async def get_model():
    if mlflow_inst.model_name not in models:
        models[mlflow_inst.model_name] = mlflow_inst.get_pytorch_model()
    
    if mlflow_inst.scaler_name not in scalers:
        scalers[mlflow_inst.scaler_name] = mlflow_inst.get_scaler()

    return models[mlflow_inst.model_name], scalers[mlflow_inst.scaler_name]


@app.post("/forecast/", status_code=200)
async def return_forecasts(request:Request):
    result = {}
    n_periods = forecast_period(request.n_periods)
    model, scaler = await get_model()

    # Get untransformed and transformed forecasts.
    forecasts, tf, mom, yoy = score(model, scaler, seq_length, n_periods)
    tf = tf.to_dict()

    result["request"] = request.model_dump()
    result["Untransformed forecasts"] = forecasts
    result['Forecasts'] = tf
    result["MoM_Inflation"] = mom
    result["YoY_Inflation"] = yoy

    return result