# Monetary-Policy-Forecasting-LSTM

LSTM model and web endpoint built with PyTorch and FastAPI for multivariate forecasting inflation, unemployment, and the Federal Funds effective rate.
The model was compared to a baseline vector autoregression which LSTM performed better on the validation set by around 8-9%.
Parameter tuning was done using Optuna's Bayesian Optimization.

Model development is contained in the notebook while "src/" folder contains code for the FastAPI endpoint.

A Dockerfile is also available so the FastAPI endpoint can be run in a container.
A Dockerfile and Docker compose is also available for setting up a MLflow server in Docker.

### Notes: 

- Uses MLflow as the model store so requires a MLflow server.
- Requires a FRED api key in "configs/fred_key.txt".
- Post request body schema: {
  "n_periods": int, "round_untransformed_forecasts": bool}
  . Where n_periods is the number of forecasts to return.

### Running the FastAPI App:
In a terminal, navigate to the project and enter:
```
uvicorn src.app:app --host 0.0.0.0 --port 8000
```
