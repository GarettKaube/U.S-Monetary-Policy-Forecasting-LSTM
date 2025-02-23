# Monetary-Policy-Forecasting-LSTM

Neural network with LSTM cells and web endpoint built with PyTorch and FastAPI for multivariate forecasting inflation, unemployment, and the Federal Funds effective rate. Right now, these three variables may not be enough as Okun's law fails when inflation expectations beome anchored.
The model was compared to a baseline vector autoregression which RNN's with LSTM cells performed better on the validation set by around 8-9%.
Parameter tuning was done using Optuna's Bayesian Optimization. The data was first differenced and inflation was Box-Cox transformed to stabalize variance. This was done primarily for the VAR as it typically requires stationary data. The data was kept in this form for the RNN's for easy comparison in performance. the RNN's though, could potentially perform better on undifferenced data as we lose long-term dependencies when we difference the data. This is something to explore more.

Later, variational inference with Pyro was used to estimate a Bayesian RNN which performs a bit better than vannila and has the bonus advantage of forecast uncertainty.
Plot of differenced inflation forecasts on validation set using posterior predictive from the BRNN:
![output](https://github.com/user-attachments/assets/55b59536-184c-46e5-97fe-5d625603b56a)



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
