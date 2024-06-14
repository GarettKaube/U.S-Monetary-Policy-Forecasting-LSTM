import mlflow
import torch
import numpy as np
import pandas as pd
from scipy.special import inv_boxcox

# mlflow tracking
tracking_uri = "http://127.0.0.1:8080/"

# Device for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_predict(model, x):
    """ Make prediction with model using x.

    Parameters
    ----------
    model : torch.nn.Module
    x: torch.tensor

    Returns
    -------
    torch.tensor
    """
    with torch.no_grad():
        return model(x.to(device).float())


def get_differenced_data():
    """ Gets the data and transforms/differences it

    Returns
    -------
    pd.DataFrame
    """
    from src.data import get_data, difference_data, fred_data
    global lambda_
    data, lambda_ = get_data(fred_data=fred_data)
    difference_data = difference_data(data, fred_data)
    return difference_data


def generate_forecasts(X, model, n_periods):
    """ Generates raw forecasts with the PyTorch model

    Parameters
    ----------
    X : torch.tensor
      Latest observed data so the model can forecast into the future.

    Returns
    -------
    torch.tensor
    """
    forecasts = []
    outputs = []
    first_data = X
    for i in range(n_periods):
        out = model_predict(model, first_data)
        new_value = torch.concat(
            [
                first_data[-1].unsqueeze(dim=0), 
                out[-1].reshape(1,1,3)
            ], 
            dim=1
        )[:, 1:, :]
        forecasts.append(new_value[:, -1, :].unsqueeze(dim=0))
        outputs.append(new_value)
        first_data = outputs[-1]
    return torch.concat(forecasts, dim=1).to(device)


def generate_latest_prediction(forecasts, scaler):
    """ Untransforms the forecasts by undifferencing and unlogging
    """
    increment = pd.DateOffset(months=1)
    
    most_recent = data_diff.iloc[-2:][["CPI", "fed_rate", "Unemployment"]]
    most_recent_logs = most_recent.copy()

    # Log transform data so we can add the forecasts to them
    most_recent_logs["fed_rate"] = np.exp(np.log(most_recent_logs["fed_rate"]))
    untransformed_forecasts = []
    for forecast in forecasts[0]:
        # Untransform forecasts and add them to the latest observation 
        # since the model forecasts the change in the variables.
        
        forecast = scaler.inverse_transform(
            forecast.unsqueeze(dim=0).cpu().numpy()
        )
        prediction = forecast + (most_recent_logs.iloc[[-1],:].mul({"CPI": 2, "fed_rate":1, "Unemployment":1})  
                                               - [most_recent_logs.iloc[-2]["CPI"].item(), 0, 0])
        
        increment = most_recent_logs.index[-1] + pd.DateOffset(months=1)
        
        prediction.index = [increment]
        most_recent_logs.loc[increment] = prediction.iloc[0]
        
        # undo logs
        prediction["CPI"] =(prediction["CPI"]*(lambda_) + 1)**(1/lambda_)
        
        untransformed_forecasts.append(prediction)

    return pd.concat(untransformed_forecasts, axis=0)


def round_predictions(predictions):
    predictions["fed_rate"] = round(predictions["fed_rate"], 2)
    predictions["CPI"] = round(predictions["CPI"], 3)
    predictions["Unemployment"] = round(predictions["Unemployment"], 1)



def calculate_inflation(predictions, untransformed_data, mode="both"):
    """ Calculates month over month and year over year inflation rate.
    """
    most_recent = pd.concat([untransformed_data, predictions],axis=0)
    inflation_mom_list = []
    inflation_yoy_list = []
    for pred_index in reversed(range(1, predictions.shape[0] + 1)):

        most_recent_cpi = most_recent.iloc[[-(pred_index + 1)]][["CPI", "fed_rate", "Unemployment"]]
        yr_ago_cpi = most_recent.iloc[[-(12 + pred_index + 1)]][["CPI"]]
        
        # MoM  inflation
        inflation_mom = ((predictions.iloc[-pred_index]["CPI"].item() - most_recent_cpi["CPI"].item()) 
                        / most_recent_cpi["CPI"].item())*100
        # YoY Inflation
        inflation_yoy = ((predictions.iloc[-pred_index]["CPI"].item() - yr_ago_cpi["CPI"].item()) 
                        / yr_ago_cpi["CPI"].item())*100
        inflation_mom_list.append(inflation_mom)
        inflation_yoy_list.append(inflation_yoy)

    if mode == "yoy":
        return inflation_yoy_list
    elif mode == "mom":
        return inflation_mom_list
    else:
        return inflation_mom_list, inflation_yoy_list


def score(model, scaler, seq_length, n_periods):
    """ Generates raw forecasts, untransformed forecasts, MoM and YoY inflation.

    Parameters
    ----------
    model : torch.nn.Module
      LSTM PyTorch model
    scaler : sklearn.preprocessing.StandardScaler
      loaded scaler used to scale data
    seq_length : int
      Length of the sequence used in the model
    n_periods : int
      number of periods to forecast into the future
    
    Returns
    -------
    list
        List of transformed forecasts
    pd.DataFrame
        dataframe of the untransformed forecasts
    list
        list of MoM inflation rates
    list 
        list of YoY inflation rates
    """
    global data_diff
    global most_recent_diff

    # Difference the data
    data_diff = get_differenced_data()

    model = model.to(device)

    # Get latest data for untransforming the forecasts
    most_recent_diff = data_diff.iloc[-seq_length:]\
        [["log_inflation_MoM", "fed_rate_diff", "Unemployment_diff"]]
    
    data = torch.tensor(
        scaler.transform(most_recent_diff)
    ).to(device).unsqueeze(dim=0)
    
    # Raw forecasts
    forecasts = generate_forecasts(data, n_periods)
    # Untransformed forecasts
    tf = generate_latest_prediction(forecasts, model, scaler)
    
    # Untransform CPI
    untransformed = data_diff.copy()
    untransformed["CPI"] = inv_boxcox(untransformed["CPI"], lambda_)

    mom, yoy = calculate_inflation(tf, untransformed, mode="both")
    return forecasts.cpu().numpy().tolist(), tf, mom, yoy
    

if __name__ == "__main__":
    mlflow.set_tracking_uri(tracking_uri)
    model_name = "LSTMProduction@production"
    scaler_name = "standardscaler/latest"

    model = mlflow.pytorch.load_model(
        model_uri = f"models:/{model_name}"
    )

    scaler = mlflow.sklearn.load_model(
        model_uri = f"models:/{scaler_name}"
    )
    
    print(score(model, scaler, 3, 5))