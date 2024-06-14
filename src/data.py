import pandas as pd
import numpy as np
from fredapi import Fred
import yaml
from pathlib import Path
from scipy.stats import boxcox
from pathlib import Path


def get_fred_api_key(path):
    with open(path, 'r') as f:
        key = f.read()
        return key

    
def get_data_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config
        
config_path = Path("./configs")

key = get_fred_api_key(config_path / "fred_key.txt")
fred = Fred(api_key=key)

configs = get_data_config(config_path / "time_series_config.yaml")
fred_data = configs["FRED"]


def box_cox_transform_cpi(data:pd.DataFrame):
    """ Box-cox transforms the CPI variable in data dataframe

    Parameters
    ----------
    data : pd.DataFrame
      Dataframe with "CPI" variable to be transformed

    Returns
    -------
    data : pd.DataFrame
    lambda_ : float
    """
    transformed = boxcox(data["CPI"].dropna())
    lambda_ = transformed[1]
    data.loc[~data['CPI'].isna(),'CPI'] = transformed[0]
    return data, lambda_


def get_data(fred_data:dict):
    """ Gets latest version of the data specified in the config.

    Parameters
    ----------
    fred_data : dict
      dictionary containing the FRED codes and other attributes

    Returns
    -------
    data : pd.DataFrame
    lambda_ : float
    """
    data = {fred_data[code].get('name'): fred.get_series_latest_release(code) 
            for code in fred_data}
    data = pd.DataFrame(data)
    data, lambda_ = box_cox_transform_cpi(data)
    return data, lambda_
    

def difference_data(df:pd.DataFrame, config:dict):
    """ Differences data based on the rules defined in the config
    Returns the dataframe with new columns which are the differenced variables.

    Parameters
    ----------
    df : pd.DataFrame
      data to be differenced
    config : dict
      dictionary containing the FRED codes and their differencing/tranformations
    
    Returns
    -------
    pd.DataFrame  
    """
    for item in config:
        series = config[item]
        col = series['name']
        
        try:
            diff_name = series['diff_name'] 
        except Exception:
            diff_name = col + "_diff"
        
        if series['log'] and series['diff']:
            
            df[diff_name] =  np.log(df[col])
            for i in range(series["n_diffs"]):
                df[diff_name] = df[diff_name].diff()
                
        elif series['diff']:
            df[diff_name] = df[col]
            for i in range(series["n_diffs"]):
                df[diff_name] = df[diff_name].diff()

    return df.dropna()