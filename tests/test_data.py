import sys
sys.path.insert(0, "./src")
import pytest
import pandas as pd
from data import get_data, get_data_config, difference_data

@pytest.fixture
def config():
    return get_data_config("./configs/time_series_config.yaml")


@pytest.fixture
def data(config):
    data, lambda_ = get_data(config["FRED"])
    return data, lambda_


def test_get_data(data, config):
    expected_columns = [config['FRED'].get(key).get("name") 
                        for key in config['FRED'].keys()]
    data, lambda_ = data
    assert lambda_ > 0
    assert type(data) == pd.DataFrame
    assert type(data.index) == pd.DatetimeIndex
    assert expected_columns == list(data.columns)

    dtypes = data.dtypes
    for col in data.columns:
        assert dtypes[col] == float



def test_difference_data(data, config):
    expected_columns = [config['FRED'].get(key).get("name") 
                        for key in config['FRED'].keys()]
    data, lambda_ = data

    diff_names = []
    for col in config['FRED']:
        col = config['FRED'].get(col)
        diff_name = col.get("diff_name")
        if diff_name is None:
            diff_name = col.get("name") + "_diff"
        diff_names.append(diff_name)
    
    differenced = difference_data(data, config["FRED"])

    assert type(differenced) == pd.DataFrame
    assert len(differenced.columns) == 2*len(expected_columns)
    assert list(differenced.columns) == expected_columns + diff_names
    assert type(data.index) == pd.DatetimeIndex

    dtypes = data.dtypes
    for col in data.columns:
        assert dtypes[col] == float
    
