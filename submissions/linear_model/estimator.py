import pandas as pd
import os
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def _merge_external_data(X):
    filepath_one = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )
    # Make sure that DateOfDeparture is of dtype datetime
    X = X.copy()  # modify a copy of X
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    # Parse date to also be of dtype datetime
    data_weather = pd.read_csv(filepath_one, parse_dates=["Date"])

    X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC']]
    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'}
    )

    X_merged_one = pd.merge(
        X, X_weather, how='left', on=['DateOfDeparture', 'Arrival'], sort=False
    )
    ######################## other data ###################################

    filepath_two = os.path.join(
        os.path.dirname(__file__), 'External_Data_2.csv')
    data_stocks_fuel = pd.read_csv(filepath_two, parse_dates=["Date"],index_col=0)
    data_stocks_fuel = data_stocks_fuel.rename(
        columns={'Date': 'DateOfDeparture'})
    X_merged = pd.merge(
        X_merged_one, data_stocks_fuel, how='left', on='DateOfDeparture', sort=False)
	
    return X_merged


def _encode_dates(X):
    # Make sure that DateOfDeparture is of dtype datetime
    X = X.copy()  # modify a copy of X
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.isocalendar().week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["DateOfDeparture"])


def get_estimator():
    data_merger = FunctionTransformer(_merge_external_data)
    date_encoder = FunctionTransformer(_encode_dates)

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = [
        "Arrival", "Departure", "year", "month", "day",
        "weekday", "week", "n_days"
    ]

    numerical_scaler = StandardScaler()
    numerical_cols = ["WeeksToDeparture", "std_wtd","Open","prix"]

    preprocessor = make_column_transformer(
        (categorical_encoder, categorical_cols),
        (numerical_scaler, numerical_cols)
    )

    regressor = LinearRegression()

    return make_pipeline(date_encoder,data_merger, preprocessor, regressor)
