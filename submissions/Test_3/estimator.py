import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor


def _merge_external_data(X):


    ########################### meteo data ##################################

    filepath_one = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )

    ####### Make sure that DateOfDeparture is of dtype datetime #######

    X = X.copy()  # modify a copy of X
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    # Parse date to also be of dtype datetime
    data_weather = pd.read_csv(filepath_one, parse_dates=["Date"])
    
    ####### Take data for the departure airport #######

    X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC','Events']]

    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Departure', 'Max TemperatureC':'temperature_depart','Events':'Events_depart'}
    )

    X_merged = pd.merge(
        X, X_weather, how='left', on=['DateOfDeparture', 'Departure'], sort=False
    )

    ####### Take data for the arrival airport #######

    X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC','Events']]

    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival', 'Max TemperatureC':'temperature_arrival', 'Events':'Events_arrival'}
    )

    X_merged_meteo = pd.merge(
        X_merged, X_weather, how='left', on=['DateOfDeparture', 'Arrival'], sort=False
    )


    ########################### airport data ##################################

    filepath_two = os.path.join(
        os.path.dirname(__file__), 'airport_key.csv'
    )

    data_airport = pd.read_csv(filepath_two,index_col=0)


    ####### Take data for the departure airport #######

    X = X_merged_meteo.copy() 

    X_airport = data_airport[['Aeroport', 'wage median','beach','passenger per year','population','latitude_deg','longitude_deg','score']]
    X_airport = X_airport.rename(
        columns={'Aeroport':'Departure', 'wage median':'wage_median_depart', 'beach':'beach_depart', 'passenger per year':'passenger_per_year_depart',
            'population':'population_depart', 'latitude_deg':'latitude_deg_depart', 'longitude_deg':'longitude_deg_depart','score':'score'}
    )

    X_merged_airport = pd.merge(
        X, X_airport, how='left', on='Departure', sort=False
    )


    ####### Take data for the arrival airport #######

    X = X_merged_airport.copy()


    X_airport = data_airport[['Aeroport', 'wage median','beach','passenger per year','population','latitude_deg','longitude_deg','score']]
    X_airport = X_airport.rename(
        columns={'Aeroport':'Arrival', 'wage median':'wage_median_arrival', 'beach':'beach_arrival', 'passenger per year':'passenger_per_year_arrival',
            'population':'population_arrival','latitude_deg':'latitude_deg_arrival','longitude_deg':'longitude_deg_arrival','score':'score'}
    )

    X_merged_airport = pd.merge(
        X, X_airport, how='left', on='Arrival', sort=False
    )


    ######################## stocks and fuel data ###################################

    filepath_tree = os.path.join(
        os.path.dirname(__file__), 'External_Data_2.csv')

    X = X_merged_airport.copy()

    data_stocks_fuel = pd.read_csv(filepath_tree, parse_dates=["Date"],index_col=0)
    data_stocks_fuel = data_stocks_fuel.rename(
        columns={'Date': 'DateOfDeparture'})
    X_merged = pd.merge(
        X, data_stocks_fuel, how='left', on='DateOfDeparture', sort=False)
	
    return X_merged


def _encode_data(X):

    ################# encoding the date #########################

    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.isocalendar().week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )

    ################# encoding the meteo #########################

    # From the meteo data we supply more information
    X["precipitations_depart"]=X.apply(lambda x: 0 if pd.isnull(x['Events_depart']) else 1, axis=1)
    X["precipitations_arrival"]=X.apply(lambda x: 0 if pd.isnull(x['Events_arrival']) else 1, axis=1)
    X["diff_temp"]=X['temperature_depart']-X['temperature_arrival']

    ################# encoding the distance #########################

    # calculer la distance entre les deux aeroports

    ################# ending the encoding #########################

    # Finally we can drop the original columns from the dataframe
    X.drop(columns=["DateOfDeparture"],inplace=True)
    X.drop(columns=["Events_depart"],inplace=True)
    X.drop(columns=["Events_arrival"],inplace=True)
    X.drop(columns=["latitude_deg_depart"],inplace=True)
    X.drop(columns=["latitude_deg_arrival"],inplace=True)
    X.drop(columns=["longitude_deg_depart"],inplace=True)
    X.drop(columns=["longitude_deg_arrival"],inplace=True)
    return X


def get_estimator():

    # Data augmentation transformer (add a column)
    data_merger = FunctionTransformer(_merge_external_data)
    data_encoder = FunctionTransformer(_encode_data)

    # preprocessor for categorical variables
    categorical_encoder = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OrdinalEncoder()
    )
    categorical_cols = ['Arrival', 'Departure']

    preprocessor = make_column_transformer(
        (categorical_encoder, categorical_cols),
        remainder='passthrough',  # passthrough numerical columns as they are
    )

    # Regressor to do the prediction
    regressor = RandomForestRegressor(
        n_estimators=100, max_depth=None, max_features="auto", n_jobs=-1
    )

    # Create a pipeline to return a scikit-learn estimator that will
    # be used in ramp-test to do across validation
    return make_pipeline(
        data_merger, data_encoder,
        preprocessor, regressor
    )
