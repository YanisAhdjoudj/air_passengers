import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor



def _merge_external_data(X):


    ########################### Date data ##################################

    filepath_one = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )

    ####### Make sure that DateOfDeparture is of dtype datetime #######

    X = X.copy()  # modify a copy of X
    X.rename
    X.loc[:, "DateOfDeparture"] = pd.to_datetime(X['DateOfDeparture'])
    # Parse date to also be of dtype datetime
    data=pd.read_csv(filepath_one,index_col=0,parse_dates=["Date"])
    
    ####### Take data for the departure airport #######

    X_weather = data[['Date', 'AirPort', 'Max TemperatureC','Events']]
    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Departure', 'Max TemperatureC':'temperature_depart','Events':'Events_depart'}
    )

    X_merged = pd.merge(
        X, X_weather, how='left', on=['DateOfDeparture', 'Departure'], sort=False
    )

    ####### Take data for the arrival airport #######

    X_weather = data[['Date', 'AirPort', 'Max TemperatureC','Events']]

    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival', 'Max TemperatureC':'temperature_arrival', 'Events':'Events_arrival'}
    )

    X_merged_meteo = pd.merge(
        X_merged, X_weather, how='left', on=['DateOfDeparture', 'Arrival'], sort=False
    )
    
    
    ########################### airport data ##################################

    X = X_merged_meteo.copy() 

    X_airport = data[['Date','AirPort', 'wage median','beach','passenger per year','population','latitude_deg','longitude_deg','score']]
    X_airport = X_airport.rename(
        columns={'Date': 'DateOfDeparture','AirPort':'Departure', 'wage median':'wage_median_depart', 'beach':'beach_depart', 'passenger per year':'passenger_per_year_depart',
            'population':'population_depart', 'latitude_deg':'latitude_deg_depart', 'longitude_deg':'longitude_deg_depart','score':'score_depart'}
    )

    X_merged_airport = pd.merge(
        X, X_airport, how='left', on='Departure', sort=False
    ).drop(columns=['DateOfDeparture_y']).drop_duplicates().rename(
        columns={'DateOfDeparture_x': 'DateOfDeparture'})
    
    

      ####### Take data for the arrival airport #######
    
    
    X = X_merged_airport.copy()


    X_airport = data[['Date','AirPort', 'wage median','beach','passenger per year','population','latitude_deg','longitude_deg','score']]
    X_airport = X_airport.rename(
        columns={'Date':'DateOfDeparture','AirPort':'Arrival', 'wage median':'wage_median_arrival', 'beach':'beach_arrival', 'passenger per year':'passenger_per_year_arrival',
            'population':'population_arrival','latitude_deg':'latitude_deg_arrival','longitude_deg':'longitude_deg_arrival','score':'score_arrival'}
    )

    X_merged_airport = pd.merge(
        X, X_airport, how='left', on=['DateOfDeparture','Arrival'], sort=False
    )
    
      ######################## stocks fuel holiday data ###################################

    X = X_merged_airport.copy()

    data_stocks_fuel_holiday = data[['Date', 'AirPort', 'Open','prix','is_holiday']]
    data_stocks_fuel_holiday = data_stocks_fuel_holiday.rename(
        columns={'Date': 'DateOfDeparture','AirPort':'Arrival'})
    X_merged = pd.merge(
        X, data_stocks_fuel_holiday, how='left', on=['DateOfDeparture'], sort=False)
    
    
    ######################## stocks fuel holiday data ###################################

    X = X_merged_airport.copy()

    data_stocks_fuel_holiday = data[['Date', 'AirPort', 'Open','prix','is_holiday']]
    data_stocks_fuel_holiday = data_stocks_fuel_holiday.rename(
        columns={'Date': 'DateOfDeparture','AirPort':'Arrival'})
    X_merged = pd.merge(
        X, data_stocks_fuel_holiday, how='left', on=['DateOfDeparture','Arrival'], sort=False)
    
    return X_merged
  
def _encode_data(X):

    ################# encoding the date #########################

    # Encode the date information from the DateOfDeparture columns
    X_encoded = X.copy()

    # Make sure that DateOfDeparture is of datetime format
    X_encoded.loc[:, 'DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
    # Encode the DateOfDeparture
    X_encoded.loc[:, 'year'] = X_encoded['DateOfDeparture'].dt.year
    X_encoded.loc[:, 'month'] = X_encoded['DateOfDeparture'].dt.month
    X_encoded.loc[:, 'day'] = X_encoded['DateOfDeparture'].dt.day
    X_encoded.loc[:, 'weekday'] = X_encoded['DateOfDeparture'].dt.weekday
    X_encoded.loc[:, 'week'] = X_encoded['DateOfDeparture'].dt.week
    X_encoded.loc[:, 'n_days'] = X_encoded['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )

    
    X_encoded['weekend'] = np.where(X_encoded['weekday'].isin([0,1]),1,0)
    X_encoded['ete'] = np.where(X_encoded['month'].isin([10,5,6]),1,0)
    
    X_encoded['beach_ete_dep'] = X_encoded['ete'] * X_encoded['beach_depart']
    X_encoded['beach_ete_arr'] = X_encoded['ete'] * X_encoded['beach_arrival']
    
    ################# encoding the meteo #########################

    # From the meteo data we supply more information
    X_encoded["precipitations_depart"]=X_encoded.apply(lambda x: 0 if pd.isnull(x['Events_depart']) else 1, axis=1)
    X_encoded["precipitations_arrival"]=X_encoded.apply(lambda x: 0 if pd.isnull(x['Events_arrival']) else 1, axis=1)
    X_encoded["diff_temp"]=X_encoded['temperature_depart']-X_encoded['temperature_arrival']

    ################# encoding the distance #########################

    # calculer la distance entre les deux aeroports


    # approximate radius of earth in km

    def haversine_vectorize(lon1, lat1, lon2, lat2):

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        newlon = lon2 - lon1
        newlat = lat2 - lat1
        haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

        dist = 2 * np.arcsin(np.sqrt(haver_formula ))
        km = 6367 * dist #6367 for distance in KM for miles use 3958
        return km
    
    X_encoded["distance"]=haversine_vectorize(X_encoded['longitude_deg_depart'],X_encoded['latitude_deg_depart'],X_encoded['longitude_deg_arrival'],X_encoded['latitude_deg_arrival'])

 

    ################# ending the encoding #########################

    # Finally we can drop the original columns from the dataframe
    
    #print(X_encoded.columns)
    return X_encoded.drop(columns=["DateOfDeparture","Events_depart","Events_arrival","latitude_deg_depart",
                                   "latitude_deg_arrival","longitude_deg_arrival","longitude_deg_depart"])
                                   


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
        n_estimators=500, max_depth=10, max_features="auto", n_jobs=-1
    )

    # Create a pipeline to return a scikit-learn estimator that will
    # be used in ramp-test to do across validation
    

    
    return make_pipeline(
        data_merger, data_encoder,
        preprocessor, regressor
    )                           