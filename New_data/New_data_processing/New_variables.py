# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 21:10:36 2021

@author: yanis
"""


import os
import pandas as pd
import numpy as np
os.chdir(r"C:\\Users\\yanis\\01 Projets\\01 Python Projects\\Projet_Air_Passenger\\air_passengers-master\\New_data")


# American airline stocks
AAL_stocks=pd.read_csv("New_data_brut\AAL.csv")
AAL_stocks=AAL_stocks[["Date","Open"]]

Date=pd.date_range(start='27/09/2005', end='02/05/2021').to_frame().rename(columns={0: 'Date'})
AAL_stocks.loc[:, 'Date'] = pd.to_datetime(AAL_stocks.loc[:, 'Date'])
AAL_stocks=Date.merge(AAL_stocks,on="Date",how="left")


def fullfil_date(X,column_name):
    """
    Parameters
    ----------
    X : Dataframe 
    column_name : String, the name of the column to filled
    

    Returns
    -------
    
    The data frame with all the daily dates from the begining filled with
    the precedent 

    """
    for i in range(0,len(X)):
        if np.isnan(X.loc[i,column_name])==True:
            X.loc[i,column_name]=X.loc[i-1,column_name]
        
    return X


AAL_stocks=fullfil_date(AAL_stocks,"Open")


#fuel price
Fuel_Prices=pd.read_excel("C:\\Users\\yanis\\01 Projets\\01 Python Projects\\Projet_Air_Passenger\\air_passengers-master\\New_data\\New_data_brut\\EMD_EPD2DXL0_PTE_NUS_DPGw.xls",header=0)
Date=pd.date_range(start='02/05/2007', end='02/05/2021').to_frame().rename(columns={0: 'date'})
Fuel_Prices.loc[:, 'date'] = pd.to_datetime(Fuel_Prices.loc[:, 'date'])
Fuel_Prices=Date.merge(Fuel_Prices,on="date",how="left")


Fuel_Prices=fullfil_date(Fuel_Prices,"prix")

Fuel_Prices=pd.read_csv(r"C:\Users\yanis\01 Projets\01 Python Projects\Projet_Air_Passenger\air_passengers\New_data\New_data_cleaned\Fuel_Prices.csv",index_col=0)
AAL_stocks=pd.read_csv(r"C:\Users\yanis\01 Projets\01 Python Projects\Projet_Air_Passenger\air_passengers\New_data\New_data_cleaned\AAL_Stocks.csv",index_col=0)
Fuel_Prices.rename(columns={"date": "Date"},inplace=True)


External_Data=AAL_stocks.merge(Fuel_Prices,on="Date",how="left")
test_merge.to_csv(r"C:\Users\yanis\01 Projets\01 Python Projects\Projet_Air_Passenger\air_passengers\New_data\New_data_cleaned\FULL.csv")


#meteo data
meteo_data=pd.read_csv(r"C:\Users\yanis\01 Projets\01 Python Projects\Projet_Air_Passenger\air_passengers\New_data\New_data_cleaned\external_data.csv", parse_dates=["Date"])
meteo_data["Pluie"]=meteo_data.apply(lambda x: 0 if pd.isnull(x['Events']) else 1, axis=1)


data_airport = pd.read_csv(r'C:\Users\yanis\01 Projets\01 Python Projects\Projet_Air_Passenger\air_passengers\submissions\Test_1\airport_key.csv',index_col=0)

external_data=pd.read_csv(r'C:\Users\yanis\01 Projets\01 Python Projects\Projet_Air_Passenger\air_passengers\New_data\New_data_cleaned\External_Data_2.csv',index_col=0)
external_data.loc[:, 'Date'] = pd.to_datetime(external_data.loc[:, 'Date'])
test=external_data.merge(meteo_data,on="Date",how="left")

test2.rename(columns={"Aeroport":"AirPort"},inplace=True)
test2=test.merge(data_airport,on="Aeroport",how="inner")
external_data=test.merge(meteo_data,on="Date",how="left")


external_data=pd.read_csv(r'C:\Users\yanis\01 Projets\01 Python Projects\Projet_Air_Passenger\air_passengers\New_data\New_data_cleaned\external_data_i.csv',index_col=0)
testplus=external_data[["Date","AirPort","Dernier","is_holiday","scheduled_service"]]
testplus.loc[:, 'Date'] = pd.to_datetime(testplus.loc[:, 'Date'])
external_data.info()
test4.rename(columns={"Date":"DateofDeparture"},inplace=True)
test4=testplus.merge(test2,on=["Date","AirPort"],how="left")

test4.to_csv(r"C:\Users\yanis\01 Projets\01 Python Projects\Projet_Air_Passenger\air_passengers\New_data\New_data_cleaned\External_Data_Full.csv")
