# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 21:10:36 2021

@author: yanis
"""


import os
import pandas as pd
import numpy as np
os.chdir("C:\\Users\\yanis\\01 Projets\\01 Python Projects\\Projet_Air_Passenger\\air_passengers-master\\New_data")


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

Fuel_Prices.to_csv(r"C:\Users\yanis\01 Projets\01 Python Projects\Projet_Air_Passenger\air_passengers-master\New_data\New_data_cleaned\Fuel_Prices.csv")
AAL_stocks.to_csv(r"C:\Users\yanis\01 Projets\01 Python Projects\Projet_Air_Passenger\air_passengers-master\New_data\New_data_cleaned\AAL_Stocks.csv")
