import sqlite3

import pandas as pd

import settings

from quiniela import transform_data

def load_matchday(season, division, matchday):
    # this function has been changed in order to be able to implement our model, since we need data previous to the matchday
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        all_data = pd.read_sql("SELECT * FROM Matches ", conn)
    
    seasons = all_data['season'].unique().tolist()
    index= seasons.index(season)
    data = all_data[(all_data['season']==seasons[index-1]) | (all_data['season']==season)].copy()
    merge_colummns = ['season','division','matchday','home_team','away_team']
    data_new = transform_data.transform_data_matchday(data)
    data = pd.merge(data,data_new, how='left', on=merge_colummns)
    data_final = data[(data['season']==season) & (data['division']==division) & (data['matchday']==matchday)].copy()
    if data_final.empty:
        raise ValueError("There is no matchday data for the values given")
    return data_final


def load_historical_data(seasons):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        if seasons == "all":
            data = pd.read_sql("SELECT * FROM Matches", conn)
        else:
            data = pd.read_sql(f"""
                SELECT * FROM Matches
                    WHERE season IN {tuple(seasons)}
            """, conn)
    if data.empty:
        raise ValueError(f"No data for seasons {seasons}")
    return data

def save_predictions(predictions):
    predictions = predictions[['season','division','matchday','date','time','home_team','away_team','score','pred']]
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        predictions.to_sql(name="Predictions", con=conn, if_exists="append", index=False)
