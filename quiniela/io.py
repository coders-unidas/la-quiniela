import sqlite3

import pandas as pd

import settings

from quiniela import transform_data

def load_matchday(season, division, matchday):
    with sqlite3.connect(settings.DATABASE_PATH) as conn:
        data = pd.read_sql("SELECT * FROM Matches", conn)
    
    features = ['away_team_rank','home_team_rank','matchday','home_team_matchday_rank', 'away_team_matchday_rank','match_result']
    data_new = transform_data.transform_data_both(data)
    data[features] = data_new[features].copy()
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
