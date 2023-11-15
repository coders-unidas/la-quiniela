import pandas as pd
import numpy as np


def transform_data(df):
    df = df.dropna(subset=['score'])
    df["score_home_team"] = df["score"].str.split(":").str[0].astype(float)
    df["score_away_team"] = df["score"].str.split(":").str[1].astype(float)
    df["goal_difference"] = df["score_home_team"] - df["score_away_team"]

    df["match_result"] = np.where(df['score_home_team'] > df['score_away_team'], '1', np.where(df['score_home_team'] < df['score_away_team'], '2', 'X'))
    
    df_class_home = df.groupby(['division', 'season', 'matchday', 'home_team','match_result']).agg(
        GF_safe = pd.NamedAgg(column='score_home_team', aggfunc='sum'),
        GA_safe = pd.NamedAgg(column='score_away_team', aggfunc='sum')
        ).reset_index()
    df_class_away = df.groupby(['division', 'season', 'matchday', 'away_team', 'match_result']).agg(
        GF_safe = pd.NamedAgg(column='score_away_team', aggfunc='sum'),
        GA_safe = pd.NamedAgg(column='score_home_team', aggfunc='sum')
        ).reset_index()

    df_class_home['W_safe'] = np.where(df_class_home['match_result'] == '1' , 1 ,0)
    df_class_home['L_safe'] = np.where(df_class_home['match_result'] == '2' , 1 ,0)
    df_class_home['T_safe'] = np.where(df_class_home['match_result'] == 'X' , 1 ,0)

    df_class_away['W_safe'] = np.where(df_class_away['match_result'] == '2' , 1 ,0)
    df_class_away['L_safe'] = np.where(df_class_away['match_result'] == '1' , 1 ,0)
    df_class_away['T_safe'] = np.where(df_class_away['match_result'] == 'X' , 1 ,0)

    df_class_away.rename(columns={'away_team':'team'}, inplace=True)
    df_class_home.rename(columns={'home_team':'team'}, inplace=True)
    df_classification = df_class_away.merge(df_class_home,how='outer')
    df_classification = df_classification.groupby(['season', 'division','matchday','team']).sum().reset_index()

    df_classification[['W','L','T','GF','GA']] = df_classification.groupby([ 'division','season','team'])[['W_safe','L_safe','T_safe','GF_safe','GA_safe']].cumsum()
    df_classification['result_matchday'] = np.where(df_classification['W_safe']==1,'W',np.where(df_classification['L_safe']==1,'L','T'))

    for i in range(5):
        df_classification[f"last_{i}"] = df_classification.groupby(['division','season' ,'team'])['result_matchday'].shift(i+1)

    df_classification['GD'] = df_classification['GF'] - df_classification['GA']
    df_classification['Pts'] = (df_classification['W']) * 3 + df_classification['T']
    df_classification['year_of_start']=df_classification['season'].str.split("-").str[0].astype(int)

    df_classification["last_5"] = df_classification[[f"last_{i}" for i in range(5)]].agg(lambda x: [i for i in x if not pd.isna(i)], axis=1)

    df_classification_ordered = df_classification.sort_values(by=['year_of_start'], ascending=False)
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start','division'],ascending=[False,True])
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start','division','matchday'], ascending=[False,True,True])
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start','division','matchday', 'Pts'],ascending=[False,True,True,False])
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start','division','matchday', 'Pts','GD'],ascending=[False,True,True,False,False])
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start','division','matchday', 'Pts', 'GD', 'GF'],ascending=[False,True,True,False,False,False]).reset_index(drop=True)

    df_classification_ordered['rank']=df_classification_ordered.groupby(['year_of_start','division','matchday']).cumcount()+1
    df_classification_ordered['delayed_rank'] = df_classification_ordered.groupby(['year_of_start','division','team'])['rank'].shift(1)

    df_with_rank = df_classification_ordered[['season','division','matchday','team','delayed_rank']]
    df_useful = df[['season','division','matchday','home_team','match_result','away_team']]

    home_team_rank = df_useful.merge(df_with_rank, left_on=['season','division', 'matchday', 'home_team'], right_on=['season','division', 'matchday', 'team'], how='left')
    home_team_rank.rename(columns={'delayed_rank': 'home_team_rank'}, inplace=True)
    home_team_rank.drop(columns=['team'], inplace=True)

    away_team_rank = df_useful.merge(df_with_rank, left_on=['season', 'division', 'matchday', 'away_team'], right_on=['season','division', 'matchday', 'team'], how='left')
    away_team_rank.rename(columns={'delayed_rank': 'away_team_rank'}, inplace=True)
    away_team_rank.drop(columns=['team'], inplace=True)

    df_new = away_team_rank.merge(home_team_rank, on=['season', 'division', 'matchday', 'home_team','away_team'], how='left')

    df_new.rename(columns={'match_result_x': 'match_result'},inplace=True)

    df_to_train = df_new[['season','home_team','away_team','away_team_rank','home_team_rank','match_result','matchday']]
    df_to_train = df_to_train.fillna(0)
    return df_to_train