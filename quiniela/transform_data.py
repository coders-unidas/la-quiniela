import pandas as pd
import numpy as np


def transform_data_matchday(df):

    df_tochange = df.dropna(subset=['score'])
    df_tochange["score_home_team"] = df_tochange["score"].str.split(":").str[0].astype(float)
    df_tochange["score_away_team"] = df_tochange["score"].str.split(":").str[1].astype(float)
    df_tochange["goal_difference"] = df_tochange["score_home_team"] - df_tochange["score_away_team"]

    df_tochange["match_result"] = np.where(df_tochange['score_home_team'] > df_tochange['score_away_team'], '1', np.where(df_tochange['score_home_team'] < df_tochange['score_away_team'], '2', 'X'))
    
    df_class_home = df_tochange.groupby(['division', 'season', 'matchday', 'home_team','match_result']).agg(
        GF_safe = pd.NamedAgg(column='score_home_team', aggfunc='sum'),
        GA_safe = pd.NamedAgg(column='score_away_team', aggfunc='sum')
        ).reset_index()
    df_class_away = df_tochange.groupby(['division', 'season', 'matchday', 'away_team', 'match_result']).agg(
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
    df_useful = df_tochange[['season','division','matchday','home_team','match_result','away_team']]

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

def transform_data_season(df):
    df_tochange = df.dropna(subset=['score'])
    df_tochange["score_home_team"] = df_tochange["score"].str.split(":").str[0].astype(float)
    df_tochange["score_away_team"] = df_tochange["score"].str.split(":").str[1].astype(float)
    df_tochange["goal_difference"] = df_tochange["score_home_team"] - df_tochange["score_away_team"]

    df_tochange["match_result"] = np.where(df_tochange['score_home_team'] > df_tochange['score_away_team'], '1', np.where(df_tochange['score_home_team'] < df_tochange['score_away_team'], '2', 'X'))

    def func_home_wins(data):
        return (data[data == '1']).count()

    def func_away_wins(data):
        return (data[data == '2']).count()

    def func_tie(data):
        return (data[data == 'X']).count()

    df_class_home = df_tochange.groupby(['division', 'season', 'home_team']).agg(
        GF=pd.NamedAgg(column='score_home_team', aggfunc='sum'),
        GA=pd.NamedAgg(column='score_away_team', aggfunc='sum'),
        W=pd.NamedAgg(column='match_result', aggfunc=func_home_wins),
        L=pd.NamedAgg(column='match_result', aggfunc=func_away_wins),
        T=pd.NamedAgg(column='match_result', aggfunc=func_tie)
    ).reset_index()

    df_class_away = df_tochange.groupby(['division', 'season', 'away_team']).agg(
        GF=pd.NamedAgg(column='score_away_team', aggfunc='sum'),
        GA=pd.NamedAgg(column='score_home_team', aggfunc='sum'),
        W=pd.NamedAgg(column='match_result', aggfunc=func_away_wins),
        L=pd.NamedAgg(column='match_result', aggfunc=func_home_wins),
        T=pd.NamedAgg(column='match_result', aggfunc=func_tie)
    ).reset_index()

    df_class_away.rename(columns={'away_team': 'team'}, inplace=True)
    df_class_home.rename(columns={'home_team': 'team'}, inplace=True)

    df_classification = df_class_away.merge(df_class_home, how='outer')
    df_classification = df_classification.groupby(['season', 'team', 'division']).sum().reset_index()

    df_classification['GD'] = df_classification['GF'] - df_classification['GA']
    df_classification['Pts'] = (df_classification['W']) * 3 + df_classification['T']

    df_classification['year_of_start'] = df_classification['season'].str.split("-").str[0].astype(int)

    df_classification_ordered = df_classification.sort_values(by=['year_of_start'], ascending=False)
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start', 'division'], ascending=[False, True])
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start', 'division', 'Pts'], ascending=[False, True, False])
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start', 'division', 'Pts', 'GD'], ascending=[False, True, False, False])
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start', 'division', 'Pts', 'GD', 'GF'], ascending=[False, True, False, False, False])

    df_classification_ordered = df_classification_ordered.reset_index(drop=True)
    df_classification_ordered['rank'] = df_classification_ordered.groupby(['year_of_start', 'division']).cumcount() + 1

    df_classification_1_div = df_classification_ordered[df_classification_ordered['division']==1]
    df_classification_2_div = df_classification_ordered[df_classification_ordered['division']==2]
    num_teams_1_div = df_classification_1_div.groupby('season')['rank'].max().reset_index()

    merged_df = df_classification_2_div.merge(num_teams_1_div, on='season')
    merged_df['rank'] = merged_df['rank_x'] + merged_df['rank_y']
    merged_df = merged_df.drop(columns={'rank_x','rank_y'})
    df_classification_ordered_next = merged_df.merge(df_classification_1_div,how='outer')

    df_classification_ordered_next = df_classification_ordered_next.sort_values(by=['year_of_start', 'division', 'Pts', 'GD', 'GF'], ascending=[False, True, False, False, False])
    df_classification_ordered_next['delayed_rank'] = df_classification_ordered_next.groupby(['team'])['rank'].shift(-1)

    df_with_rank = df_classification_ordered_next[['season','division','team','delayed_rank']]

    df_useful = df_tochange[['season','division','home_team','match_result','away_team']]

    home_team_rank = df_useful.merge(df_with_rank, left_on=['season','division', 'home_team'], right_on=['season','division', 'team'], how='left')
    home_team_rank.rename(columns={'delayed_rank': 'home_team_rank'}, inplace=True)
    home_team_rank.drop(columns=['team'], inplace=True)

    away_team_rank = df_useful.merge(df_with_rank, left_on=['season', 'division', 'away_team'], right_on=['season','division', 'team'], how='left')
    away_team_rank.rename(columns={'delayed_rank': 'away_team_rank'}, inplace=True)
    away_team_rank.drop(columns=['team'], inplace=True)

    df_new = away_team_rank.merge(home_team_rank, on=['season', 'division', 'home_team','away_team'], how='left')

    df_new.rename(columns={'match_result_x': 'match_result'},inplace=True)

    df_to_train = df_new[['season','home_team','away_team','home_team_rank','away_team_rank','match_result']]
    df_to_train_season = df_to_train.fillna(0)
    return df_to_train_season


def transform_data_both(df):
    
    df_tochange = df.dropna(subset=['score']).copy()
    df_tochange["score_home_team"] = df_tochange["score"].str.split(":").str[0].astype(float)
    df_tochange["score_away_team"] = df_tochange["score"].str.split(":").str[1].astype(float)
    df_tochange["goal_difference"] = df_tochange["score_home_team"] - df_tochange["score_away_team"]

    df_tochange["match_result"] = np.where(df_tochange['score_home_team'] > df_tochange['score_away_team'], '1', np.where(df_tochange['score_home_team'] < df_tochange['score_away_team'], '2', 'X'))
    
    df_class_home = df_tochange.groupby(['division', 'season', 'matchday', 'home_team','match_result']).agg(
        GF_safe = pd.NamedAgg(column='score_home_team', aggfunc='sum'),
        GA_safe = pd.NamedAgg(column='score_away_team', aggfunc='sum')
        ).reset_index()
    df_class_away = df_tochange.groupby(['division', 'season', 'matchday', 'away_team', 'match_result']).agg(
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
    df_useful = df_tochange[['season','division','matchday','home_team','match_result','away_team']]

    home_team_rank = df_useful.merge(df_with_rank, left_on=['season','division', 'matchday', 'home_team'], right_on=['season','division', 'matchday', 'team'], how='left')
    home_team_rank.rename(columns={'delayed_rank': 'home_team_matchday_rank'}, inplace=True)
    home_team_rank.drop(columns=['team'], inplace=True)

    away_team_rank = df_useful.merge(df_with_rank, left_on=['season', 'division', 'matchday', 'away_team'], right_on=['season','division', 'matchday', 'team'], how='left')
    away_team_rank.rename(columns={'delayed_rank': 'away_team_matchday_rank'}, inplace=True)
    away_team_rank.drop(columns=['team'], inplace=True)

    df_new = away_team_rank.merge(home_team_rank, on=['season', 'division', 'matchday', 'home_team','away_team'], how='left')

    df_new.rename(columns={'match_result_x': 'match_result'},inplace=True)

    df_to_train_matchday = df_new[['season','home_team','away_team','away_team_matchday_rank','home_team_matchday_rank','match_result','matchday']]
    df_to_train_matchday = df_to_train_matchday.fillna(0)


    def func_home_wins(data):
        return (data[data == '1']).count()

    def func_away_wins(data):
        return (data[data == '2']).count()

    def func_tie(data):
        return (data[data == 'X']).count()

    df_class_home = df_tochange.groupby(['division', 'season', 'home_team']).agg(
        GF=pd.NamedAgg(column='score_home_team', aggfunc='sum'),
        GA=pd.NamedAgg(column='score_away_team', aggfunc='sum'),
        W=pd.NamedAgg(column='match_result', aggfunc=func_home_wins),
        L=pd.NamedAgg(column='match_result', aggfunc=func_away_wins),
        T=pd.NamedAgg(column='match_result', aggfunc=func_tie)
    ).reset_index()

    df_class_away = df_tochange.groupby(['division', 'season', 'away_team']).agg(
        GF=pd.NamedAgg(column='score_away_team', aggfunc='sum'),
        GA=pd.NamedAgg(column='score_home_team', aggfunc='sum'),
        W=pd.NamedAgg(column='match_result', aggfunc=func_away_wins),
        L=pd.NamedAgg(column='match_result', aggfunc=func_home_wins),
        T=pd.NamedAgg(column='match_result', aggfunc=func_tie)
    ).reset_index()

    df_class_away.rename(columns={'away_team': 'team'}, inplace=True)
    df_class_home.rename(columns={'home_team': 'team'}, inplace=True)

    df_classification = df_class_away.merge(df_class_home, how='outer')
    df_classification = df_classification.groupby(['season', 'team', 'division']).sum().reset_index()

    df_classification['GD'] = df_classification['GF'] - df_classification['GA']
    df_classification['Pts'] = (df_classification['W']) * 3 + df_classification['T']

    df_classification['year_of_start'] = df_classification['season'].str.split("-").str[0].astype(int)

    df_classification_ordered = df_classification.sort_values(by=['year_of_start'], ascending=False)
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start', 'division'], ascending=[False, True])
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start', 'division', 'Pts'], ascending=[False, True, False])
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start', 'division', 'Pts', 'GD'], ascending=[False, True, False, False])
    df_classification_ordered = df_classification_ordered.sort_values(by=['year_of_start', 'division', 'Pts', 'GD', 'GF'], ascending=[False, True, False, False, False])

    df_classification_ordered = df_classification_ordered.reset_index(drop=True)
    df_classification_ordered['rank'] = df_classification_ordered.groupby(['year_of_start', 'division']).cumcount() + 1

    df_classification_1_div = df_classification_ordered[df_classification_ordered['division']==1]
    df_classification_2_div = df_classification_ordered[df_classification_ordered['division']==2]
    num_teams_1_div = df_classification_1_div.groupby('season')['rank'].max().reset_index()

    merged_df = df_classification_2_div.merge(num_teams_1_div, on='season')
    merged_df['rank'] = merged_df['rank_x'] + merged_df['rank_y']
    merged_df = merged_df.drop(columns={'rank_x','rank_y'})
    df_classification_ordered_next = merged_df.merge(df_classification_1_div,how='outer')

    df_classification_ordered_next = df_classification_ordered_next.sort_values(by=['year_of_start', 'division', 'Pts', 'GD', 'GF'], ascending=[False, True, False, False, False])
    df_classification_ordered_next['delayed_rank'] = df_classification_ordered_next.groupby(['team'])['rank'].shift(-1)

    df_with_rank = df_classification_ordered_next[['season','division','team','delayed_rank']]

    df_useful = df_tochange[['season','division','home_team','match_result','away_team']]

    home_team_rank = df_useful.merge(df_with_rank, left_on=['season','division', 'home_team'], right_on=['season','division', 'team'], how='left')
    home_team_rank.rename(columns={'delayed_rank': 'home_team_rank'}, inplace=True)
    home_team_rank.drop(columns=['team'], inplace=True)

    away_team_rank = df_useful.merge(df_with_rank, left_on=['season', 'division', 'away_team'], right_on=['season','division', 'team'], how='left')
    away_team_rank.rename(columns={'delayed_rank': 'away_team_rank'}, inplace=True)
    away_team_rank.drop(columns=['team'], inplace=True)

    df_new = away_team_rank.merge(home_team_rank, on=['season', 'division', 'home_team','away_team'], how='left')
    df_new.rename(columns={'match_result_x': 'match_result'},inplace=True)

    df_to_train_season = df_new[['home_team_rank','away_team_rank']]
    df_to_train_season = df_to_train_season.fillna(0)

    df_to_train_matchday[['home_team_rank','away_team_rank']] = df_to_train_season

    return df_to_train_matchday




