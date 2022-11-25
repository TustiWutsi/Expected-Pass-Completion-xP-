import warnings
warnings.filterwarnings('ignore')
import requests
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MultiLabelBinarizer


# get the list of match_id from a given competition and season if you have Wyscout API access (otherwise, you can use free available data)
# credentials = username:password encoded in Base64
def get_match_ids(competition_id, season_id, credentials):
    
    URL = f"https://apirest.wyscout.com/v3/competitions/{competition_id}/matches"
    r = requests.get(url = URL, headers={'Authorization': f'Basic {credentials}'})
    data = r.json()
    
    match_ids = [data['matches'][i]['matchId'] for i in range(len(data['matches'])) if data['matches'][i]['seasonId'] == season_id]
    
    return match_ids



# Transform the dataframe at the right format for modelling
# Those transformations are based on Wyscout V3, some updates should be done for former versions
def prepare_pass_data(match_df):
    
    match_df['match_id'] = [match_df.loc[i,'matchId'] for i in range(len(match_df))]
    match_df['match_period'] = [match_df.loc[i,'matchPeriod'] for i in range(len(match_df))]
    match_df['minute'] = [match_df.loc[i,'minute'] for i in range(len(match_df))]
    match_df['team'] = [match_df.loc[i,'team']['name'] for i in range(len(match_df))]
    match_df['opponent_team'] = [match_df.loc[i,'opponentTeam']['name'] for i in range(len(match_df))]
    match_df['player_name'] = [match_df.loc[i,'player']['name'] for i in range(len(match_df))]
    match_df['player_id'] = [match_df.loc[i,'player']['id'] for i in range(len(match_df))]
    match_df['player_position'] = [match_df.loc[i,'player']['position'] for i in range(len(match_df))]
    
    match_df['eventName'] = [match_df.loc[i,'type']['primary'] for i in range(len(match_df))]
    match_df['subEventName'] = [match_df.loc[i,'type']['secondary'] for i in range(len(match_df))]
    match_df['x_start'] = [match_df.loc[i,'location']['x'] if match_df.loc[i,'location'] != None else None for i in range(len(match_df))]
    match_df['y_start'] = [match_df.loc[i,'location']['y'] if match_df.loc[i,'location'] != None else None for i in range(len(match_df))]
    match_df['carry_progression'] =[match_df.loc[i,'carry']['progression'] if match_df.loc[i,'carry'] != None else None for i in range(len(match_df))]
    match_df['possession_id'] = [match_df.loc[i,'possession']['id'] if match_df.loc[i,'possession'] != None else None for i in range(len(match_df))]
    match_df['possession_team'] = [match_df.loc[i,'possession']['team']['name'] if match_df.loc[i,'possession'] != None else None for i in range(len(match_df))]
    
    match_df['previous_event'] = match_df.eventName.shift(1)
    match_df['previous_subEvent'] = match_df.subEventName.shift(1)
    match_df['previous_carry_progression'] = match_df.carry_progression.shift(1)
    match_df['previous_possession_team'] = match_df.possession_team.shift(1)
    
    match_df['previous_action_teammate'] = match_df.apply(lambda x : 1 if x.possession_team == x.previous_possession_team else 0, axis=1)
    
    passes = match_df[match_df['eventName'] == 'pass'].reset_index(drop=True)
    
    passes['accurate'] = [passes.loc[i,'pass']['accurate'] for i in range(len(passes))]
    passes['angle'] = [passes.loc[i,'pass']['angle'] for i in range(len(passes))]
    passes['height'] = [passes.loc[i,'pass']['height'] for i in range(len(passes))]
    passes['length'] = [passes.loc[i,'pass']['length'] for i in range(len(passes))]
    passes['x_end'] = [passes.loc[i,'pass']['endLocation']['x'] for i in range(len(passes))]
    passes['y_end'] = [passes.loc[i,'pass']['endLocation']['y'] for i in range(len(passes))]
    
    passes['number_passes_possession'] = passes.groupby(['possession_id']).cumcount()+1
    
    passes = passes.drop(columns=['carry'])
    
    mlb = MultiLabelBinarizer(sparse_output=True)

    passes = pd.concat([passes,
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(passes.pop('subEventName')),
                index=passes.index,
                columns=mlb.classes_)], axis=1)
    
    passes['result'] = passes.accurate.apply(lambda x : 1 if x == True else 0)
    
    passes = pd.concat([passes, pd.get_dummies(passes.previous_event, prefix='previous_event')], axis=1)

    passes.height = passes.height.apply(lambda x : x if x != None else 'No')
    passes = pd.concat([passes, pd.get_dummies(passes.height, prefix='height')], axis=1)
    
    passes.previous_carry_progression = passes.previous_carry_progression.fillna(0)
    
    # Here we keep pass attributes that are available both for 2021 and 2022 data
    
    passes_vf = passes[[
       'match_id', 'match_period', 'minute', 'team', 'opponent_team', 'player_name', 'player_id', 'player_position',
       'x_start', 'y_start',
       'previous_carry_progression',
       'previous_action_teammate',
       'angle', 'length', 'x_end', 'y_end',
       'number_passes_possession', 'back_pass',
       'counterpressing_recovery', 'cross',
       'deep_completion', 'forward_pass', 'hand_pass',
       'head_pass', 'key_pass', 'lateral_pass', 'linkup_play', 'long_pass',
       'pass_to_final_third', 'pass_to_penalty_area',
       'progressive_pass', 'recovery', 'short_or_medium_pass',
       'shot_assist', 'through_pass', 'touch_in_box',
       'under_pressure',
       'previous_event_duel', 'previous_event_free_kick',
       'previous_event_goal_kick',
       'previous_event_interception',
       'previous_event_pass', 'previous_event_shot_against',
       'previous_event_throw_in', 'previous_event_touch',
        'height_No', 'height_high', 'height_low','result']]
    
    return passes_vf



def load_and_prepare_pass_data(match_ids, credentials):

    df_pass = pd.DataFrame()
    
    for match in tqdm(match_ids):
        
        URL = f"https://apirest.wyscout.com/v3/matches/{match}/events"
        r = requests.get(url = URL, headers={'Authorization': f'Basic {credentials}'}) #username:password encoded in Base64
        data = r.json()
        df = pd.DataFrame(data['events'])
        df = prepare_pass_data(df)
        df_pass = pd.concat([df_pass,df])
    
    return df_pass



def change_column_type(df):
    for col in df.columns[15:]:
        if df[col].dtypes == 'uint8':
            df[col] = df[col].astype('int')
        else:
            try:
                df[col] = df[col].values.to_dense().astype(np.int64)
            except:
                pass   
    
    return df



# This function returns a dataframe with the total number of minutes played by players
def get_season_minutes_played(df_pass, competition_id, credentials):
    
    df = pd.DataFrame({'player_id':[], 'total_minutes_played':[]})
    players_ids = list(df_pass.player_id.unique())
    
    for player_id in tqdm(players_ids):
        
        URL = f"https://apirest.wyscout.com/v3/players/525236/advancedstats?compId={competition_id}"
        r = requests.get(url = URL, headers={'Authorization': f'Basic {credentials}'})
        data = r.json()
        
        df_player = pd.DataFrame({'player_id': [data['playerId']], 'total_minutes_played': [data['total']['minutesOnField']]})
        df = pd.concat([df, df_player])
    
    return df
