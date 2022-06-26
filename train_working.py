# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 14:37:33 2022

@author: hbinn
"""

#Imports
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


#Load Data
print("Load Data")
load_data = pd.read_csv("C:/Users/hbinn/Documents/Projects/AFL_Fantasy_Predict/Data/afl_database.csv", index_col=[0], low_memory=False)
dob = pd.read_csv("C:/Users/hbinn/Documents/Projects/AFL_Fantasy_Predict/Data/player_dob_draft.csv")
lineups = pd.read_csv("C:/Users/hbinn/Documents/Projects/AFL_Fantasy_Predict/Data/lineups.csv")

# Merge DOB data into dataset and create age variable
print("Merge External Data")
db = load_data.merge(dob, how='inner')

# DateTime Data
print("Manage DateTime Data")
db['dob'] = pd.to_datetime(db['dob'],format='%d/%m/%Y')
db['match_date'] = pd.to_datetime(db['match_date'], format='%d/%m/%Y')
db['match_local_time'] = pd.to_datetime(db['match_local_time'], format='%H:%M:%S')

db['year'] = db['match_date'].dt.year

db['time_category'] = (db['match_local_time'].dt.hour % 24 + 4) // 4
db['time_category'].replace({1: 'Late Night',
                      2: 'Early Morning',
                      3: 'Morning',
                      4: 'Noon',
                      5: 'Evening',
                      6: 'Night'}, inplace=True)

#Limit Data to minimize processing times
print("Time and TOG Limit Data")
db = db[db['year'] > 1980]
db = db[db['time_on_ground_percentage'] > 0 ]
db = db.drop(['player_is_retired','brownlow_votes','supercoach_score','rating_points','subbed'],axis=1)

# Player Data
print("Preprocess Player Demographic Data")
db['name_merge'] = db['player_last_name'] + "_" + db['player_first_name']
db["games_played_total"] = db.groupby("player_id")["player_id"].rank(method="first", ascending=True)
db['player_age'] = db['match_date'] - db['dob']
db['player_age'] = db['player_age'].dt.days / 365
db["games_played_season"] = db.groupby(["player_id",'year'])["player_id"].rank(method="first", ascending=True)
db['days_since_last_game'] = db.groupby('player_id')['match_date'].diff().astype('timedelta64[D]')
db['days_since_last_game'] = db['days_since_last_game'].fillna(7)
db['seasons_played'] = db.groupby(['player_id'])['year'].rank(method='dense')

# Player Position Data
print("Preprocess Player Position Data")
db['player_position_merge'] = db['player_position'].replace(dict.fromkeys(['FPL','FPR'] ,'FP'))
db['player_position_merge'] = db['player_position_merge'].replace(dict.fromkeys(['BPL','BPR'] ,'BP'))
db['player_position_merge'] = db['player_position_merge'].replace(dict.fromkeys(['HBFL','HBFR'] ,'HBF'))
db['player_position_merge'] = db['player_position_merge'].replace(dict.fromkeys(['HFFL','HFFR'] ,'HFF'))
db['player_position_merge'] = db['player_position_merge'].replace(dict.fromkeys(['WL','WR'] ,'W'))

db['player_position_fantasy'] = db['player_position_merge'].replace(dict.fromkeys(['FP','HFF','FF','CHF'],'FWD'))
db['player_position_fantasy'] = db['player_position_fantasy'].replace(dict.fromkeys(['BP','HBF','FB','CHB'],'BACK'))
db['player_position_fantasy'] = db['player_position_fantasy'].replace(dict.fromkeys(['C','R','RR','W'],'MID'))

# Team Data
print("Preprocess Team Data")
db['home_team'] = np.where(db['player_team'] == db['match_home_team'], 1, 0)
db['winning_team'] = np.where(db['player_team'] == db['match_winner'], 1, 0)
db['opposition_team'] = np.where(db['player_team'] == db['match_home_team'], db['match_away_team'], db['match_home_team'])
db['match_id_retain'] = db['match_id']
db['club_game_number'] = db.groupby(["player_team"])["match_id"].rank(method="dense", ascending=True)

# Feature Engineering
print("Feature Transformation")
db['afl_fantasy_score'] = (db['kicks'] * 3) + (db['handballs'] *2) + (db['marks'] * 3) + db['hitouts'] + (db['tackles'] * 4) + (db['free_kicks_for']) + (db['free_kicks_against'] * -3) + (db['goals'] * 6) + (db['behinds'])
#db['kick_hb_ratio'] = (db['kicks'] / db['handballs'])
db['goal_accuracy'] = (db['goals'] / db['shots_at_goal'])
db['score_accuracy'] = ((db['goals'] + db['behinds']) / db['shots_at_goal'])
db['free_kick_diff'] = (db['free_kicks_for'] - db['free_kicks_against']).fillna(0)
db['cont_off_pcnt'] = (db['contest_off_wins'] / db['contest_off_one_on_ones'])
db['cont_def_pcnt'] = ((db['contest_def_one_on_ones'] - (db['contest_def_losses'])) / db['contest_def_one_on_ones'])
db['kick_eff_pcnt'] = (db['effective_kicks'] / db['kicks'])
db['effective_handballs'] = (db['effective_disposals'] - db['effective_kicks']).fillna(0)
db['handball_eff_pcnt'] = ((db['effective_disposals'] - db['effective_kicks']) / db['handballs'])
db['contested_possession_rate'] = (db['contested_possessions'] / (db['contested_possessions'] + db['uncontested_possessions']))
db['hitout_win_percentage'] = np.nan
db['hitout_win_percentage'] = (db['hitouts'] / db['ruck_contests'])
db['hitout_eff_pcnt'] = (db['hitouts_to_advantage'] / db['hitouts'])
db['hitout_adv_pcnt'] = (db['hitouts_to_advantage'] / db['ruck_contests'])

# Select Numeric Variables for Operations 
player_variables_numeric = [
                    'kicks',
                    'marks',
                    'handballs',
                    'disposals',
                    'effective_disposals',
                    'goals',
                    'behinds',
                    'hitouts',
                    'tackles',
                    'rebounds',
                    'inside_fifties',
                    'clearances',
                    'clangers',
                    'free_kicks_for',
                    'free_kicks_against',
                    'contested_possessions',
                    'uncontested_possessions',
                    'contested_marks',
                    'marks_inside_fifty',
                    'one_percenters',
                    'bounces',
                    'goal_assists',
                    'centre_clearances',
                    'afl_fantasy_score',
                    'centre_clearances',
                    'stoppage_clearances',
                    'score_involvements',
                    'metres_gained',
                    'turnovers',
                    'intercepts',
                    'tackles_inside_fifty',
                    'contest_def_losses',
                    'contest_def_one_on_ones',
                    'contest_off_one_on_ones',
                    'contest_off_wins',
                    'def_half_pressure_acts',
                    'effective_kicks',
                    'f50_ground_ball_gets',
                    'ground_ball_gets',
                    'hitouts_to_advantage',
                    'intercept_marks',
                    'marks_on_lead',
                    'pressure_acts',
                    'ruck_contests',
                    'score_launches',
                    'shots_at_goal',
                    'spoils',
                    'effective_handballs'
                    ]

player_variables_pcnts = [
                    'disposal_efficiency_percentage',
                    'time_on_ground_percentage',
                    'hitout_win_percentage',
                    'goal_accuracy',
                    'score_accuracy',
                    'cont_off_pcnt',
                    'cont_def_pcnt',
                    'kick_eff_pcnt',
                    'handball_eff_pcnt',
                    'hitout_eff_pcnt',
                    'winning_team'
                    ]

team_variables_numeric = [
                    
                    'match_attendance',
                    'guernsey_number',
                    'match_weather_temp_c'
                    ]

misc_drop = [
                    'player_id',
                ]

# Select, Shift and Stack Player Level Data for Model Training
players = []

    # FUnction to Select Data by Player if they played in 2012 or beyond.
def player_stats(data, player_id):
    player_data = data[data['player_id'] == player_id]
    player_data = player_data.sort_values(by=['year'])
    if player_data['year'].max() < 2012:
        pass
    else:
        return player_data
    
    # Select player data
print("Select and Shift Player Data")
for player in db['player_id'].unique():
    dset = player_stats(db, player)
    dset = dset.sort_values(by=['match_id'])
    print(dset['name_merge'].iloc[0])

    # Add pre game - 1 row for predictions
    row_dict = {'name_merge':[dset['name_merge'].iloc[0]],'player_id':dset['player_id'].iloc[0],'games_played_total':[0]}
    first_game_row = pd.DataFrame(row_dict)
    dset = first_game_row.append(dset, ignore_index=True)
        
    # Forward shift certain variables to be pre or post game.
    dset['afl_fantasy_score_next_match'] = dset['afl_fantasy_score'].shift(-1)
    dset['venue_name'] = dset['venue_name'].shift(-1)
    dset['match_attendance'] = dset['match_attendance'].shift(-1)
    dset['match_id'] = dset['match_id'].shift(-1)
    dset['match_home_team'] = dset['match_home_team'].shift(-1)
    dset['match_away_team'] = dset['match_away_team'].shift(-1)
    dset['match_date'] = dset['match_date'].shift(-1)
    dset['match_local_time'] = dset['match_local_time'].shift(-1)
    dset['match_round'] = dset['match_round'].shift(-1)
    dset['match_weather_temp_c'] = dset['match_weather_temp_c'].shift(-1)
    dset['match_weather_type'] = dset['match_weather_type'].shift(-1)
    dset['player_id'] = dset['player_id'].shift(-1).fillna(method='ffill')
    dset['player_first_name'] = dset['player_first_name'].shift(-1).fillna(method='ffill')
    dset['player_last_name'] = dset['player_last_name'].shift(-1).fillna(method='ffill')
    dset['player_height_cm'] = dset['player_height_cm'].shift(-1).fillna(method='ffill')
    dset['player_weight_kg'] = dset['player_weight_kg'].shift(-1).fillna(method='ffill')
    dset['player_team'] = dset['player_team'].shift(-1).fillna(method='ffill')
    dset['guernsey_number'] = dset['guernsey_number'].shift(-1).fillna(method='ffill').fillna(method='ffill')
    dset['player_position'] = dset['player_position'].shift(-1)
    dset['name_merge_push'] = dset['name_merge'].fillna(method='ffill')
    dset['name_merge'] = dset['name_merge'].shift(-1).fillna(method='ffill')
    dset['dob'] = dset['dob'].shift(-1).fillna(method='ffill')
    dset['year'] = dset['year'].shift(-1)
    dset['year_separate'] = dset['year'].shift(-1)
    dset['player_position_merge'] = dset['player_position_merge'].shift(-1)
    dset['player_position_fantasy'] = dset['player_position_fantasy'].shift(-1)
    dset['opposition_team'] = dset['opposition_team'].shift(-1)
    dset['home_team'] = dset['home_team'].shift(-1)
    dset['days_since_last_game'] = dset['days_since_last_game'].shift(-1)
    dset['club_game_number_club'] = dset['club_game_number'].fillna(method='bfill')
    dset.at[0,'club_game_number_club'] = dset.at[0,'club_game_number'] - 1
    dset['player_age'] = dset['player_age'].shift(-1)
    dset['games_played_season'] = dset['games_played_season'].fillna(0)
    dset['seasons_played'] = dset['seasons_played'].fillna(0)
    dset['time_category'] = dset['time_category'].shift(-1)
    players.append(dset)

df_players = pd.concat(players)    
df_players = df_players.copy()

df_club_max_player = df_players.groupby(['player_team','match_id'])['club_game_number'].max().reset_index()
df_club_max_player = pd.DataFrame(df_club_max_player)
df_club_max_player = df_club_max_player.rename(columns={"club_game_number":"club_game_number_player"})
df_players = df_players.merge(df_club_max_player,how='left', on = ['player_team','match_id'])

# Create Rolling Statistics

# Compute Rolling Average for Player Level Numerical Variables  
def rolling_var_player(df, var, window):
    new_col = var + "_" + str(window)
    df[new_col] = df.groupby(['player_id'])[str(var)].transform(lambda x: x.rolling(window,min_periods=1).mean())
    
print("Compute Playing Rolling Average Data - Player")     
for player_variable in player_variables_numeric:
    print("Processing: ", player_variable)
    for x in range(5,11,5):
        rolling_var_player(df_players,player_variable, x)

for player_variable in player_variables_pcnts:
    print("Processing: ", player_variable)
    for x in range(5,11,5):
        rolling_var_player(df_players,player_variable, x)

# Create Team Level Statistics - LineUps

print("Create Team Comparison Stats")
df_lineups = df_players.groupby(by=['match_id','player_team'],dropna=False).agg(['sum','mean'])
df_lineups.columns = df_lineups.columns.map('{0[0]}_{0[1]}'.format)
df_lineups = df_lineups.reset_index()  
df_lineups = df_lineups.rename(columns=lambda x: "team_players_" + x).rename(columns={"team_players_player_team":"player_team","team_players_match_id":"match_id",
                                                                               "team_players_opposition_team":"opposition_team",'team_players_year':'year',            
                                                                               "match_id":"match_id_old"
                                                                               })

for var in player_variables_pcnts:
    drop_var = "team_players_" + var + "_sum"
    print("Dropping: ",drop_var)
    df_lineups = df_lineups.drop(drop_var,axis=1)
    
for var in team_variables_numeric:
    drop_var = "team_players_" + var + "_sum"
    drop_var2 = "team_players_" + var + "_mean"
    print("Dropping: ",drop_var)
    df_lineups = df_lineups.drop(drop_var,axis=1)
    df_lineups = df_lineups.drop(drop_var2,axis=1)
    
for var in misc_drop:
    drop_var = "team_players_" + var + "_sum"
    drop_var2 = "team_players_" + var + "_mean"
    print("Dropping: ",drop_var)
    df_lineups = df_lineups.drop(drop_var,axis=1)
    df_lineups = df_lineups.drop(drop_var2,axis=1)
    
# Create Team Level Statistics - Team

print("Create Team Comparison Stats")
df_team = db.groupby(by=['match_id','player_team','club_game_number'], dropna=False).agg(['sum','mean'])
df_team.columns = df_team.columns.map('{0[0]}_{0[1]}'.format)
df_team = df_team.reset_index()
df_team = df_team.rename(columns=lambda x: "team_match_" + x).rename(columns={"team_match_player_team":"player_team","team_match_match_id":"match_id_retain",
                                                                               "team_match_opposition_team":"opposition_team",'team_match_year':'year',
                                                                               "team_match_club_game_number":"club_game_number","opposition_team_y":"opposition_team",
                                                                              
                                                                             
                                                                              })

for var in player_variables_pcnts:
    drop_var = "team_match_" + var + "_sum"
    #print("Dropping: ",drop_var)
    df_team = df_team.drop(drop_var,axis=1)
    
for var in team_variables_numeric:
    drop_var = "team_match_" + var + "_sum"
    drop_var2 = "team_match_" + var + "_mean"
    #print("Dropping: ",drop_var)
    df_team = df_team.drop(drop_var,axis=1)
    df_team = df_team.drop(drop_var2,axis=1)
    
for var in misc_drop:
    drop_var = "team_match_" + var + "_sum"
    drop_var2 = "team_match_" + var + "_mean"
    #print("Dropping: ",drop_var)
    df_team = df_team.drop(drop_var,axis=1)
    df_team = df_team.drop(drop_var2,axis=1)
      
# Compute Rolling Average for Team Level Numerical Variables
def rolling_var_team(df, var, window):
    new_col = var + "_" + str(window)
    df[new_col] = df.groupby(['player_team'])[str(var)].transform(lambda x: x.rolling(window,min_periods=1).mean())
    
for player_variable in player_variables_numeric:
    #print("Processing: ","team_players_" + player_variable)
    team_var_roll = "team_players_" + player_variable + "_sum"
    for x in range(5,16,5):
        rolling_var_team(df_lineups,team_var_roll, x)

for player_variable in player_variables_numeric:
    #print("Processing: ","team_players_" + player_variable)
    team_var_roll = "team_players_" + player_variable + "_mean"
    for x in range(5,16,5):
        rolling_var_team(df_lineups,team_var_roll, x)
        
for player_variable in player_variables_pcnts:
    #print("Processing: ","team_players_" + player_variable)
    team_var_roll = "team_players_" + player_variable + "_mean"
    for x in range(5,16,5):
        rolling_var_team(df_lineups,team_var_roll, x)
        
# Compute Rolling Average for Team - Match Level Numerical Variables        
        
for player_variable in player_variables_numeric:
    #print("Processing: ","team_match_" + player_variable)
    team_var_roll = "team_match_" + player_variable + "_sum"
    for x in range(5,16,5):
        rolling_var_team(df_team,team_var_roll, x)

for player_variable in player_variables_numeric:
    #print("Processing: ","team_match_" + player_variable)
    team_var_roll = "team_match_" + player_variable + "_mean"
    for x in range(5,16,5):
        rolling_var_team(df_team,team_var_roll, x)
        
for player_variable in player_variables_pcnts:
    print("Processing: ","team_match_" + player_variable)
    team_var_roll = "team_match_" + player_variable + "_mean"
    for x in range(5,16,5):
        rolling_var_team(df_team,team_var_roll, x)
        
# # Create Opposition Level Statistics - Per Game
# print("Create Opposition Statistics Row")
# opp_stats = df_team.rename(columns=lambda x: "opp_player_" + x)
# opp_stats = opp_stats.rename(columns={"opp_player_player_team":"opposition_team","opp_player_opposition_team":"player_team",'opp_player_match_id':'match_id'})

# def rolling_var_opp(df, var, window):
#     new_col = var + "_" + str(window)
#     df[new_col] = df.groupby(['player_team'])[str(var)].transform(lambda x: x.rolling(window,min_periods=1).mean())

# for player_variable in player_variables_numeric:
#     print("Processing: ","opp_player_team_match_" + player_variable)
#     team_var_roll = "opp_player_team_match_" + player_variable + "_sum"
#     for x in range(5,16,5):
#         rolling_var_team(opp_stats,team_var_roll, x)

# for player_variable in player_variables_numeric:
#     print("Processing: ","opp_player_team_match_" + player_variable)
#     team_var_roll = "opp_player_team_match_" + player_variable + "_mean"
#     for x in range(5,16,5):
#         rolling_var_team(opp_stats,team_var_roll, x)
        
# for player_variable in player_variables_pcnts:
#     print("Processing: ","opp_player_team_match_" + player_variable)
#     team_var_roll = "opp_player_team_match_" + player_variable + "_mean"
#     for x in range(5,16,5):
#         rolling_var_team(opp_stats,team_var_roll, x)

#input("check home team and winning team in df_players")

df_players = df_players.drop(columns = ['match_home_team_goals','match_home_team_behinds','match_home_team_score',
                                        'match_away_team_goals','match_away_team_behinds','match_away_team_score',
                                        'match_margin','match_winner','winning_team','name_merge_push',
                                        'match_id_retain',])

df_players['club_game_number'] = df_players['club_game_number_player']

mysubset = ['player_id','name_merge', 'player_first_name','player_last_name','player_height_cm','player_weight_kg','guernsey_number','dob','player_age','games_played_total',
            'games_played_season','seasons_played','days_since_last_game','club_game_number','club_game_number_player',
            'player_team','opposition_team','player_position','player_position_merge','player_position_fantasy','match_id','venue_name','match_date','match_local_time','year','year_separate','time_category','match_round',
            'home_team','afl_fantasy_score','afl_fantasy_score_next_match']
othercols = [c for c in df_players.columns if c not in mysubset]
df_players = df_players[mysubset+othercols].reset_index()



df_team = df_team.drop(columns = ['team_match_match_home_team_goals_sum','team_match_match_home_team_behinds_sum','team_match_match_home_team_score_sum',
                                  'team_match_match_away_team_goals_sum','team_match_match_away_team_behinds_sum','team_match_match_away_team_score_sum',
                                  'team_match_match_margin_sum','team_match_player_height_cm_sum','team_match_player_weight_kg_sum',
                                  'team_match_initial_draft_position_sum','team_match_initial_draft_position_mean','team_match_initial_draft_year_sum',
                                  'team_match_initial_draft_year_mean','team_match_recent_draft_position_sum','team_match_recent_draft_position_mean',
                                  'team_match_recent_draft_year_sum','team_match_recent_draft_year_mean','team_match_year_sum','team_match_year_mean',
                                  'team_match_player_age_sum','team_match_days_since_last_game_sum','team_match_seasons_played_sum',
                                  'team_match_home_team_sum','team_match_match_id_retain_sum'])

df_team = df_team.rename(columns= {"club_game_number_player":"club_game_number"})

df_lineups = df_lineups.drop(columns = ['team_players_match_home_team_goals_sum','team_players_match_home_team_behinds_sum','team_players_match_home_team_score_sum',
                                  'team_players_match_away_team_goals_sum','team_players_match_away_team_behinds_sum','team_players_match_away_team_score_sum',
                                  'team_players_match_margin_sum','team_players_player_height_cm_sum','team_players_player_weight_kg_sum',
                                  'team_players_initial_draft_position_sum','team_players_initial_draft_position_mean','team_players_initial_draft_year_sum',
                                  'team_players_initial_draft_year_mean','team_players_recent_draft_position_sum','team_players_recent_draft_position_mean',
                                  'team_players_recent_draft_year_sum','team_players_recent_draft_year_mean','team_players_year_sum','team_players_year_mean',
                                  'team_players_player_age_sum','team_players_days_since_last_game_sum','team_players_seasons_played_sum',
                                  'team_players_home_team_sum','team_players_match_id_retain_sum'])

df_team_opp = df_team.rename(columns = {'player_team':'opposition_team'})

df_lineups_opp = df_lineups.rename(columns = {'player_team':'opposition_team'})

# Produce Team Level % Stats - Override Others

# Merge Player, Team and Game Data    
print("Merge Team and Match Comparison Stats")
df_final = df_players.merge(df_team, how='outer', on=['player_team','club_game_number'])
df_final = df_final.merge(df_lineups, how='left', on=['player_team','match_id'])
#df_final = df_final.merge(df_team_opp, how='left', on=['opposition_team','match_id_retain'],suffixes=['','_opp_team'])
#df_final = df_final.merge(df_lineups_opp, how='left', on=['opposition_team','match_id'],suffixes=['','_opp_lineups'])

# # Compute Player v Team and Player v Match Comparison Variables
# print("Compute Player v Team and Player v Match Proportional Stats")
# def compute_proportional_stats(df,var,comparison):
#    new_col = comparison + "_" + var + "_proportion"
#    prop_col = comparison + "_" + var + "_sum" 
#    df[new_col] = df[var] / df[prop_col]
   
# for var in player_variables_numeric:
#     compute_proportional_stats(db,var,"team")
#     compute_proportional_stats(db,var,"match")
    
# for player_variable in player_variables_numeric:
#     print("Processing: ","team_" + player_variable + "_proportion")
#     team_var_roll = "team_" + player_variable + "_proportion"
#     for x in range(5,16,5):
#         rolling_var_team(db,team_var_roll, x)
        
# for player_variable in player_variables_numeric:
#     print("Processing: ","team_" + player_variable + "_proportion")
#     team_var_roll = "team_" + player_variable + "_proportion"
#     for x in range(5,16,5):
#         rolling_var_match(db,match_var_roll, x)

# Remove data prior to 2012
print("Time Limit Player Data > 2011")
df_final = df_final[df_final['year'] > 2011]

#Categorical Variables to Dummy Variables
print("Compute Dummies for Categorical Variables")
categorical_col_updated = ['player_position_merge','player_position_fantasy','year_separate','venue_name','match_weather_type','time_category']
df_final = pd.get_dummies(df_final, columns=categorical_col_updated)    

# Produce Correlation Table to Check Relationships
#print("Compute Variables Correlations")
#correlations = df_players.corr()

df_reset_point = df_final.copy()

# Drop Unused Variables from DataFrame
print("Selecting Variables for Use in Model Training")
# Select Variables for Use in Model
df_final = df_reset_point.copy()

df_final = df_final.drop(columns=[
                              'index','player_first_name','player_last_name','dob','club_game_number','club_game_number_player',
                              'player_team','opposition_team','player_position','match_date','match_local_time','afl_fantasy_score',
                              'match_home_team','match_away_team','match_attendance','hitout_win_percentage','handball_eff_pcnt',
                              'initial_draft_type','Signing','initial_draft_position','initial_draft_year','recent_draft_type','recent_draft_signing',
                              'recent_draft_position','recent_draft_year','goal_accuracy','score_accuracy','cont_off_pcnt','cont_def_pcnt',
                              'hitout_eff_pcnt','hitout_adv_pcnt','club_game_number_club','hitout_win_percentage_5','hitout_win_percentage_10',
                              'goal_accuracy_5','goal_accuracy_10','score_accuracy_5','score_accuracy_10','cont_off_pcnt_5','cont_off_pcnt_10',
                              'cont_def_pcnt_5','cont_def_pcnt_10','hitout_eff_pcnt_5','hitout_eff_pcnt_10','match_id_retain',
                              'match_round',
                              'team_players_afl_fantasy_score_next_match_sum','team_players_afl_fantasy_score_next_match_mean',

                              'player_height_cm','player_weight_kg'
                              ])
    
    
#df_players = df_players.fillna(0)
# df_players = df_players[[
#          'year','match_id','home_team',
         
#          'year_separate_2012.0','year_separate_2013.0','year_separate_2014.0','year_separate_2020.0',
         
#          'name_merge','player_id','games_played_total','player_age','guernsey_number','player_height_cm','player_weight_kg','seasons_played',#'days_since_last_game',
         
#          'kicks','kicks_5','kicks_10','kicks_15',
#          'marks_5','marks_10','marks_15',
#          'handballs','handballs_5','handballs_10','handballs_15',
#          'disposals','disposals_5','disposals_10','disposals_15',
#          'effective_disposals','effective_disposals_5','effective_disposals_10','effective_disposals_15',
#          'goals','goals_5','goals_10','goals_15',
#          'behinds','behinds_5','behinds_10','behinds_15',
#          'hitouts','hitouts_5','hitouts_10','hitouts_15',
#          'tackles','tackles_5','tackles_10','tackles_15',
#          'rebounds','rebounds_5','rebounds_10','rebounds_15',
#          'inside_fifties','inside_fifties_5','inside_fifties_10','inside_fifties_15',
#          'clearances','clearances_5','clearances_10','clearances_15',
#          'clangers','clangers_5','clangers_10','clangers_15',
#          'free_kicks_for','free_kicks_for_5','free_kicks_for_10','free_kicks_for_15',
#          'free_kicks_against','free_kicks_against_5','free_kicks_against_10','free_kicks_against_15',
#          'contested_possessions','contested_possessions_5','contested_possessions_10','contested_possessions_15',
#          'uncontested_possessions','uncontested_possessions_5','uncontested_possessions_10','uncontested_possessions_15',
#          'time_on_ground_percentage','time_on_ground_percentage_5','time_on_ground_percentage_10','time_on_ground_percentage_15',
#          'afl_fantasy_score','afl_fantasy_score_5','afl_fantasy_score_10','afl_fantasy_score_15',
#          'metres_gained','metres_gained_5','metres_gained_10','metres_gained_15',
#          'centre_clearances','centre_clearances_5','centre_clearances_10','centre_clearances_15',
#          'contest_def_one_on_ones_5','contest_def_one_on_ones_10','contest_def_one_on_ones_15',
#          'ground_ball_gets','ground_ball_gets_5','ground_ball_gets_10','ground_ball_gets_15',
#          'f50_ground_ball_gets','f50_ground_ball_gets_5','f50_ground_ball_gets_10','f50_ground_ball_gets_15',
#          'pressure_acts','pressure_acts_5','pressure_acts_10','pressure_acts_15',
#          'spoils','spoils_5','spoils_10','spoils_15',
#          'goal_accuracy','goal_accuracy_5','goal_accuracy_10','goal_accuracy_15',
#          'score_accuracy','score_accuracy_5','score_accuracy_10','score_accuracy_15',
#          #'free_kick_diff','free_kick_diff_5','free_kick_diff_10','free_kick_diff_15',
#          'cont_off_pcnt_5','cont_off_pcnt_10',
#          'cont_def_pcnt_5','cont_def_pcnt_10',
#          'kick_eff_pcnt_5','kick_eff_pcnt_10',
#          'effective_handballs_5','effective_handballs_10',
#          'handball_eff_pcnt_5','handball_eff_pcnt_10',
#          #'contested_possession_rate_5','contested_possession_rate_10',
#          'hitout_eff_pcnt','hitout_eff_pcnt_5','hitout_eff_pcnt_10','hitout_eff_pcnt_15',
         
#          'player_position_fantasy_BACK','player_position_fantasy_MID','player_position_fantasy_FWD','player_position_fantasy_RK','player_position_fantasy_INT',
#          'player_position_fantasy_SUB',
         
#          'player_position_merge_FF','player_position_merge_FP','player_position_merge_CHF','player_position_merge_HFF',
#          'player_position_merge_FB','player_position_merge_BP','player_position_merge_CHB','player_position_merge_HBF',
#          'player_position_merge_W','player_position_merge_C','player_position_merge_R','player_position_merge_RR','player_position_merge_RK',
#          'player_position_merge_SUB',
         
#          'time_category_Noon','time_category_Evening',
         
#          'match_weather_type_MOSTLY_SUNNY','match_weather_type_OVERCAST','match_weather_type_RAIN','match_weather_type_SUNNY','match_weather_type_MOSTLY_CLEAR',
#          'match_weather_type_CLEAR_NIGHT','match_weather_type_WINDY','match_weather_type_ROOF_CLOSED','match_weather_type_THUNDERSTORMS','match_weather_type_WINDY_RAIN',
         
#          'team_spoils_proportion','team_spoils_proportion_5','team_spoils_proportion_10','team_spoils_proportion_15',
#          'team_afl_fantasy_score_proportion','team_afl_fantasy_score_proportion_5','team_afl_fantasy_score_proportion_10','team_afl_fantasy_score_proportion_15',
#          'team_disposals_proportion','team_disposals_proportion_5','team_disposals_proportion_10','team_disposals_proportion_15',
#          'team_effective_disposals_proportion','team_effective_disposals_proportion_5','team_effective_disposals_proportion_10','team_effective_disposals_proportion_15',
#          'team_contested_possessions_proportion','team_contested_possessions_proportion_5','team_contested_possessions_proportion_10','team_contested_possessions_proportion_15',
#          'team_clearances_proportion','team_clearances_proportion_5','team_clearances_proportion_10','team_clearances_proportion_15',
         
#          'team_afl_fantasy_score_sum','team_afl_fantasy_score_sum_5','team_afl_fantasy_score_sum_10','team_afl_fantasy_score_sum_15',
#          'team_disposals_sum','team_disposals_sum_5','team_disposals_sum_10','team_disposals_sum_15',
         
#          'team_afl_fantasy_score_mean','team_afl_fantasy_score_mean_5','team_afl_fantasy_score_mean_10','team_afl_fantasy_score_mean_15',
#          'team_disposals_mean','team_disposals_mean_5','team_disposals_mean_10','team_disposals_mean_15',
         
#          'winning_team','winning_team_5','winning_team_10','winning_team_15',
         
#          'opp_team_afl_fantasy_score_sum','opp_team_afl_fantasy_score_sum_5','opp_team_afl_fantasy_score_sum_10','opp_team_afl_fantasy_score_sum_15',
#          'opp_team_disposals_sum','opp_team_disposals_sum_5','opp_team_disposals_sum_10','opp_team_disposals_sum_15',
#          'opp_team_tackles_sum','opp_team_tackles_sum_5','opp_team_tackles_sum_10','opp_team_tackles_sum_15',
         
#          'opp_team_afl_fantasy_score_mean','opp_team_afl_fantasy_score_mean_5','opp_team_afl_fantasy_score_mean_10','opp_team_afl_fantasy_score_mean_15',
#          'opp_team_disposals_mean','opp_team_disposals_mean_5','opp_team_disposals_mean_10','opp_team_disposals_mean_15',
#          'opp_team_tackles_mean','opp_team_tackles_mean_5','opp_team_tackles_mean_10','opp_team_tackles_mean_15',
         
#          'opp_team_winning_team_mean','opp_team_winning_team_mean_5','opp_team_winning_team_mean_10','opp_team_winning_team_mean_15',
         
#          'afl_fantasy_score_next_match'
#          ]]

# Extract Next Game Rows for Predictions
players_list = []
next_game_list = []

print("Separate Player Next Game for Predictions")
for player in df_final['player_id'].unique():
    dset = player_stats(df_final, player)
    dset = dset.sort_values(by=['match_id'])
    dset[0:1] = dset[0:1].fillna(0)
    players_list.append(dset[:-1])
    next_game_list.append(dset.iloc[-1])

df_final = pd.concat(players_list)    
df_final = df_final.copy()

# Remove Rows with Missing Data
print("Drop Rows with Missing Values")
# = df_players.copy()
df_final = df_final.sort_values(by=['player_id','games_played_total'])
df_final = df_final.dropna()

#TBA - to Handle
df_predict = pd.concat(next_game_list,axis=1).T
df_predict = df_predict.copy()


# Extract 3x Games for Test Data
df_final = df_final[df_final['time_on_ground_percentage'] > 0]
df_test_games = df_final[(df_final['match_id'] == 15806) | (df_final['match_id'] == 15801) | (df_final['match_id'] == 15802)]
df_final = df_final[(df_final['match_id'] != 15806) | (df_final['match_id'] != 15801) | (df_final['match_id'] != 15802)]

# Prepare Dataset for Training
print("Prepare Data for Training")
X = df_final
y = X['afl_fantasy_score_next_match']
X = X.drop(['name_merge','player_id','afl_fantasy_score_next_match','year','match_id'],axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1,random_state=1)

X_test = df_test_games
y_test = X_test['afl_fantasy_score_next_match']
X_test = X_test.drop(['name_merge','player_id','afl_fantasy_score_next_match','year','match_id'],axis=1)

model = XGBRegressor(n_estimators=100, max_depth=3,eta = 0.27,colsample_bytree=0.5)
model.fit(X_train, y_train)
score = model.score(X_train, y_train)

print("XGBModel: ", score)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

mae_val = mean_absolute_error(y_val, y_val_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_val = mean_squared_error(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
r2_val = r2_score(y_val, y_val_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Validation Set: - ")
print("   -  MAE: ", mae_val)
print("   -  MSE: ", mse_val)
print("   -  RMSE: ", rmse_val)
print("   -  R2 Score: ", r2_val)

print("Test Set: - ")
print("   -  MAE: ", mae_test)
print("   -  MSE: ", mse_test)
print("   -  RMSE: ", rmse_test)
print("   -  R2 Score: ", r2_test)

importances = model.feature_importances_
df3 = pd.DataFrame(importances).T
df3.columns = X.columns
df3.reset_index()
df3 = df3.T
df3 = df3.rename(columns={0:'importances'})
df3 = df3.sort_values(by='importances')


pred_arr = pd.DataFrame(y_test_pred)
pred_arr.columns = ['prediction']
test_out_r = df_test_games.reset_index()
test_out = pd.concat([test_out_r,pred_arr],axis=1).reset_index()
test_out = test_out[['name_merge','match_id','home_team','player_id','prediction','afl_fantasy_score_next_match']]
test_out = test_out.sort_values(by=['match_id','home_team','player_id'])


extract_team = test_out[test_out['match_id'] == 15806]
extract_team = extract_team[extract_team['home_team'] == 1]
extract_team = extract_team[['name_merge','afl_fantasy_score_next_match','prediction']]
extract_team = extract_team.sort_values(by='name_merge')

df_predict_lineups = lineups.merge(df_predict,on=[])
df_predict.to_csv("next_game_predict.csv")

# next_player_predict_data = pd.read__test_pred)
# 

#csv("C:/Users/hbinn/Documents/Projects/AFL_Fantasy_Predict/Data/next_game_predict.csv", index_col=[0], low_memory=False)

# 

# sns.scatterplot(data=df_players, x='games_played_total',y='afl_fantasy_score_next_match')

#corr, _ = pearsonr(df_players['afl_fantasy_score_next_match'],df_players['games_played_total'])
#print('Games Played Total - Fantasy Next Game - Pearsons correlation: %.3f' % corr)
#corr, _ = pearsonr(df_players['afl_fantasy_score_next_match'],df_players['kicks'])
#print('Kicks - Fantasy Next Game - Pearsons correlation: %.3f' % corr)

# good_vars = corr_check['afl_fantasy_score'].sort_values()

# def next_game_predict(data, player_name, venue_name,home_team, away_team,date, game_time,match_round,temp_c,weather,position):
#     dset = data[data['name_merge'] == player_name]
    
#     return dset
    
#     dset.at[[0], "venue_name"] = venue_name
#     dset.at[[0], "match_home_team"] = home_team
#     dset.at[[0], "match_away_team"] = away_team
#     dset.at[[0], "match_date"] = date
#     dset.at[[0], "game_time"] = game_time
#     dset.at[[0], "match_round"] = match_round
#     dset.at[[0], "match_weather_temp_c"] = temp_c
#     dset.at[[0], "match_weather_type"] = weather
#     dset.at[[0], "player_position"] = position
    
    
#     return dset
    
# new_row_predict_test = next_game_predict(df_players, 'Amon_Karl')
    

    
    
    

