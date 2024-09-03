# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utility import TeamSeason
from scipy.stats import probplot
for dirname, _, filenames in os.walk('./data'):
    print(dirname)
    print(filenames)
    for filename in filenames:
        if 'events' in filename:
            events = pd.read_csv(os.path.join(dirname, filename))
        if 'ginf' in filename:
            games = pd.read_csv(os.path.join(dirname, filename))

# %%
games.head()

# %%
# games data set plots

# no. of games played per year
year_wise_game_count = games.groupby('season').size()
year_wise_game_count.plot(kind='bar', rot=0, color='lightblue')
plt.title('# of games played per year')
plt.xlabel('Year')
plt.ylabel('Games played')
plt.show()


# %%
# no. of games per year per country

countrywise = games.groupby(['country', 'season']).size().reset_index(name='count')
sns.lineplot(x="season", y="count", hue="country", data=countrywise, marker="o")

plt.xlabel("Year")
plt.ylabel("Number of games")
plt.title("Number of Games per Year for Each Country")


plt.show()

# %%
# total wins, losses and draws
# total wins losses and draws per country
games['is_draw'] = games['ftag'] == games['fthg']
games['outcome'] = games.apply(lambda row: 'Away Team won' if row['ftag'] > row['fthg'] else ('Home Team won' if row['ftag'] < row['fthg'] else 'Draw'), axis=1)
draw_splits = games['is_draw'].value_counts()
outcome_splits = games['outcome'].value_counts()


fig, axs = plt.subplots(1, 2, figsize=(10, 4))


axs[0].pie(draw_splits, labels=['No draw' if not x else 'Draw' for x in draw_splits.index], autopct='%1.1f%%', startangle=90)
axs[0].set_title('Win vs Draw Ratio')


axs[1].pie(outcome_splits, labels=outcome_splits.index, autopct='%1.1f%%', startangle=90)
axs[1].set_title('Home vs Away Team win splits')

plt.tight_layout()

plt.show()

# %%
# country wise plot for the above
countries = games['country'].unique()
seasons = games['season'].unique()
fig, axes = plt.subplots(nrows=len(countries), ncols=len(seasons), figsize=(25, 5 * len(countries)))

for i, country in enumerate(countries):
    country_data = games[games['country'] == country]
    # Iterate through unique years for the current country
    for j,season in enumerate(seasons):
        year_data = country_data[country_data['season'] == season]
        outcome_counts = year_data['outcome'].value_counts()

        axes[i][j].pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%',
                    startangle=90, wedgeprops=dict(width=0.3), textprops={'fontsize': 8})
        
        axes[i][j].set_title(f"{country.title()} - {season}")
        axes[i][j].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.tight_layout()
plt.show()

# %%
# no. of games with particular scoreline
games['scoreline'] = games['fthg'].astype(str) + '-' + games['ftag'].astype(str)
scoreline_counts = games['scoreline'].value_counts()

# Plotting
plt.figure(figsize=(10, 6))
scoreline_counts.sort_values(ascending=False).head(10).plot(kind='bar', color='skyblue')

# Adding count labels on top of the bars
for i, count in enumerate(scoreline_counts.sort_values(ascending=False).head(10) ):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

plt.xlabel('Scoreline')
plt.ylabel('Count')
plt.title('Occurrences of Each Scoreline')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()

# %%
# dist plot of home teams goals vs away teams goals

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

# Plotting the distribution of home team goals
sns.histplot(games['fthg'], kde=True, color='skyblue', ax=axes[0])
axes[0].set_title('Distribution of Home Team Goals')
axes[0].set_xlabel('Number of Home Team Goals')
axes[0].set_ylabel('Frequency')

# Plotting the distribution of away team goals
sns.histplot(games['ftag'], kde=True, color='orange', ax=axes[1])
axes[1].set_title('Distribution of Away Team Goals')
axes[1].set_xlabel('Number of Away Team Goals')
axes[1].set_ylabel('Frequency')


plt.tight_layout()

plt.show()

# %%
# form dataframe for the above team's season wise stats

team_info = dict()
for index, row in games.iterrows():
    home_team = row['ht'] + '-' + str(row['season'])
    away_team = row['at'] + '-' + str(row['season'])
    
    # add in team_info dict
    if home_team not in team_info:
        team_info[home_team] = TeamSeason(row['ht'], row['season'])
    if away_team not in team_info:
        team_info[away_team] = TeamSeason(row['at'], row['season'])
        
    # adding to their statistic
    team_info[home_team].addGoalsScored(row['fthg'])
    team_info[home_team].addGoalsConceded(row['ftag'])
    team_info[away_team].addGoalsScored(row['ftag'])
    team_info[away_team].addGoalsConceded(row['fthg'])
    
    if row['ftag'] > row['fthg']:
        team_info[away_team].addWin()
        team_info[home_team].addLoss()
    elif row['ftag'] < row['fthg']:
        team_info[away_team].addLoss()
        team_info[home_team].addWin()
    else:
        team_info[away_team].addDraw()
        team_info[home_team].addDraw()

        
teams_list = []
for teams, data in team_info.items():
    teams_list.append(data.export())
    
teams_seasons = pd.DataFrame(teams_list)
teams_seasons.head()

# %%
# pair plot to check if wins and losses are related to goals scored and conceded

wins_losses = teams_seasons[['wins', 'losses',  'goals_scored', 'goals_conceded']]
sns.pairplot(wins_losses, height=1.5)
plt.show()

# %%
sns.heatmap(teams_seasons[teams_seasons['name'] == 'Real Madrid'][['goals_scored', 'goals_conceded']], annot=True, fmt='.0f', yticklabels=False, cbar=True)
plt.title('Real madrid goals scored vs conceded over the year')
plt.show()

# %%
# hist plot with kde
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
colors = ["#3498db", "#e74c3c"]

# Plot histogram with KDE for 'wins'
sns.histplot(data=teams_seasons, x='wins', kde=True, ax=axes[0], color=colors[0])
axes[0].set_title('Histogram with KDE for Wins')

# Plot histogram with KDE for 'losses'
sns.histplot(data=teams_seasons, x='losses', kde=True, ax=axes[1],  color=colors[1])
axes[1].set_title('Histogram with KDE for Losses')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# %%
# qq plot for wins

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

probplot(teams_seasons['wins'], plot=axes[0, 0])
axes[0, 0].set_title('QQ Plot for Wins')

# QQ plot for 'losses'
probplot(teams_seasons['losses'], plot=axes[0, 1])
axes[0, 1].set_title('QQ Plot for Losses')

# QQ plot for 'goals_scored'
probplot(teams_seasons['goals_scored'], plot=axes[1, 0])
axes[1, 0].set_title('QQ Plot for Goals Scored')

# QQ plot for 'goals_conceded'
probplot(teams_seasons['goals_conceded'], plot=axes[1, 1])
axes[1, 1].set_title('QQ Plot for Goals Conceded')


plt.tight_layout()

plt.show()

# %%
sns.kdeplot(teams_seasons['goals_scored'], fill=True, alpha=0.6, color='lightgreen', linewidth=2)
plt.title('Goals Scored KDE Plot')
plt.show()

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Scatter plot with regression line for 'goals_scored' vs 'wins'
sns.regplot(x='goals_scored', y='wins', data=teams_seasons,  ax=axes[0],  scatter_kws={'alpha': 0.6})
axes[0].set_title('Wins vs Goals Scored')

# Scatter plot with regression line for 'goals_conceded' vs 'losses'
sns.regplot(x='goals_conceded', y='losses', data=teams_seasons, ax=axes[1], scatter_kws={'alpha': 0.6})
axes[1].set_title('Losses vs Goals Conceded')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# %%
sns.heatmap(teams_seasons[teams_seasons['name'] == 'Real Madrid'][['goals_scored', 'goals_conceded']], annot=True, fmt='.0f', yticklabels=False, cbar=True)
plt.title('Real madrid goals scored vs conceded over the year')
plt.show()

# %%
# Boxen plot
sns.boxenplot(data=teams_seasons[['wins', 'losses', 'draws', 'goals_scored', 'goals_conceded']], palette="Set3")
plt.title('Multivariate Boxen Plot for Football Data')
plt.show()

# %%
# area plot

plt.figure(figsize=(10, 6))

# Plot the area for 'wins'
plt.fill_between(teams_seasons['season'], teams_seasons['wins'], label='Wins', alpha=0.5)

# Plot the area for 'losses'
plt.fill_between(teams_seasons['season'], teams_seasons['losses'], label='Losses', alpha=0.5)

# Plot the area for 'draws'
plt.fill_between(teams_seasons['season'], teams_seasons['draws'], label='Draws', alpha=0.5)

plt.title('Area Plot for Wins, Losses, Draws Over Seasons')
plt.xlabel('Season')
plt.ylabel('Count')
plt.legend(title='Outcome', loc='upper left')
plt.show() 



# %%
goal_events = events[events['is_goal'] == 1]
yearwise_goals = goal_events.merge(games[['id_odsp', 'season']], on='id_odsp', how='left' ) 

# %%
yearwise_goals.columns

# %%
# player wise goals
player_goals = yearwise_goals[['player', 'season', 'is_goal']].groupby(['season', 'player']).sum().reset_index()
player_goals = player_goals.rename(columns={'is_goal': 'goals'})
player_goals.head()


# %%
import seaborn.objects as so

# %%
selected_players = ['cristiano ronaldo', 'karim benzema', 'luis suarez', 'lionel messi', 'robert lewandowski', 'sergio aguero']

# Filter player_goals DataFrame for selected players
selected_player_goals = player_goals[player_goals['player'].isin(selected_players)]

# Create a stacked area plot for selected players
plt.figure(figsize=(10, 6))


(
    so.Plot(selected_player_goals, "season", "goals", color="player", )
    .add(so.Area(alpha=.7), so.Stack()).label(title="Goals of top 5 strikers")
)

# for player, color in zip(selected_player_goals['player'].unique(), sns.color_palette('Set2', n_colors=len(selected_player_goals['player'].unique()))):
#     subset = selected_player_goals[selected_player_goals['player'] == player]
#     plt.fill_between(subset['season'], subset.groupby('season')['goals'].cumsum(), label=player, alpha=0.7, color=color)

# plt.title('Stacked Number of Goals Over Seasons for Selected Players')
# plt.xlabel('Season')
# plt.ylabel('Number of Goals')
# plt.legend(title='Player', loc='upper left')
# plt.show()

# %%
# violin plot

plt.figure(figsize=(10, 6))
sns.violinplot(data=player_goals, x="goals", y="season")

plt.title('Violin Plot of Number of Goals for Players')
plt.xlabel('Player')
plt.ylabel('Number of Goals')
plt.show()

# %%
sns.jointplot(x='wins', y='goals_scored', data=teams_seasons, kind='kde')

plt.title('Joint Plot with KDE and Scatter Representation for Team wins vs goals scored')
plt.tight_layout()
plt.show()

# %%
sns.set(style="whitegrid")

# Create a rug plot on player_goals
plt.figure(figsize=(10, 6))
sns.rugplot(x='goals', data=player_goals, height=0.5, color='skyblue')

plt.title('Rug Plot on Player Goals')
plt.xlabel('Number of Goals')
plt.show()

# %%
data_for_cluster = teams_seasons[['wins', 'losses', 'draws', 'goals_scored', 'goals_conceded']]

# Create a cluster map
plt.figure(figsize=(10, 8))
sns.clustermap(data_for_cluster, cmap='viridis', method='complete', metric='euclidean', figsize=(10, 8))

plt.title('Cluster Map for Teams Seasons')
plt.show()

# %%
plt.figure(figsize=(10, 8))
sns.jointplot(x='goals_scored', y='goals_conceded', data=teams_seasons, kind='hex', cmap='viridis')

plt.title('Hexbin Plot for Goals Scored vs Goals Conceded')
plt.show()

# %%

sns.stripplot(x='goals', y='season', data=player_goals, cmap='viridis')

plt.title('Strip plot for goals scored over years')
plt.xticks(range(-1, 55, 5))
plt.show()

# %%
top5_scorers_per_season = player_goals.groupby('season').apply(lambda x: x.nlargest(5, 'goals')).reset_index(drop=True)
plt.figure(figsize=(10, 8))
sns.swarmplot(x='goals', y='season', data=top5_scorers_per_season, hue="player", cmap='viridis')

plt.title('Swarm plot for Top goal scorers every year years')
plt.xticks(range(0, 55, 5))
plt.tight_layout()
plt.show()

# %%
#outlier
q1_temp = player_goals['goals'].quantile(0.25)
q3_temp = player_goals['goals'].quantile(0.75)
iqr_height = q3_temp - q1_temp

lower_h = q1_temp - 1.5 * iqr_height
upper_h = q3_temp + 1.5 * iqr_height

print(
    f'Q1 and Q3 of the goals by players is {str(q1_temp)} to {str(q3_temp)} \n')
print(f'IQR for goals scored is {str(iqr_height) } \n')

print(f'Any goals scored more than {str(upper_h)} or less than {str(lower_h)}  is an outlier')

# %%
#df2=teams_seasons[['rain_1h', 'snow_1h', 'clouds_all', 'temp'] ]
correlation_matrix = teams_seasons.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True,  fmt=".2f")
plt.title("Correlation Coefficient Matrix")
plt.show()


# %%
teams_seasons.describe().round(2)



