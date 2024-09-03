from dash import Dash, dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math
from scipy.stats import norm, kstest, shapiro, normaltest
from utility import TeamSeason
import pandas as pd
from datetime import date
import json
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']




def get_player_goals(events, games):
    goal_events = events[events['is_goal'] == 1]
    yearwise_goals = goal_events.merge(games[['id_odsp', 'season']], on='id_odsp', how='left' ) 
    player_goals = yearwise_goals[['player', 'season', 'is_goal']].groupby(['season', 'player']).sum().reset_index()
    return player_goals.rename(columns={'is_goal': 'goals'})



def get_team_seasons(games):
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
            
    return pd.DataFrame(teams_list)


def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    print('='*50)
    x=  f'K-S test: {title} dataset: statistics= {stats:.2f} p-value = {p:.2f}'

    if p <= 0.05:
        x+=' this indicates that variable is quite significant.'
    else:
        x+=' this indicates that enough evidence cannot be found that varianle is significant'
    return x

def shapiro_test(x, title):
    stats, p = shapiro(x)
    if p <= 0.05:
        x+=' this indicates that variable is quite significant.'
    else:
        x+=' this indicates that enough evidence cannot be found that varianle is significant'
    return x

def da_k_squared_test(x, title):
    stats, p = normaltest(x)
    if p <= 0.05:
        x+=' this indicates that variable is quite significant.'
    else:
        x+=' this indicates that enough evidence cannot be found that varianle is significant'
    return x

games = pd.read_csv('ginf.csv')
events = pd.read_csv('events.csv')
player_goals = get_player_goals(events, games)
team_seasons = get_team_seasons(games)

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

server = app.server


app.layout = html.Div([
    html.H1('Term Project : Football Dataset'),
    dcc.Tabs(id="plot-tabs", value='plot1', children=[
        dcc.Tab(label='About', value='plot1'),
        dcc.Tab(label='Team Data', value='plot2'),
        dcc.Tab(label='Players goals', value='plot3'),
        dcc.Tab(label='Data Table', value='plot4'),
        dcc.Tab(label='Normality Tests', value='plot5'),
    ]),
    html.Div(id='plots-output')
])

@callback(Output('plots-output', 'children'),
              Input('plot-tabs', 'value'))
def render_content(tab):
    if tab == 'plot1':
        return render_p1()
    elif tab == 'plot2':
        return render_p2()
    elif tab == 'plot3':
        return render_p3()
    elif tab == 'plot4':
        return render_p4()
    elif tab == 'plot5':
        return render_p5()




# P1

def render_p1():
    
    
  

    return html.Div(
        [
            html.Img(src=app.get_asset_url('top5strikers.png'), style={'width':500, 'height': 300}),
            
          
            html.H4("About the dataset", style={'font-weight': 'bold'}),
            html.P('The dataset has been sourced from Kaggle. The dataset is called Football Events. The dataset gives information of football games from across different leagues in the world. The events include goals, assists, saves and various other data points. Also information like game scores, game counts are also retrieved from the dataset. The dataset includes two files:'),
            html.Br(),
            html.Ul([
                html.Li("events.csv: Consisting of more than 900000 records. Gives more indepth dataevents of the games like goal scored, who scored against whom and assists made etc."),
                html.Li("ginf.csv: Consisting of more than 10000 records. Gives surface level information about the games."),
            ]),

            html.H4("Aim of the project", style={'font-weight': 'bold'}),
            html.P('Following points were planned about what information can be gained from the dataset'),
            html.Ol([
                html.Li("What has been the team records in general ? What are the splits of wins, losses and draws"),
                html.Li("How the teams perform at home vs away ?"),
                html.Li("Has football gotten any more attacking or less attacking ?"),
                html.Li("Who have been the best teams and most attacking teams ?"),
                html.Li("What is the correlation of goals scored vs games won?"),
                html.Li("Which players have been the most dominant throughout?"),
            ]),

        ], 
    )




# P2

def render_p2():
    top_teams = [
    'Manchester City', 'Manchester United', 'Liverpool', 'Chelsea', 'Arsenal',  # EPL
    'Real Madrid', 'Barcelona', 'Atlético Madrid', 'Sevilla', 'Valencia',  # La Liga
    'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Borussia Mönchengladbach',  # Bundesliga
    'Juventus', 'Inter Milan', 'AC Milan', 'AS Roma', 'Napoli',  # Serie A
    'Paris Saint Germain', 'Olympique de Marseille', 'AS Monaco', 'Lyon', 'Lille' ]


    

    

    return html.Div(
        [
            html.H3("View season wise stats of top teams"),
            html.H4("Pick a team"),
            html.Br(),
            dcc.Dropdown(id = "p2_team",
                 options = [
                     {"label" : i, "value":i} for i in top_teams],
                     multi=False,
                     placeholder="Select a Team..."
                         , style={'width':'50%'}),

            html.Br(),
            html.H4("Pick a Season"),
            
            html.Br(),

            dcc.RadioItems([2012, 2013, 2014, 2015, 2016, 2017], inline=True , id="p2_year",  style={'width':'50%'}),
            # dcc.Dropdown(id = "p2_year",
            #      options = [
            #          {"label" : i, "value":i} for i in [2012, 2013, 2014, 2015, 2016, 2017]],
            #          multi=False,
            #          placeholder="Select an Year..."
                        #  , style={'width':'50%'}),
            html.Br(),
            dcc.Graph(id='p2_graph', style={'width':'30%'}),

            html.Br(),
            dcc.Graph(id='p2_pie', style={'width':'30%'}),
    
            

            

        ], 
    )


@app.callback(
    Output(component_id='p2_graph', component_property='figure'),
    [Input(component_id='p2_team', component_property='value'),
     Input(component_id='p2_year', component_property='value'),
     ]
)
def update(team, season):


    if team == None or season == None:
        return px.line()
    teams_seasons = get_team_seasons(games)
    teamseason = teams_seasons[(teams_seasons['name'] == team) & (teams_seasons['season'] == int(season) )]

    return px.bar(teamseason, x='name', y=['wins', 'losses', 'draws'],
             color_discrete_map={'wins': 'green', 'losses': 'red', 'draws': 'yellow'},
             labels={'value': 'Count', 'variable': 'Result'},
             title=f'Wins, Losses, and Draws for {team} in {season}')


@app.callback(
    Output(component_id='p2_pie', component_property='figure'),
    [Input(component_id='p2_team', component_property='value'),
     Input(component_id='p2_year', component_property='value'),
     ]
)
def updatePie(team, season):
  

    if team == None or season == None:
        return px.line()
    teams_seasons = get_team_seasons(games)
    teamseason = teams_seasons[(teams_seasons['name'] == team) & (teams_seasons['season'] == int(season) )]

    return go.Figure(data=[go.Pie(labels=['Goals Scored', 'Goals Conceded'], values=[teamseason.iloc[0]['goals_scored'], teamseason.iloc[0]['goals_conceded']], title=f'Goals Scored and Conceded for {team} in {season}',
                             insidetextorientation='radial'
                            )])





# P3

def render_p3():

    all_top_scorers = list(set(player_goals.nlargest(30, 'goals')['player'].tolist()))
    

    return html.Div(
        [
            html.H3("View year by year goals of these top players."),
            html.P("Type a players name: Partial name could work. All Players matching query would be returned.\n Add atleast 4 characters"),
            html.Br(),
            dcc.Textarea(
        id='p3_player',
        value='',
        style={'width': '10%', 'height': 30},),
            html.Br(),
            dcc.Checklist(
            [2012, 2013,2014, 2015,2016,2017],
            [2012, 2013, 2014, 2015, 2016, 2017],
            id='p3_seasons',
            style={'width':'50%'},
            inline=True),
            html.Br(),
            dcc.Graph(id='p3_graph', style={'width':'50%'}),

            # html.Br(),
            # dcc.Graph(id='p2_pie', style={'width':'30%'}),
    
            

            

        ], 
    )


@app.callback(
    Output(component_id='p3_graph', component_property='figure'),
    [Input(component_id='p3_player', component_property='value'),
     Input(component_id='p3_seasons', component_property='value'),
     ]
)
def update(player, season):
    if player == '' or player is None or len(player) < 4:
        return px.line()
    
    if season == [] or season is None:
        season = [2012, 2013, 2014, 2015, 2016, 2017]

    filtered_players = player_goals[player_goals['player'].str.startswith(player) & player_goals['season'].isin(season)]



    fig =  px.line(filtered_players, x='season', y='goals', color='player',
             title='Goals Scored by Players Across Seasons',
             labels={'goals_scored': 'Goals Scored', 'season': 'Season'})
    fig.update_traces(marker=dict(size=16))

    # Set xticks from 2015 to 2022
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[2012, 2013, 2014, 2015, 2016, 2017],
            ticktext=['2012', '2013', '2014', '2015', '2016', '2017']
        )
    )
    return fig




# P4

def render_p4():

    temp = player_goals[player_goals['season'] == 2012]
    

    return html.Div(
        [
            
            html.H3("View all the players who have scored particular number of goals during selected season"),
            html.P("Select a season"),
            html.Br(),
            dcc.Slider(2012, 2017, 1, value=2012,  
                       marks={
                   2012: "2012",
                   2013: "2013",
                   2014: "2014",
                   2015: "2015",
                   2016:"2016",
                   2017:"2017",
               },
                       
                        id='p4_season'),
            html.P("Select a Range of goals"),
            html.Br(),
            dcc.RangeSlider(15, 54, 3, value=[15, 24], id='p4_goals'),
            html.Br(),
            html.H3("The players with selected # of goals for the year"),
            dash_table.DataTable(
                id='p4_datatable',
                columns=[{'name': col, 'id': col} for col in temp.columns],
                data=temp.head().to_dict('records')
                    ),

            dcc.Download(id="p4_download"),
            html.Button("Download CSV", id="p4_button", n_clicks=0),
            
            
        ], 
    )


@app.callback(
    Output(component_id='p4_datatable', component_property='data'),
    [Input(component_id='p4_season', component_property='value'),
     Input(component_id='p4_goals', component_property='value'),
     ]
)
def update(season, goals):
    filtered_player_goals = player_goals[(player_goals['season'] == season) & (player_goals['goals'].between(goals[0], goals[1]))]
    sorted_player_goals = filtered_player_goals.sort_values(by='goals', ascending=False)

    return sorted_player_goals.to_dict('records')


@app.callback(
        Output("p4_download", "data"),
    [Input("p4_button", "n_clicks"),
    Input(component_id='p4_season', component_property='value'),
     Input(component_id='p4_goals', component_property='value'),
     ],
)
def downloadJson(n, season, goals):
    if n != 0:
        filtered_player_goals = player_goals[(player_goals['season'] == season) & (player_goals['goals'].between(goals[0], goals[1]))]
        sorted_player_goals = filtered_player_goals.sort_values(by='goals', ascending=False).to_dict('records')
        jsonstr = json.dumps(sorted_player_goals, indent=2)
        return dcc.send_data_frame(filtered_player_goals.to_csv , "filtereddata.csv")




# P5

def render_p5():


    return html.Div(
        [
           
            
            html.H4("Pick a variable", id='p5_tt'),
            dcc.Dropdown(id = "p5_variable",
                options = [
                    {"label" : column, "value":column} for column in ['wins', 'losses', 'draws', 'goals_scored', 'goals_conceded']],
                    multi=False,
                    placeholder="Select a variable"
                        ),
            
            # ====================
            # selecting feature
            # ======================
            html.Br(),
            html.Br(),
            html.H4('Pick a test'),
            dcc.Dropdown(id = "p5_test",
                options = [
                    {"label" : i, "value":i} for i in ['Da K Squared', 'KS Test', 'Shapiro Test']],
                    multi=False,
                    placeholder="Select Normality test"
                        ),
            # ====================
            # date selection
            # ======================
            html.Br(),
            html.Br(),
            html.Div(id='p5_output')
            
            
        ], 
    )


@app.callback(
    Output(component_id='p5_output', component_property='children'),
    [Input(component_id='p5_variable', component_property='value'),
     Input(component_id='p5_test', component_property='value'),
     ]
)
def update(variable, test):
    if variable == None:
        return 'Variable is none'
    elif test == None:
        return 'Test is none'
    else:
        
        if test == 'Da K Squared':
            return da_k_squared_test(team_seasons[variable], 'Raw')
        elif test == 'KS Test':
            return ks_test(team_seasons[variable], 'Raw')
        else:
            return shapiro_test(team_seasons['variable'], 'Raw')





if __name__ == '__main__':
    app.run_server(
        debug=False,
        port=8080,
        host='0.0.0.0'
    )

# app.run_server(
#         debug=True
#         # port=8045,
#         # host='0.0.0.0'
#     )