# making a class to deal with team seasons easily
class TeamSeason:
    def __init__(self, name,season, wins=0, losses=0, draws=0, goalsScored=0, goalsConceded=0):
        # Instance variables
        self.name = name
        self.season = season
        self.wins = wins
        self.losses = losses
        self.draws = draws
        self.goalsScored = goalsScored
        self.goalsConceded = goalsConceded

    def addWin(self):
        self.wins+=1
        
    def addLoss(self):
        self.losses+=1
        
    def addDraw(self):
        self.draws+=1
        
    def getTotalGames(self):
        return self.wins + self.losses + self.draws
    
    def addGoalsScored(self,n):
        self.goalsScored+=n
        
    def addGoalsConceded(self,n):
        self.goalsConceded+=n
    
    def export(self):
        return {
            'name':self.name,
            'season': self.season,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'goals_scored': self.goalsScored,
            'goals_conceded': self.goalsConceded
        }
        
