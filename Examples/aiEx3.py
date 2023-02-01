import requests 
import json 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

# request data from the api 
response = requests.get("https://www.balldontlie.io/api/v1/games")

# parse json response
json_data = json.loads(response.text)

# create a list of all the games
games = json_data["data"]

# create a list of all the players
players = []
for game in games:
    if game["status"] == "Final": # only include finished games
        players.append(game["home_team_players"])
        players.append(game["away_team_players"])

# create a new list that contains the stats of each player
player_stats = []
for player in players:
    player_stats.append([player["player_id"],
                        player["min"],
                        player["pts"],
                        player["fgm"],
                        player["fga"],
                        player["ftm"],
                        player["fta"],
                        player["oreb"],
                        player["dreb"],
                        player["ast"],
                        player["stl"],
                        player["blk"],
                        player["turnover"],
                        player["pf"]
                        ])

# convert the list of player stats into a numpy array
player_stats_array = np.array(player_stats)

# separate the data into input and output
input_data = player_stats_array[:,0:14]
output_data = player_stats_array[:,1:14]

# define the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(14, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 14)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# train the neural network
net = Net()
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters())

X = torch.from_numpy(input_data).float()
y = torch.from_numpy(output_data).float()

for epoch in range(1000):  # loop over the dataset multiple times
    
    optimizer.zero_grad()   # zero the parameter gradients
    
    # forward + backward + optimize
    outputs = net(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# request data from the api for today's games
response = requests.get("https://www.balldontlie.io/api/v1/games?date=today")

# parse json response
json_data = json.loads(response.text)

# create a list of all the games
games = json_data["data"]

# create a list of all the players for each game
for game in games:
    players = []
    players.append(game["home_team_players"])
    players.append(game["away_team_players"])

    # create a new list that contains the stats of each player
    player_stats = []
    for player in players:
        player_stats.append([player["player_id"],
                            player["min"],
                            player["pts"],
                            player["fgm"],
                            player["fga"],
                            player["ftm"],
                            player["fta"],
                            player["oreb"],
                            player["dreb"],
                            player["ast"],
                            player["stl"],
                            player["blk"],
                            player["turnover"],
                            player["pf"]
                            ])
    
    # convert the list of player stats into a numpy array
    player_stats_array = np.array(player_stats)

    # separate the data into input and output
    input_data = player_stats_array[:,0:14]
    output_data = player_stats_array[:,1:14]

    # predict the stats for each player in the game
    for player in input_data:
        player_input = torch.from_numpy(player).float()
        player_output = net(player_input)
        print(player_output)