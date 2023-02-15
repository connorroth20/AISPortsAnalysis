import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

path = '../cbb.csv'
data = pd.read_csv(path, header=None)
data.columns = ['TEAM','CONF','G','W','ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O','3P_D','ADJ_T','WAB','POSTSEASON','SEED', 'YEAR']
#We only want to analyze the teams in the tournament
data = data.loc[data['SEED'].notnull()]
#Get just the columns that we want (Adjusted Offensive & Defensive Efficiency, Power Rating, Effective Field Goal %, Turnover Rate, Adjusted Tempo)
X = data.loc[:, ['ADJOE','ADJDE','BARTHAG','EFG_O','TOR','ADJ_T']].values
Y = data.loc[:, 'W'].values
#Strip off the headers
X=X[1:]
Y=Y[1:]