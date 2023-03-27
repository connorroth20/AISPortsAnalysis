import torch.nn as nn
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class cbbDataset(Dataset):
    
    def __init__(self, split='train', shuffle=True):
        path1 = '../cbb.csv'
        data = pd.read_csv(path1, header=None)
        data.columns = ['TEAM','CONF','G','W','ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O','3P_D','ADJ_T','WAB','POSTSEASON','SEED', 'YEAR']
        #We only want to analyze the teams in the tournament
        data = data.loc[data['SEED'].notnull()]
        #Get just the columns that we want
        X = data.loc[:, ['ADJOE','ADJDE','BARTHAG','EFG_O','TOR','ADJ_T']].values
        Y = data.loc[:, 'POSTSEASON'].values
        #Strip off the headers
        X=X[1:]
        Y=Y[1:]

        #Change Y to fit following model:
        #0 - NA
        #1 - R64 and R68
        #2 - R32
        #3 - S16
        #4 - E8
        #5 - F4
        #6 - 2ND
        #7 - Champions

        for i,y in enumerate(Y):
            if y == 'NA':
                Y[i] = 0
            
            elif (y == 'R64' or y == 'R68'):
                Y[i] = 1
            
            elif (y == 'R32'):
                Y[i] = 2

            elif (y == 'S16'):
                Y[i] = 3

            elif (y == 'E8'):
                Y[i] = 4
            
            elif (y == 'F4'):
                Y[i] = 5

            elif (y == '2ND'):
                Y[i] = 6

            elif (y == 'Champions'):
                Y[i] = 7

        X = X.astype(float)
        Y = Y.astype(float)

        #Scale the input data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        #Separate into data sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.2, random_state = 0, shuffle = True)

        # print(X_train)
        # print(y_train)
        if split == 'train':
            self.x_data = X_train
            self.y_data = y_train
            print('Training dataset loaded')
        elif split == 'test':
            self.x_data = X_test
            self.y_data = y_test
            print('Test dataset loaded')
        else:
            self.x_data = np.concatenate((X_train,X_test),axis=0)
            self.y_data = np.concatenate((y_train,y_test),axis=0)

    def __getitem__(self, i):
        return torch.tensor(self.x_data[i]).float().to(device), torch.tensor(self.y_data[i]).float().to(device)

    def __len__(self):
        return len(self.y_data)

    @staticmethod
    def test(model, test_dl):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        accuracy = None
        total = 0
        correct = 0
        for (X, y) in test_dl:
            X = X.to(device)
            Y = y.to(device)
            prediction = model(X)
            null, prediction_label = torch.max(prediction.data, 0)
            correct += (prediction_label == Y.data).sum()
            total += 1
        accuracy = correct / total
        return accuracy

    @staticmethod
    def train(model, lr, momentum, num_epochs, train_dl, test_dl):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        loss_fn = nn.MSELoss(reduction='mean')

        for epoch in range(1, num_epochs + 1):
            for (X, y) in train_dl:
                opt.zero_grad()
                X = X.to(device)
                Y = y.to(device)
                guess = model(X)
                loss = loss_fn(guess, Y)
                loss.backward()
                opt.step()
            test_accuracy = cbbDataset.test(model, test_dl)
            print(f"Test accuracy at epoch {epoch}: {test_accuracy:.4f}")

class cbb_linear_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features = 6, out_features = 32, bias=True)
        self.fc2 = nn.Linear(in_features = 32, out_features = 64, bias=True)
        self.fc3 = nn.Linear(in_features = 64, out_features = 32, bias=True)
        self.fc4 = nn.Linear(in_features = 32, out_features = 6, bias=True)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def createModel(self):
        lr = 0.01
        momentum = 0.2
        num_epochs = 1

        dataset_train = cbbDataset(split='train')
        dataset_test = cbbDataset(split='test')

        model = cbb_linear_model()
        model = model.to(device)
        cbbDataset.train(model, lr, momentum, num_epochs, dataset_train, dataset_test)
        test_accuracy = cbbDataset.test(model, dataset_test)
        return test_accuracy


    def useModel(self):
        # Predictions for 2021 season
        model = self.to(device)
        path = '../cbb19.csv'
        data = pd.read_csv(path)
        #We only want to analyze the teams in the tournament
        data = data.loc[data['SEED'].notnull()]
        #Grab the values tha we want
        x = data.loc[:, ['ADJOE','ADJDE','BARTHAG','EFG_O','TOR','ADJ_T']].values

        #Get the teams and results
        teams = data.loc[:, 'TEAM']
        actual = data.loc[:, 'POSTSEASON']

        #Scale the data
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        results = ['NA','R64','R32','S16','E8','F4','2ND','Champions']
        complete_results = []

        for t in range(len(teams)):
            torch_data = torch.tensor(x_scaled[t]).float().to(device)
            prediction = model(torch_data)
            null, prediction_index = torch.max(prediction.data, 0)
            prediction_label = results[prediction_index]
            complete_results.append(dict(team = teams[t], prediction = prediction_label, actual = actual[t]))

        return complete_results
            
run_model = cbb_linear_model()

acc = run_model.createModel()
print(f"Accuracy: {acc}")

results = run_model.useModel()

print(results)