import requests
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# fetch data from API
url = "https://www.balldontlie.io/api/v1/stats?start_date=2022-12-31&end_date=2023-01-07"
data = requests.get(url).json()

# create a pandas dataframe from the API data
df = pd.DataFrame(data['data'])

# select relevant columns for training
train_cols = ['ast', 'pts', 'reb']
train_data = df[train_cols].values

# convert to tensor and normalize
train_data = torch.tensor(train_data, dtype=torch.float32)
train_data = (train_data - train_data.mean(dim=0)) / train_data.std(dim=0)

# define the model
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.linear(x)

# initialize the model, loss function and optimizer
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# train the model
num_epochs = 1000
for epoch in range(num_epochs):
    inputs = train_data
    targets = train_data[:, 2].reshape(-1, 1)
    
    # forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# use the trained model to make predictions
with torch.no_grad():
    test_data = train_data[0:5]
    predictions = model(test_data)
    print(predictions)
#Note that this is just a sample code to give you an idea of how to train a PyTorch model using the API data. You may need to adjust the code to suit your specific requirements and improve the model's performance.




