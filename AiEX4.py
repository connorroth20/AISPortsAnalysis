# deep learning example found using AI
#import necessary libraries 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import random_split
import torch.nn.functional as F

#sample dataset
# Assume we have a 10x2 input dataset
train = torch.tensor([[1,2],[3,4],[5,6],[7,8],[9,10]], dtype=torch.float)
test = torch.tensor([[11,12],[13,14],[15,16],[17,18],[19,20]], dtype=torch.float)

# Create a model
class Model(nn.Module):
  #added a p for hyperparameter
  def __init__(self, p=0.5):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(2, 10)
    self.fc2 = nn.Linear(10, 5)
    self.fc3 = nn.Linear(5, 2)
    self.fc4 = nn.Linear(2, 2)
    #p used as probability and dropout so that bad values are used
    self.dropout = nn.Dropout(p=p)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    x = F.relu(self.fc3(x))
    x = self.dropout(x)
    x = self.fc4(x)
    return x

# Create an instance of the model
model = Model(p=0.5)

# Choose an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
#loss function to train the model 

criterion = nn.MSELoss()

total = 0
correct = 0
# Train and test the model
for epoch in range(3000):
  # Clear the gradients
  optimizer.zero_grad()
  
  # Forward pass
  output = model(train)
  
  # Calculate loss
  loss = (output - torch.sum(train)).pow(2).mean()

  #alternatively, use this, where we use our criterion to
  #set up the training model, in a similar way.
  #loss = criterion(output, torch.sum(train)).pow(2).mean() ?

  # Backward pass
  loss.backward()
 
  # Update weights
  optimizer.step()

  #evaluate with the test set-> 
  
  with torch.no_grad():
    prediction = model(test)

    prediction_label_x = torch.mean(prediction.data[1,:])
    actual_label_x = torch.mean(test.data[1,:])
    print(f"predicted X: {prediction_label_x}")
    print(f"actual X: {actual_label_x}")

    prediction_label_y = torch.mean(prediction.data[:,1])
    actual_label_y = torch.mean(test.data[:,1])
    print(f"predicted Y: {prediction_label_y}")
    print(f"actual Y: {actual_label_y}")

    x_similarity = 1 - (abs(prediction_label_x - actual_label_x)/actual_label_x)
    y_similarity = 1 - (abs(prediction_label_y - actual_label_y)/actual_label_y)
    correct += (x_similarity + y_similarity)/2
    total += 1

    accuracy = (correct/total)*100
    print(f"Accuracy: {accuracy}%")

  #Above is the example with the same model just using a dropout model
  # Per:  Here, we use the nn.MSELoss() as the loss function and train the model with dropout by including it in the forward method of the model. We also use torch.no_grad() to evaluate the model on the test set without computing gradients. The rest of the training code remains the same.





  # Start testing
  # prediction =  model(test)

  # prediction_label_x = torch.mean(prediction.data[1,:])
  # actual_label_x = torch.mean(test.data[1,:])
  # print(f"predicted X: {prediction_label_x}")
  # print(f"actual X: {actual_label_x}")

  # prediction_label_y = torch.mean(prediction.data[:,1])
  # actual_label_y = torch.mean(test.data[:,1])
  # print(f"predicted Y: {prediction_label_y}")
  # print(f"actual Y: {actual_label_y}")

  # x_similarity = 1 - (abs(prediction_label_x - actual_label_x)/actual_label_x)
  # y_similarity = 1 - (abs(prediction_label_y - actual_label_y)/actual_label_y)
  # correct += (x_similarity + y_similarity)/2
  # total += 1

  # accuracy = (correct/total)*100
  # print(f"Accuracy: {accuracy}%")

# Print the model's weights
# for param in model.parameters():
#   print(param)