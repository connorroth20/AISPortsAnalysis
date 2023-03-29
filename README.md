# AiSportsAnalysis
Ai Sports Analysis Tool for Exelaration

Tool uses a react frotnend to display findings and Ai predicted statistics. Machine larning used to find and analyze stats for upcmoning games
Examples includes code generated examples from AI that display pytorch functionalitiy

## Getting Started
Start by running the following commands in a Linux environment:

```
pip install torch
pip install pandas
pip install -U scikit-learn
```

## Running the Django App Locally
First, in the top directory called `storefront` run the command:

```
pipenv install django
pipenv shell
```

Then, you need to repeat the earlier steps laid out in `Getting Started`

Project built by Connor, Matthew, Santiago, Dan

Ways to improve 
There are several ways to improve this code:

Use a larger and more complex dataset to train the model. The current dataset is very small and simple, which limits the ability of the model to learn complex patterns.

Use regularization techniques such as dropout or weight decay to prevent overfitting. Currently, the model is likely to overfit to the small dataset provided. --> implemented

Use cross-validation to evaluate the performance of the model. The current implementation only tests the model on a separate test set after training, which can result in overfitting to the test set.

Monitor the loss function during training to ensure that the model is actually learning and not getting stuck in a local minimum.

Experiment with different hyperparameters such as the learning rate, number of epochs, and number of layers to find the best settings for the model.

Use a more sophisticated optimizer, such as AdamW or Adagrad, to improve the convergence speed and generalization performance of the model.

Try different activation functions for the hidden layers, such as LeakyReLU or Tanh, to see if they improve the performance of the model.