import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

data = pd.read_csv('/kaggle/input/bank-personal-loan-modelling/Bank_Personal_Loan_Modelling.csv')
X = data.iloc[:, [0,1,2,3,4,5,6,7,9]].values
y = data.iloc[:, -1].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(9, 4)
        self.layer2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.sigmoid(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out

def fitness_function(weights):
    net = NeuralNetwork()
    net.load_state_dict(weights)
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(50):
        running_loss = 0.0
        for i in range(len(X_train)):
            inputs = torch.tensor(X_train[i], dtype=torch.float32)
            label = torch.tensor(y_train[i], dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(X_val)):
            inputs = torch.tensor(X_val[i], dtype=torch.float32)
            label = torch.tensor(y_val[i], dtype=torch.float32)
            
            outputs = net(inputs)
            predicted = torch.round(outputs)
            total += 1
            correct += (predicted == label).sum().item()

    accuracy = 100 * correct / total
    return accuracy

class Ant:
    def __init__(self, weights):
        self.weights = weights
        self.fitness = fitness_function(weights)

    def update(self, pheromone):
        new_weights = {}
        for key in self.weights.keys():
            if np.random.rand() < pheromone[key]:
                new_weights[key] = self.weights[key] + np.random.normal(0, 0.1, self.weights[key].shape)
            else:
                new_weights[key] = self.weights[key]
        return Ant(new_weights)

population_size = 20
num_iterations = 50
evaporation_rate = 0.1
pheromone_initial = 0.1

weights = NeuralNetwork().state_dict()
pheromone = {}
for key in weights.keys():
    pheromone[key] = pheromone

ants = [Ant(weights) for i in range(population_size)]
best_ant = ants[0]

for iteration in range(num_iterations):
    for ant in ants:
        for key in pheromone.keys():
            pheromone[key] += evaporation_rate * pheromone[key]
            if np.random.rand() < ant.fitness / best_ant.fitness:
                pheromone[key] += pheromone_initial

    new_ants = [ant.update(pheromone) for ant in ants]

    for ant in new_ants:
        if ant.fitness > best_ant.fitness:
            best_ant = ant

    ants = new_ants

print("Accuracy:", best_ant.fitness)
