import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def fitness_function(weights):
    
    input_size = 9
    hidden_size = 10
    output_size = 1
    num_epochs = 100
    learning_rate = 0.001

    neural_network = NeuralNetwork(input_size, hidden_size, output_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(neural_network.parameters(), lr=learning_rate)

   
    for epoch in range(num_epochs):
        weights_tensor = torch.FloatTensor(weights)

        outputs = neural_network(weights_tensor)
        loss = criterion(outputs, target_tensor)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    with torch.no_grad():
        weights_tensor = torch.FloatTensor(weights)
        outputs = neural_network(weights_tensor)
        predicted_labels = (outputs > 0.5).float()
        accuracy = (predicted_labels == target_tensor).float().mean()

    return accuracy.item()
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim


data = np.loadtxt("Bank_Personal_Loan_Modelling.csv", delimiter=",", skiprows=1)
X = data[:, 1:-1]
y = data[:, -1]
target_tensor = torch.FloatTensor(y)


X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def fitness_function(weights):
    input_size = 9
    hidden_size = 10
    output_size = 1
    num_epochs = 100
    learning_rate = 0.001

    neural_network = NeuralNetwork(input_size, hidden_size, output_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(neural_network.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        weights_tensor = torch.FloatTensor(weights)

        outputs = neural_network(weights_tensor)
        loss = criterion(outputs, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        weights_tensor = torch.FloatTensor(weights)
        outputs = neural_network(weights_tensor)
        predicted_labels = (outputs > 0.5).float()
        accuracy = (predicted_labels == target_tensor).float().mean()

    return accuracy.item()

def cultural_algorithm(fitness_function, pop_size, chromo_size, num_iter, num_parents):
    population = np.random.uniform(-1, 1, size=(pop_size, chromo_size))
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = fitness_function(population[i])

    for it in range(num_iter):
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]

        mean = np.mean(population, axis=0)
        std = np.std(population, axis=0)

        parents = population[:num_parents]
        for i in range(num_parents, pop_size):
            child = np.zeros(chromo_size)
            for j in range(chromo_size):
                if np.random.uniform() < 0.5:
                    child[j] = np.random.normal(mean[j], std[j])
                else:
                    child[j] = parents[np.random.randint(num_parents)][j]
            population[i] = child
            fitness
