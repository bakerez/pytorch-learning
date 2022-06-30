import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        # ...idk about this line...
        super(NeuralNetwork, self).__init__()
        # I think flattens an array/tensor??
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # arguments of nn.Linear are (# of input nodes, # of output nodes)
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        # Passes input data throught the model and returns the output
        logits = self.linear_relu_stack(x)
        return logits

# Create an instance of the model, put onto gpu
model = NeuralNetwork().to(device)
print(model)

# Create random input and pass it through the (untrained) model
X = torch.rand(1, 28, 28, device=device)
logits = model(X)

# Not sure about these two lines...??
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


# What are the automatically assigned weights? Biases?
print(f"First Linear weights: {model.linear_relu_stack[0].weight} \n")

print(f"First Linear weights: {model.linear_relu_stack[0].bias} \n")

input_image = torch.rand(3,28,28)
print(input_image.size())