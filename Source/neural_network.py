import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np



class ApproxNN(nn.Module):
    def __init__(self):
        super(ApproxNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)
        print("NN created")

    def forward(self, x):
        x = torch.tanh_(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the neural network, loss function, and optimizer
def trainNN(model, x_tensor, y_tensor):
    print("Training model")

    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training the neural network
    epochs = 500
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        predictions = model(x_tensor)
        loss = criterion(predictions, y_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()


        # Print the loss every 100 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    print("Finished Training")


def train_and_animate(model, x_tensor, y_tensor, x, y, epochs=1000, interval=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-1.5, 1.5)
    scatter_true = ax.scatter(x, y, color='blue', label='True Function')
    # Scatter plot for model predictions (initially empty)
    scatter_pred = ax.scatter([], [], color='red', label='Model Prediction')
    ax.legend()

    def init():
        scatter_pred.set_offsets(np.c_[[], []])  # Initialize with empty points
        return scatter_pred,

    def update(epoch):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        y_pred = []
        for i, a in enumerate(x_tensor):
            y_pred.append(predictions[i].item())
        scatter_pred.set_offsets(np.c_[x, y_pred])
        ax.set_title(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
        # plt.title('Epoch: ', epoch)
        return scatter_pred,

    ani = FuncAnimation(fig, update, frames=epochs, init_func=init, blit=True, interval=interval, repeat=False)
    plt.show()
    print("Finished")

    return ani
