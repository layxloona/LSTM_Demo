import torch
import numpy as np
import matplotlib.pyplot as plt
from model import LSTM
from utils import r_squared, get_device


class LSTMTrainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_func = torch.nn.MSELoss()
        self.times = []
        self.losses = []

    def train_step(self, x_np, y_np, step):
        x = torch.from_numpy(x_np[np.newaxis, np.newaxis, :]).to(self.device)
        y = torch.from_numpy(y_np[np.newaxis, np.newaxis, :]).to(self.device)

        prediction = self.model(x)
        loss = self.loss_func(prediction, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        mae = torch.mean(torch.abs(prediction - y)).item()
        r2 = r_squared(y, prediction)

        self.times.append(step)
        self.losses.append(loss.item())

        return {
            'loss': loss.item(),
            'mae': mae,
            'r2': r2,
            'prediction': prediction.cpu().flatten().detach().numpy()
        }

    def plot_progress(self):
        plt.plot(self.times, self.losses)
        plt.pause(0.05)