from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

class BaseModel(nn.Module, ABC):
    def __init__(self, in_dim=2, out_dim=2):
        super(BaseModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def test(self, x):
        pass


class BaseLoss(ABC):
    @abstractmethod
    def transform_output(self):
        pass
    
    @abstractmethod
    def get_loss(self):
        pass


class BaseOptimizer(ABC):
    def __init__(self, model, lr=0.001):
        self.model = model
        self.lr = lr

    @abstractmethod
    def get_optimizer(self):
        pass


class ModelDriver:
    def __init__(self, model, optimizer, loss_fn, batch_size=8, num_epochs=25):
        """
        Model driver for training and evaluating models.

        Args:
        - model: Instance of a model inheriting from BaseModel.
        - optimizer: Instance of a class inheriting from BaseOptimizer.
        - loss_fn: Instance of a class inheriting from BaseLoss.
        - batch_size: Batch size for training.
        - num_epochs: Number of training epochs.
        """
        self.model = model
        self.optimizer = optimizer.get_optimizer()
        self.transform = loss_fn
        self.loss_fn = loss_fn.get_loss()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss_history = []

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for features, labels in dataloader:
                self.optimizer.zero_grad()
                output = self.model(features)
                print(output)
                output = self.transform.transform_output(output)
                print(output)
                
                loss = self.loss_fn(output, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            self.loss_history.append(avg_loss)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}")

        self._plot_loss()

    def evaluate(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for features, labels in dataloader:
                output = self.model(features)
                loss = self.loss_fn(output, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss

    def save_model(self, path="model_weights.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Model loaded from {path}")

    def _plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.show()

class ModelValidator:
    def __init__(self, model, dataset, batch_size=1, results_dir=f"results/"):
        """
        Initializes the model validator.

        Args:
        - model: Trained model instance.
        - dataset: Dataset object (must be a PyTorch Dataset).
        - batch_size: Batch size for evaluation.
        """
        self.results_dir = f"{results_dir}{model.__class__.__name__}/"
        self.model = model
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        os.makedirs(self.results_dir, exist_ok=True)

    def validate(self):
        self.model.eval()
        t1_loss = 0
        t2_loss = 0
        total_samples = 0

        with torch.no_grad():
            for features, labels in self.dataloader:
                output = self.model(features)

                t1_pred, t2_pred = output[:, 0].cpu().numpy(), output[:, 1].cpu().numpy()
                t1_true, t2_true = labels[:, 0].cpu().numpy(), labels[:, 1].cpu().numpy()

                t1_loss += np.sum((t1_true - t1_pred) ** 2)
                t2_loss += np.sum((t2_true - t2_pred) ** 2)
                total_samples += len(labels)

        avg_t1_mse = t1_loss / total_samples
        avg_t2_mse = t2_loss / total_samples

        print(f"Theta 1 MSE: {avg_t1_mse:.6f}")
        print(f"Theta 2 MSE: {avg_t2_mse:.6f}")

        return avg_t1_mse, avg_t2_mse

    def plot_predictions(self):
        self.model.eval()
        theta1_pred, theta2_pred, theta1_true, theta2_true = [], [], [], []

        with torch.no_grad():
            for features, labels in self.dataloader:
                output = self.model(features)

                theta1_pred.extend(output[:, 0].cpu().numpy())
                theta2_pred.extend(output[:, 1].cpu().numpy())
                theta1_true.extend(labels[:, 0].cpu().numpy())
                theta2_true.extend(labels[:, 1].cpu().numpy())

        theta1_pred, theta2_pred = np.array(theta1_pred), np.array(theta2_pred)
        theta1_true, theta2_true = np.array(theta1_true), np.array(theta2_true)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(theta1_true, theta1_pred, alpha=0.5, label="Theta1")
        plt.plot([-np.pi/2, np.pi/2], [-np.pi/2, np.pi/2], 'r--')
        plt.xlabel("True Theta1")
        plt.ylabel("Predicted Theta1")
        plt.title("Theta1 Predictions")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(theta2_true, theta2_pred, alpha=0.5, label="Theta2")
        plt.plot([-np.pi/2, np.pi/2], [-np.pi/2, np.pi/2], 'r--')
        plt.xlabel("True Theta2")
        plt.ylabel("Predicted Theta2")
        plt.title("Theta2 Predictions")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_robot_arm_predictions(self, input_features, names, L1=1.0, L2=0.8):
        """
        Plots the 2-link robot arm for each predicted theta value.
        
        Args:
        - input_features: Input tensor to the model.
        - names: List of names for the end-effector positions.
        - L1: Length of the first link.
        - L2: Length of the second link.
        """
        self.model.eval()
        output = self.model(input_features)
        theta1_pred, theta2_pred = output[:, 0].detach().numpy(), output[:, 1].detach().numpy()

        for theta1, theta2_relative, name in zip(theta1_pred, theta2_pred, names):
            self._plot_robot_arm(theta1, theta2_relative, name, L1, L2)

    def _plot_robot_arm(self, theta1, theta2_relative, name, L1=1.0, L2=0.8):
        """
        Plots a 2-link robot arm where the second angle is relative to the first link.
        """
        x0, y0 = 0, 0
        x1 = L1 * np.cos(theta1)
        y1 = L1 * np.sin(theta1)
        theta2_absolute = theta1 + theta2_relative
        x2 = x1 + L2 * np.cos(theta2_absolute)
        y2 = y1 + L2 * np.sin(theta2_absolute)

        plt.figure(figsize=(5, 5))
        plt.plot([x0, x1], [y0, y1], 'bo-', label="Link 1")
        plt.plot([x1, x2], [y1, y2], 'ro-', label="Link 2")
        plt.scatter(name[0], name[1])
        plt.scatter([x0, x1, x2], [y0, y1, y2], c='black', zorder=3)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.title(f"2-Link Robot Arm (θ1={theta1:.1f}°, θ2_relative={theta2_relative:.1f}°)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(f"{self.results_dir}/end_effector_{name}.png")
        plt.close()
