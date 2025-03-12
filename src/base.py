from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sympy as sm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(f"Using device: {device}")

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
    def __init__(self, model, lr=0.01):
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
        self.model = model.to(device)
        self.optimizer = optimizer.get_optimizer()
        self.transform = loss_fn.transform_output
        self.loss_fn = loss_fn.get_loss()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.loss_history = []

    def train(self, dataset):
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, (features, labels) in enumerate(dataloader):
                features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                self.optimizer.zero_grad()
                output = self.model(features)
                out_pos = self.transform(output)

                loss = self.loss_fn(out_pos, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            self.loss_history.append(avg_loss)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')

        self._plot_loss()

    def evaluate(self, dataset):
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for features, labels in dataloader:
                features, labels = features.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
    def __init__(self, model, loss_fn, dataset, batch_size=1, results_dir=f"results/"):
        """
        Initializes the model validator.

        Args:
        - model: Trained model instance.
        - loss_fn: Loss function instance.
        - dataset: Dataset object (must be a PyTorch Dataset).
        - batch_size: Batch size for evaluation.
        """
        self.results_dir = f"{results_dir}{model.__class__.__name__}/"
        self.model = model.to(device)
        self.transform = loss_fn.transform_output
        self.loss_fn = loss_fn.get_loss()
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        os.makedirs(self.results_dir, exist_ok=True)

    def validate(self):
        # self.model.eval()
        total_loss = 0.0
        total_samples = 0
        criterion = self.loss_fn

        with torch.no_grad():
            for features, _ in self.dataloader:
                features = features.to(device) 
                outputs = self.model(features)

                predicted_positions = self.transform(outputs)

                loss = criterion(predicted_positions, features[:3])
                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)

        avg_loss = total_loss / total_samples
        print(f"Validation Loss (MSE): {avg_loss:.6f}")
        return avg_loss
    
    def plot_robot_arm_predictions(self, input_features):
        """
        Plots the 2-link robot arm for each predicted theta value.

        Args:
        - input_features: Input tensor to the model.
        """
        self.model.eval()
        features = input_features.to(device)
        output = self.model(features)
        goal = tuple(input_features[:3].tolist())
        self._plot_robot_arm(output, goal)

    def _plot_robot_arm(self, theta, goal):
        theta = theta.to(device).detach().cpu().numpy()
        t_s, a_s, r_s, d_s = sm.symbols('θ α a d')

        T = sm.Matrix([[sm.cos(t_s), -sm.sin(t_s)*sm.cos(a_s),  sm.sin(t_s)*sm.sin(a_s), r_s*sm.cos(t_s)],
                       [sm.sin(t_s),  sm.cos(t_s)*sm.cos(a_s), -
                        sm.cos(t_s)*sm.sin(a_s), r_s*sm.sin(t_s)],
                       [0,          sm.sin(a_s),
                        sm.cos(a_s),        d_s],
                       [0,            0,                 0,        1]])

        params = sm.Matrix([t_s, a_s, r_s, d_s])
        T_i_i1 = sm.lambdify((params,), T, modules='numpy')

        # __________________________________________
        alpha = np.array([np.radians(90),0,np.radians(-90),np.radians(90),np.radians(-90),0])
        d= np.array([-50,-130,5.5,0,0,0,])
        r = np.array([104.5,0,0,102.5,0,23])

        theta = np.column_stack([ 
                            np.radians(180) + theta[0],
                            np.radians(90) + theta[1],
                            theta[2],
                            theta[3],
                            theta[4],
                            theta[5],
                            ])

        params = np.array([theta[0], alpha, r, d])
        params = np.transpose(params)

        points = np.array([[0, 0, 0]])
        Tt = np.eye(4)
        for par in params:
            Tt = Tt @ T_i_i1(par)
            points = np.vstack((points, Tt[:3, 3]))

        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
        X, Y, Z = X, -Y, Z

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X, Y, Z, '-o', markersize=8, label="Robot Arm")
        ax.scatter(X[0], Y[0], Z[0], color='g', s=100, label="Base Joint")
        ax.scatter(X[1:], Y[1:], Z[1:], color='r', s=50)
        ax.scatter(goal[0], goal[1], goal[2], color='y', s=100)
        # Label axes
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title("3D Robot Arm Visualization")
        ax.legend()
        plt.savefig('marius_template/test_plot/4hidden100_2e-06_transform.png')
        plt.show()
