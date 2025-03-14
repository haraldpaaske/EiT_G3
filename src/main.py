from dataset.dataset import DataFrameDataset
from model.multihead_nn import MultiHeadKinematicNN
from model.kinematic_nn import KinematicNN, BNKinematicNN, LNKinematicNN
from model.residual_nn import ResidualKinematicNN
from optimizer.adam_optimizer import AdamOptimizer
from loss.mse_loss import MSELoss
from base import ModelDriver, ModelValidator
import torch

train_dataset = DataFrameDataset("KUKA/data/dataset/dataset100000/train.json")

neurons = 512
num_layers = 8
lr = 5e-4
num_epochs = 50
batch_size = 64

# Initialize model
# model = KinematicNN(num_layers=num_layers, neurons=neurons)
# model = ResidualKinematicNN(num_blocks=6, neurons=neurons)
# model = BNKinematicNN(num_layers=num_layers, neurons=neurons)
# model = LNKinematicNN(num_layers=num_layers, neurons=neurons)
model = MultiHeadKinematicNN(num_layers=num_layers, neurons=neurons)

# Initialize optimizer and loss function
optimizer = AdamOptimizer(model, lr=lr)
loss_fn = MSELoss()

# Initialize ModelDriver
driver = ModelDriver(model, optimizer, loss_fn, batch_size=batch_size, num_epochs=num_epochs)

# Train model
driver.train(train_dataset)

# Save model
driver.save_model(f"models/{model.__class__.__name__}.pth")

# Evaluate model
eval_loss = driver.evaluate(train_dataset)

# imported_model = torch.load(f"models/{model.__class__.__name__}.pth")
# model.load_state_dict(imported_model)

# Load training dataset
test_dataset = DataFrameDataset('KUKA/data/dataset/dataset100000/val.json')


# Initialize ModelValidator
validator = ModelValidator(model, loss_fn, test_dataset)

# Run validation
validator.validate()

# Plot robot arm movement
input_features = torch.Tensor([200, 50, 100, 0, 0, 0])
validator.plot_robot_arm_predictions(input_features)

input_features = torch.Tensor([110, 120, 20, 0, 0, 0])
validator.plot_robot_arm_predictions(input_features)