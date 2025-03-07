from dataset.dataset import DataFrameDataset
from model.kinematic_nn import KinematicNN
from optimizer.adam_optimizer import AdamOptimizer
from loss.mse_loss import MSELoss
from base import ModelDriver, ModelValidator
import torch

train_dataset = DataFrameDataset("KUKA/data/dataset/dataset10000.json")

# Initialize model
model = KinematicNN()

# Initialize optimizer and loss function
optimizer = AdamOptimizer(model, lr=0.0001)
loss_fn = MSELoss()

# Initialize ModelDriver
driver = ModelDriver(model, optimizer, loss_fn, batch_size=4, num_epochs=50)

# Train model
driver.train(train_dataset)

# Save model
driver.save_model("models/kinematic_nn1.pth")

# Evaluate model
eval_loss = driver.evaluate(train_dataset)

# Load training dataset
test_dataset = DataFrameDataset("dataset/dataset1000_test.json")

# Initialize ModelValidator
validator = ModelValidator(model, test_dataset)

# Run validation
validator.validate()

# Plot predictions
validator.plot_predictions()

# Plot robot arm movement
input_features = torch.Tensor([[-1,1.5], [1,1], [1,-1.5], [-0.5,-0.5], [0.5,0.5]])
names = [(-1,1.5), (1,1), (1,-1.5), (-0.5,-0.5), (0.5,0.5)]
validator.plot_robot_arm_predictions(input_features, names)