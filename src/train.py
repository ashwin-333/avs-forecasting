import torch
from torch import nn
from snntorch import surrogate
import snntorch as snn
from torch.utils.data import DataLoader
import torch.optim as optim
from snntorch import utils
import preprocessing

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BoundingBoxPredictor(nn.Module):
    def __init__(self, img_width=346, img_height=260):
        super(BoundingBoxPredictor, self).__init__()
        self.spike_grad = surrogate.atan()
        self.beta = 0.5
        self.img_width = img_width
        self.img_height = img_height
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.spike1 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 128, 5)
        self.spike2 = snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 62 * 83, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.spike1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.spike2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        scaling = torch.tensor([self.img_width, self.img_height, self.img_width, self.img_height]).to(x.device)
        x_rescaled = x * scaling
        return x_rescaled

def train(trainloader, model, optimizer, loss_fn, num_epochs=3, num_steps=50):
    model.train()
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        for i, (data, targets) in enumerate(trainloader):
            data, targets = data.to(device), targets.to(device)
            if targets.dim() > 2:
                targets = targets.squeeze(1)
            optimizer.zero_grad()
            utils.reset(model)
            for step in range(num_steps - 1):
                _ = model(data)
            final_output = model(data)
            loss_val = loss_fn(final_output, targets)
            loss_val.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Batch {i+1} - Train Loss: {loss_val.item():.2f}")
            print("Predicted Bounding Box:", final_output.detach().cpu().numpy())
            print("Actual Bounding Box:", targets.detach().cpu().numpy())
            print(f"Epoch {epoch+1}, Batch {i+1} - Train Loss: {loss_val.item():.2f}")
            print("\n")

def train_model(train_loader):
    model = BoundingBoxPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()
    train(train_loader, model, optimizer, loss_fn)
