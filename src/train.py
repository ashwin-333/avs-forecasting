import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from torchvision import transforms
from torch import nn
from snntorch import functional as SF
from snntorch import surrogate
import snntorch as snn
from snntorch import utils



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def train(trainloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# neuron and simulation parameters
    spike_grad = surrogate.atan()
    beta = 0.5

#  Initialize Network
    net = nn.Sequential(nn.Conv2d(1, 16, 5),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(2),
                        nn.Conv2d(16, 128, 5),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(5146, 4),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.9, incorrect_rate=0.1)

    num_epochs = 3
    num_steps = 50
    #num_iters = 50

    loss_hist = []
    train_acc_hist = []

    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            data = data.to(device)
            targets = targets.to(device)
            print(data.size())

            net.train()
            spk_rec = forward_pass(net, data, num_steps)
            print(spk_rec.size())
            for i in range(len(spk_rec)):
                loss_val = loss_fn(spk_rec[i], targets)
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

            


            # Gradient calculation + weight update
            
            
            

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            acc = SF.accuracy_rate(spk_rec, targets)
            train_acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")


def forward_pass(net, data, num_steps):
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, _ = net(data)
      print(spk_out.size())
      spk_rec.append(spk_out)

  return torch.stack(spk_rec)