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

from torchvision.ops import box_iou



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def train(trainloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# neuron and simulation parameters
    spike_grad = surrogate.atan()
    beta = 0.5

#  Initialize Network
    net = nn.Sequential(nn.Conv2d(1, 16, 3),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(2),
                        nn.Conv2d(16, 128, 3),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(5355, 4),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()

    num_epochs = 1
    num_steps = 1
    #num_iters = 50

    loss_hist = []
    train_acc_hist = []

    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            print(i)
            data = data.to(device)
            targets = targets.to(device)

            net.train()
            spk_rec = forward_pass(net, data, num_steps)

            #Gradient calculation + weight update
            loss_vals = []
            for j in range(len(spk_rec[0])):
                loss_vals.append(loss_fn(spk_rec[0][j], targets[0][j]))
            
            total_loss = sum(loss_vals)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
      

            # Store loss history for future plotting
            loss_hist.append(total_loss.item())

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {total_loss.item():.2f}")
            print("output spikes (bounding box: ) ", end = "")
            print(spk_rec[0])
            print("target: ", end = "")
            print(targets[0])

            #TODO figure out how to calculate accuracy using intersection over union the code below needs changing

            #acc = box_iou(spk_rec, targets) # check what box_iou returns
            #rain_acc_hist.append(acc)
            #print(f"Accuracy: {acc * 100:.2f}%\n")


def forward_pass(net, data, num_steps):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, _ = net(data)
        spk_rec.append(spk_out)
  
    spk_rec = torch.sum(torch.stack(spk_rec), 1)
    return spk_rec