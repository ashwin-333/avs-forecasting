import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils


class FIL(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, beta, spike_grad):
        super(FIL, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, mid_channels),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(mid_channels, out_channels),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism="none"),
        )
    def forward(self, x: torch.Tensor):
        x = torch.sum(x, dim=0)
        _, mem_rec = self.net(x)
        return mem_rec
class SCNN(nn.Module):
    def __init__(self, beta, spike_grad):
        super(SCNN, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(2, 32, 1),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 1),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 1),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 1),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 1),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(2))
    def forward(self, x: torch.Tensor):
        spk_rec = []
        utils.reset(self.net)
        for step in range(x.size(0)):
            out = self.net(x[step])
            spk_rec.append(out)
        return torch.stack(spk_rec)
    
class Model(nn.Module):
    def __init__(self, beta, spike_grad):
        super(Model, self).__init__()
        
        # Define the layers and components in the constructor (__init__)
        self.scnn = SCNN(beta, spike_grad)
        self.fil = FIL(10240, 4096, 4, beta=beta, spike_grad=spike_grad)
        
    def forward(self, x):
        # Define the forward pass
        x = self.scnn(x)  # Pass through SCNN layer
        x = self.fil(x)   # Pass through FIL layer
        return x