import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn



# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        in_channels = 1
        out_channels = 4
        kernel_size = 3
        max_pool = 2
        avg_pool = 2
        flattened_input = 49 * 16
        num_outputs = 4
        beta = 0.5

        spike_grad_lstm = snn.surrogate.straight_through_estimator()
        spike_grad_fc = snn.surrogate.fast_sigmoid(slope=5)

        # initialize layers
        self.sclstm1 = snn.SConv2dLSTM(
            in_channels,
            out_channels,
            kernel_size,
            max_pool=max_pool,
            spike_grad=spike_grad_lstm,
        )
        self.sclstm2 = snn.SConv2dLSTM(
            out_channels,
            out_channels,
            kernel_size,
            avg_pool=avg_pool,
            spike_grad=spike_grad_lstm,
        )
        self.fc1 = nn.Linear(flattened_input, num_outputs)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad_fc)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        syn1, mem1 = self.lif1.reset_mem()
        syn2, mem2 = self.lif1.reset_mem()
        mem3 = self.lif3.init_leaky()

        # Record the final layer
        spk3_rec = []
        mem3_rec = []

        # Number of steps assuming x is [N, T, C, H, W] with
        # N = Batches, T = Time steps, C = Channels,
        # H = Height, W = Width

        spk1, syn1, mem1 = self.sclstm1(x, syn1, mem1)
        spk2, syn2, mem2 = self.sclstm2(spk1, syn2, mem2)
        cur = self.fc1(spk2.flatten(1))
        spk3, mem3 = self.lif1(cur, mem3)

        spk3_rec.append(spk3)
        mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec)
    