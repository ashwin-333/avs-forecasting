import torch
from torch import nn
from snntorch import surrogate
import snntorch as snn
from torch.utils.data import DataLoader
import torch.optim as optim
from snntorch import utils
import preprocessing
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BoundingBoxPredictor(nn.Module):
    def __init__(self, img_width=346, img_height=260):
        super(BoundingBoxPredictor, self).__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.beta = 0.5
        self.img_width = img_width
        self.img_height = img_height
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.spike1 = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 128, 5)
        self.spike2 = snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True)
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

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_coverage(pred_box, actual_box):
    xA = max(pred_box[0], actual_box[0])
    yA = max(pred_box[1], actual_box[1])
    xB = min(pred_box[2], actual_box[2])
    yB = min(pred_box[3], actual_box[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    actualArea = max(1, (actual_box[2] - actual_box[0]) * (actual_box[3] - actual_box[1]))
    coverage = (interArea / actualArea) * 100
    return coverage

def forward_pass(model, data, num_steps):
    mem_rec = []
    spk_rec = []
    utils.reset(model)
    
    for step in range(num_steps):
        output = model(data)
        spk_rec.append(output)

    spk_rec = torch.stack(spk_rec)
    return spk_rec

def train(trainloader, model, optimizer, loss_fn, num_epochs=3, num_steps=50):
    loss_hist = []
    iou_hist = []
    coverage_hist = []
    expected_num_batches = None

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        
        # Determine the number of batches
        current_num_batches = len(trainloader)
        
        if epoch == 0:
            expected_num_batches = current_num_batches
            print(f"Number of batches in Epoch {epoch+1}: {current_num_batches}")
        else:
            print(f"Number of batches in Epoch {epoch+1}: {current_num_batches}")
            if current_num_batches == expected_num_batches:
                print(f"Batch count matches Epoch 1: {current_num_batches} batches.\n")
            else:
                print(f"Batch count differs from Epoch 1: Epoch {epoch+1} has {current_num_batches} batches vs {expected_num_batches} batches in Epoch 1.\n")
        
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_coverage = 0.0
        num_batches = 0
        for i, (data, targets) in enumerate(trainloader):
            data, targets = data.to(device), targets.to(device)
            if targets.dim() > 2:
                targets = targets.squeeze(1)
            optimizer.zero_grad()
            spk_rec = forward_pass(model, data, num_steps)
            final_output = spk_rec[-1]
            loss_val = loss_fn(final_output, targets)
            loss_val.backward()
            optimizer.step()

            pred_box = final_output.detach().cpu().numpy()[0]
            actual_box = targets.detach().cpu().numpy()[0]

            iou = calculate_iou(pred_box, actual_box)
            coverage = calculate_coverage(pred_box, actual_box)

            epoch_loss += loss_val.item()
            epoch_iou += iou
            epoch_coverage += coverage
            num_batches += 1

            print(f"Epoch {epoch+1}, Batch {i+1} - Train Loss: {loss_val.item():.4f}")
            print(f"Predicted Bounding Box: {pred_box}")
            print(f"Actual Bounding Box: {actual_box}")
            print(f"Batch IoU: {iou * 100:.2f}%")
            print(f"Batch Coverage: {coverage:.2f}%\n")
        
        avg_loss = epoch_loss / num_batches
        avg_iou = (epoch_iou / num_batches) * 100
        avg_coverage = epoch_coverage / num_batches
        loss_hist.append(avg_loss)
        iou_hist.append(avg_iou)
        coverage_hist.append(avg_coverage)
        print(f"Epoch {epoch+1} Completed:")
        print(f"Average Train Loss: {avg_loss:.4f}")
        print(f"Average IoU: {avg_iou:.2f}%")
        print(f"Average Coverage: {avg_coverage:.2f}%\n")
    
    return loss_hist, iou_hist, coverage_hist

def plot_metrics(loss_hist, iou_hist, coverage_hist):
    epochs = range(1, len(loss_hist) + 1)
    
    plt.figure(figsize=(18, 5))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss_hist, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Average Loss per Epoch')
    plt.legend()
    
    # Plot IoU
    plt.subplot(1, 3, 2)
    plt.plot(epochs, iou_hist, marker='o', color='g', label='Average IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU (%)')
    plt.title('Average IoU per Epoch')
    plt.legend()
    
    # Plot Coverage
    plt.subplot(1, 3, 3)
    plt.plot(epochs, coverage_hist, marker='o', color='r', label='Average Coverage')
    plt.xlabel('Epoch')
    plt.ylabel('Coverage (%)')
    plt.title('Average Coverage per Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def train_model(train_loader):
    model = BoundingBoxPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()
    loss_hist, iou_hist, coverage_hist = train(train_loader, model, optimizer, loss_fn)
    plot_metrics(loss_hist, iou_hist, coverage_hist)
    return model, loss_hist, iou_hist, coverage_hist
