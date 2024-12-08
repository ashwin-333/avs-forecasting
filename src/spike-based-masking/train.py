import numpy as np
import torch
from torch import nn
from snntorch import surrogate
import matplotlib.pyplot as plt
from torchvision.ops import distance_box_iou_loss
from model import Model
import matplotlib.pyplot as plt
import utils



def train_model(trainloader, valloader, device):
    
    loss_fn = distance_box_iou_loss
    accuracy_fn = utils.calculate_iou

    net = Model().to(device)
    net.apply(initialize_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    num_epochs = 40

    train_loss_hist = []
    train_iou_hist = []

    val_loss_hist = []
    val_iou_hist = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0
        epoch_train_iou = 0
        for i, (data, targets) in enumerate(iter(trainloader)):
            batch_size = data.size(1)
            data = data.to(device) # (T x B x C x H x W)
            targets = targets.squeeze(1) 
            targets = targets.to(device) # B x 4

            net.train()
            out = net(data)

            optimizer.zero_grad()
            train_loss = torch.sum(loss_fn(out, targets)) / batch_size
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item()

            pred_box = out.detach().cpu().numpy()
            actual_box = targets.detach().cpu().numpy()

            train_iou = np.sum(accuracy_fn(pred_box, actual_box), axis=0) / batch_size
            epoch_train_iou += train_iou


            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {train_loss.item():.2f}")
            print("Sample predicted Bounding Box: ", pred_box[0])
            print("Sample actual Bounding Box: ", actual_box[0])
            print(f"Train IoU: {train_iou * 100:.2f}%")
            print()

        avg_train_loss = epoch_train_loss / len(trainloader)
        avg_train_iou = epoch_train_iou / len(trainloader)

        train_loss_hist.append(avg_train_loss)
        train_iou_hist.append(avg_train_iou)

        val_loss, val_iou = validate(net, valloader, loss_fn, accuracy_fn, device)
        val_loss_hist.append(val_loss)
        val_iou_hist.append(val_iou)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_iou:.4f}")
        if (epoch+1) % 10 == 0:
            utils.save_checkpoint(net, optimizer, epoch, train_loss_hist[-1].item(), checkpoint_path=f"checkpoint{epoch}.pth")
    
    utils.save_checkpoint(net, optimizer, epoch, train_loss_hist[-1].item(), checkpoint_path=f"model.pth")

    epochs = list(range(1, num_epochs + 1))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_hist, label="Average train loss per Epoch")
    plt.plot(epochs, val_loss_hist, label="Average val loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_iou_hist, label="Average train IoU per Epoch")
    plt.plot(epochs, val_iou_hist, label="Average val IoU per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.tight_layout()
    plt.show()


def validate(net, val_loader, loss_fn, accuracy_fn, device):
    net.eval()
    val_loss = 0
    val_iou = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for i, (data, targets) in enumerate(iter(val_loader)):
            batch_size = data.size(1)
            # Move inputs and targets to the appropriate device
            data, targets = data.to(device), targets.squeeze(1).to(device)

            # Forward pass
            out = net(data)

            loss = torch.sum(loss_fn(out, targets)) / batch_size
            val_loss += loss.item()  

            pred_box = out.detach().cpu()
            actual_box = targets.detach().cpu()
            iou = np.sum(accuracy_fn(pred_box, actual_box).numpy(), axis=0) / batch_size
            val_iou += iou.item()
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader) 

    return avg_val_loss, avg_val_iou


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        #if m.bias is not None:
            #nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        #if m.bias is not None:
            #nn.init.constant_(m.bias, 0)


