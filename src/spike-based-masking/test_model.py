from model import Model
import torch
from torchvision.ops import distance_box_iou_loss
import utils
import numpy as np

def test_model(test_loader, iterations):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    loss_fn = distance_box_iou_loss
    accuracy_fn = utils.calculate_iou

    net = Model().to(device)
    epoch, loss = utils.load_checkpoint(model=net, optimizer=None, checkpoint_path="model.pth")
    net.eval()

    print(f"Model trained for: {epoch} epochs")
    print(f"Last epoch loss: {loss}")
    test_loss = 0
    test_iou = 0

    sample_targets = []
    sample_predictions = []
    sample_images = []

    with torch.no_grad():
        for i, (data, targets) in enumerate(iter(test_loader)):
            if i != -1 and i >= iterations:
                break
            batch_size = data.size(1)
            data = data.to(device) # (T x B x C x H x W)
            targets = targets.squeeze(1) 
            targets = targets.to(device) # B x 4

            out = net(data)
            loss = torch.sum(loss_fn(out, targets)) / batch_size
            test_loss += loss.item()

            pred_box = out.detach().cpu()
            actual_box = targets.detach().cpu()

            iou = np.sum(accuracy_fn(pred_box, actual_box).numpy(), axis=0) / batch_size
            test_iou += iou.item()

            #take one prediction/target per batch
            image = data[:, 0].cpu()
            image = torch.sum(image, dim = 0)

            sample_images.append(image)
            sample_predictions.append(pred_box[0])
            sample_targets.append(actual_box[0])

    avg_test_loss = test_loss / len(test_loader)
    avg_test_iou = test_iou / len(test_loader)

    print(f"Average test loss: {avg_test_loss}")
    print(f"Average test IoU: {avg_test_iou}")

    utils.visualize_bounding_boxes(sample_images, sample_predictions, sample_targets)

    




