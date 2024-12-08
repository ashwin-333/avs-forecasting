from preprocessing import preprocess_train
from preprocessing import preprocess_test

from train import train_model
from test_model import test_model
from utils import test_masking  
import torch

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # uncomment to test masking function
    #data_dir = 'PEDRo-dataset/numpy'
    #test_masking(data_dir, grid_size=10, threshold=0.45)

    # preprocessing, training, and testing
    train_loader, val_loader = preprocess_train(timesteps=4, device=device)
    train_model(train_loader, val_loader, device)
    test_loader = preprocess_test(timesteps=1)
    test_model(test_loader)
