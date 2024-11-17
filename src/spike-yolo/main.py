from preprocessing import preprocess

if __name__ == "__main__":
    train_loader, val_loader = preprocess(timesteps=4)
    
