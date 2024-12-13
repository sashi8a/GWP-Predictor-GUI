import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model import CustomNN

def load_data(x_path, y_path):
    X = pd.read_csv(x_path).values
    y = pd.read_csv(y_path).values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset

def train_model(model, dataloader, criterion, optimizer, epochs, device):
    model.to(device)
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()

    print(f"Best Loss: {best_loss:.4f}")
    return best_model_state

def main():
    # Paths to data
    x_path = "rev_train_X_final.csv"
    y_path = "train_y.csv"

    # Load data
    dataset = load_data(x_path, y_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model and training configuration
    input_dim = dataset[0][0].shape[0]
    print('inputdim: ',input_dim)
    output_dim = dataset[0][1].shape[0]
    print('output: ',output_dim)
    model = CustomNN(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    best_model_state = train_model(model, dataloader, criterion, optimizer, epochs=100, device=device)

    # Save the best trained model
    torch.save(best_model_state, "best_custom_nn_model.pth")

if __name__ == "__main__":
    main()
