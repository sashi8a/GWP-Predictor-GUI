import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model import CustomNN

def load_test_data(x_path, y_path):
    X = pd.read_csv(x_path).values
    y = pd.read_csv(y_path).values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset

def evaluate_model(model, dataloader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")

def main():
    # Paths to data
    test_x_path = "rev_test_X_final.csv"  
    test_y_path = "test_y.csv" 

    # Load test data
    dataset = load_test_data(test_x_path, test_y_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Model configuration
    input_dim = dataset[0][0].shape[0]
    output_dim = dataset[0][1].shape[0]
    model_path = "best_custom_nn_model.pth"  

    # Load the best model
    model = CustomNN(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluation criterion
    criterion = torch.nn.MSELoss()

    # Evaluate the model
    evaluate_model(model, dataloader, criterion, device)

if __name__ == "__main__":
    main()
