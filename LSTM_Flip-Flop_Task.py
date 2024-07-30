import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import pandas as pd

def tokenize_batch(batch):
    mapping = {'w': 0, 'r': 1, 'i': 2, '0': 3, '1': 4}
    tokenized_batch = [[mapping[char] for char in s] for s in batch['text']]
    return {'tokens': torch.tensor(tokenized_batch, dtype=torch.int64)}

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x=x[:,:,None].type(torch.cuda.FloatTensor)
        # Initialize the hidden state and cell state with the correct dimensions for batched input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Apply the fully connected layer to get the output
        out = self.fc(out)

        return out

def evaluate(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_errors = 0
    total_r_count = 0  # Count occurrences of 'r'

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['tokens'].to(device)
            targets = inputs.roll(shifts=-1, dims=1)

            # Forward pass
            outputs = model(inputs)
            outputs = outputs[:, :-1, :]
            targets = targets[:, :-1]

            # Focus only on positions where the current token is 'r'
            is_r_mask = (inputs == 1)[:, :-1]  # Assuming 'r' is mapped to index 1

            # Count total 'r' occurrences to normalize error rate later
            total_r_count += is_r_mask.sum().item()

            # Predictions and targets for 'r' positions
            pred_at_r = outputs[is_r_mask]
            true_at_r = targets[is_r_mask]

            # Since we're predicting the next token, compare shifted targets
            # Incorrect predictions are where the argmax of prediction doesn't match the target
            errors = (pred_at_r.argmax(dim=-1) != true_at_r).sum().item()
            total_errors += errors

    # Calculate error rate as the ratio of incorrect predictions to total 'r' occurrences
    error_rate = total_errors / total_r_count if total_r_count > 0 else 0
    print(f"Total Errors: {total_errors}")
    print(f"Error Rate: {error_rate}")

    return error_rate

def LSTM_flip_flop_task(train_type: str, rdseed: int, num_epochs: int, batch_size: int, hidden_size: int, num_layers: int):
    # Hyperparameters
    input_size = 1
    num_classes = 5
    learning_rate = 3e-4


    # Instantiate the LSTM model
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes).cuda()

    # Load your dataset and DataLoader (assuming you have the same dataset setup as before)
    # ...

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    result = {'error_rate_val': np.zeros(num_epochs),
              'error_rate_val_sparse': np.zeros(num_epochs),
              'error_rate_val_dense': np.zeros(num_epochs)}

    error_rate_val = np.zeros(num_epochs)
    error_rate_val_sparse = np.zeros(num_epochs)
    error_rate_val_dense = np.zeros(num_epochs)

    dataset = datasets.load_dataset('synthseq/flipflop')

    dataset.set_transform(tokenize_batch)

    # Splitting the dataset into training and evaluation subsets
    if train_type == 'train_mix':
        train_dataset = dataset['train'].shuffle(rdseed).select(range(40000))
    else:
        train_dataset = dataset['train'].shuffle(rdseed).select(range(40000))  # seed= 42, 142, 1142, 11142, 111142

    # Create DataLoader instances

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    eval_dataloader = DataLoader(dataset['val'].select(range(4000)), batch_size=batch_size, shuffle=False)
    eval_sparse_dataloader = DataLoader(dataset['val_sparse'].select(range(40000)), batch_size=batch_size,
                                        shuffle=False)
    eval_dense_dataloader = DataLoader(dataset['val_dense'].select(range(1000)), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = batch['tokens'].to(device)
            targets = inputs.roll(shifts=-1, dims=1)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Identify positions where the current token is 'r'
            is_r_mask = (inputs == 1)[:, :]  # Assuming 'r' maps to index 1

            # Prepare the targets and outputs for loss calculation
            # Since the targets are already shifted, you can use them directly
            outputs = outputs[is_r_mask]  # Filter outputs where the previous token was 'r'

            targets = targets[is_r_mask]  # Corresponding filtered targets

            # Calculate the loss
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Evaluate the model after each epoch (optional but recommended)
        error_rate_val[epoch] = evaluate(model, eval_dataloader, device)
        error_rate_val_sparse[epoch] = evaluate(model, eval_sparse_dataloader, device)
        error_rate_val_dense[epoch] = evaluate(model, eval_dense_dataloader, device)

    result['error_rate_val'] = error_rate_val
    result['error_rate_val_sparse'] = error_rate_val_sparse
    result['error_rate_val_dense'] = error_rate_val_dense
    result = pd.DataFrame(result)
    print(result)
    result.to_csv('output_LSTM.csv', index=False)
    return result

train_type='train'
rdseed = 42
num_epochs = 10
batch_size=32
hidden_size=128
num_layers=1
LSTM_flip_flop_task(train_type, rdseed, num_epochs, batch_size, hidden_size, num_layers)
