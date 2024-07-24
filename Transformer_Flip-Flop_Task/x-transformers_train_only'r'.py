import torch
import datasets
from x_transformers import TransformerWrapper, Decoder
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

def tokenize_batch(batch):
    mapping = {'w': 0, 'r': 1, 'i': 2, '0': 3, '1': 4}
    tokenized_batch = [[mapping[char] for char in s] for s in batch['text']]
    return {'tokens': torch.tensor(tokenized_batch, dtype=torch.int64)}

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

dataset = datasets.load_dataset('synthseq/flipflop')

dataset.set_transform(tokenize_batch)

model = TransformerWrapper(
    num_tokens = 5,
    max_seq_len = 512,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8
    )
).cuda()

# Splitting the dataset into training and evaluation subsets

train_dataset = dataset['train'].shuffle(seed=142).select(range(40000)) #seed= 42, 142, 1142, 11142, 111142

eval_dataset = dataset['val_sparse'].select(range(10000))


# Create DataLoader instances

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-4

optimizer = optim.Adam(model.parameters(), lr=learning_rate)



num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Ensure the model is on the right device

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs = batch['tokens'].to(device)  # Move input to the same device as the model
        targets = inputs.roll(shifts=-1, dims=1)  # Assuming next-token prediction, shift targets right

        # Identify positions where the current token is 'r'
        is_r_mask = (inputs == 1)[:, :]  # Assuming 'r' maps to index 1

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass focusing on 'r' positions

        outputs = model(inputs)

        outputs = outputs[is_r_mask]  # Filter outputs where the previous token was 'r'

        targets = targets[is_r_mask]  # Corresponding filtered targets

        # Calculate loss
        loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

    # Evaluate the model after each epoch (optional but recommended)
    evaluate(model, eval_dataloader, device)