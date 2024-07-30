import numpy as np
import torch
import datasets
from x_transformers import TransformerWrapper, Decoder
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

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

def transformer_flip_flop_task(train_type: str, rdseed: int, num_epochs: int, batch_size: int, dim: int, depth: int, heads: int):
    result={'error_rate_val':np.zeros(num_epochs),
            'error_rate_val_sparse':np.zeros(num_epochs),
            'error_rate_val_dense':np.zeros(num_epochs)}

    error_rate_val=np.zeros(num_epochs)
    error_rate_val_sparse = np.zeros(num_epochs)
    error_rate_val_dense = np.zeros(num_epochs)

    dataset = datasets.load_dataset('synthseq/flipflop')

    dataset.set_transform(tokenize_batch)

    model = TransformerWrapper(
        num_tokens=5,
        max_seq_len=512,
        attn_layers=Decoder(
            dim=dim,
            depth=depth,
            heads=heads
        )
    ).cuda()

    # Splitting the dataset into training and evaluation subsets
    if train_type=='train_mix':
        train_dataset = dataset['train'].shuffle(rdseed).select(range(40000))
    else:
        train_dataset = dataset['train'].shuffle(rdseed).select(range(40000))  # seed= 42, 142, 1142, 11142, 111142

    # Create DataLoader instances

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    eval_dataloader = DataLoader(dataset['val'].select(range(4000)), batch_size=batch_size, shuffle=False)
    eval_sparse_dataloader = DataLoader(dataset['val_sparse'].select(range(40000)), batch_size=batch_size, shuffle=False)
    eval_dense_dataloader = DataLoader(dataset['val_dense'].select(range(1000)), batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 3e-4

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Ensure the model is on the right device

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
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
        error_rate_val[epoch]=evaluate(model, eval_dataloader, device)
        error_rate_val_sparse[epoch]=evaluate(model, eval_sparse_dataloader, device)
        error_rate_val_dense[epoch]=evaluate(model, eval_dense_dataloader, device)

    result['error_rate_val']=error_rate_val
    result['error_rate_val_sparse']=error_rate_val_sparse
    result['error_rate_val_dense']=error_rate_val_dense
    result=pd.DataFrame(result)
    print(result)
    result.to_csv('output.csv',index=False)
    return result


if __name__ == "__main__":
    from pathlib import Path
    import argparse
    import json
    import submitit

    # set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query", help="path to json file containing query", default="query.json"
    )
    args = parser.parse_args()

    # read in query
    if Path(args.query).resolve().exists():
        query_path = Path(args.query).resolve()
    else:
        # throw
        raise ValueError(
            f"Could not locate {args.query} in query directory or as absolute path"
        )
    with open(query_path) as f:
        query = json.load(f)
    # save query parameters to variables. if you want a default, better to put
    # at the outermost call to a function.

    default_train_type='train'
    train_type=query.get("train_type",default_train_type)
    default_random_seed=42 # 42, 142, 1142, 11142, 111142
    rdseed=query.get("rdseed",default_random_seed)
    default_num_epochs=10
    num_epochs=query.get("num_epochs",default_num_epochs)
    default_batch_size=32
    batch_size=query.get("batch_size",default_batch_size)
    default_dim = 512
    dim = query.get("dim", default_dim)
    default_depth = 6
    depth = query.get("depth", default_depth)
    default_heads = 8
    heads = query.get("heads", default_heads)
    output_directory = Path("results").resolve()
    executor = submitit.AutoExecutor(folder=output_directory)
    # here we unpack the query dictionary and pull any slurm commands that
    # are in 'slurm' key. For more info on the ** syntax, see:
    # https://stackoverflow.com/a/36908. The slurm options here are the same
    # as those you use on the command line but instead of prepending with '--'
    # we prepend with 'slurm_'
    executor.update_parameters(**query.get("slurm", {}))
    print(train_type,rdseed,num_epochs,batch_size,dim,depth,heads)

    # if submitit is true in our query json, we'll use submitit
    if query.get("submitit", False):
        executor.submit(
            transformer_flip_flop_task,
            train_type,
            rdseed,
            num_epochs,
            batch_size,
            dim,
            depth,
            heads,
        )
    else:
        transformer_flip_flop_task(
            train_type,
            rdseed,
            num_epochs,
            batch_size,
            dim,
            depth,
            heads,
        )
