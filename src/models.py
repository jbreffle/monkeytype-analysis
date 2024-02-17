"""Torch models for the monkeytype project."""

# Imports
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pyprojroot

import hashlib

MODEL_PATH = pyprojroot.here("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

STREAMLIT_MODE_PATH = pyprojroot.here("streamlit/streamlit-data")
STREAMLIT_MODE_PATH.mkdir(parents=True, exist_ok=True)


# Functions
def load_loss(model_name, model_path=MODEL_PATH):
    """Load a model from disk."""
    model_hash = model_name.split("_")[1].split(".")[0]
    train_loss = np.load(model_path / f"train_loss_{model_hash}.npy")
    test_loss = np.load(model_path / f"test_loss_{model_hash}.npy")
    return train_loss, test_loss


def load_model(model_name, model_path=MODEL_PATH):
    """Load a model from disk."""
    model = torch.load(model_path / model_name)
    return model


def save_for_streamlit(
    model,
    train_loss,
    test_loss,
    X_test,
    y_test,
    streamlit_model_path=STREAMLIT_MODE_PATH,
):
    # Save to streamlit
    model_streamlit = torch.jit.script(model)
    model_streamlit.save(streamlit_model_path / "streamlit_model.pt")
    # Save to streamlit
    np.save(streamlit_model_path / "streamlit_train_loss", train_loss)
    np.save(streamlit_model_path / "streamlit_test_loss", test_loss)
    np.save(streamlit_model_path / "X_test", X_test.cpu())
    np.save(streamlit_model_path / "y_test", y_test.cpu())
    return None


def save_model(
    model,
    model_name,
    train_loss=None,
    test_loss=None,
    model_path=MODEL_PATH,
    streamlit_model_path=STREAMLIT_MODE_PATH,
    save_to_streamlit=False,
):
    """Save a model to disk."""
    torch.save(model, model_path / model_name)
    model_hash = model_name.split("_")[1].split(".")[0]
    if train_loss is not None:
        np.save(model_path / f"train_loss_{model_hash}", train_loss)
    if test_loss is not None:
        np.save(model_path / f"test_loss_{model_hash}", test_loss)
    return None


def model_exists(model_name, model_path=MODEL_PATH):
    """Check if a model exists."""
    return (model_path / model_name).is_file()


def get_model_name(params, str_length=20):
    """Create a unique hash based on model hyperparameters, to use as a filename."""
    string_to_hash = ""
    for key, value in vars(params).items():
        string_to_hash += f"{key}{value}"
    hash_str = hashlib.md5(string_to_hash.encode()).hexdigest()[:str_length]
    model_name = f"model_{hash_str}.pt"
    return model_name


def generate_dummy_df(X_df, trial_type_id=None):
    """Generate a dummy dataframe for testing the models."""
    # TODO this still needs cleaned up
    if trial_type_id is None:
        trial_type_labels = [col for col in X_df.columns if "trial_type_id" in col]
        n_trial_types = len(trial_type_labels)
        # Mean accuracy for each trial type in trial_type_labels
        trial_type_mean_acc = np.zeros(n_trial_types)
        for i, trial_type_label in enumerate(trial_type_labels):
            trial_type_mean_acc[i] = X_df[X_df[trial_type_labels[i]] == 1]["acc"].mean()
        n_trial_num = 200
        n_trial_type_num = 200
        # Create a matrix of all combinations of 1:n_trial_types, 1:n_trial_num, 1:n_trial_type_num
        X = np.meshgrid(
            range(1, n_trial_types + 1),
            range(1, n_trial_num + 1),
            range(1, n_trial_type_num + 1),
            indexing="ij",
            sparse=False,
        )
        flattened_grid = np.array(X).reshape(3, -1).T
        # Remove rows where the third column exceeds the second column
        flattened_grid = flattened_grid[flattened_grid[:, 2] <= flattened_grid[:, 1]]
        # Based on the first column, fill in binary categorical values for trial_type_id
        dummy_df = pd.DataFrame(columns=X_df.columns)
        for i, trial_type_label in enumerate(trial_type_labels):
            dummy_df[trial_type_label] = 1 * (flattened_grid[:, 0] == (i + 1))
        dummy_df["acc"] = trial_type_mean_acc[flattened_grid[:, 0] - 1]
        dummy_df["trial_num"] = flattened_grid[:, 1]
        dummy_df["trial_type_num"] = flattened_grid[:, 2]
        dummy_df
    else:
        trial_column_label = f"trial_type_id_{trial_type_id}"
        dummy_df = X_df.copy()
        dummy_df["acc"] = X_df[X_df[trial_column_label] == 1]["acc"].mean()
        for col in dummy_df.columns:
            if col.startswith("trial_type_id"):
                dummy_df[col] = 0
        # Set the the column combined_id_+num2str(mode_trial_id) to 1
        dummy_df[trial_column_label] = 1
        # Set trial_num to the index+1
        dummy_df["trial_num"] = dummy_df.index + 1
        # Set trial_type_num to equal trial_num
        dummy_df["trial_type_num"] = dummy_df["trial_num"]
    return dummy_df


def get_default_params():
    """Function to initilize default hyperparameters"""
    params = type("Params", (), {})()
    # Data
    params.batch_size = 250
    params.train_size = 0.8
    # Model
    params.n_hidden_units = 10
    # Training
    params.log_interval = 100  # How often to log results in epochs
    params.lr = 1e-3  # Learning rate
    params.weight_decay = 1e-5
    params.n_epochs = 1000
    params.gamma = 0.999  # Learning rate decay
    # Device
    params.use_cuda = True
    params.device = torch.device(
        "cuda" if torch.cuda.is_available() and params.use_cuda else "cpu"
    )
    # Not implemented yet
    params.dry_run = None  # not implemented yet
    params.save_model = None  # not implemented yet
    params.test_batch_size = None  # not implemented yet
    return params


def train_and_evaluate(
    model, train_loader, test_loader, scheduler, optimizer, loss_function, params
):
    """..."""
    train_loss = []
    test_loss = []
    start_time = time.time()
    for epoch_idx in range(1, params.n_epochs + 1):
        step_train_loss = train(
            train_loader,
            model,
            optimizer,
            loss_function,
            params,
        )
        step_test_loss = test(model, params.device, test_loader, loss_function)
        scheduler.step()
        # Append losses
        train_loss.append(step_train_loss)
        test_loss.append(step_test_loss)
        # print loss and runtime
        if (epoch_idx + 1) % params.log_interval == 0:
            txt = "Epoch [{:4d}/{:4d}], Train loss: {:07.4f}, Test loss: {:07.4f}, Run Time: {:05.2f}"
            print(
                txt.format(
                    epoch_idx + 1,
                    params.n_epochs,
                    step_train_loss,
                    step_test_loss,
                    time.time() - start_time,
                )
            )
    return train_loss, test_loss


def train(
    train_loader,
    model,
    optimizer,
    loss_function,
    params,
):
    model.train()
    # Train set
    step_train_loss = 0
    for X, y in train_loader:
        X, y = X.to(params.device), y.to(params.device)
        preds = model(X)
        loss = loss_function(preds, y)
        optimizer.zero_grad()
        loss.backward()
        step_train_loss += loss.item()
        optimizer.step()
    step_train_loss = step_train_loss / len(train_loader)

    return step_train_loss


def test(model, device, test_loader, loss_function, silent=True):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()
    # test_loss /= len(test_loader.dataset)

    if not silent:
        print(f"Test loss: {test_loss:.4f}")
    return test_loss


# Models
class MLPCondensedVariable(nn.Module):
    """
    Multi-layer perceptron for non-linear regression.
    """

    def __init__(self, input_n, n_hidden_units, output_n, nLayers):
        super().__init__()
        # Input layer
        layers = [nn.Linear(input_n, n_hidden_units), nn.ReLU()]
        # Hidden layers
        for _ in range(nLayers):
            layers.extend([nn.Linear(n_hidden_units, n_hidden_units), nn.ReLU()])
        # Output layer
        layers.append(nn.Linear(n_hidden_units, output_n))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLPcondensed(nn.Module):
    """
    Multi-layer perceptron for non-linear regression.
    """

    def __init__(self, input_n, n_hidden_units, output_n):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_n, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, output_n),
        )

    def forward(self, x):
        return self.layers(x)


class SLPnet(nn.Module):
    """
    Single layer perceptron for non-linear regression.
    """

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(in_features=3, out_features=9)
        self.hidden_1 = nn.Linear(in_features=9, out_features=9)
        self.output = nn.Linear(in_features=9, out_features=1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        return self.output(x)
