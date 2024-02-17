import os

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pyprojroot
import torch
from torch import Tensor

from src import plot

import Home

MODEL_PATH = pyprojroot.here("streamlit/streamlit-data/")


@st.cache_resource
def load_model(model_path=MODEL_PATH):
    """Load a model from disk."""
    model = torch.jit.load(os.path.join(model_path, "streamlit_model.pt"))
    train_loss = np.load(model_path / "streamlit_train_loss.npy")
    test_loss = np.load(model_path / "streamlit_test_loss.npy")
    X_test = np.load(model_path / "X_test.npy")
    y_test = np.load(model_path / "y_test.npy")
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_test = torch.tensor(X_test, dtype=torch.float).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float).to(device)

    return model, train_loss, test_loss, X_test, y_test


def main():
    """New page."""

    # Page configuration
    Home.configure_page(page_title="Predicting performance")

    # Data set up
    data_df, _ = Home.load_data()

    # Page introduction
    st.title("Predicting performance")
    st.write(
        """
        Can we train a model to predict a user's performance on a trial?
        We don't have a direct measure of a trial type's difficulty,
        but we can use the user's performance on other trials to predict
        their performance on a new trial.
        """
    )
    st.divider()

    st.subheader("Neural network model")
    st.write(
        f"""
        One way to predict performance is to use a neural network model.
        """
    )
    model, train_loss, test_loss, X_test, y_test = load_model()
    # Plot losses over time
    fig = plt.figure(figsize=(6, 3))
    ax = plot.model_loss(train_loss, test_loss)
    st.pyplot(fig, use_container_width=True, transparent=True)
    # Plot actual vs predicted
    test_predictions = model(X_test)
    fig = plt.figure(figsize=(6, 3))
    ax = plot.model_scatter(
        y_test.cpu().detach().numpy(), test_predictions.cpu().detach().numpy()
    )
    st.pyplot(fig, use_container_width=True, transparent=True)
    # Plot actual and predicted across feature values
    fig = plt.figure(figsize=(6, 3))
    fig, ax0, ax1 = plot.model_feature_scatter(
        y_test.cpu(), test_predictions.detach().cpu().numpy(), X_test.cpu(), fig=fig
    )
    # Remove legend background
    st.pyplot(fig, use_container_width=True, transparent=True)

    st.divider()

    nb_url_1 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/4_nn_predict.ipynb"
    st.write(
        f"""
        Click here
        [./notebooks/4_nn_predict.ipynb]({nb_url_1})
        for additional training and additional plots.
        """
    )
    nb_url_2 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/5_nn_hyperopti.ipynb"
    st.write(
        f"""
        Click here
        [./notebooks/5_nn_hyperopti.ipynb]({nb_url_2})
        for results from hyperparameter optimization.
        """
    )


if __name__ == "__main__":
    main()
