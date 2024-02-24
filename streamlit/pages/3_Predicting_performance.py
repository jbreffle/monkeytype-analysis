import os

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pyprojroot
import torch

from src import plot

import Home

MODEL_PATH = pyprojroot.here("streamlit/streamlit-data/")


@st.cache_data
def plot_model_feature_scatter(actual_values, predicted_values, _x_test):
    fig = plt.figure(figsize=(6, 3))
    fig, _, _ = plot.model_feature_scatter(
        actual_values, predicted_values, _x_test, fig=fig
    )
    st.pyplot(fig, use_container_width=True, transparent=True)
    return None


@st.cache_data
def plot_model_scatter(actual_values, predicted_values):
    fig = plt.figure(figsize=(6, 3))
    _ = plot.model_scatter(actual_values, predicted_values)
    st.pyplot(fig, use_container_width=True, transparent=True)
    return None


@st.cache_data
def plot_model_loss(train_loss, test_loss):
    fig = plt.figure(figsize=(6, 3))
    _ = plot.model_loss(train_loss, test_loss)
    st.pyplot(fig, use_container_width=True, transparent=True)
    return None


@st.cache_resource(show_spinner=False)
def load_predictions(_model, _x_test, _y_test):
    """Load predictions from a model."""
    # Predictions
    predicted_values = _model(_x_test)
    actual_values = _y_test
    return predicted_values.cpu().detach().numpy(), actual_values.cpu().detach().numpy()


@st.cache_resource(show_spinner=False)
def load_model(model_path=MODEL_PATH):
    """Load a model from disk."""
    device = "cpu"  # No cuda on streamlit community cloud
    model = torch.jit.load(
        os.path.join(model_path, "streamlit_model.pt"), map_location=device
    )
    train_loss = np.load(model_path / "streamlit_train_loss.npy")
    test_loss = np.load(model_path / "streamlit_test_loss.npy")
    x_test = np.load(model_path / "X_test.npy")
    y_test = np.load(model_path / "y_test.npy")
    # Select device
    model.to(device)
    x_test = torch.tensor(x_test, dtype=torch.float).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float).to(device)
    return model, train_loss, test_loss, x_test, y_test


def main():
    """New page."""

    # Page configuration
    Home.configure_page(page_title="Predicting performance")

    # Page introduction
    st.title("Predicting performance")
    st.write(
        """
        Given the large differences in difficulty between different trial types,
        it is difficult to compare performance across trials.
        Can we train a model that is able to predict a user's performance over time
        and across different trial types?
        """
    )
    st.divider()

    # Network loss and predictions
    st.subheader("Neural network model")
    st.write(
        """
        One way to predict performance is to use a neural network model.
        Here we train a simple feedforward neural network with a single output node
        to predict the user's typing speed (wpm).

        The features for each trial are the following:
        - Which trial type it is
        - The trial's accuracy (%)
        - The number of all trials completed so far
        - The number of trials of the same type completed so far
        
        Here are the Train and Test losses over time from training one such network:
        """
    )
    with st.spinner("Loading model..."):
        model, train_loss, test_loss, x_test, y_test = load_model()
        predicted_values, actual_values = load_predictions(model, x_test, y_test)
        x_test = x_test.cpu()
    plot_model_loss(train_loss, test_loss)
    st.write(
        """        
        We are able to predict the user's typing speed with a reasonable degree of
        accuracy.
        Here is the predicted typing speed vs. the actual typing speed for the test set:
        """
    )
    plot_model_scatter(actual_values, predicted_values)
    st.write(
        """
        Here is the same data, but now we are plotting the actual values in blue
        and the predicted values in orange, across the different feature values.
        We see that the model learned the shape of performance curves for the different
        trial types.
        """
    )
    plot_model_feature_scatter(actual_values, predicted_values, x_test)
    st.divider()

    # Links to notebooks
    nb_url_1 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/4_nn_predict.ipynb"
    nb_url_2 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/5_nn_hyperopti.ipynb"
    st.write(
        f"""
        Click here
        [./notebooks/4_nn_predict.ipynb]({nb_url_1})
        for additional analysis and plots from training the neural network,
        including analysis of dummy data to evaluate the model's learned
        performance curves.

        Click here
        [./notebooks/5_nn_hyperopti.ipynb]({nb_url_2})
        for results from hyperparameter optimization,
        including plots of loss as a function of hyperparameter values.
        """
    )


if __name__ == "__main__":
    main()
