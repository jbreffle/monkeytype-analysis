import streamlit as st

import Home


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
    nb_url_1 = "?"
    st.write(
        """
        One way to predict performance is to use a neural network model.

        Click here
        [./notebooks/4_nn_predict.ipynb]({nb_url_1})
        for plots that will eventually be included in this app.
        """
    )
    # TODO
    st.divider()

    nb_url_2 = "?"
    st.write(
        """
        Click here
        [./notebooks/5_nn_hyperopti.ipynb]({nb_url_2})
        for results from hyperparameter optimization.
        """
    )


if __name__ == "__main__":
    main()
