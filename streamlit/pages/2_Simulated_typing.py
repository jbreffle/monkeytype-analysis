import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src import util
from src import plot

import Home


@st.cache_data
def run_poisson_sim(**kwargs):
    np.random.seed(1)
    if "silent" not in kwargs:
        kwargs["silent"] = True
    wpm, acc, n_mistakes = util.run_simulation_poisson(**kwargs)
    return wpm, acc, n_mistakes


@st.cache_data
def run_simple_sim(**kwargs):
    np.random.seed(1)
    if "silent" not in kwargs:
        kwargs["silent"] = True
    wpm, acc, n_mistakes = util.run_simulation_simple(**kwargs)
    return wpm, acc, n_mistakes


def main():
    """New page."""

    # Page configuration
    Home.configure_page(page_title="Simulated typing")

    # Data set up
    data_df, _ = Home.load_data()

    # Page introduction
    st.title("Simulated typing")
    st.write(
        """
        One important question when practicing typing is, 
        "how carefully should one try to avoid mistakes"?
        In the raw data we can see that performance (wpm) is strongly correlated with
        accuracy. But is this correlation causal?

        One hypothesis is that mistakes are i.i.d. random, 
        and each mistake requires a fixed time to correct
        (time that otherwise would have been spent typing).
        This alone might cause the degree of correlation observed in the data.
        Is the best approach for practicing typing to balance the probability of making
        a mistake with the time it takes to correct it?
        """
    )
    st.divider()

    st.subheader("Simulated typing: random mistake draws")
    st.write(
        f"""
        One method to simulate typing is to randomly draw mistakes.
        """
    )
    # TODO
    avg_wpm = 60
    avg_acc = 0.95
    n_trials = 1000
    wpm, acc, n_mistakes = run_simple_sim(
        avg_wpm=avg_wpm, avg_acc=avg_acc, n_trials=n_trials
    )
    # Plot scatter_hist of wpm and acc
    fig = plt.figure(figsize=(6, 4))
    ax, ax_histx, ax_histy = plot.sim_scatter_hist(wpm, acc, fig=fig)
    ax.axvline(avg_wpm, color="grey", linestyle="--", alpha=0.5)
    ax.axhline(avg_acc, color="grey", linestyle="--", alpha=0.5)
    ax.plot(np.mean(wpm), np.mean(acc), "ro")
    st.pyplot(fig, use_container_width=True, transparent=True)
    # TODO
    st.divider()

    st.subheader("Simulated typing: Poisson process")
    st.write(
        f"""
        An alternative simulation method is to use a Poisson process.
        """
    )
    avg_wpm = 60
    avg_acc = 0.95
    wpm, acc, n_mistakes = run_poisson_sim(avg_wpm=avg_wpm, avg_acc=avg_acc)
    # Plot scatter_hist of wpm and acc
    fig = plt.figure(figsize=(6, 4))
    ax, ax_histx, ax_histy = plot.sim_scatter_hist(wpm, acc, fig=fig)
    ax.axvline(avg_wpm, color="grey", linestyle="--", alpha=0.5)
    ax.axhline(avg_acc, color="grey", linestyle="--", alpha=0.5)
    ax.plot(np.mean(wpm), np.mean(acc), "ro")
    st.pyplot(fig, use_container_width=True, transparent=True)
    st.divider()

    nb_url_1 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/3a_sim_simple.ipynb"
    nb_url_2 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/3b_sim_poisson.ipynb"
    st.write(
        f"""
        Click here 
        [./notebooks/3a_sim_simple.ipynb]({nb_url_1})
        for the simple simulation method notebook.

        Click here
        [./notebooks/3a_sim_poisson.ipynb]({nb_url_2})
        for the Poisson simulation notebook.
        """
    )


if __name__ == "__main__":
    main()
