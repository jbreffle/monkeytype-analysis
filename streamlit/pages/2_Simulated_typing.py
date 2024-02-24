import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src import util
from src import plot

import Home


@st.cache_data
def plot_sim(wpm, acc, avg_wpm, avg_acc):
    fig = plt.figure(figsize=(6, 4))
    ax, _, _ = plot.sim_scatter_hist(wpm, acc, fig=fig)
    ax.axvline(avg_wpm, color="grey", linestyle="--", alpha=0.5)
    ax.axhline(avg_acc, color="grey", linestyle="--", alpha=0.5)
    ax.plot(np.mean(wpm), np.mean(acc), "ro")
    st.pyplot(fig, use_container_width=True, transparent=True)
    return None


@st.cache_data(show_spinner=False)
def run_poisson_sim(sim_seed=1, **kwargs):
    np.random.seed(sim_seed)
    if "silent" not in kwargs:
        kwargs["silent"] = True
    wpm, acc, n_mistakes = util.run_simulation_poisson(**kwargs)
    return wpm, acc, n_mistakes


@st.cache_data(show_spinner=False)
def run_simple_sim(sim_seed=1, **kwargs):
    np.random.seed(sim_seed)
    if "silent" not in kwargs:
        kwargs["silent"] = True
    wpm, acc, n_mistakes = util.run_simulation_simple(**kwargs)
    return wpm, acc, n_mistakes


def main():
    """New page."""

    # Page configuration
    Home.configure_page(page_title="Simulated typing")

    # Page introduction
    st.title("Simulated typing")
    iid_url = "https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables"
    st.write(
        f"""
        One important question when practicing typing is, 
        "how carefully should one try to avoid mistakes"?
        In the raw data we can see that performance (wpm) is strongly correlated with
        accuracy. But is this correlation causal?

        One hypothesis is that mistakes are [i.i.d.](<{iid_url}>), 
        and each mistake requires a fixed time to correct
        (time that otherwise would have been spent typing).
        This alone might cause the degree of correlation observed in the data.
        We can study the causal relationship between accuracy and wpm through 
        simulations.
        """
    )
    # Run simulations, with a button to re-run with a new seed
    avg_wpm = 60
    avg_acc = 0.95
    n_trials = 1000
    st.write("Click the button below to re-run the simulations with a new seed.")
    seed = st.button("Re-run simulations")
    if seed:
        seed = np.random.randint(1e6)
    else:
        seed = 1  # Default seed
    with st.spinner("Running simulations..."):
        wpm_simple, acc_simple, _ = run_simple_sim(
            sim_seed=seed, avg_wpm=avg_wpm, avg_acc=avg_acc, n_trials=n_trials
        )
        wpm_possion, acc_possion, _ = run_poisson_sim(
            sim_seed=seed, avg_wpm=avg_wpm, avg_acc=avg_acc, n_trials=n_trials
        )
    st.divider()

    # Simple simulation
    st.subheader("Simulated typing: random mistake draws")
    st.write(
        """
        A simple method of simulating a typing session is to draw a random
        number of mistakes from a Poisson distribution and then assume each of those
        mistakes takes some random amount of time to correct.
        The WPM and accuracy can then be calculated based on those random values.

        Here we see results from 1000 such simulated trials.
        If we assume an average performance of 60 WPM and 95\% accuracy
        then we can reproduce the $R^2$ that is observed in the actual data
        when we assume each mistake takes an average of 0.5 seconds to correct
        with a standard deviation of 0.45.
        """
    )
    plot_sim(wpm_simple, acc_simple, avg_wpm, avg_acc)
    st.write(
        """
        The dashed grey lines show the target WPM and accuracy.
        The red dot is the mean WPM and accuracy over all trial simulations.
        The red line is the linear regression.
        """
    )
    st.divider()

    # Poisson simulation
    st.subheader("Simulated typing: Poisson process")
    st.write(
        """
        A more complicated but more realistic simulation approach would be to simulate
        typing behavior across time within each trial using a Poisson process.
        
        We see that we reproduce similar results to the simple method.
        Here we model mistakes as a Poisson process and
        assume each mistake takes 0.75 seconds to fix.
        """
    )
    plot_sim(wpm_possion, acc_possion, avg_wpm, avg_acc)
    st.write(
        """
        The dashed grey lines show the target WPM and accuracy.
        The red dot is the mean WPM and accuracy over all trial simulations.
        The red line is the linear regression.
        """
    )
    st.divider()

    # Links to notebooks
    nb_url_1 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/3a_sim_simple.ipynb"
    nb_url_2 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/3b_sim_poisson.ipynb"
    st.write(
        f"""
        Click here 
        [./notebooks/3a_sim_simple.ipynb]({nb_url_1})
        for the simple simulation notebook.

        Click here
        [./notebooks/3a_sim_poisson.ipynb]({nb_url_2})
        for the Poisson simulation notebook.
        """
    )


if __name__ == "__main__":
    main()
