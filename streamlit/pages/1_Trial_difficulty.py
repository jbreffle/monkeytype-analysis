import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pyprojroot

from src import plot
import Home


def main():
    """Trial difficulty page.
    Shows trial types, z-scoring, and log-fit normalization.
    """

    # Page configuration
    Home.configure_page(page_title="Trial difficulty")

    # Data set up
    data_df, user_processed_df = Home.load_data()

    # Page introduction
    st.title("Trial difficulty")
    st.write(
        """
        Monkeytype has a wide variety of typing test options.
        You can vary the trial durations, the language, and size of the vocabulary.
        Yuu can include puncation, numbers, and special characters,
        and there are a number of special modes.

        One of the easiest trial types would be 15 second trials of "English"
        (which is only the 200 most common words in English)
        with no punctuation, numbers, or special characters.
        """
    )
    st.image(
        str(pyprojroot.here("./assets/english_600x150.png")),
        caption="Example of an English trial from monkeytype.com",
        use_column_width=True,
    )
    st.write(
        """
        One of the hardest trial types would be 120 second trials of the "ASCII" mode,
        which includes all characters on a standard keyboard.
        """
    )
    st.image(
        str(pyprojroot.here("./assets/ascii_600x150.png")),
        caption="Example of an ASCII trial from monkeytype.com",
        use_column_width=True,
    )
    st.write(
        """
        With such a wide range of difficulty,
        how can we compare performance across different trial types?
        """
    )
    st.text("")

    use_user_data = st.checkbox(
        """As on the Home page, you can choose to use your own data or the example data.
        Check this box to use your uploaded data in the analyses,
        or leave it unchecked to use the app's example data.
        """,
        value=False,
    )
    if use_user_data:
        # Give warning if user data is not uploaded
        if user_processed_df is None:
            st.warning("No user data uploaded yet. See Home page.")
        else:
            data_df = user_processed_df
    st.divider()

    st.subheader("Comparing trials of different difficulty")
    st.write(
        """
        When plotting the WPM across time for all trials
        we see that WPM jumps between different average values,
        suggesting that the user is completing trials of different difficulty.
        """
    )
    fig = plt.figure(figsize=(6, 3))
    ax = plot.df_scatter(data_df, "datetime", "wpm", plot_regression=False, s=3)
    st.pyplot(fig, use_container_width=True, transparent=True)
    st.write(
        """
        By defining a trial type as a unique combination of all trial options,
        we can group trials by trial type and compare performance across those
        trial types.

        Here is a histogram of the number of trials completed for each trial type.
        We will focus our following analyses on the several trial types which have
        the most trials completed.
        """
    )
    fig = plt.figure(figsize=(6, 3))
    ax = plt.gca()
    ax.hist(
        data_df["trial_type_id"],
        bins=data_df["trial_type_id"].nunique(),
        edgecolor="black",
        align="left",
    )
    ax.set_xlabel("Trial type ID (sorted)")
    ax.set_ylabel("Trials completed (count)")
    st.pyplot(fig, use_container_width=True, transparent=True)
    # Coloring by trial type
    st.write(
        """
        Now that we have defined trial types,
        we can plot the same graph of WPM across time as before,
        but color the points based on trial type.
        The most common trial types are each given a unique color,
        and the rest are labeled in grey.
        """
    )
    fig = plt.figure(figsize=(6, 3))
    ax = plot.df_scatter(
        data_df, "datetime", "wpm", plot_regression=False, s=3, n_colors=5
    )
    st.pyplot(fig, use_container_width=True, transparent=True)
    st.divider()

    # Z-scoring
    st.subheader("Z-scoring by trial type")
    st.write(
        """
        Z-scoring is a way of normalizing data to compare across different distributions.
        We can Z-score the WPM for each trial type to compare them.
        Z-scoring involves shifting the mean of the data to 0
        and scaling the standard deviation to 1,
        following the equation:
        $$
        z = \\frac{x - \\mu}{\\sigma}
        $$
        where $x$ is the original value,
        $\mu$ is the mean,
        $\sigma$ is the standard deviation,
        and $z$ is the resulting z-scored value.
        """
    )
    fig = plt.figure(figsize=(6, 3))
    ax = plot.df_scatter(
        data_df, "datetime", "z_wpm", plot_regression=False, s=3, n_colors=5
    )
    st.pyplot(fig, use_container_width=True, transparent=True)
    st.write(
        """
        There is a problem with our use of z-scoring,
        which is that performance is non-stationary.
        As a user completes more trials,
        they improve and their performance increases.
        This means that the mean WPM for each trial type is not constant.
        """
    )
    st.divider()

    st.subheader("Logarithmic learning curves")
    st.write(
        r"""A standard model for skill learning is that the rate of improvement
        decreases as the skill level increases.
        A simple model for this is the logarithmic learning curve,
        given by the equation:
        $$
        y = x^\alpha + c
        $$
        where $y$ is the performance,
        $x$ is the skill level,
        and $\alpha$ and $c$ are constants.
        """
    )
    st.write(
        """
        Here is an example of a logarithmic learning curve,
        showing the WPM as a function of the number of trials completed
        for the trial type with the most trials completed.
        """
    )
    fig = plt.figure(figsize=(6, 3))
    ax = plot.log_fit_scatter(
        data_df, "wpm", silent=True, legend_on=False, n_trial_types=1
    )
    ax.set_title("")
    st.pyplot(fig, use_container_width=True, transparent=True)
    st.write(
        """
        Here is a similar plot,
        but showing the four most common trial types.
        """
    )
    fig = plt.figure(figsize=(6, 3))
    ax = plot.log_fit_scatter(
        data_df, "wpm", silent=True, legend_on=False, n_trial_types=4
    )
    ax.set_title("")
    st.pyplot(fig, use_container_width=True, transparent=True)
    # Log-fit normalized WPM
    st.write(
        """
        Rather than z-scoring to make the mean 0 and the standard deviation 1,
        we can normalize the data based on the fitted log curve.
        We will use the fitted log curve to perform the normalization
        $$
        y = \\frac{x - c}{\sigma_{res}}
        $$
        where $y$ is the log-fit normalized value,
        $x$ is the original value,
        $c$ is the log-fit initial value parameter,
        and $\sigma_{res}$ is standard deviation of the residuals that result
        from subtracting the fitted curve from the original values.

        Those normalized values are show in the following plot.
        """
    )
    fig = plt.figure(figsize=(6, 3))
    ax = plot.df_scatter(
        data_df,
        "trial_type_num",
        "log_norm_wpm",
        plot_regression=False,
        s=3,
        n_colors=5,
    )
    st.pyplot(fig, use_container_width=True, transparent=True)
    st.write(
        """
        Here are the same data, but plotted across time.
        """
    )
    fig = plt.figure(figsize=(6, 3))
    ax = plot.df_scatter(
        data_df, "datetime", "log_norm_wpm", plot_regression=False, s=3, n_colors=5
    )
    st.pyplot(fig, use_container_width=True, transparent=True)

    st.write(
        """
        When we plot across time
        we see that there is one remaining issue,
        which is that we'd expect there to be some
        general underlying typing skill that transfers across trial types.
        How can we account for this?
        """
    )

    st.divider()

    nb_url_1 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/2a_z_scoring.ipynb"
    nb_url_2 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/2b_learning_curve.ipynb"
    st.write(
        f"""
        For additional trial-type specific exploratory anlyses
        click here
        [./notebooks/2a_z_scoring.ipynb]({nb_url_1})
        and here
        [./notebooks/2b_learning_curve.ipynb]({nb_url_2}).
        """
    )


if __name__ == "__main__":
    main()
