import streamlit as st

import Home


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
    nb_url_1 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/3a_sim_simple.ipynb"
    st.write(
        f"""
        One method to simulate typing is to randomly draw mistakes.

        Click here 
        [./notebooks/3a_sim_simple.ipynb]({nb_url_1})
        for plots that will eventually be included in this app.
        """
    )
    # TODO
    st.divider()

    st.subheader("Simulated typing: Poisson process")
    nb_url_2 = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/3b_sim_poisson.ipynb"
    st.write(
        f"""
        An alternative simulation method is to use a Poisson process.

        Click here
        [./notebooks/3a_sim_poisson.ipynb]({nb_url_2})
        for plots that will eventually be included in this app.
        """
    )
    # TODO


if __name__ == "__main__":
    main()
