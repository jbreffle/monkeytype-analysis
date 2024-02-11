# pylint: disable=invalid-name
"""Home page of the app"""

# Imports
import os
import sys

import matplotlib.pyplot as plt
import streamlit as st
import pyprojroot.here
import numpy as np
import pandas as pd

sys.path.insert(0, str(pyprojroot.here()))  # Add parent directory to path
from src import process  # pylint: disable=wrong-import-position
from src import plot  # pylint: disable=wrong-import-position
from src import util  # pylint: disable=wrong-import-position


# Constants used across pages
APP_ICON_PATH = os.path.join(
    pyprojroot.here(), "assets/images/favicons/jb-favicon-512x512.png"
)
GH_ICON_PATH = os.path.join(
    pyprojroot.here(), "assets/images/favicons/github-mark-white.svg"
)


def add_sidebar_links():
    column_dimensions = [0.6, 3.4]
    with st.sidebar:
        logo_column_1 = st.columns(column_dimensions)
        with logo_column_1[0]:
            st.image(APP_ICON_PATH)
        with logo_column_1[1]:
            st.link_button(
                " jbreffle.github.io   ",
                "https://jbreffle.github.io/",
                type="primary",
                use_container_width=True,
            )
        logo_column_2 = st.columns(column_dimensions)
        with logo_column_2[0]:
            st.image(GH_ICON_PATH)
        with logo_column_2[1]:
            st.link_button(
                "github.com/jbreffle",
                "https://github.com/jbreffle",
                type="primary",
                use_container_width=True,
            )
    return


@st.cache_data
def hide_imgae_fullscreen():
    """Hides the fullscreen button for images."""
    hide_img_fs = """
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        """
    st.markdown(hide_img_fs, unsafe_allow_html=True)
    return


@st.cache_data
def set_plt_style():
    # white text and axes, Grid on, thinner axes lines, smaller font
    params = {
        "ytick.color": "w",
        "xtick.color": "w",
        "axes.labelcolor": "w",
        "axes.edgecolor": "w",
        "text.color": "w",
        "grid.color": "w",
        "grid.linewidth": 0.1,
        "axes.linewidth": 0.25,
        "axes.grid": True,
        "font.size": 8,
        "axes.titlepad": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 12,
    }
    plt.rcParams.update(params)
    return


@st.cache_data
def expand_sidebar_pages_vertical_expander():
    st.markdown(
        """
    <style>
    div[data-testid='stSidebarNav'] ul {max-height:none}</style>
    """,
        unsafe_allow_html=True,
    )
    return


def configure_page(page_title=None):
    """Convenience function to configure the page layout."""
    st.set_page_config(
        page_title=page_title,
        page_icon=APP_ICON_PATH,
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    add_sidebar_links()
    hide_imgae_fullscreen()
    expand_sidebar_pages_vertical_expander()
    set_plt_style()
    return


@st.cache_data
def load_data():
    process.combine_raw_results()
    if "user_processed_df" in st.session_state:
        user_processed_df = st.session_state
    else:
        user_processed_df = None
        st.session_state.user_processed_df = user_processed_df
    if "data_df" in st.session_state:
        loaded_df = st.session_state.data_df
    else:
        loaded_df = process.load_processed_results()
        st.session_state.data_df = loaded_df
    return loaded_df, user_processed_df


@st.cache_data
def user_upload_form():
    raise NotImplementedError


def main():
    """Main script for Home.py"""

    # Page configuration
    configure_page(page_title="Monketype Analysis")

    # Data set up
    data_df, user_processed_df = load_data()
    feature_name_df = pd.DataFrame(
        {
            "column_name": data_df.columns.values,
            "feature_name": [
                util.get_label_string(str) for str in data_df.columns.values
            ],
        }
    )
    feature_name_df.sort_values(
        "feature_name", inplace=True, key=lambda col: col.str.lower()
    )
    feature_name_df.reset_index(drop=True, inplace=True)
    scatter_options_df = feature_name_df[
        feature_name_df["column_name"].isin(
            [
                "acc",
                "z_acc",
                "consistency",
                "is_pb",
                "raw_wpm",
                "test_duration",
                "time_of_day_sec",
                "trial_type_num",
                "wpm",
                "z_wpm",
            ]
        )
    ]

    # Introduction
    st.title("monkeytype.com Data Analysis")
    st.write(
        """
        [monkeytype.com](<https://monkeytype.com>) is a popular typing test website.
        If you create an account,
        the website tracks your typing performance over time and displays some basic
        figures and statistics at
        [monkeytype.com/account](<https://monkeytype.com/account>).

        However, the website does not provide any tools for more in-depth analysis of
        your typing data.
        Thankfully, the website allows you to download your typing data in a
        `results.csv` file.

        This app is designed to help you analyze your typing data from Monkeytype.
        The app has default data from a user to demonstrate the app's capabilities,
        but you can also upload your own results file to analyze your own data.
        All plots on this page and on the Trial difficulty page can analyze uploaded
        Monkeytype data files.

        This app can be found at
        [jbreffle.github.io](<https://jbreffle.github.io/monkeytype-app>),
        and the code for the app is at
        [github.com/jbreffle/monkeytype-analysis](<https://github.com/jbreffle/monkeytype-analysis>).
        The code repository includes additional plots from supplementary analyses 
        in Jupyter notebooks.

        This app has multiple pages, each of which address different aspects of the
        data. The pages are accessible via the sidebar on the left.

        - Home (this page): Data upload form and feature correlations
        - Trial difficulty: Analyses comparing trial types of different difficulty
        - Simulated typing: Analysis of simulated typing data
        - Predicting performance: Training a model to predict typing performance from experience and trial type
        """
    )
    st.divider()

    # User file upload option
    st.header("User file upload")
    st.write(
        """
        All plots on this page can analyze any uploaded Monkeytype data file.
        """
    )
    # TODO add validation to the file upload, check if conforms to the expected format
    with st.form("User file upload form"):
        uploaded_files = st.file_uploader(
            "Upload your own Monkeytype `results.csv` file to analyze your own data.",
            accept_multiple_files=False,
            type=["csv"],
            key="user_file_upload",
        )
        submit_button = st.form_submit_button(label="Process user's raw data")
        if submit_button and uploaded_files is not None:
            user_raw_df = pd.read_csv(uploaded_files, sep="|")
            user_data_is_valid = util.validate_user_data(data_df, user_processed_df)
            if user_data_is_valid:
                user_processed_df = process.process_combined_results(user_raw_df)
                st.session_state.user_raw_df = user_raw_df
                st.session_state.user_processed_df = user_processed_df
            else:
                st.warning("User data is not in the expected format.")
        # Use expander to show users uploaded file
        with st.expander("User resutlts file", expanded=False):
            # If user_processed_df does not exist, show message
            if user_processed_df is None:
                st.write("No file uploaded and processed yet.")
            else:
                st.write("Processed user data")
                st.dataframe(user_processed_df)
    # Add user data to the session state, regardless of whether it was uploaded
    if "user_processed_df" in st.session_state:
        user_processed_df = st.session_state.user_processed_df
    use_user_data = st.checkbox(
        """Check this box to use your uploaded data in the analyses,
        or leave it unchecked to use the app's example data.""",
        value=False,
    )
    if use_user_data:
        # Give warning if user data is not uploaded
        if user_processed_df is None:
            st.warning("No user data uploaded yet.")
        else:
            data_df = user_processed_df
    st.divider()

    # Scatter and regression across time
    st.header("Feature correlations over time")
    st.write(
        """
        In the following scatter plot you can select a feature to plot against time
        and calculate its trend.
        You can also select a trial type to plot a subset of the data.
        """
    )
    # Selectbox for scatter
    option_default_index = int(
        np.where(scatter_options_df["column_name"] == "wpm")[0][0]
    )
    time_scatter_columns = st.columns(2)
    with time_scatter_columns[0]:
        feature_to_plot = st.selectbox(
            "Select a feature to plot",
            scatter_options_df["feature_name"],
            index=option_default_index,
        )
    # Option to plot by trial type
    trial_options = [None] + sorted(data_df["trial_type_id"].unique())
    with time_scatter_columns[1]:
        trial_type = st.selectbox(
            "Select to plot a trial-type subset",
            trial_options,
            index=None,
        )
    # Scatter plot
    column_to_plot = scatter_options_df.loc[
        scatter_options_df["feature_name"] == feature_to_plot, "column_name"
    ].values[0]
    fig = plt.figure(figsize=(5, 3))

    _ = plot.df_scatter(
        data_df,
        "datetime",
        column_to_plot,
        trial_type_id=trial_type,
        plot_regression=True,
        s=3,
    )
    st.pyplot(fig, use_container_width=True, transparent=True)
    st.divider()

    # Correlate features, independent of time
    st.header("Cross-feature correlations")
    st.write(
        """
        In the following scatter plot, 
        you can select two features to plot against each other and calculate
        their linear correlation.
        You can also select a trial type to plot a subset of the data.
        """
    )
    # Selectbox for scatter
    option_default_index_x = int(
        np.where(scatter_options_df["column_name"] == "acc")[0][0]
    )
    option_default_index_y = int(
        np.where(scatter_options_df["column_name"] == "wpm")[0][0]
    )
    scatter_corr_columns = st.columns(3)
    with scatter_corr_columns[0]:
        feature_x = st.selectbox(
            "Select a feature for the x-axis",
            scatter_options_df["feature_name"],
            index=option_default_index_x,
        )
    with scatter_corr_columns[1]:
        feature_y = st.selectbox(
            "Select a feature for the y-axis",
            scatter_options_df["feature_name"],
            index=option_default_index_y,
        )
        # Option to plot by trial type
    with scatter_corr_columns[2]:
        trial_type_2 = st.selectbox(
            "Select to plot a trial-type subset",
            trial_options,
            index=None,
            key="trial_type_2",
        )
    # Scatter plot
    column_x = scatter_options_df.loc[
        scatter_options_df["feature_name"] == feature_x, "column_name"
    ].values[0]
    column_y = scatter_options_df.loc[
        scatter_options_df["feature_name"] == feature_y, "column_name"
    ].values[0]
    fig = plt.figure(figsize=(5, 3))
    _ = plot.df_scatter(
        data_df,
        column_x,
        column_y,
        trial_type_id=trial_type_2,
        plot_regression=True,
        s=3,
    )
    st.pyplot(fig, use_container_width=True, transparent=True)
    st.divider()

    nb_url = "https://github.com/jbreffle/monkeytype-analysis/blob/main/notebooks/1_explore.ipynb"
    st.write(
        f"""
        For additional exploratory data anlyses,
        such as performance by day of the week,
        performance autocorrelation,
        and performance by trial feature,
        click here
        [./notebooks/1_explore.ipynb]({nb_url}).
        """
    )


if __name__ == "__main__":
    main()
