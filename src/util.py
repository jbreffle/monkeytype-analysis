"""Small utility functions for the monkeytype project"""

# Imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Functions
def validate_user_data(data_df, user_processed_df):
    """Returns boolean indicating whether the user data is valid.
    The user data is valid if it is a pandas DataFrame and has the same columns
    and same data types as the example data.
    """
    if user_processed_df is None:
        return False
    if not isinstance(user_processed_df, pd.DataFrame):
        return False
    if user_processed_df.columns.to_list() != data_df.columns.to_list():
        return False
    #if not user_processed_df.dtypes.equals(data_df.dtypes):
    #    return False
    return True



def get_label_string(label):
    """..."""
    match label:
        case "log_norm_wpm":
            return "WPM (Log-fit norm.)"
        case "acc":
            return "Accuracy (%)"
        case "z_acc":
            return "Accuracy (z-score)"
        case "wpm":
            return "Words per minute"
        case "z_wpm":
            return "WPM (z-score)"
        case "datetime":
            return "Date"
        case "mode2":
            return "Trial duration (s)"
        case "consistency":
            return "WPM consistency (%)"
        case "is_pb":
            return "Personal best"
        case "raw_wpm":
            return "Raw WPM"
        case "test_duration":
            return "Test duration (s)"
        case "trial_type_id":
            return "Trial type"
        case "trial_type_num":
            return "Trial type completed"
        case "time_of_day_sec":
            return "Time of day (s)"
        case "iti_all":
            return "Inter-trial interval (s, all trials)"
        case "iti_session":
            return "Inter-trial interval (s, session)"
        case _:
            return label


def add_log_norm_wpm(df, min_n_trials=10):
    """Add a column to the dataframe with the log normalised WPM.
    This function optimizes the previous approach by grouping by 'combined_id'
    and applying calculations per group, rather than per row.
    """
    # Initialize the column to avoid SettingWithCopyWarning
    df["log_norm_wpm"] = 0

    # Group by 'combined_id' and loop over each group
    for combined_id, group in df.groupby("combined_id"):
        num_trials = group.shape[0]
        if num_trials <= min_n_trials:
            df.loc[df["combined_id"] == combined_id, "log_norm_wpm"] = np.nan
            continue

        y_vec = group["wpm"].values
        x_vec = np.arange(1, len(y_vec) + 1)

        # Assuming y0_guess and alpha_guess are defined elsewhere
        y0_guess, alpha_guess = 10, 0.5
        try:
            # pylint: disable-next=unbalanced-tuple-unpacking
            popt, _ = curve_fit(
                lambda t, y0, alpha: y0 + t**alpha,
                x_vec,
                y_vec,
                p0=(y0_guess, alpha_guess),
            )
        except RuntimeError:
            # Handle case where curve fitting fails
            df.loc[df["combined_id"] == combined_id, "log_norm_wpm"] = np.nan
            continue

        y0, alpha = popt
        y_fitted = y0 + x_vec**alpha
        y_residuals = y_vec - y_fitted
        y_residuals_std = np.std(y_residuals)
        y_vec_fit_offset = y0
        y_vec_norm = (y_vec - y_vec_fit_offset) / y_residuals_std

        # Apply normalized values to the dataframe
        for trial_type_num, value in zip(group["trial_type_num"], y_vec_norm):
            df.loc[
                (df["combined_id"] == combined_id)
                & (df["trial_type_num"] == trial_type_num),
                "log_norm_wpm",
            ] = value

    return df
