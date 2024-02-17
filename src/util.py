"""Small utility functions for the monkeytype project"""

# Imports
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Functions
def run_simulation_poisson(
    avg_wpm=60,
    avg_acc=0.95,
    duration=60,
    n_trials=1000,
    error_cost=0.75,
    dt=0.005,
    silent=False,
):
    """Simulate a typing test as a Poisson process.

    Across time steps, the probability of typing a correct or incorrect letter is
    modelled as a Poisson process.

    Parameters:
        avg_wpm (float): The average wpm.
        avg_acc (float): The average accuracy.
        duration (int): The duration of each test.
        n_trials (int): The number of tests.
        error_cost (float): The cost of making an error.
        dt (float): The time step.
        silent (bool): Whether to print the results.
    """
    # Dependent parameters
    n_timesteps = int(duration / dt)  # number of time steps
    word_length = 5  # Standardized word length
    error_cost_dt = round(error_cost / dt)  # cost of making an error (in time steps)
    cps_correct = (
        avg_wpm * word_length / 60
    )  # correct clicks per second, not including error_cost
    cps_incorrect = (
        avg_wpm * word_length / 60 * (1 - avg_acc)
    )  # incorrect clicks per second, not including error_cost
    cps_total = (cps_correct + cps_incorrect) / (
        1 - (cps_incorrect * error_cost)
    )  # Total clicks per second, including error_cost
    p_letter = cps_total * dt  # probability of typing a letter in a time step
    # Set up arrays
    t = np.arange(0, duration, dt)  # time array
    correct_typed = np.zeros(
        (n_trials, n_timesteps)
    )  # 1 for correct letter, 0 otherwise
    incorrect_typed = np.zeros(
        (n_trials, n_timesteps)
    )  # 1 for incorrect letter, 0 otherwise
    n_mistakes = np.zeros((n_trials))  # number of mistakes
    wpm = np.zeros((n_trials))  # Trial wpm
    acc = np.zeros((n_trials))  # trial acc
    # Loop over trials
    for ithTrial in range(n_trials):
        ithTimeStep = 0
        while ithTimeStep < n_timesteps:
            # Did we type a letter?
            if np.random.rand() < p_letter:
                # Did we type it correctly?
                if np.random.rand() < avg_acc:
                    correct_typed[ithTrial, ithTimeStep] = 1
                else:
                    incorrect_typed[ithTrial, ithTimeStep] = 1
                    ithTimeStep += error_cost_dt  # Jump ahead in time
            ithTimeStep += 1  # Increment time step

        # Calculate words per minute for the trial
        wpm[ithTrial] = (
            np.sum(correct_typed[ithTrial, :]) / word_length / duration
        ) * 60

        # Calculate accuracy for the trial
        acc[ithTrial] = np.sum(correct_typed[ithTrial, :]) / (
            np.sum(correct_typed[ithTrial, :]) + np.sum(incorrect_typed[ithTrial, :])
        )
        n_mistakes[ithTrial] = np.sum(incorrect_typed[ithTrial, :])
    # Print the results
    if not silent:
        print("Average WPM: " + str(np.mean(wpm)))
        print("Average Accuracy: " + str(np.mean(acc)))
    return wpm, acc, n_mistakes


def run_simulation_simple(
    avg_wpm=60,
    avg_acc=0.95,
    duration=60,
    n_trials=10000,
    error_mean=0.5,
    error_std=0.45,
    silent=False,
    use_lognormal=True,
):
    """Simulate a typing test with random mistakes.

    For each trial, the number of mistakes is drawn from a Poisson distribution,
    and the duration of each mistake is drawn from a normal distribution.

    Parameters:
        avg_wpm (float): The average wpm.
        avg_acc (float): The average accuracy.
        duration (int): The duration of each test.
        n_trials (int): The number of tests.
        error_mean (float): Mean duration of each mistake.
        error_std (float): STD of the duration of each mistake.
        silent (bool): Whether to print the results.

    Returns:
        wpm (np.array): The wpm for each test.
        acc (np.array): The accuracy for each test.
        n_mistakes (np.array): The number of mistakes for each test.
    """
    # Dependent parameters
    word_length = 5  # Standardized word length
    expected_total_mistakes = (
        avg_wpm / 60 * word_length * (1 - avg_acc) * duration
    )  # Expected number of mistakes
    expected_total_mistake_duration = expected_total_mistakes * error_mean
    effective_duration = duration - expected_total_mistake_duration
    avg_cps = (
        avg_wpm * word_length / 60 * (duration / effective_duration)
    )  # average correct clicks per second
    # Mistakes occur as a poisson process
    mistake_lambda = avg_cps * (1 - avg_acc) * effective_duration
    # Set up
    wpm = np.zeros(n_trials)
    acc = np.zeros(n_trials)
    n_mistakes = np.random.poisson(mistake_lambda, n_trials)
    # Simulate each trial
    for i in range(n_trials):
        # Number of mistakes in this trial
        n = np.random.poisson(mistake_lambda)
        # Durations to fix each mistake
        if use_lognormal:
            mu = np.log(error_mean**2 / np.sqrt(error_mean**2 + error_std**2))
            sigma = np.sqrt(np.log(1 + error_std**2 / error_mean**2))
            all_mistake_durations = np.random.lognormal(mean=mu, sigma=sigma, size=n)
        else:
            all_mistake_durations = np.random.normal(error_mean, error_std, n)

        # Total time spent fixing mistakes
        total_correction_time = np.sum(all_mistake_durations)

        # Total characters typed correctly (assumes all corrected characters are eventually correct)
        total_chars_correct = avg_cps * (duration - total_correction_time)

        # Adjust WPM based on effective duration
        wpm[i] = (total_chars_correct / word_length) / (duration / 60)

        # Calculate accuracy as ratio of correct characters to total attempted characters (including mistakes)
        total_chars_attempted = total_chars_correct + n
        acc[i] = total_chars_correct / total_chars_attempted
    # Print the results
    if not silent:
        print("Average WPM: " + str(np.mean(wpm)))
        print("Average Accuracy: " + str(np.mean(acc)))
    return wpm, acc, n_mistakes


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
    # if not user_processed_df.dtypes.equals(data_df.dtypes):
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
        case "trial_num":
            return "Trial number"
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
