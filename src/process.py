"""Module for processing monkeytype data"""

# Imports
import glob
import hashlib
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pyprojroot
import requests
import yaml
from dotenv import load_dotenv

from src import util

# Config
with open(pyprojroot.here("config.yaml"), "r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

RAW_DATA_FOLDER = pyprojroot.here(config["RAW_DATA_FOLDER"])
PROCESSED_DATA_FOLDER = pyprojroot.here(config["PROCESSED_DATA_FOLDER"])
COMBINED_DATA_PATH_PATTERN = str(pyprojroot.here(config["COMBINED_DATA_PATH_PATTERN"]))
PROCESSED_DF_PATH_PATTERN = str(pyprojroot.here(config["PROCESSED_DF_PATH_PATTERN"]))

# Create default folders if they don't exists
Path(PROCESSED_DATA_FOLDER).mkdir(parents=True, exist_ok=True)
Path(RAW_DATA_FOLDER).mkdir(parents=True, exist_ok=True)

SILENT_DEFAULT = True
SESSION_THRESHOLD_DEFAULT = 600  # seconds between trials to define a new session
DEFAULT_TIMEOUT = 10  # seconds

# Load environment variables
load_dotenv()

# GitHub credentials and repo info (repo is private, so you'll need a token)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = "jbreffle"
GITHUB_REPO = "monkeytype-data"
GITHUB_BRANCH = "main"  # or whatever branch is relevant
# Base URL for raw files in your repo
RAW_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/data/raw"


def download_datafiles_from_github():
    """Download all .psv and .csv files from 'data/raw' in the GitHub repo into
    RAW_DATA_FOLDER.
    Returns a list of local file paths."""
    base_url = (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/data/raw"
    )
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    resp = requests.get(base_url, headers=headers)
    resp.raise_for_status()
    items = resp.json()

    downloaded_paths = []
    for item in items:
        if item["type"] == "file" and (
            item["name"].endswith(".csv") or item["name"].endswith(".psv")
        ):
            file_url = item["download_url"]
            file_resp = requests.get(file_url, headers=headers)
            file_resp.raise_for_status()

            local_path = os.path.join(RAW_DATA_FOLDER, item["name"])
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(file_resp.text)
            downloaded_paths.append(local_path)

    return downloaded_paths


# Functions
def parse_processesd_file_pattern(file_name_pattern=COMBINED_DATA_PATH_PATTERN):
    """Find the most recent combined-results file and return the number of raw files and
    the hash of the raw file names.

    Looks for files that match a pattern file_name_pattern, which is a string like
        "./<path>/combined-results-*.parquett"
    """
    # Incrememt through all files that match the pattern, then return i and hash
    # combined-results-i-hash.csv that are associated with maximum i value
    file_paths = glob.glob(file_name_pattern)
    num_raw_files = 0
    hash_str = ""
    for file_path in file_paths:
        # get basename without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_name_split = file_name.split("-")
        file_extension = os.path.splitext(file_path)[1][1:]
        if (
            len(file_name_split) == 4
            and file_name_split[0] == "combined"
            and file_extension == "parquet"
        ):
            num_raw_files_tmp = int(file_name_split[2])
            hash_str_tmp = file_name_split[3]
            if num_raw_files_tmp > num_raw_files:
                num_raw_files = num_raw_files_tmp
                hash_str = hash_str_tmp
    return num_raw_files, hash_str


def combine_raw_results(
    raw_data_folder=RAW_DATA_FOLDER,
    processed_data_folder=PROCESSED_DATA_FOLDER,
    silent=True,
    force_github_data=False,
):
    """Merge all .psv and .csv results files from data/raw/ into one combined file (.csv + .parquet)."""

    # 1) Check if local data already exists
    local_paths = glob.glob(os.path.join(raw_data_folder, "*.csv"))
    local_paths += glob.glob(os.path.join(raw_data_folder, "*.psv"))
    local_data_exists = len(local_paths) > 0

    # 2) Decide whether to use local or GitHub data
    if local_data_exists and not force_github_data:
        raw_file_paths = local_paths
    else:
        Path(raw_data_folder).mkdir(parents=True, exist_ok=True)
        raw_file_paths = download_datafiles_from_github()
        if not silent:
            print(f"Downloaded {len(raw_file_paths)} files from GitHub.")

    # 3) Compute a hash of the file names to create a unique combined filename
    file_names_no_ext = [
        os.path.splitext(os.path.basename(p))[0] for p in raw_file_paths
    ]
    file_names_no_ext = sorted(file_names_no_ext)
    hash_str = hashlib.md5("".join(file_names_no_ext).encode()).hexdigest()[:8]
    combined_name = f"combined-results-{len(raw_file_paths)}-{hash_str}.csv"
    combined_path = os.path.join(processed_data_folder, combined_name)

    # 4) If already combined, return early
    if os.path.exists(combined_path):
        if not silent:
            print(f"File {combined_name} already exists in {processed_data_folder}.")
        return

    # 5) Otherwise read each file with the proper delimiter and combine
    df_list = []
    for fpath in raw_file_paths:
        if fpath.endswith(".psv"):
            df_list.append(pd.read_csv(fpath, sep="|"))
        elif fpath.endswith(".csv"):
            df_list.append(pd.read_csv(fpath, sep=","))
    df = pd.concat(df_list, ignore_index=True).drop_duplicates()
    df["mode2"] = df["mode2"].astype(str)  #  errors when saving to parquet

    # 6) Write out combined data
    df.to_csv(combined_path, index=False)
    df.to_parquet(combined_path.replace(".csv", ".parquet"))
    if not silent:
        print(f"Created {combined_name} in {processed_data_folder}.")


def process_combined_results(
    df, exclude_older_than_2023=False, new_sesh_thresh=SESSION_THRESHOLD_DEFAULT
):
    """Process the combined results dataframe.

    WARNING: If this function gets modified then you must call
    df = process.load_processed_results(force_update=True) to replace the old processed
    results dataframe file (named "processed-results-<n_raw_files>-<hash>.parquet").
    """
    # Rename the default columns
    df = df.rename(
        columns={
            "_id": "id",
            "isPb": "is_pb",
            "rawWpm": "raw_wpm",
            "charStats": "char_stats",
            "quoteLength": "quote_length",
            "restartCount": "restart_count",
            "testDuration": "test_duration",
            "afkDuration": "afk_duration",
            "incompleteTestSeconds": "incomplete_test_seconds",
            "lazyMode": "lazy_mode",
            "blindMode": "blind_mode",
            "bailedOut": "bailed_out",
        }
    )
    # Invert the rows, so first row is oldest and last row is newest
    df = df.sort_values(by=["timestamp"]).reset_index(drop=True)
    # Round test_duration to integer
    df["test_duration"] = df["test_duration"].round().astype(int)
    # Convert is_pb from 1/None to 1/0
    df["is_pb"] = df["is_pb"].apply(lambda x: 1 if x == 1 else 0)
    # Add processed columns
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["day_of_week"] = pd.Categorical(
        df["datetime"].dt.day_name(),
        categories=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        ordered=True,
    )
    df["time_of_day_sec"] = (
        df["datetime"].dt.hour * 3600
        + df["datetime"].dt.minute * 60
        + df["datetime"].dt.second
    )
    # Time difference between trials in units of seconds
    df["time_diff_sec"] = df["timestamp"].diff() / 1000
    # Trial is start of new session if the time difference between trials
    # is > new_sesh_thresh
    df["new_sesh_ind"] = df["time_diff_sec"].apply(
        lambda x: 0 if x < new_sesh_thresh else 1
    )
    # Inter-trial interval: same as time_diff_sec, but the previous trials' test_duration
    df["iti_all"] = df["time_diff_sec"] - df["test_duration"]
    # iti_session: same as iti_all but make all cross-session intervals nan
    df["iti_session"] = df["iti_all"]
    df.loc[df["new_sesh_ind"] == 1, "iti_session"] = None
    # Remove data that is older than 2023
    if exclude_older_than_2023:
        df = df.drop(df[df["datetime"] < "2023-01-01"].index).reset_index(drop=True)
    # time_since_last_new_sesh: time since previous within-session trial
    df["time_since_last_new_sesh"] = 0
    for i in range(1, len(df) - 1):
        if df.loc[i, "new_sesh_ind"] == 0:
            df.loc[i, "time_since_last_new_sesh"] = (
                df.loc[i, "timestamp"]
                - df.loc[i - 1, "timestamp"]
                + df.loc[i - 1, "time_since_last_new_sesh"]
            )
        else:
            df.loc[i, "time_since_last_new_sesh"] = 0
    # Create group ID for z-scoring of same-settings trials
    df["combined_id"] = df.groupby(
        [
            "mode",
            "mode2",
            "punctuation",
            "numbers",
            "language",
            "funbox",
            "difficulty",
            "lazy_mode",
            "blind_mode",
        ],
        sort=False,
    ).ngroup()
    # df["trial_type_id"] is the same as df["combined_id"] but re-ordered by group size
    group_sizes = df["combined_id"].value_counts().sort_index()
    group_ranks = group_sizes.rank(ascending=False, method="first").astype(int)
    df["trial_type_id"] = df["combined_id"].map(group_ranks)
    # Increment through all the rows to calculate z-scored wpm and acc
    for index, row in df.iterrows():
        # Find all rows that match the current row's combined_id
        tmp = df.loc[df["combined_id"] == row["combined_id"]]
        # z-scored wpm
        class_wpm_mean = tmp["wpm"].mean()
        class_wpm_std = tmp["wpm"].std()
        z_wpm = (row["wpm"] - class_wpm_mean) / class_wpm_std
        df.at[index, "z_wpm"] = z_wpm
        # z-scored acc
        class_acc_mean = tmp["acc"].mean()
        class_acc_std = tmp["acc"].std()
        z_acc = (row["acc"] - class_acc_mean) / class_acc_std
        df.at[index, "z_acc"] = z_acc
    # For classes with only one trial, the std is nan, so fill with 0
    # Fill NaN values with 0 in z_wpm and z_acc
    df["z_wpm"] = df["z_wpm"].fillna(0)
    df["z_acc"] = df["z_acc"].fillna(0)
    # Add column for the ith trial (equal to the index + 1)
    df["trial_num"] = df.index + 1
    # Add column for the ith trial of the combined_id trial-type
    df["trial_type_num"] = df.groupby(["combined_id"]).cumcount() + 1
    # Add log fit normalized wpm, requires trial_type_num
    df = util.add_log_norm_wpm(df)
    return df


def load_processed_results(
    combined_results_path=None,
    exclude_older_than_2023=True,
    new_sesh_thresh=SESSION_THRESHOLD_DEFAULT,
    force_update=False,
):
    """Returns the processed dataframe.

    If combined_results_path is None, then the most recent combined-results file is
    loaded and processed. The processed dataframe is then saved to a parquett file.
    """
    # If combined_results_path is None, then load the most recent combined-results file
    if combined_results_path is None:
        # Get the most recent combined-results file parameters
        num_raw_files, hash_str = parse_processesd_file_pattern()
        # Get expected file paths for the combined results and processed results
        processesd_df_path = os.path.join(
            PROCESSED_DF_PATH_PATTERN.replace("*", f"{num_raw_files}-{hash_str}"),
        )
        combined_results_path = os.path.join(
            COMBINED_DATA_PATH_PATTERN.replace("*", f"{num_raw_files}-{hash_str}"),
        )
    # If processesd_df_path exists then load it
    # Otherwise, load the combined_results_path and process it
    if os.path.exists(processesd_df_path) and not force_update:
        processed_df = pq.read_table(processesd_df_path).to_pandas()
    elif force_update:
        combinde_df = pq.read_table(combined_results_path).to_pandas()
        processed_df = process_combined_results(
            combinde_df,
            exclude_older_than_2023=exclude_older_than_2023,
            new_sesh_thresh=new_sesh_thresh,
        )
        processed_df.to_parquet(processesd_df_path)
    else:
        combinde_df = pq.read_table(combined_results_path).to_pandas()
        processed_df = process_combined_results(
            combinde_df,
            exclude_older_than_2023=exclude_older_than_2023,
            new_sesh_thresh=new_sesh_thresh,
        )
        processed_df.to_parquet(processesd_df_path)
    return processed_df
