"""Module for processing monkeytype data
"""

# Imports
import os
from pathlib import Path
import hashlib
import glob

import boto3
from dotenv import load_dotenv  # python-dotenv
import pandas as pd
import pyprojroot
import yaml
import pyarrow.parquet as pq

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


# Load environment variables
load_dotenv()

AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_RAW_DATA_FOLDER = os.path.join(
    "s3://", AWS_S3_BUCKET, config["RAW_DATA_FOLDER"].replace("./", "")
)


# AWS S3 client
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
s3 = session.resource("s3")
my_bucket = s3.Bucket("monkeytype-analysis")


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
    silent=SILENT_DEFAULT,
    force_aws_data=False,
    aws_raw_data_folder=AWS_RAW_DATA_FOLDER,
):
    """Merge all results files from /data/raw/ into one .csv file."""
    # 0) Decide if loading locally or from AWS
    # Are there files that match the pattern os.path.join(raw_data_folder, "results-*.csv")?
    local_raw_data_exists = (
        len(glob.glob(os.path.join(raw_data_folder, "results-*.csv"))) > 0
    )
    if local_raw_data_exists and not force_aws_data:
        # 1) Identify the desired file based on hash of the file names
        # Get all file_names in /data/raw/ that match the format 'results-*.csv'
        path = os.path.join(raw_data_folder, "results-*.csv")
        raw_file_paths = glob.glob(path)
    else:
        # Switch from local to AWS stored data
        raw_data_folder = os.path.normpath(aws_raw_data_folder)
        # Get all file_names in /data/raw/ that match the format 'results-*.csv'
        path = os.path.join(raw_data_folder, "results-*.csv")
        raw_files = [
            my_bucket_object.key
            for my_bucket_object in my_bucket.objects.filter(Prefix="data/raw/")
            if "results-" in my_bucket_object.key
        ]
        raw_file_paths = [f"s3://{AWS_S3_BUCKET}/{file}" for file in raw_files]
        if not silent:
            print(f"Loading raw data from {raw_data_folder} on AWS.")

    # Extract just the filenames, dropping the path and extensions in one line
    raw_file_names = [os.path.splitext(os.path.basename(f))[0] for f in raw_file_paths]
    raw_file_names = sorted(raw_file_names, key=lambda x: str(x))
    # Create an 8 digit hash of the file names
    hash_str = hashlib.md5("".join(raw_file_names).encode()).hexdigest()[:8]
    # Create a new file name for the combined results file
    # combined-results-<number of raw files>-<hash of raw file names>.csv
    combined_file_name = f"combined-results-{len(raw_file_paths)}-{hash_str}.csv"
    combined_file_path = os.path.join(processed_data_folder, combined_file_name)

    # 2) If combined_file_name exists then load it, otherwise create it
    if os.path.exists(combined_file_path):
        if not silent:
            print(
                f"File {combined_file_name} already exists in {processed_data_folder}."
            )
        return None
    else:
        # Load and combine all of the files into one dataframe
        if local_raw_data_exists and not force_aws_data:
            df = pd.concat(
                [pd.read_csv(f, sep="|") for f in raw_file_paths], ignore_index=True
            )
        else:  # Load from AWS
            df = pd.concat(
                [
                    pd.read_csv(
                        f,
                        storage_options={
                            "key": AWS_ACCESS_KEY_ID,
                            "secret": AWS_SECRET_ACCESS_KEY,
                        },
                        sep="|",
                    )
                    for f in raw_file_paths
                ],
                ignore_index=True,
            )
        df = df.drop_duplicates()  # Remove duplicate rows
        df.to_csv(combined_file_path, index=False, sep="|")
        # Also to parquett
        df.to_parquet(combined_file_path.replace(".csv", ".parquet"))
        if not silent:
            print(f"File {combined_file_name} created in {processed_data_folder}.")
    return None


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
