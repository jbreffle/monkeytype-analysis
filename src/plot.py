"""Module for plotting monkeytype data"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import scipy
import numpy as np
from scipy.optimize import curve_fit

from src import util


def sim_n_mistakes(n_mistakes, ax=None, **hist_kwargs):
    if ax is None:
        ax = plt.gca()
    ax.hist(n_mistakes, **hist_kwargs)
    ax.set_xlabel("Number of Mistakes")
    ax.set_ylabel("Count (trial)")
    ax.set_title("Simulated Number of Mistakes")
    return ax


def sim_scatter_hist(wpm, acc, fig=None, **hist_kwargs):
    """..."""
    if fig is None:
        fig = plt.figure(figsize=(6, 4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax, **hist_kwargs)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax, **hist_kwargs)
    # Plot data
    ax.scatter(wpm, acc, s=2.5)
    ax_histx.hist(wpm, bins=20)
    ax_histy.hist(acc, bins=20, orientation="horizontal")
    # Plot linear regression
    wpm_acc_linregres = scipy.stats.linregress(wpm, acc)
    ax.plot(
        wpm,
        wpm_acc_linregres.intercept + wpm_acc_linregres.slope * wpm,
        "r",
        label="fitted line",
    )
    ax.text(
        0.05,
        0.95,
        # f"R={wpm_acc_linregres.rvalue:.2f}, p={wpm_acc_linregres.pvalue:.2f}",
        rf"$R^2={np.square(wpm_acc_linregres.rvalue):.2f},\  p={wpm_acc_linregres.pvalue:.2f}$",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=8,
        color="black",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    # Configure plot
    ax.set_xlabel("WPM")
    ax.set_ylabel("Accuracy")
    ax.set_title("Simulated WPM vs Accuracy")
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.set_ylabel("Count")
    ax_histy.set_xlabel("Count")
    return ax, ax_histx, ax_histy


def log_fit_scatter(
    df,
    y_label,
    plot_residuals=False,
    n_trial_types=None,
    min_trials=None,
    silent=False,
    legend_on=True,
    ax=None,
    **scatter_kwargs,
):
    """Scatter plot of two columns in a dataframe, with a log fit."""
    # Fit a log curve to all trial-types with a minimum number of trials
    y0_guess = 0.5
    alpha_guess = 1 / 0.3
    absolute_min_trials = 10
    if ax is None:
        ax = plt.gca()
    if n_trial_types is None and min_trials is None:
        min_trials = 100
    if min_trials is not None:
        trial_types = df["trial_type_id"].value_counts()
        trial_types = trial_types[trial_types > min_trials].index
    else:
        trial_types = range(1, n_trial_types + 1)
    # If any trial in trial_types has less than absolute_min_trials, remove it
    trial_types = [
        trial_type
        for trial_type in trial_types
        if len(df[df["trial_type_id"] == trial_type]) > absolute_min_trials
    ]
    for trial_type in trial_types:
        y_vec = df.loc[df["trial_type_id"] == trial_type][y_label].values
        x_vec = np.arange(1, len(y_vec) + 1)
        # pylint: disable=unbalanced-tuple-unpacking
        popt, _ = curve_fit(
            lambda t, y0, alpha: y0 + x_vec**alpha,
            x_vec,
            y_vec,
            p0=(y0_guess, alpha_guess),
        )
        y0 = popt[0]
        alpha = popt[1]
        y_fitted = y0 + x_vec**alpha
        if plot_residuals:
            ax.scatter(x_vec, y_vec - y_fitted, s=5, alpha=0.33)
            ax.axhline(0, color="black", lw=1)
        else:
            ax.plot(x_vec, y_fitted, label=trial_type, linewidth=2)
            ax.scatter(x_vec, y_vec, s=5, alpha=0.33)
        if not silent:
            print(f"Fit parameters: x_0={y0:.4f}, alpha={alpha:.4f}")
    ax.set_title(r"$y=x^\alpha+x_0$")
    if plot_residuals:
        ax.set_title("Residuals of log fit")
    ax.set_ylabel(util.get_label_string(y_label))
    ax.set_xlabel("Trial type completed")
    # Add legend if more than 1 trial type
    if len(trial_types) > 1 and legend_on:
        ax.legend()
    return ax


def performance_autocorrelation(
    df,
    trial_type_id=None,
    n_lags=None,
    ax=None,
    **plot_kwargs,
):
    if ax is None:
        ax = plt.gca()
    if trial_type_id is not None and trial_type_id != "All":
        df = df[df["trial_type_id"] == trial_type_id]
        if n_lags is None:
            n_lags = 100
        title_str = f"Performance autocorrelation, trial-type {trial_type_id}"
    else:
        if n_lags is None:
            n_lags = 500
        title_str = "Performance autocorrelation, all trials"
    # Set up data
    x_vals = np.arange(n_lags * -1, n_lags + 1)
    z_wpm_autocorr = np.correlate(df["z_wpm"], df["z_wpm"], mode="full")
    y_vals_wpm = z_wpm_autocorr[
        len(z_wpm_autocorr) // 2 - n_lags : len(z_wpm_autocorr) // 2 + n_lags + 1
    ]
    y_vals_wpm_norm = y_vals_wpm / y_vals_wpm.max()
    z_acc_autocorr = np.correlate(df["z_acc"], df["z_acc"], mode="full")
    y_vals_acc = z_acc_autocorr[
        len(z_acc_autocorr) // 2 - n_lags : len(z_acc_autocorr) // 2 + n_lags + 1
    ]
    y_vals_acc_norm = y_vals_acc / y_vals_acc.max()
    # Plot autocorrelation
    ax.plot(x_vals, y_vals_wpm_norm, **plot_kwargs)
    ax.plot(x_vals, y_vals_acc_norm, **plot_kwargs)
    ax.set_xlabel("Lag (trials)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title_str)
    ax.legend(["WPM (z-score)", "Accuracy (z-score)"])
    return ax


def histogram_by_type(
    df, feature_plot, feature_split, ax=None, alpha=0.5, min_trials=10, **hist_kwargs
):
    """Plot a histogram of a feature, split by another feature."""
    if ax is None:
        ax = plt.gca()
    # Filter out trials with too few data points
    feature_split_values = df[feature_split].unique()
    for split in feature_split_values:
        if len(df[df[feature_split] == split]) < min_trials:
            df = df[df[feature_split] != split]
    # Calculate histogram, using standard bins for all splits
    bins = np.histogram_bin_edges(df[feature_plot], bins="auto")
    hist_data = [
        df[df[feature_split] == split][feature_plot]
        for split in df[feature_split].unique()
    ]
    # Plot histogram
    ax.hist(
        hist_data,
        bins=bins,
        label=df[feature_split].unique(),
        alpha=alpha,
        histtype="stepfilled",
        **hist_kwargs,
    )
    ax.set_xlabel(util.get_label_string(feature_plot))
    ax.set_ylabel("Count (trials)")
    ax.set_title(
        f"{util.get_label_string(feature_plot)} by {util.get_label_string(feature_split)}"
    )
    ax.legend()
    return ax


def df_scatter(
    df,
    x_label,
    y_label,
    ax=None,
    trial_type_id=None,
    plot_regression=False,
    show_legend=False,
    n_colors=1,
    **scatter_kwargs,
):
    """Scatter plot of two columns in a dataframe."""
    # Filter by trial type
    if trial_type_id is not None and trial_type_id != "All":
        df = df[df["trial_type_id"] == trial_type_id].copy()
    else:
        df = df.copy()
    if ax is None:
        ax = plt.gca()
    # Color the scatter plot by trial_type_id
    # Only color the top n_colors-1 trial types, the rest are other_color
    if n_colors == 1:
        df.loc[:, "color"] = "tab:blue"
    else:
        other_color = "grey"
        colors = plt.cm.get_cmap("Set1", n_colors)
        df["color"] = df["trial_type_id"].apply(
            lambda x: colors(x - 1) if x < n_colors else other_color
        )
    # Plot scatter
    ax.scatter(df[x_label], df[y_label], c=df["color"], **scatter_kwargs)
    ax.set_xlabel(util.get_label_string(x_label))
    ax.set_ylabel(util.get_label_string(y_label))
    # Add legend if necessary
    if n_colors > 1 and show_legend:
        handles = []
        labels = [f"Type {i}" for i in range(1, n_colors + 1)] + ["Other"]
        for i, label in enumerate(labels, start=1):
            color = colors(i - 1) if i <= n_colors else other_color
            handle = mlines.Line2D(
                [],
                [],
                color=color,
                marker="o",
                linestyle="None",
                markersize=3,
                label=label,
            )
            handles.append(handle)
        ax.legend(handles=handles)
    # Feature-specific modifications
    if x_label == "time_of_day_sec":
        # TODO adjust for timezone
        gmt_to_est_adjustment = 0
        ticks = ax.get_xticks()  # Get the current x axis ticks
        ticks_hours = [int(tick / 3600) for tick in ticks]  # Convert the ticks to hours
        ticks_hours_est = [tick + gmt_to_est_adjustment for tick in ticks_hours]
        ax.set_xticks(ticks, ticks_hours_est)
        ax.set_xlabel("Time of day (hours, UTC)")
    if y_label == "time_of_day_sec":
        gmt_to_est_adjustment = 0
        ticks = ax.get_yticks()
        ticks_hours = [int(tick / 3600) for tick in ticks]
        ticks_hours_est = [tick + gmt_to_est_adjustment for tick in ticks_hours]
        ax.set_yticks(ticks, ticks_hours_est)
        ax.set_ylabel("Time of day (hours, UTC)")
    if x_label == "datetime":
        plt.gcf().autofmt_xdate()
    # If is_pb,
    if y_label == "is_pb":
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["False", "True"])
        ax.set_ylim([-0.1, 1.1])
    # Add linear regression line and statistics
    if plot_regression:
        if x_label == "datetime":
            regression_x_plot = pd.date_range(
                df[x_label].min(), df[x_label].max(), freq="min"
            )
            regression_x_fit = df["timestamp"]
            regression_x_evaluate = regression_x_plot.astype(int) / 10**6
        else:
            regression_x_plot = df[x_label]
            regression_x_fit = df[x_label]
            regression_x_evaluate = df[x_label]
        regression_y_fit = df[y_label]
        nan_mask = ~np.isnan(regression_x_fit) & ~np.isnan(regression_y_fit)
        x_y_linregres = scipy.stats.linregress(
            regression_x_fit[nan_mask], regression_y_fit[nan_mask]
        )
        regression_y = (
            x_y_linregres.slope * regression_x_evaluate + x_y_linregres.intercept
        )
        ax.plot(regression_x_plot, regression_y, color="red")
        # Add R^2 and p value annotation
        if x_y_linregres.pvalue < 0.0001:
            format_str = ".4e"
        else:
            format_str = ".4f"
        annotation_str = (
            rf"$R^2={x_y_linregres.rvalue:.4f}, p={x_y_linregres.pvalue:{format_str}}$"
        )
        ax.text(
            0.01,
            0.99,
            annotation_str,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )
        ymin, ymax = plt.ylim()
        y_range = ymax - ymin
        ax.set_ylim([ymin, ymax + (y_range * 0.05)])
        # Remove the color column that was created
        df.drop(columns="color", inplace=True)
    return ax
