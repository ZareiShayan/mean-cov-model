from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from scipy.stats import wilcoxon
from sklearn.decomposition import PCA

def save_figure(fig, file_name, file_path, ext=".png", transparent=False, **savefig_kwargs):
    file_path = Path(file_path)
    file_path.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        file_path / f"{file_name}{ext}",
        dpi=100,
        bbox_inches="tight",
        pad_inches=0.25,
        transparent=transparent,
        **savefig_kwargs,
    )

    plt.close(fig)

    return file_path


def save_shap_plot(make_plot, title, file_name, file_path, figsize=(10, 6)):
    plt.close("all")

    plot_result = make_plot()

    if hasattr(plot_result, "figure"):
        fig = plot_result.figure
        ax = plot_result
    else:
        fig = plt.gcf()
        ax = plt.gca()

    ax.set_title(title, pad=20)
    fig.canvas.draw()

    return save_figure(fig, file_name, file_path)


def plot_unit_spikes(
    spikes,
    unit_idx,
    title,
    file_name,
    file_path,
    bin_times,
    bin_size,
    figsize=(4, 6),
    show=False,
):
    unit_spikes = spikes[:, unit_idx, :] / bin_size

    mean_by_bin = np.nanmean(unit_spikes, axis=0)
    sem_by_bin = np.nanstd(unit_spikes, axis=0) / np.sqrt(unit_spikes.shape[0])
    ci_low = mean_by_bin - 1.96 * sem_by_bin
    ci_high = mean_by_bin + 1.96 * sem_by_bin

    xticks = [
        bin_times[0],
        0,
        bin_times[-1],
    ]

    fig = plt.figure(figsize=figsize, layout="constrained")

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[4, 1],
        hspace=0.05,
    )

    ax_heatmap = fig.add_subplot(gs[0])

    ax_psth = fig.add_subplot(
        gs[1],
        sharex=ax_heatmap,
    )

    im = ax_heatmap.imshow(
        unit_spikes,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap='Greys',
        extent=[
            bin_times[0],
            bin_times[-1],
            0,
            unit_spikes.shape[0],
        ],
    )

    ax_heatmap.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax_heatmap.set_ylabel(
        "Trial",
    )

    ax_heatmap.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    ax_heatmap.spines[["top", "right"]].set_visible(False)

    ax_psth.fill_between(
        bin_times,
        ci_low,
        ci_high,
        color="black",
        alpha=0.25,
    )

    ax_psth.plot(
        bin_times,
        mean_by_bin,
        color="black",
        linewidth=2,
    )

    ax_psth.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax_psth.set(
        xlabel="Time from press onset (s)",
        ylabel="Mean\nspikes per s",
        xticks=xticks,
        xticklabels=[
            f"{bin_times[0]:g}",
            "0",
            f"{bin_times[-1]:g}",
        ],
    )

    ax_psth.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax_heatmap,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        "Spikes per s",
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


def plot_spike_summary(
    spikes,
    file_name,
    file_path,
    title=None,
    metric="mean",
    k=4.0,
    unit_threshold=None,
    trial_threshold=None,
    unit_names=None,
    trial_names=None,
    cmap="viridis",
    figsize=(8, 5),
    show=False,
):
    to_numpy = lambda x: (
        x.detach().cpu().numpy()
        if torch.is_tensor(x)
        else np.asarray(x)
    )

    metric_functions = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin,
        "std": np.nanstd,
        "var": np.nanvar,
        "sum": np.nansum,
    }

    if metric not in [*metric_functions.keys(), "outlier_rate"]:
        raise ValueError(
            "metric must be one of: "
            f"{[*metric_functions.keys(), 'outlier_rate']}."
        )

    def parse_threshold(threshold):
        if threshold is None:
            return None, None

        if np.isscalar(threshold):
            return None, float(threshold)

        threshold = tuple(threshold)

        if len(threshold) != 2:
            raise ValueError(
                "threshold must be None, a scalar, or (low, high)."
            )

        low, high = threshold

        low = None if low is None else float(low)
        high = None if high is None else float(high)

        return low, high

    spikes = to_numpy(spikes).astype(float)

    if spikes.ndim != 3:
        raise ValueError(
            "spikes must have shape (n_trials, n_units, n_bins). "
            f"Received shape {spikes.shape}."
        )

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    n_trials, n_units, n_bins = spikes.shape

    if unit_names is None:
        unit_names = np.arange(n_units)

    if trial_names is None:
        trial_names = np.arange(n_trials)

    unit_names = np.asarray(unit_names)
    trial_names = np.asarray(trial_names)

    if unit_names.size != n_units:
        raise ValueError("unit_names must contain one name per unit.")

    if trial_names.size != n_trials:
        raise ValueError("trial_names must contain one name per trial.")

    if metric == "outlier_rate":
        median = np.nanmedian(
            spikes,
            axis=0,
            keepdims=True,
        )

        mad = np.nanmedian(
            np.abs(spikes - median),
            axis=0,
            keepdims=True,
        )

        robust_scale = np.maximum(
            1.4826 * mad,
            1.0,
        )

        flagged = spikes > median + k * robust_scale

        summary = np.nanmean(
            flagged,
            axis=2,
        )

        unit_score = np.nanmean(
            flagged,
            axis=(0, 2),
        )

        trial_score = np.nanmean(
            flagged,
            axis=(1, 2),
        )

        metric_name = f"Outlier rate"
        colorbar_label = "Proportion of flagged bins"

    else:
        metric_function = metric_functions[metric]

        summary = metric_function(
            spikes,
            axis=2,
        )

        unit_score = metric_function(
            spikes,
            axis=(0, 2),
        )

        trial_score = metric_function(
            spikes,
            axis=(1, 2),
        )

        metric_name = metric.title()
        colorbar_label = f"{metric.title()} spike value"

    unit_low, unit_high = parse_threshold(
        unit_threshold,
    )

    trial_low, trial_high = parse_threshold(
        trial_threshold,
    )

    bad_units = np.zeros(
        n_units,
        dtype=bool,
    )

    bad_trials = np.zeros(
        n_trials,
        dtype=bool,
    )

    if unit_low is not None:
        bad_units |= unit_score < unit_low

    if unit_high is not None:
        bad_units |= unit_score > unit_high

    if trial_low is not None:
        bad_trials |= trial_score < trial_low

    if trial_high is not None:
        bad_trials |= trial_score > trial_high

    units_to_remove = np.where(
        bad_units,
    )[0]

    trials_to_remove = np.where(
        bad_trials,
    )[0]

    unit_tick_idx = np.linspace(
        0,
        n_units - 1,
        min(10, n_units),
        dtype=int,
    )

    trial_tick_idx = np.linspace(
        0,
        n_trials - 1,
        min(10, n_trials),
        dtype=int,
    )

    fig = plt.figure(
        figsize=figsize,
        layout="constrained",
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.25, 1],
        height_ratios=[1, 0.25],
        wspace=0.03,
        hspace=0.03,
    )

    ax_main = fig.add_subplot(gs[0, 1])

    ax_left = fig.add_subplot(
        gs[0, 0],
        sharey=ax_main,
    )

    ax_bottom = fig.add_subplot(
        gs[1, 1],
        sharex=ax_main,
    )

    ax_corner = fig.add_subplot(gs[1, 0])
    ax_corner.axis("off")

    im = ax_main.imshow(
        summary.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )

    ax_main.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    ax_main.tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )

    ax_main.spines[["top", "right"]].set_visible(False)

    unit_idx = np.arange(n_units)
    trial_idx = np.arange(n_trials)

    ax_left.scatter(
        unit_score,
        unit_idx,
        color="C0",
        s=8,
        alpha=0.7,
    )

    if unit_low is not None:
        ax_left.axvline(
            unit_low,
            color="tab:red",
            linewidth=1,
        )

    if unit_high is not None:
        ax_left.axvline(
            unit_high,
            color="tab:red",
            linewidth=1,
        )

    ax_left.scatter(
        unit_score[bad_units],
        unit_idx[bad_units],
        color="tab:red",
        s=18,
        zorder=3,
    )

    ax_left.set(
        xlabel=metric_name,
        ylabel="Unit",
        yticks=unit_tick_idx,
        yticklabels=unit_names[unit_tick_idx],
    )

    ax_left.tick_params(
        axis="x",
        labelsize=7,
    )

    ax_left.tick_params(
        axis="y",
        labelsize=7,
    )

    ax_left.spines[["top", "right"]].set_visible(False)

    ax_bottom.scatter(
        trial_idx,
        trial_score,
        color="C0",
        s=8,
        alpha=0.7,
    )

    if trial_low is not None:
        ax_bottom.axhline(
            trial_low,
            color="tab:red",
            linewidth=1,
        )

    if trial_high is not None:
        ax_bottom.axhline(
            trial_high,
            color="tab:red",
            linewidth=1,
        )

    ax_bottom.scatter(
        trial_idx[bad_trials],
        trial_score[bad_trials],
        color="tab:red",
        s=18,
        zorder=3,
    )

    ax_bottom.set(
        xlabel="Trial",
        ylabel=metric_name,
        xticks=trial_tick_idx,
        xticklabels=trial_names[trial_tick_idx],
    )

    ax_bottom.tick_params(
        axis="x",
        labelrotation=90,
        labelsize=7,
    )

    ax_bottom.tick_params(
        axis="y",
        labelsize=7,
    )

    ax_bottom.spines[["top", "right"]].set_visible(False)

    for unit_idx in units_to_remove:
        ax_main.axhline(
            unit_idx - 0.5,
            color="tab:red",
            linewidth=0.6,
        )

        ax_main.axhline(
            unit_idx + 0.5,
            color="tab:red",
            linewidth=0.6,
        )

    for trial_idx in trials_to_remove:
        ax_main.axvline(
            trial_idx - 0.5,
            color="tab:red",
            linewidth=0.6,
        )

        ax_main.axvline(
            trial_idx + 0.5,
            color="tab:red",
            linewidth=0.6,
        )

    cbar = fig.colorbar(
        im,
        ax=ax_main,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        colorbar_label,
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    fig.suptitle(title)

    if show:
        plt.show()

    save_figure(
        fig,
        file_name,
        file_path,
    )

    return trials_to_remove, units_to_remove


def plot_training_history(
    trainer,
    title,
    file_name,
    file_path,
    show=False,
):

    h = trainer.history
    train = np.asarray(h.train_loss_epoch)
    valid = np.asarray(h.valid_loss_epoch)
    lr = np.asarray(h.lr)
    epochs = np.arange(1, len(valid) + 1)

    trend = lambda x: np.diff(x) / np.maximum(np.abs(x[:-1]), 1e-8)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")

    ax[0].plot(epochs, valid, color="tab:orange", label="Validation")
    ax[0].plot(epochs, train, color="tab:blue", label="Train")
    ax[0].set(title="Loss", xlabel="Epoch", ylabel="NLL loss")
    ax[0].legend(frameon=False)

    ax[1].axhline(0, color="black", ls="--", lw=1)
    ax[1].plot(epochs[1:], trend(valid), color="tab:orange", label="Validation")
    ax[1].plot(epochs[1:], trend(train), color="tab:blue", label="Train")
    ax[1].set(title="Loss trend", xlabel="Epoch", ylabel="Relative change")
    ax[1].legend(frameon=False)

    ax[2].plot(epochs, lr, color="tab:green")
    ax[2].set(title="Learning rate", xlabel="Epoch", ylabel="LR")
    ax[2].set_yscale("log")

    fig.text(
        -0.05,
        0.5,
        title,
        rotation="vertical",
        va="center",
        ha="left",
        fontsize=12,
    )

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def get_cov_loading_matrices(cov_model):
    if hasattr(cov_model, "lambda_matrix"):
        return {
            "shared": cov_model.lambda_matrix
        }

    if hasattr(cov_model, "component_loadings"):
        return {
            component_name: cov_model.full_loading_matrix(component_name)
            for component_name in cov_model.component_loadings
        }

    raise TypeError(
        f"Unsupported covariance model: {type(cov_model).__name__}."
    )

def get_cov_length_scales(cov_model):
    if hasattr(cov_model, "length_scales"):
        return {
            "shared": F.softplus(cov_model.length_scales)
        }

    if hasattr(cov_model, "component_length_scales"):
        return {
            component_name: F.softplus(length_scales)
            for component_name, length_scales
            in cov_model.component_length_scales.items()
        }

    raise TypeError(
        f"Unsupported covariance model: {type(cov_model).__name__}."
    )

def get_covariance_matrix(cov_model):
    if hasattr(cov_model, "covariance_matrix"):
        covariance = cov_model.covariance_matrix()

    elif hasattr(cov_model, "build_component_covariance"):
        covariance = cov_model.build_component_covariance(
            cov_model.lambda_matrix,
            cov_model.length_scales,
        )

    elif hasattr(cov_model, "build_covariance_matrix"):
        lambda_matrix = cov_model.lambda_matrix.unsqueeze(0)

        cov_tensor = cov_model.build_covariance_matrix(
            lambda_matrix,
        )

        covariance = (
            cov_tensor @ cov_tensor.transpose(-1, -2)
        ).squeeze(0)

        return covariance

    else:
        raise TypeError(
            f"Unsupported covariance model: "
            f"{type(cov_model).__name__}."
        )

    covariance = covariance + cov_model.I

    noise_diag = torch.sigmoid(cov_model.noise.flatten())
    covariance.diagonal().add_(noise_diag)

    return covariance

def plot_cov_loading_matrix(
    mean_cov_lit_model,
    title,
    file_name,
    file_path,
    bin_times=None,
    vmax=None,
    show=False,
):
    cov_model = mean_cov_lit_model.full_model.cov_model
    loading_matrices = get_cov_loading_matrices(cov_model)

    if bin_times is None:
        bin_times = np.arange(cov_model.n_bins)

    bin_times = np.asarray(bin_times)

    loading_matrix = np.concatenate(
        [
            lambda_matrix.detach().cpu().numpy().transpose(2, 0, 1).reshape(
                lambda_matrix.size(-1),
                -1,
            )
            for lambda_matrix in loading_matrices.values()
        ],
        axis=0,
    )

    component_names = [
        component_name
        for component_name, lambda_matrix in loading_matrices.items()
        for _ in range(lambda_matrix.size(-1))
    ]

    component_boundaries = np.cumsum(
        [
            lambda_matrix.size(-1)
            for lambda_matrix in loading_matrices.values()
        ]
    )[:-1]

    if vmax is None:
        vmax = np.nanmax(np.abs(loading_matrix))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    vmin = -vmax

    fig, ax = plt.subplots(
        figsize=(
            max(4, cov_model.n_bins * 0.4),
            max(2, loading_matrix.shape[0] * 0.175),
        ),
        layout="constrained",
    )

    im = ax.imshow(
        loading_matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    for bin_idx in range(1, cov_model.n_bins):
        ax.axvline(
            bin_idx * cov_model.n_units - 0.5,
            color="black",
            linewidth=0.7,
            alpha=0.7,
        )

    for boundary in component_boundaries:
        ax.axhline(
            boundary - 0.5,
            color="black",
            linewidth=1.0,
        )

    bin_centers = (
        np.arange(cov_model.n_bins) * cov_model.n_units
        + (cov_model.n_units - 1) / 2
    )

    ax.set(
        xlabel="Time bin",
        ylabel="Latent",
        xticks=bin_centers,
        xticklabels=np.round(bin_times, 2),
        yticks=np.arange(loading_matrix.shape[0]),
        yticklabels=component_names,
    )

    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.015,
        pad=0.01,
    )
    cbar.set_label(r"$\Lambda$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def plot_cov_noise(
    mean_cov_lit_model,
    title,
    file_name,
    file_path,
    bin_times=None,
    unit_names=None,
    vmax=None,
    show=False,
):
    noise = mean_cov_lit_model.full_model.cov_model.noise
    noise = torch.sigmoid(noise).detach().cpu().numpy()

    n_bins, n_units = noise.shape

    if bin_times is None:
        bin_times = np.arange(n_bins)

    if unit_names is None:
        unit_names = np.arange(n_units)

    if vmax is None:
        vmax = np.max(noise)

    xticks = [
        0,
        np.where(bin_times == 0)[0][0],
        n_bins - 1,
    ]

    fig, ax = plt.subplots(
        figsize=(5, 4),
        layout="constrained",
    )

    im = ax.imshow(
        noise.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="Reds",
        vmin=0,
        vmax=vmax,
    )

    ax.axvline(
        np.where(bin_times == 0)[0][0],
        color="black",
        linestyle="--",
        linewidth=1,
    )

    max_yticks = 12

    ytick_idx = np.unique(
        np.linspace(
            0,
            n_units - 1,
            min(max_yticks, n_units),
            dtype=int,
        )
    )

    ax.set(
        xlabel="Time",
        ylabel="Unit",
        xticks=xticks,
        xticklabels=[
            f"{bin_times[0]:g}",
            "0",
            f"{bin_times[-1]:g}",
        ],
        yticks=ytick_idx,
        yticklabels=np.asarray(unit_names)[ytick_idx],
    )

    ax.tick_params(
        axis="x",
        labelsize=7,
    )

    ax.tick_params(
        axis="y",
        labelsize=7,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        "Noise variance",
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


def plot_cov_length_scales(
    mean_cov_lit_model,
    title,
    file_name,
    file_path,
    ymax=None,
    show=False,
):
    cov_model = mean_cov_lit_model.full_model.cov_model
    length_scales_dict = get_cov_length_scales(cov_model)

    length_scales = np.concatenate(
        [
            length_scale.detach().cpu().numpy()
            for length_scale in length_scales_dict.values()
        ]
    )

    component_names = [
        component_name
        for component_name, length_scale in length_scales_dict.items()
        for _ in range(length_scale.numel())
    ]

    component_boundaries = np.cumsum(
        [
            length_scale.numel()
            for length_scale in length_scales_dict.values()
        ]
    )[:-1]

    if ymax is None:
        ymax = np.max(length_scales)

    fig, ax = plt.subplots(
        figsize=(
            max(4, len(length_scales) * 0.225),
            3.5,
        ),
        layout="constrained",
    )

    latent_idx = np.arange(len(length_scales))

    ax.bar(
        latent_idx,
        length_scales,
        color="C0",
    )

    for boundary in component_boundaries:
        ax.axvline(
            boundary - 0.5,
            color="black",
            linewidth=1.0,
        )

    ax.set(
        xlabel="Latent component",
        ylabel="Length scale",
        xticks=latent_idx,
        xticklabels=component_names,
        ylim=(0, 1.05 * ymax),
    )

    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def plot_covariance_matrix(
    mean_cov_lit_model,
    title,
    file_name,
    file_path,
    bin_times=None,
    time_indices=None,
    vmax=None,
    linthresh=None,
    show=False,
):
    cov_model = mean_cov_lit_model.full_model.cov_model

    covariance_matrix = get_covariance_matrix(cov_model)
    covariance_matrix = covariance_matrix.detach().cpu().numpy()

    n_bins = cov_model.n_bins
    n_units = cov_model.n_units

    if bin_times is None:
        bin_times = np.arange(n_bins)

    if time_indices is None:
        time_indices = np.arange(n_bins)
    else:
        time_indices = np.atleast_1d(time_indices)

    feature_indices = np.concatenate(
        [
            np.arange(
                time_idx * n_units,
                (time_idx + 1) * n_units,
            )
            for time_idx in time_indices
        ]
    )

    covariance_matrix = covariance_matrix[
        np.ix_(feature_indices, feature_indices)
    ]

    bin_times = np.asarray(bin_times)[time_indices]
    n_bins = len(time_indices)

    if vmax is None:
        vmax = np.nanmax(np.abs(covariance_matrix))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    vmin = -vmax

    if linthresh is None:
        linthresh = max(vmax * 0.01, 1e-8)

    norm = mcolors.SymLogNorm(
        linthresh=linthresh,
        vmin=vmin,
        vmax=vmax,
        base=10,
    )

    fig, ax = plt.subplots(
        figsize=(
            max(5, n_bins * 0.25),
            max(5, n_bins * 0.25),
        ),
        layout="constrained",
    )

    im = ax.imshow(
        covariance_matrix,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=norm,
    )

    for bin_idx in range(1, n_bins):
        line_pos = bin_idx * n_units - 0.5

        ax.axvline(
            line_pos,
            color="black",
            linewidth=0.5,
            alpha=0.5,
        )

        ax.axhline(
            line_pos,
            color="black",
            linewidth=0.5,
            alpha=0.5,
        )

    bin_centers = (
        np.arange(n_bins) * n_units
        + (n_units - 1) / 2
    )

    tick_labels = np.round(bin_times, 2)

    ax.set(
        xticks=bin_centers,
        xticklabels=tick_labels,
        yticks=bin_centers,
        yticklabels=tick_labels,
    )

    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label(
        "Covariance (SymLog Scale)",
        fontsize=8,
    )
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def plot_unit_covariance(
    mean_cov_lit_model,
    unit_idx,
    title,
    file_name,
    file_path,
    bin_times=None,
    unit_names=None,
    cmap="RdBu_r",
    vmax=None,
    ymax=None,
    figsize=(5, 3),
    show=False,
):
    cov_model = mean_cov_lit_model.full_model.cov_model
    covariance = get_covariance_matrix(cov_model)
    covariance = covariance.detach().cpu().numpy()

    n_bins = cov_model.n_bins
    n_units = cov_model.n_units

    if bin_times is None:
        bin_times = np.arange(n_bins)

    if unit_names is None:
        unit_names = np.arange(n_units)

    bin_times = np.asarray(bin_times)
    unit_names = np.asarray(unit_names)

    if unit_idx < 0 or unit_idx >= n_units:
        raise ValueError(
            f"unit_idx must be in [0, {n_units - 1}], got {unit_idx}."
        )

    if bin_times.size != n_bins:
        raise ValueError("bin_times must contain one value per bin.")

    if unit_names.size != n_units:
        raise ValueError("unit_names must contain one name per unit.")

    other_unit_idx = np.delete(np.arange(n_units), unit_idx)

    unit_cov = np.empty((other_unit_idx.size, n_bins), dtype=float)

    for bin_idx in range(n_bins):
        feature_idx = bin_idx * n_units + unit_idx
        block_start = bin_idx * n_units
        block_end = (bin_idx + 1) * n_units

        diag_block_values = covariance[
            feature_idx,
            block_start:block_end,
        ]
        unit_cov[:, bin_idx] = diag_block_values[other_unit_idx]

    mean_by_bin = np.nanmean(unit_cov, axis=0)

    if vmax is None:
        vmax = np.nanmax(np.abs(unit_cov))

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    vmin = -vmax

    if ymax is None:
        ymax = np.nanmax(np.abs(mean_by_bin))

    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0

    xticks = [
        bin_times[0],
        0,
        bin_times[-1],
    ]

    fig = plt.figure(
        figsize=figsize,
        layout="constrained",
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[4, 1],
        hspace=0.05,
    )

    ax_heatmap = fig.add_subplot(gs[0])
    ax_mean = fig.add_subplot(
        gs[1],
        sharex=ax_heatmap,
    )

    im = ax_heatmap.imshow(
        unit_cov,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            bin_times[0],
            bin_times[-1],
            0,
            unit_cov.shape[0],
        ],
    )

    ax_heatmap.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    max_yticks = 12

    ytick_pos = np.unique(
        np.linspace(
            0,
            unit_cov.shape[0] - 1,
            min(max_yticks, unit_cov.shape[0]),
            dtype=int,
        )
    )

    ax_heatmap.set(
        ylabel="Other unit",
        yticks=ytick_pos + 0.5,
        yticklabels=unit_names[other_unit_idx][ytick_pos],
    )

    ax_heatmap.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    ax_heatmap.tick_params(
        axis="y",
        labelsize=7,
    )

    ax_heatmap.spines[["top", "right"]].set_visible(False)

    ax_mean.plot(
        bin_times,
        mean_by_bin,
        color="tab:blue",
        linewidth=2,
    )

    ax_mean.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1,
    )

    ax_mean.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax_mean.set(
        xlabel="Time from press onset (s)",
        ylabel="Mean\ncovariance",
        xticks=xticks,
        xticklabels=[
            f"{bin_times[0]:g}",
            "0",
            f"{bin_times[-1]:g}",
        ],
        ylim=(-ymax, ymax),
    )

    ax_mean.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(
        im,
        ax=ax_heatmap,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label(
        "Covariance",
        fontsize=8,
    )

    cbar.ax.tick_params(
        labelsize=7,
    )

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(
        fig,
        file_name,
        file_path,
    )


def plot_prediction(
    Y,
    Y_hat,
    trial_idx,
    unit_idx,
    bin_times,
    title,
    filename,
    filepath,
    figsize=(10, 2),
    show=False,
):
    to_numpy = lambda x: (
        x.detach().cpu().numpy()
        if torch.is_tensor(x)
        else np.asarray(x)
    )

    y = to_numpy(Y[trial_idx])[:, unit_idx]
    y_hat = to_numpy(Y_hat[trial_idx])[:, unit_idx]
    bin_times = np.asarray(bin_times)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    ax.plot(
        bin_times,
        y,
        color="tab:green",
        linewidth=2,
        label="Ground truth",
    )
    ax.plot(
        bin_times,
        y_hat,
        color="tab:purple",
        linewidth=2,
        label="Model prediction",
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Time from press onset (s)",
        ylabel="Neural activity",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, filename, filepath)


def plot_train_valid_metrics_comparison(
    correlations_train,
    correlations_valid,
    r2s_train,
    r2s_valid,
    mses_train,
    mses_valid,
    title,
    filename,
    filepath,
    figsize=(7, 4),
    show=False,
):
    metrics = [
        (
            "Correlation",
            np.asarray(correlations_train, dtype=float),
            np.asarray(correlations_valid, dtype=float),
            "tab:brown",
        ),
        (
            r"$R^2$",
            np.asarray(r2s_train, dtype=float),
            np.asarray(r2s_valid, dtype=float),
            "tab:brown",
        ),
        (
            "MSE",
            np.asarray(mses_train, dtype=float),
            np.asarray(mses_valid, dtype=float),
            "tab:brown",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")

    for idx, (metric_name, train, valid, color) in enumerate(metrics):
        finite = np.isfinite(train) & np.isfinite(valid)
        train = train[finite]
        valid = valid[finite]

        ax = axes[idx]

        box = ax.boxplot(
            [train, valid],
            labels=["Train", "Validation"],
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black", linewidth=1.0),
            capprops=dict(color="black", linewidth=1.0),
            boxprops=dict(linewidth=1.0, color="black"),
            flierprops=dict(
                marker="o",
                markersize=4,
                markerfacecolor=color,
                markeredgecolor="none",
                alpha=0.35,
            ),
        )

        box["boxes"][0].set_facecolor("tab:blue")
        box["boxes"][0].set_alpha(0.45)
        box["boxes"][1].set_facecolor("tab:orange")
        box["boxes"][1].set_alpha(0.45)

        ax.set_title(metric_name, x=0.15, y=0.98, ha="left")
        ax.set_ylabel(metric_name)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, filename, filepath)


def plot_model_metrics_comparison(
    correlations_x_train,
    correlations_x_valid,
    r2s_x_train,
    r2s_x_valid,
    mses_x_train,
    mses_x_valid,
    correlations_y_train,
    correlations_y_valid,
    r2s_y_train,
    r2s_y_valid,
    mses_y_train,
    mses_y_valid,
    x_label,
    y_label,
    file_name,
    file_path,
    title=None,
    bins=100,
    figsize=(14, 7),
    show=False,
):
    metrics = [
        (
            "Correlation",
            correlations_x_train,
            correlations_x_valid,
            correlations_y_train,
            correlations_y_valid,
        ),
        (
            r"$R^2$",
            r2s_x_train,
            r2s_x_valid,
            r2s_y_train,
            r2s_y_valid,
        ),
        (
            "MSE",
            mses_x_train,
            mses_x_valid,
            mses_y_train,
            mses_y_valid,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")

    for idx, (metric_name, x_train, x_valid, y_train, y_valid) in enumerate(metrics):
        x_train = np.asarray(x_train, dtype=float)
        x_valid = np.asarray(x_valid, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        y_valid = np.asarray(y_valid, dtype=float)

        finite_train = np.isfinite(x_train) & np.isfinite(y_train)
        finite_valid = np.isfinite(x_valid) & np.isfinite(y_valid)

        x_train, y_train = x_train[finite_train], y_train[finite_train]
        x_valid, y_valid = x_valid[finite_valid], y_valid[finite_valid]

        lower = min(x_train.min(), x_valid.min(), y_train.min(), y_valid.min())
        upper = max(x_train.max(), x_valid.max(), y_train.max(), y_valid.max())
        span = max(upper - lower, 1e-8)
        xlim = (lower - 0.1 * span, upper + 0.4 * span)
        ylim = (lower - 0.1 * span, upper + 0.4 * span)

        ax = axes[0, idx]

        ax.scatter(
            x_train,
            y_train,
            s=32,
            color="tab:blue",
            edgecolor="none",
            label="Training",
            zorder=1,
        )
        ax.scatter(
            x_valid,
            y_valid,
            s=32,
            color="tab:orange",
            edgecolor="none",
            label="Validation",
            zorder=2,
        )
        ax.plot(xlim, ylim, color="gray", linestyle="--", linewidth=1.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{x_label} {metric_name}")
        ax.set_ylabel(f"{y_label} {metric_name}")
        ax.set_title(metric_name, x=0.15, y=0.98, ha="left")
        ax.legend(frameon=False, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        difference_train = x_train - y_train
        difference_valid = x_valid - y_valid
        limit = max(
            np.max(np.abs(difference_train)),
            np.max(np.abs(difference_valid)),
            1e-8,
        )
        hist_limit = 4 * limit
        edges = np.linspace(-hist_limit, hist_limit, bins + 1)

        ax = axes[1, idx]

        counts_train, _, _ = ax.hist(
            difference_train,
            bins=edges,
            color="tab:blue",
            alpha=0.5,
            edgecolor="none",
            label="Training",
        )
        counts_valid, _, _ = ax.hist(
            difference_valid,
            bins=edges,
            color="tab:orange",
            alpha=0.5,
            edgecolor="none",
            label="Validation",
        )
        ymax = max(np.max(counts_train), np.max(counts_valid), 1.0)
        ax.set_xlim(-hist_limit, hist_limit)
        ax.set_ylim(0, 2 * ymax)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.set_title(metric_name, x=0.15, y=0.98, ha="left")
        ax.legend(frameon=False, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path, transparent=True)


def plot_model_metric_improvement(
    mse_trial_baseline,
    mse_trial_model,
    corr_trial_baseline,
    corr_trial_model,
    r2_trial_baseline,
    r2_trial_model,
    title,
    file_name,
    file_path,
    alpha=0.05,
    show=False,
):
    n_units = len(mse_trial_baseline)
    p_values_mse, p_values_corr, p_values_r2 = [], [], []
    better_mse, better_corr, better_r2 = [], [], []

    for unit_idx in range(n_units):
        mse_b = mse_trial_baseline[unit_idx]
        mse_m = mse_trial_model[unit_idx]
        corr_b = corr_trial_baseline[unit_idx]
        corr_m = corr_trial_model[unit_idx]
        r2_b = r2_trial_baseline[unit_idx]
        r2_m = r2_trial_model[unit_idx]

        try:
            _, p_corr = wilcoxon(corr_b, corr_m)
        except ValueError:
            p_corr = 1.0

        try:
            _, p_r2 = wilcoxon(r2_b, r2_m)
        except ValueError:
            p_r2 = 1.0

        try:
            _, p_mse = wilcoxon(mse_b, mse_m)
        except ValueError:
            p_mse = 1.0

        p_values_corr.append(p_corr)
        p_values_r2.append(p_r2)
        p_values_mse.append(p_mse)

        better_corr.append(np.nanmean(corr_m) > np.nanmean(corr_b))
        better_r2.append(np.nanmean(r2_m) > np.nanmean(r2_b))
        better_mse.append(np.nanmean(mse_m) < np.nanmean(mse_b))

    def get_proportions(pvals, is_better):
        sig = sum(1 for p, better in zip(pvals, is_better) if p < alpha and better)
        total = len(pvals)
        return sig / total, (total - sig) / total

    sig_mse, nonsig_mse = get_proportions(p_values_mse, better_mse)
    sig_corr, nonsig_corr = get_proportions(p_values_corr, better_corr)
    sig_r2, nonsig_r2 = get_proportions(p_values_r2, better_r2)

    metrics = ["MSE", "Correlation", r"$R^2$"]
    significant = [sig_mse, sig_corr, sig_r2]
    no_significant_improvement = [nonsig_mse, nonsig_corr, nonsig_r2]

    fig, ax = plt.subplots(figsize=(6, 6))

    p1 = ax.bar(
        metrics,
        significant,
        color="#4558C4",
        label="Significant improvement",
    )

    p2 = ax.bar(
        metrics,
        no_significant_improvement,
        bottom=significant,
        color="#C42A2F",
        label="No significant improvement",
    )

    ax.bar_label(p1, label_type="center", fmt="%.2f", fontsize=12)
    ax.bar_label(p2, label_type="center", fmt="%.2f", fontsize=12)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of units", fontsize=14)

    ax.tick_params(axis="both", labelsize=14)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.yaxis.grid(True, linestyle="--", alpha=0.7, color="grey", linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, fontsize=14)

    fig.suptitle(
        title,
        x=0.5,
        y=1.02,
        fontsize=18,
    )
    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)


def plot_shap(
    shap_values,
    title,
    file_name, 
    file_path,
    bin_times=None,
    unit_names=None,
    cmap="RdBu_r",
    center_zero=False,
    figsize=(5, 4),
    show=False,
):
    shap_values = np.asarray(shap_values, dtype=float)


    if shap_values.ndim != 2:
        raise ValueError(
            "shap_values must have shape (n_bins, n_units). "
            f"Received shape {shap_values.shape}."
        )


    n_bins, n_units = shap_values.shape


    if bin_times is None:
        bin_times = np.arange(n_bins)
    if unit_names is None:
        unit_names = np.arange(n_units)


    bin_times = np.asarray(bin_times)
    unit_names = np.asarray(unit_names)


    if bin_times.size != n_bins:
        raise ValueError("bin_times must contain one value per bin.")
    if unit_names.size != n_units:
        raise ValueError("unit_names must contain one name per unit.")


    if center_zero:
        vmin = -np.nanmax(np.abs(shap_values))
        vmax = np.nanmax(np.abs(shap_values))
    else:
        vmin = 0
        vmax = np.nanmax(shap_values)


    mean_by_unit = np.nanmean(shap_values, axis=0)
    mean_by_bin = np.nanmean(shap_values, axis=1)


    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.25, 1],
        height_ratios=[1, 0.25],
        wspace=0.03,
        hspace=0.03,
    )


    ax_main = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[1, 1])
    ax_corner = fig.add_subplot(gs[1, 0])
    ax_corner.axis("off")


    im = ax_main.imshow(
        shap_values.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


    ax_main.set_title(title, y=1.02)
    ax_main.set_xticks([])
    max_yticks = 12
    ytick_idx = np.unique(
        np.linspace(0, n_units - 1, min(max_yticks, n_units), dtype=int)
    )


    ax_main.set_yticks(ytick_idx)
    ax_main.tick_params(axis="y", left=False, labelleft=False)
    ax_main.spines[["top", "right"]].set_visible(False)


    unit_idx = np.arange(n_units)


    ax_left.axvline(0, color="black", linewidth=0.8, zorder=0)


    ax_left.hlines(
        y=unit_idx,
        xmin=0,
        xmax=mean_by_unit,
        color="black",
        linewidth=1.1,
        zorder=1,
    )


    ax_left.scatter(
        mean_by_unit,
        unit_idx,
        color="black",
        s=14,
        zorder=2,
    )


    left_limit = max(np.nanmax(np.abs(mean_by_unit)), np.finfo(float).eps)
    ax_left.set_xlim(-1.1 * left_limit, 1.1 * left_limit)


    ax_left.set_ylabel("Unit")
    ax_left.set_yticks(ytick_idx, unit_names[ytick_idx], fontsize=7)
    ax_left.set_xlabel("Mean SHAP", fontsize=8)
    ax_left.tick_params(axis="x", labelsize=7)
    ax_left.tick_params(axis="y", labelsize=7)
    ax_left.spines[["top", "right"]].set_visible(False)


    ax_bottom.axhline(0, color="black", linewidth=0.8, zorder=0)


    ax_bottom.plot(
        bin_times,
        mean_by_bin,
        color="black",
        linewidth=1.5,
    )


    ax_bottom.fill_between(
        bin_times,
        0,
        mean_by_bin,
        color="black",
        alpha=0.20,
    )


    ax_bottom.set_xlabel("Time")
    ax_bottom.set_ylabel("Mean\nSHAP", fontsize=8)
    ax_bottom.tick_params(axis="both", labelsize=7)
    ax_bottom.spines[["top", "right"]].set_visible(False)


    cbar = fig.colorbar(
        im,
        ax=ax_main,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("SHAP value", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


    if show:
        plt.show()


    return save_figure(fig, file_name, file_path)


def plot_shap_hist(
    unit_bin_shap_shuffles,
    unit_bin_shap_permutations,
    variable_name,
    unit_name,
    bin_time,
    color,
    title,
    filename,
    filepath,
    bins=50,
    show=False,
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    shuffle_values = unit_bin_shap_shuffles[np.isfinite(unit_bin_shap_shuffles)]
    permutation_values = unit_bin_shap_permutations[np.isfinite(unit_bin_shap_permutations)]
    values = np.concatenate([shuffle_values, permutation_values])
    bin_edges = np.linspace(values.min(), values.max(), bins + 1)
    ax.hist(shuffle_values, bins=bin_edges, density=True, color="gray", edgecolor=None, alpha=0.5, label="Shuffle null")
    ax.hist(permutation_values, bins=bin_edges, density=True, color=color, edgecolor=None, alpha=0.5, label="Permutation")
    ax.axvline(shuffle_values.mean(), color="gray", linestyle="--", linewidth=2, label="Shuffle mean")
    ax.axvline(permutation_values.mean(), color=color, linestyle="--", linewidth=2, label="Permutation mean")
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_variable_selectivity_hist(
    unit_selectivity_shuffles,
    unit_selectivity_permutations,
    variable_name,
    unit_name,
    color,
    title,
    filename,
    filepath,
    bins=50,
    show=False,
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    shuffle_values = unit_selectivity_shuffles[np.isfinite(unit_selectivity_shuffles)]
    permutation_values = unit_selectivity_permutations[np.isfinite(unit_selectivity_permutations)]
    values = np.concatenate([shuffle_values, permutation_values])
    bin_edges = np.linspace(values.min(), values.max(), bins + 1)
    ax.hist(shuffle_values, bins=bin_edges, density=True, color="gray", edgecolor=None, alpha=0.5, label="Shuffle null")
    ax.hist(permutation_values, bins=bin_edges, density=True, color=color, edgecolor=None, alpha=0.5, label="Permutation")
    ax.axvline(shuffle_values.mean(), color="gray", linestyle="--", linewidth=2, label="Shuffle mean")
    ax.axvline(permutation_values.mean(), color=color, linestyle="--", linewidth=2, label="Permutation mean")
    ax.set_xlabel("Mean absolute SHAP")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_variable_selectivity_curves(
    variable_selectivity,
    variable_name,
    significant,
    color,
    title,
    filename,
    filepath,
    show=False,
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    sort_idx = np.argsort(variable_selectivity)
    sorted_selectivity = variable_selectivity[sort_idx]
    sorted_significant = significant[sort_idx]
    x = np.arange(1, len(sorted_selectivity) + 1)
    split_idx = np.argmax(sorted_significant) if sorted_significant.any() else len(sorted_significant) - 1
    ax.plot(x[: split_idx + 1], sorted_selectivity[: split_idx + 1], color=color, linewidth=2.5, alpha=0.3)
    ax.plot(x[split_idx:], sorted_selectivity[split_idx:], color=color, linewidth=2.5, alpha=1.0)
    ax.set_xlabel("Units")
    ax.set_ylabel("Effect size")
    ax.set_xlim(1, len(sorted_selectivity))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_variable_selectivity_pca(
    variable_selectivities,
    variable_selectivity,
    variable_name,
    significant,
    color,
    title,
    filename,
    filepath,
    show=False,
):
    unit_features = variable_selectivities.T
    unit_coordinates = PCA(n_components=2).fit_transform(unit_features)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    norm = mcolors.Normalize(vmin=np.nanmin(variable_selectivity), vmax=np.nanmax(variable_selectivity))
    base_rgb = mcolors.to_rgb(color)
    alphas = np.clip(norm(variable_selectivity), 0.05, 1.0)
    colors = [(*base_rgb, alpha) for alpha in alphas]
    edgecolors = ["black" if sig else "none" for sig in significant]
    ax.scatter(
        unit_coordinates[:, 0],
        unit_coordinates[:, 1],
        color=colors,
        s=70,
        edgecolors=edgecolors,
        linewidths=1.2,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_class_selectivity_pca(
    variable_selectivity,
    class_name,
    significant,
    color,
    title,
    filename,
    filepath,
    show=False,
):
    unit_features = variable_selectivity.T
    unit_coordinates = PCA(n_components=2).fit_transform(unit_features)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    colors = [color if sig else mcolors.to_rgba(color, alpha=0.15) for sig in significant]
    edgecolors = ["black" if sig else "lightgray" for sig in significant]
    ax.scatter(
        unit_coordinates[:, 0],
        unit_coordinates[:, 1],
        c=colors,
        s=70,
        edgecolors=edgecolors,
        linewidths=1.2,
    )
    ax.set_title(class_name.replace("_", " ").title())
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_variable_selectivity_matrix(
    variable_selectivity,
    variable_names,
    unit_names,
    significant,
    variable_colors,
    title,
    filename,
    filepath,
    show=False,
):
    unit_scores = np.nanmean(variable_selectivity, axis=0)
    sort_idx = np.argsort(-unit_scores)
    sorted_matrix = variable_selectivity[:, sort_idx]
    sorted_significant = significant[:, sort_idx]
    sorted_unit_names = unit_names[sort_idx]
    vmin = np.nanmin(sorted_matrix)
    vmax = np.nanmax(sorted_matrix)
    n_vars, n_units = sorted_matrix.shape
    fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.15 * n_units), 4), layout="constrained")
    row_height = 0.7
    row_gap = 0.3
    for row_idx, color in enumerate(variable_colors):
        row_values = sorted_matrix[row_idx : row_idx + 1]
        cmap = mcolors.LinearSegmentedColormap.from_list(f"var_{row_idx}", ["white", color], N=256)
        center = row_idx * (row_height + row_gap)
        ax.imshow(
            row_values,
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            extent=[-0.5, n_units - 0.5, center - row_height / 2.0, center + row_height / 2.0],
        )
        selected_units = np.where(sorted_significant[row_idx])[0]
        for unit_idx in selected_units:
            ax.add_patch(
                plt.Rectangle(
                    (unit_idx - 0.5, center - row_height / 2.0),
                    1.0,
                    row_height,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.8,
                )
            )
    row_positions = [i * (row_height + row_gap) for i in range(n_vars)]
    ax.set_xticks(np.arange(n_units))
    ax.set_xticklabels(sorted_unit_names, rotation=90, fontsize=7)
    ax.set_yticks(row_positions)
    ax.set_yticklabels(variable_names, fontsize=9)
    ax.set_xlim(-0.5, n_units - 0.5)
    ax.set_ylim(-row_gap, (n_vars - 1) * (row_height + row_gap) + row_height + row_gap)
    ax.set_xlabel("Units")
    ax.set_ylabel("Variables")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_class_selectivity_matrix(
    class_selectivity,
    class_names,
    unit_names,
    significant,
    class_colors,
    title,
    filename,
    filepath,
    show=False,
):
    unit_scores = np.nanmean(class_selectivity, axis=0)
    sort_idx = np.argsort(-unit_scores)
    sorted_matrix = class_selectivity[:, sort_idx]
    sorted_significant = significant[:, sort_idx]
    sorted_unit_names = unit_names[sort_idx]
    n_classes, n_units = sorted_matrix.shape
    fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.15 * n_units), 4), layout="constrained")
    row_height = 0.7
    row_gap = 0.3
    for row_idx, color in enumerate(class_colors):
        center = row_idx * (row_height + row_gap)
        row_colors = [color if sig else mcolors.to_rgba(color, alpha=0.15) for sig in sorted_significant[row_idx]]
        for unit_idx, c in enumerate(row_colors):
            ax.add_patch(
                plt.Rectangle(
                    (unit_idx - 0.5, center - row_height / 2.0),
                    1.0,
                    row_height,
                    facecolor=c,
                    edgecolor="black" if sorted_significant[row_idx, unit_idx] else "none",
                    linewidth=0.8,
                )
            )
    row_positions = [i * (row_height + row_gap) for i in range(n_classes)]
    ax.set_xticks(np.arange(n_units))
    ax.set_xticklabels(sorted_unit_names, rotation=90, fontsize=7)
    ax.set_yticks(row_positions)
    ax.set_yticklabels([c.replace("_", " ").title() for c in class_names], fontsize=9)
    ax.set_xlim(-0.5, n_units - 0.5)
    ax.set_ylim(-row_gap, (n_classes - 1) * (row_height + row_gap) + row_height + row_gap)
    ax.set_xlabel("Units")
    ax.set_ylabel("Classes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    if show:
        plt.show()
    return save_figure(fig, filename, filepath)


def plot_group_influence_matrix(
    correlation_matrix,
    r2_matrix,
    mse_matrix,
    group_names,
    title,
    file_name,
    file_path,
    show=False,
):
    matrices = [
        ("Correlation", correlation_matrix),
        (r"$R^2$", r2_matrix),
        ("MSE", mse_matrix),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
        layout="constrained",
    )

    for ax, (metric_name, matrix) in zip(axes, matrices):
        im = ax.imshow(
            matrix,
            origin="upper",
            cmap="magma",
            vmin=0,
            vmax=1,
        )

        ax.set_title(metric_name)
        ax.set_xlabel("Conditioning source group")
        ax.set_ylabel("Predicted target group")
        ax.set_xticks(
            np.arange(len(group_names)),
            group_names,
            rotation=45,
            ha="right",
        )
        ax.set_yticks(np.arange(len(group_names)), group_names)

        for row_idx in range(matrix.shape[0]):
            for column_idx in range(matrix.shape[1]):
                value = matrix[row_idx, column_idx]
                label = "n/a" if np.isnan(value) else f"{value:.2f}"
                color = (
                    "white"
                    if not np.isnan(value) and value < 0.55
                    else "black"
                )

                ax.text(
                    column_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                )

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="Fraction of target units",
        )

    fig.suptitle(title)

    if show:
        plt.show()

    return save_figure(fig, file_name, file_path)

