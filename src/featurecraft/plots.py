"""Plotting utilities for FeatureCraft."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set non-interactive backend for headless environments
matplotlib.use("Agg")

from .utils import fig_to_base64


def plot_missingness(df: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Plot missingness by column."""
    fig, ax = plt.subplots(figsize=(8, 4))
    rates = df.isna().mean().sort_values(ascending=False)
    rates.plot(kind="bar", ax=ax)
    ax.set_title("Missingness by Column")
    ax.set_ylabel("Proportion Missing")
    ax.set_ylim(0, 1)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return fig, b64


def plot_distributions(df: pd.DataFrame, max_cols: int = 6) -> dict[str, tuple[plt.Figure, str]]:
    """Plot distributions for numeric columns."""
    figs: dict[str, tuple[plt.Figure, str]] = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    for c in num_cols:
        fig, ax = plt.subplots(figsize=(4, 3))
        s = df[c].dropna()
        ax.hist(s, bins=30)
        ax.set_title(f"Dist: {c}")
        b64 = fig_to_base64(fig)
        figs[c] = (fig, b64)
        plt.close(fig)
    return figs


def plot_boxplots(df: pd.DataFrame, max_cols: int = 6) -> dict[str, tuple[plt.Figure, str]]:
    """Plot boxplots for numeric columns."""
    figs: dict[str, tuple[plt.Figure, str]] = {}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    for c in num_cols:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.boxplot(df[c].dropna(), vert=True, tick_labels=[c])
        ax.set_title(f"Box: {c}")
        b64 = fig_to_base64(fig)
        figs[c] = (fig, b64)
        plt.close(fig)
    return figs


def plot_countplots(df: pd.DataFrame, max_cols: int = 6) -> dict[str, tuple[plt.Figure, str]]:
    """Plot count plots for categorical columns."""
    figs: dict[str, tuple[plt.Figure, str]] = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()[:max_cols]
    for c in cat_cols:
        fig, ax = plt.subplots(figsize=(5, 3))
        vc = df[c].astype(str).value_counts().head(20)
        vc.plot(kind="bar", ax=ax)
        ax.set_title(f"Counts: {c}")
        b64 = fig_to_base64(fig)
        figs[c] = (fig, b64)
        plt.close(fig)
    return figs


def plot_correlation_heatmap(corr: pd.DataFrame) -> tuple[plt.Figure, str]:
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr.values, interpolation="nearest")
    ax.set_title("Correlation Heatmap (numeric)")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(cax)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return fig, b64
