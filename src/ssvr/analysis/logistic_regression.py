import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample


def create_regression_design_matrix(trials_df: pd.DataFrame, n_back: int = 3) -> t.Tuple[pd.DataFrame, t.List[str]]:
    """
    Creates the design matrix for logistic regression based on trial history.
    Uses one-hot encoding for the interaction of (Sameness x Outcome).

    Regressors per lag k (6 total):
    - Same Patch & Reward
    - Same Patch & No Reward
    - Same Patch & No Choice
    - Different Patch & Reward
    - Different Patch & No Reward
    - Different Patch & No Choice

    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame containing 'session_id', 'patch_index', 'is_choice', 'is_rewarded'.
    n_back : int
        Number of trials to look back.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        Cleaned DataFrame with features (NaNs dropped) and list of feature column names.
    """
    df = trials_df.copy()

    # Ensure sorting
    if "trial_number" in df.columns:
        df = df.sort_values(["session_id", "trial_number"])
    else:
        # Fallback: assume index is roughly chronological or rely on existing order
        pass
    # TODO remove trials across sessions. For now they are too few to matter much.

    # Calculate Outcome variable for the current trial (to be shifted later)
    # 1: Reward, -1: No Reward, 0: No Choice
    outcome = np.zeros(len(df))
    is_choice = df["is_choice"].fillna(False).astype(bool)
    is_rewarded = df["is_rewarded"].fillna(False).astype(bool)

    outcome[is_choice & is_rewarded] = 1
    outcome[is_choice & ~is_rewarded] = -1
    df["outcome_val"] = outcome

    feature_cols = []
    grouped = df.groupby("session_id")
    new_cols = {}

    for i in range(1, n_back + 1):
        prev_patch = grouped["patch_index"].shift(i)
        prev_outcome = grouped["outcome_val"].shift(i)

        valid_history = prev_patch.notna() & prev_outcome.notna()

        # Sameness (Current vs Lag i)
        is_same = df["patch_index"] == prev_patch  # We shift so this is not recursive!!!!!!

        # Define the 6 one-hot categories
        # Same Patch
        new_cols[f"lag_{i}_same_rew"] = (is_same & (prev_outcome == 1)).astype(int)
        new_cols[f"lag_{i}_same_unrew"] = (is_same & (prev_outcome == -1)).astype(int)
        new_cols[f"lag_{i}_same_nochoice"] = (is_same & (prev_outcome == 0)).astype(int)

        # Different Patch
        new_cols[f"lag_{i}_diff_rew"] = (~is_same & (prev_outcome == 1)).astype(int)
        new_cols[f"lag_{i}_diff_unrew"] = (~is_same & (prev_outcome == -1)).astype(int)
        new_cols[f"lag_{i}_diff_nochoice"] = (~is_same & (prev_outcome == 0)).astype(int)

        # Apply mask to ensure we don't have valid 0s where history is missing
        cols_for_lag = [
            f"lag_{i}_same_rew",
            f"lag_{i}_same_unrew",
            f"lag_{i}_same_nochoice",
            f"lag_{i}_diff_rew",
            f"lag_{i}_diff_unrew",
            f"lag_{i}_diff_nochoice",
        ]

        for col in cols_for_lag:
            new_cols[col] = new_cols[col].mask(~valid_history, np.nan)

        feature_cols.extend(cols_for_lag)

    new_features_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_features_df], axis=1)

    df_clean = df.dropna(subset=feature_cols)

    return df_clean, feature_cols


def fit_logistic_regression(
    df: pd.DataFrame, feature_cols: t.List[str], target_col: str = "is_choice", fit_intercept: bool = True
) -> LogisticRegression:
    """
    Fits a logistic regression model.
    """
    X = df[feature_cols]
    y = df[target_col].astype(int)

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, fit_intercept=fit_intercept)
    model.fit(X, y)

    return model


def plot_regression_coefficients(
    model: LogisticRegression, n_back: int, ax: t.Optional[plt.Axes] = None
) -> t.Tuple[plt.Figure, plt.Axes]:
    """
    Plots the coefficients of the logistic regression model.
    """
    coefs = model.coef_[0]
    model.intercept_[0]

    # Reshape: n_back rows, N feats
    # Order: Same-Rew, Same-Unrew, Same-NoChoice, Diff-Rew, Diff-Unrew, Diff-NoChoice
    matrix = coefs.reshape(n_back, 6)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    lags = range(1, n_back + 1)

    # Plotting
    # Same Patch (Solid lines)
    ax.plot(lags, matrix[:, 0], label="Same - Reward", color="green", linestyle="-", marker="o")
    ax.plot(lags, matrix[:, 1], label="Same - No Reward", color="red", linestyle="-", marker="o")
    ax.plot(lags, matrix[:, 2], label="Same - No Choice", color="gray", linestyle="-", marker="o")

    # Different Patch (Dashed lines)
    ax.plot(lags, matrix[:, 3], label="Diff - Reward", color="green", linestyle="--", marker="s")
    ax.plot(lags, matrix[:, 4], label="Diff - No Reward", color="red", linestyle="--", marker="s")
    ax.plot(lags, matrix[:, 5], label="Diff - No Choice", color="gray", linestyle="--", marker="s")

    ax.set_xlabel("Lag (trials back)")
    ax.set_ylabel("Coefficient Weight")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)

    plt.tight_layout()

    return fig, ax


def perform_cross_validation(
    df: pd.DataFrame, feature_cols: t.List[str], target_col: str = "is_choice", cv: int = 5, fit_intercept: bool = True
) -> np.ndarray:
    """
    Performs k-fold cross-validation and returns the accuracy scores.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing features and target.
    feature_cols : list[str]
        List of feature column names.
    target_col : str
        Name of the target column.
    cv : int
        Number of folds.

    Returns
    -------
    np.ndarray
        Array of scores for each fold.
    """

    X = df[feature_cols]
    y = df[target_col].astype(int)

    # Create a fresh model instance (not fitted) for cross-validation.
    # cross_val_score will handle fitting on train splits and scoring on test splits
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, fit_intercept=fit_intercept)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores


def perform_bootstrap_regression(
    df: pd.DataFrame,
    feature_cols: t.List[str],
    target_col: str = "is_choice",
    n_bootstraps: int = 1000,
    fit_intercept: bool = True,
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs bootstrap resampling to estimate confidence intervals for logistic regression coefficients.
    Also calculates accuracy on Out-Of-Bag (OOB) samples for validation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing features and target.
    feature_cols : list[str]
        List of feature column names.
    target_col : str
        Name of the target column.
    n_bootstraps : int
        Number of bootstrap iterations.
    fit_intercept: bool = True,
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - coefs: Array of shape (n_bootstraps, n_features) containing coefficients from each bootstrap.
        - intercepts: Array of shape (n_bootstraps,) containing intercepts from each bootstrap.
        - scores: Array of shape (n_bootstraps,) containing OOB accuracy scores.
    """
    #  https://en.wikipedia.org/wiki/Out-of-bag_error
    n_features = len(feature_cols)
    coefs = np.zeros((n_bootstraps, n_features))
    intercepts = np.zeros(n_bootstraps)
    scores = np.zeros(n_bootstraps)

    X = df[feature_cols].values
    y = df[target_col].astype(int).values
    indices = np.arange(len(df))

    for i in range(n_bootstraps):
        # Resample indices with replacement
        resampled_idx = resample(indices)

        # OOB indices (validation set)
        oob_idx = np.setdiff1d(indices, resampled_idx)

        # If OOB is empty (unlikely with large N), just use random subset or skip
        if len(oob_idx) == 0:
            # Fallback to a random split if OOB is empty (edge case)
            from sklearn.model_selection import train_test_split

            resampled_idx, oob_idx = train_test_split(indices, test_size=0.2)

        X_train, y_train = X[resampled_idx], y[resampled_idx]
        X_test, y_test = X[oob_idx], y[oob_idx]

        # Fit model
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, fit_intercept=fit_intercept)
        model.fit(X_train, y_train)

        coefs[i, :] = model.coef_[0]
        intercepts[i] = model.intercept_[0]
        scores[i] = model.score(X_test, y_test)

    return coefs, intercepts, scores


def plot_regression_coefficients_with_ci(
    coefs: np.ndarray,
    intercepts: np.ndarray,
    n_back: int,
    ax: t.Optional[plt.Axes] = None,
    ci: float = 0.95,
) -> t.Tuple[plt.Figure, plt.Axes]:
    """
    Plots the coefficients of the logistic regression model with confidence intervals.

    Parameters
    ----------
    coefs : np.ndarray
        Array of shape (n_bootstraps, n_features) containing bootstrapped coefficients.
    intercepts : np.ndarray
        Array of shape (n_bootstraps,) containing bootstrapped intercepts.
    n_back : int
        Number of trials to look back.
    ax : plt.Axes, optional
        Axes to plot on.
    ci : float
        Confidence interval level (e.g., 0.95 for 95% CI).

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and Axes objects.
    """
    # Calculate mean and CI
    mean_coefs = np.mean(coefs, axis=0)
    np.mean(intercepts)

    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100

    ci_lower = np.percentile(coefs, lower_percentile, axis=0)
    ci_upper = np.percentile(coefs, upper_percentile, axis=0)
    yerr = np.vstack([mean_coefs - ci_lower, ci_upper - mean_coefs])
    mean_matrix = mean_coefs.reshape(n_back, 6)
    yerr_matrix = yerr.reshape(2, n_back, 6)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    lags = range(1, n_back + 1)

    def plot_series(col_idx, label, color, linestyle, marker):
        ax.errorbar(
            lags,
            mean_matrix[:, col_idx],
            yerr=yerr_matrix[:, :, col_idx],
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            capsize=3,
        )

    # Plotting
    # Same Patch (Solid lines)
    plot_series(0, "Same - Reward", "green", "-", "o")
    plot_series(1, "Same - No Reward", "red", "-", "o")
    plot_series(2, "Same - No Choice", "gray", "-", "o")

    # Different Patch (Dashed lines)
    plot_series(3, "Diff - Reward", "green", "--", "s")
    plot_series(4, "Diff - No Reward", "red", "--", "s")
    plot_series(5, "Diff - No Choice", "gray", "--", "s")

    ax.set_xlabel("Lag (trials back)")
    ax.set_ylabel("Coefficient Weight")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)

    plt.tight_layout()

    return fig, ax
