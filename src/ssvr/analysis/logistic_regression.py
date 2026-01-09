import typing as t
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample


@dataclass
class FeatureSpec:
    """Specification for a regressor in the design matrix."""

    suffix: str
    label: str
    color: str
    marker: str


# Define the feature structure in one place
FEATURE_SPECS = [
    FeatureSpec("choice", "Previous Choice", "blue", "o"),
    FeatureSpec("outcome", "Previous Outcome", "green", "s"),
    FeatureSpec("sameness", "Sameness", "purple", "^"),
    FeatureSpec("same_reward", "Same x Reward", "red", "d"),
    FeatureSpec("other_reward", "Other x Reward", "orange", "v"),
]


def get_n_features_per_lag() -> int:
    """Returns the number of features per lag."""
    return len(FEATURE_SPECS)


def get_feature_names_for_lag(k: int) -> t.List[str]:
    """Returns the feature column names for a specific lag."""
    return [f"lag_{k}_{spec.suffix}" for spec in FEATURE_SPECS]


def create_regression_design_matrix(trials_df: pd.DataFrame, n_back: int = 3) -> t.Tuple[pd.DataFrame, t.List[str]]:
    """
    Creates the design matrix for logistic regression based on trial history.

    Encodings:
    - Sameness s_{t-k} -> {+1 (same patch as current), -1 (different patch)}
    - Choice y_t -> {0, 1} (no stop, stop)
    - Outcome o_t = y_t · (2r_t - 1) -> {+1 (reward), -1 (no reward), 0 (no choice)}

    Regressors per lag k:
    - Previous choice: y_{t-k}
    - Previous outcome: o_{t-k}
    - Sameness: s_{t-k} (same=+1, different=-1)
    - Same x Reward: o_{t-k} if s_{t-k}=+1, else 0
    - Other x Reward: o_{t-k} if s_{t-k}=-1, else 0

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

    if "trial_number" in df.columns:
        df = df.sort_values(["session_id", "trial_number"])

    # TODO remove trials across sessions. For now they are too few to matter much.

    df["choice"] = df["is_choice"].fillna(False).astype(int)
    is_rewarded = df["is_rewarded"].fillna(False).astype(int)
    df["outcome"] = df["choice"] * (2 * is_rewarded - 1)

    feature_cols = []
    grouped = df.groupby("session_id")
    new_cols = {}

    for k in range(1, n_back + 1):
        prev_choice = grouped["choice"].shift(k)
        prev_outcome = grouped["outcome"].shift(k)
        prev_patch = grouped["patch_index"].shift(k)

        valid_history = prev_choice.notna() & prev_outcome.notna() & prev_patch.notna()
        sameness = (df["patch_index"] == prev_patch).astype(int) * 2 - 1

        # Same x Reward: outcome if same patch (+1), else 0
        same_reward = pd.Series(np.where(sameness == 1, prev_outcome, 0), index=df.index)

        # Other x Reward: outcome if different patch (-1), else 0
        other_reward = pd.Series(np.where(sameness == -1, prev_outcome, 0), index=df.index)

        new_cols[f"lag_{k}_outcome"] = prev_outcome.mask(~valid_history, np.nan)
        new_cols[f"lag_{k}_sameness"] = sameness.mask(~valid_history, np.nan)
        new_cols[f"lag_{k}_same_reward"] = same_reward.mask(~valid_history, np.nan)
        new_cols[f"lag_{k}_other_reward"] = other_reward.mask(~valid_history, np.nan)
        new_cols[f"lag_{k}_choice"] = prev_choice * 2 - 1  # Convert to {-1, +1}

        feature_cols.extend(get_feature_names_for_lag(k))

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
    intercept = model.intercept_[0]

    n_features = get_n_features_per_lag()
    matrix = coefs.reshape(n_back, n_features)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    lags = range(1, n_back + 1)

    for idx, spec in enumerate(FEATURE_SPECS):
        ax.plot(lags, matrix[:, idx], label=spec.label, color=spec.color, linestyle="-", marker=spec.marker)

    ax.set_xlabel("Lag k (trials back)")
    ax.set_ylabel("Coefficient Weight")
    ax.set_title(f"Logistic Regression Coefficients (β₀={intercept:.3f})")
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
    n_features = len(feature_cols)
    coefs = np.zeros((n_bootstraps, n_features))
    intercepts = np.zeros(n_bootstraps)
    scores = np.zeros(n_bootstraps)

    X = df[feature_cols].values
    y = df[target_col].astype(int).values
    indices = np.arange(len(df))

    for i in range(n_bootstraps):
        resampled_idx = resample(indices)
        oob_idx = np.setdiff1d(indices, resampled_idx)

        if len(oob_idx) == 0:
            from sklearn.model_selection import train_test_split

            resampled_idx, oob_idx = train_test_split(indices, test_size=0.2)

        X_train, y_train = X[resampled_idx], y[resampled_idx]
        X_test, y_test = X[oob_idx], y[oob_idx]

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
    mean_coefs = np.mean(coefs, axis=0)
    mean_intercept = np.mean(intercepts)

    lower_percentile = (1 - ci) / 2 * 100
    upper_percentile = (1 + ci) / 2 * 100

    ci_lower = np.percentile(coefs, lower_percentile, axis=0)
    ci_upper = np.percentile(coefs, upper_percentile, axis=0)

    n_features = get_n_features_per_lag()
    mean_matrix = mean_coefs.reshape(n_back, n_features)
    ci_lower_matrix = ci_lower.reshape(n_back, n_features)
    ci_upper_matrix = ci_upper.reshape(n_back, n_features)

    yerr_lower = mean_matrix - ci_lower_matrix
    yerr_upper = ci_upper_matrix - mean_matrix
    yerr_matrix = np.stack([yerr_lower.T, yerr_upper.T])

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    lags = range(1, n_back + 1)

    for idx, spec in enumerate(FEATURE_SPECS):
        ax.errorbar(
            lags,
            mean_matrix[:, idx],
            yerr=yerr_matrix[:, idx, :],
            label=spec.label,
            color=spec.color,
            linestyle="-",
            marker=spec.marker,
            capsize=3,
        )

    ax.set_xlabel("Lag k (trials back)")
    ax.set_ylabel("Coefficient Weight")
    ax.set_title(f"Logistic Regression Coefficients with {int(ci * 100)}% CI (β₀={mean_intercept:.3f})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linestyle=":", alpha=0.5)

    plt.tight_layout()

    return fig, ax
