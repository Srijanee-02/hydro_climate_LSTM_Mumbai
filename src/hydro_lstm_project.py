"""
Hydro-climatology self project for beginners
-------------------------------------------
Project: Forecast next-day runoff-event risk in Powai/Mumbai using NASA POWER daily
climate data and a small Long Short-Term Memory (LSTM) neural network implemented
only with NumPy.

Run from the repository root:
    python src/hydro_lstm_project.py

The script downloads/loads data, creates a simple SCS-Curve-Number runoff proxy,
trains the LSTM, evaluates it on a time-based test period, and saves plots in
results/.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

# ------------------------------
# 1. Project configuration
# ------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Powai/IIT Bombay approximate coordinates
LATITUDE = 19.13
LONGITUDE = 72.91
START_DATE = "20100101"
END_DATE = "20251231"
LOOKBACK_DAYS = 30
RUNOFF_EVENT_THRESHOLD_MM = 10.0

NASA_POWER_PARAMETERS = [
    "PRECTOTCORR",  # precipitation corrected, mm/day
    "T2M",          # temperature at 2 m, deg C
    "T2M_MAX",      # max temperature, deg C
    "T2M_MIN",      # min temperature, deg C
    "RH2M",         # relative humidity at 2 m, %
    "WS2M",         # wind speed at 2 m, m/s
]

FEATURE_COLUMNS = [
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS2M",
    "runoff_mm",
    "rain_5day_mm",
    "doy_sin",
    "doy_cos",
]


# ------------------------------
# 2. A tiny LSTM implemented in NumPy
# ------------------------------

class NumpyLSTMClassifier:
    """A compact educational LSTM for binary sequence classification.

    It uses the standard LSTM gates:
        input gate, forget gate, output gate, and candidate cell state.

    This is intentionally written without TensorFlow/PyTorch so that beginners can
    inspect the equations directly. For a larger project, PyTorch/Keras would be
    more convenient.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 16,
        learning_rate: float = 0.003,
        seed: int = 42,
        grad_clip: float = 1.0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.step = 0

        # Combined weights for four gates: input, forget, output, candidate.
        self.params = {
            "Wx": rng.normal(0, 0.5 / np.sqrt(input_size), (input_size, 4 * hidden_size)),
            "Wh": rng.normal(0, 0.5 / np.sqrt(hidden_size), (hidden_size, 4 * hidden_size)),
            "b": np.zeros((4 * hidden_size,)),
            "Wy": rng.normal(0, 0.5 / np.sqrt(hidden_size), (hidden_size, 1)),
            "by": np.zeros((1,)),
        }

        # Adam optimizer state
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -40, 40)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X: np.ndarray):
        """Forward pass.

        X shape: (batch, time_steps, features)
        returns: logits and a list of cached values for backpropagation
        """
        n, time_steps, _ = X.shape
        h = np.zeros((n, self.hidden_size))
        c = np.zeros((n, self.hidden_size))
        caches = []

        for t in range(time_steps):
            x_t = X[:, t, :]
            h_prev, c_prev = h, c

            z = x_t @ self.params["Wx"] + h_prev @ self.params["Wh"] + self.params["b"]
            H = self.hidden_size
            i = self.sigmoid(z[:, :H])
            f = self.sigmoid(z[:, H : 2 * H])
            o = self.sigmoid(z[:, 2 * H : 3 * H])
            g = np.tanh(z[:, 3 * H :])

            c = f * c_prev + i * g
            h = o * np.tanh(c)
            caches.append((x_t, h_prev, c_prev, i, f, o, g, c, h))

        logits = h @ self.params["Wy"] + self.params["by"]
        return logits, caches

    def loss_and_gradients(self, X: np.ndarray, y: np.ndarray, pos_weight: float):
        """Weighted binary cross entropy and manual backpropagation through time."""
        y = y.reshape(-1, 1).astype(float)
        n = len(y)
        H = self.hidden_size

        logits, caches = self.forward(X)
        prob = self.sigmoid(logits)
        eps = 1e-8
        weights = np.where(y == 1, pos_weight, 1.0)
        loss = -np.mean(weights * (y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)))

        # dL/dlogits for weighted BCE
        dlogits = weights * (prob - y) / n

        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        h_last = caches[-1][-1]
        grads["Wy"] = h_last.T @ dlogits
        grads["by"] = dlogits.sum(axis=0)

        dh = dlogits @ self.params["Wy"].T
        dh_next = np.zeros_like(dh)
        dc_next = np.zeros_like(dh)

        for cache in reversed(caches):
            x_t, h_prev, c_prev, i, f, o, g, c, h = cache
            dh_total = dh + dh_next
            tanh_c = np.tanh(c)

            do = dh_total * tanh_c
            dc = dh_total * o * (1 - tanh_c**2) + dc_next

            df = dc * c_prev
            dc_next = dc * f
            di = dc * g
            dg = dc * i

            dzi = di * i * (1 - i)
            dzf = df * f * (1 - f)
            dzo = do * o * (1 - o)
            dzg = dg * (1 - g**2)
            dz = np.concatenate([dzi, dzf, dzo, dzg], axis=1)

            grads["Wx"] += x_t.T @ dz
            grads["Wh"] += h_prev.T @ dz
            grads["b"] += dz.sum(axis=0)

            dh_next = dz @ self.params["Wh"].T
            dh = np.zeros_like(dh)

        # Gradient clipping prevents exploding gradients in sequence models.
        for key in grads:
            grads[key] = np.clip(grads[key], -self.grad_clip, self.grad_clip)

        return float(loss), grads

    def adam_update(self, grads: dict[str, np.ndarray]) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.step += 1
        for key, grad in grads.items():
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * grad
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * grad * grad
            m_hat = self.m[key] / (1 - beta1**self.step)
            v_hat = self.v[key] / (1 - beta2**self.step)
            self.params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 64,
        seed: int = 0,
    ) -> pd.DataFrame:
        pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-8)
        rng = np.random.default_rng(seed)
        history = []

        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(X_train))
            batch_losses = []

            for start in range(0, len(order), batch_size):
                batch_idx = order[start : start + batch_size]
                loss, grads = self.loss_and_gradients(X_train[batch_idx], y_train[batch_idx], pos_weight)
                self.adam_update(grads)
                batch_losses.append(loss)

            train_loss = float(np.mean(batch_losses))
            val_loss = weighted_bce(y_val, self.predict_proba(X_val), pos_weight)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"Epoch {epoch:02d}/{epochs} | train loss={train_loss:.4f} | val loss={val_loss:.4f}")

        return pd.DataFrame(history)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(X)
        return self.sigmoid(logits).ravel()


def weighted_bce(y_true: np.ndarray, prob: np.ndarray, pos_weight: float) -> float:
    y = y_true.reshape(-1, 1).astype(float)
    p = prob.reshape(-1, 1)
    eps = 1e-8
    weights = np.where(y == 1, pos_weight, 1.0)
    return float(-np.mean(weights * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))


# ------------------------------
# 3. Data download and hydrology features
# ------------------------------

def download_nasa_power_data() -> pd.DataFrame:
    """Download daily climate data from NASA POWER or load cached CSV."""
    csv_path = DATA_DIR / "nasa_power_powai_2010_2025.csv"
    if csv_path.exists():
        print(f"Loading cached data: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
        return df

    print("Downloading NASA POWER daily data...")
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(NASA_POWER_PARAMETERS),
        "community": "AG",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    raw = response.json()["properties"]["parameter"]

    df = pd.DataFrame({key: pd.Series(value) for key, value in raw.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.sort_index().replace(-999, np.nan).ffill().bfill()
    df.to_csv(csv_path)
    print(f"Saved data: {csv_path}")
    return df


def add_hydrology_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create runoff proxy and time-series features.

    We use the SCS Curve Number idea as a simple hydrological approximation:
    rainfall first satisfies initial abstraction; remaining water becomes runoff.

    Because this is a resume self-project without local observed discharge data, the
    target is named a 'runoff-event proxy', not measured river discharge.
    """
    df = df.copy()
    rainfall = df["PRECTOTCORR"].to_numpy(dtype=float)

    # Antecedent rainfall decides whether the catchment is dry/normal/wet.
    previous_5day = (
        pd.Series(rainfall, index=df.index)
        .rolling(5, min_periods=1)
        .sum()
        .shift(1)
        .fillna(0)
        .to_numpy()
    )

    # Educational AMC classes for an urban/suburban catchment.
    curve_number = np.where(previous_5day < 35, 80, np.where(previous_5day > 53, 92, 86))
    storage_s = 25400 / curve_number - 254  # mm
    initial_abstraction = 0.2 * storage_s
    runoff = np.where(
        rainfall > initial_abstraction,
        (rainfall - initial_abstraction) ** 2 / (rainfall + 0.8 * storage_s),
        0.0,
    )

    df["curve_number"] = curve_number
    df["runoff_mm"] = runoff
    df["rain_5day_mm"] = pd.Series(rainfall, index=df.index).rolling(5, min_periods=1).sum()
    df["runoff_event"] = (df["runoff_mm"] >= RUNOFF_EVENT_THRESHOLD_MM).astype(int)

    # Seasonal cycle as numeric variables. This helps the model learn monsoon seasonality.
    day_of_year = df.index.dayofyear.to_numpy()
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366)

    # Forecast target: whether tomorrow's runoff proxy crosses the selected threshold.
    df["target_runoff_event_tomorrow"] = df["runoff_event"].shift(-1)
    return df


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    X_train: np.ndarray
    y_train: np.ndarray
    dates_train: pd.DatetimeIndex
    X_val: np.ndarray
    y_val: np.ndarray
    dates_val: pd.DatetimeIndex
    X_test: np.ndarray
    y_test: np.ndarray
    dates_test: pd.DatetimeIndex
    persistence_test: np.ndarray
    seasonal_test: np.ndarray


def make_lstm_dataset(df: pd.DataFrame) -> DatasetBundle:
    """Create 30-day sequences and time-based train/validation/test splits."""
    df = df.dropna(subset=["target_runoff_event_tomorrow"]).copy()
    feature_array = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    target_array = df["target_runoff_event_tomorrow"].to_numpy(dtype=int)
    current_runoff_event = df["runoff_event"].to_numpy(dtype=int)
    all_dates = df.index

    X, y, forecast_dates, persistence = [], [], [], []
    for end_idx in range(LOOKBACK_DAYS - 1, len(df) - 1):
        X.append(feature_array[end_idx - LOOKBACK_DAYS + 1 : end_idx + 1])
        y.append(target_array[end_idx])
        # The forecast date is tomorrow, because y is tomorrow's event.
        forecast_dates.append(all_dates[end_idx + 1])
        persistence.append(current_runoff_event[end_idx])

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    persistence = np.array(persistence, dtype=int)

    train_mask = forecast_dates.year <= 2020
    val_mask = (forecast_dates.year >= 2021) & (forecast_dates.year <= 2022)
    test_mask = forecast_dates.year >= 2023

    scaler = StandardScaler()
    scaler.fit(X[train_mask].reshape(-1, X.shape[2]))
    X_scaled = scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

    # Simple baseline: predict event on all monsoon dates.
    seasonal = np.isin(forecast_dates.month, [6, 7, 8, 9]).astype(int)

    return DatasetBundle(
        df=df,
        X_train=X_scaled[train_mask],
        y_train=y[train_mask],
        dates_train=forecast_dates[train_mask],
        X_val=X_scaled[val_mask],
        y_val=y[val_mask],
        dates_val=forecast_dates[val_mask],
        X_test=X_scaled[test_mask],
        y_test=y[test_mask],
        dates_test=forecast_dates[test_mask],
        persistence_test=persistence[test_mask],
        seasonal_test=seasonal[test_mask],
    )


# ------------------------------
# 4. Evaluation utilities and plots
# ------------------------------

def choose_threshold(y_val: np.ndarray, prob_val: np.ndarray) -> tuple[float, dict]:
    """Choose a probability threshold that maximizes validation F1-score."""
    thresholds = np.linspace(0.05, 0.95, 181)
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in thresholds:
        pred = (prob_val >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, pred, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    return best["threshold"], best


def binary_metrics(y_true: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "event_rate": float(np.mean(y_true)),
    }
    if prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["average_precision"] = float(average_precision_score(y_true, prob))
    return out


def save_metrics(metrics: dict) -> None:
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_monthly_climatology(df: pd.DataFrame) -> None:
    monthly = df.groupby(df.index.month).agg(
        mean_rain_mm=("PRECTOTCORR", "sum"),
        mean_runoff_mm=("runoff_mm", "sum"),
        event_days=("runoff_event", "sum"),
    )
    years = df.index.year.nunique()
    monthly["mean_rain_mm"] /= years
    monthly["mean_runoff_mm"] /= years
    monthly["event_days"] /= years

    fig, ax1 = plt.subplots(figsize=(10, 5))
    months = np.arange(1, 13)
    ax1.bar(months - 0.2, monthly["mean_rain_mm"], width=0.4, label="Rainfall", color="#4C78A8")
    ax1.bar(months + 0.2, monthly["mean_runoff_mm"], width=0.4, label="Runoff proxy", color="#F58518")
    ax1.set_ylabel("Average monthly depth (mm)")
    ax1.set_xlabel("Month")
    ax1.set_xticks(months)
    ax1.set_title("Powai/Mumbai hydro-climatology: strong June–September monsoon signal")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(months, monthly["event_days"], color="#54A24B", marker="o", label="Runoff-event days")
    ax2.set_ylabel("Average event days per month")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "01_monthly_rainfall_runoff_climatology.png", dpi=180)
    plt.close(fig)


def plot_annual_trend(df: pd.DataFrame) -> None:
    annual = df.groupby(df.index.year).agg(
        rain_mm=("PRECTOTCORR", "sum"), runoff_mm=("runoff_mm", "sum"), event_days=("runoff_event", "sum")
    )
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(annual.index, annual["rain_mm"], marker="o", color="#4C78A8", label="Annual rainfall")
    ax1.plot(annual.index, annual["runoff_mm"], marker="o", color="#F58518", label="Annual runoff proxy")
    ax1.set_ylabel("Annual depth (mm)")
    ax1.set_xlabel("Year")
    ax1.set_title("Annual rainfall and runoff-proxy variability near IIT Bombay/Powai")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.bar(annual.index, annual["event_days"], color="#54A24B", alpha=0.25, label="Event days")
    ax2.set_ylabel("Runoff-event days")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "02_annual_rainfall_runoff_variability.png", dpi=180)
    plt.close(fig)


def plot_monsoon_sample(df: pd.DataFrame) -> None:
    sample = df.loc["2025-06-01":"2025-10-15"]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(sample.index, sample["PRECTOTCORR"], color="#4C78A8", alpha=0.65, label="Rainfall")
    ax1.set_ylabel("Rainfall (mm/day)")
    ax1.set_title("2025 monsoon sample: daily rainfall converted to runoff proxy")
    ax2 = ax1.twinx()
    ax2.plot(sample.index, sample["runoff_mm"], color="#F58518", linewidth=2, label="Runoff proxy")
    ax2.axhline(RUNOFF_EVENT_THRESHOLD_MM, color="red", linestyle="--", linewidth=1.4, label="Event threshold")
    ax2.set_ylabel("Runoff proxy (mm/day)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "03_2025_monsoon_rainfall_to_runoff_proxy.png", dpi=180)
    plt.close(fig)


def plot_training_loss(history: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Training loss")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted BCE loss")
    ax.set_title("LSTM learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "04_lstm_training_loss.png", dpi=180)
    plt.close(fig)


def plot_test_probability(predictions: pd.DataFrame, threshold: float) -> None:
    # Zoom to a monsoon period, where most hydrological decisions matter.
    zoom = predictions.loc["2025-06-01":"2025-10-15"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(zoom.index, zoom["lstm_probability"], color="#4C78A8", linewidth=2, label="Predicted probability")
    ax.fill_between(zoom.index, 0, zoom["actual_event"], step="mid", alpha=0.25, color="#F58518", label="Actual event (0/1)")
    ax.axhline(threshold, color="red", linestyle="--", label=f"Selected threshold = {threshold:.3f}")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Probability / event")
    ax.set_xlabel("Forecast date")
    ax.set_title("LSTM next-day runoff-event forecast: 2025 monsoon zoom")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "05_lstm_test_probabilities_2025_monsoon_zoom.png", dpi=180)
    plt.close(fig)


def plot_roc_pr(y_test: np.ndarray, prob_test: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_test, prob_test)
    precision, recall, _ = precision_recall_curve(y_test, prob_test)
    auc = roc_auc_score(y_test, prob_test)
    ap = average_precision_score(y_test, prob_test)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(fpr, tpr, color="#4C78A8", label=f"ROC-AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend()

    axes[1].plot(recall, precision, color="#F58518", label=f"Average precision = {ap:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–recall curve")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "06_lstm_roc_and_precision_recall_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrix(y_test: np.ndarray, pred_test: np.ndarray) -> None:
    cm = confusion_matrix(y_test, pred_test)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No event", "Event"],
        yticklabels=["No event", "Event"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Test confusion matrix: 2023–2025")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "07_lstm_confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_monthly_test_summary(predictions: pd.DataFrame) -> None:
    monthly = predictions.groupby(predictions.index.month).agg(
        actual_event_rate=("actual_event", "mean"), predicted_event_rate=("lstm_prediction", "mean")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    months = np.arange(1, 13)
    ax.bar(months - 0.18, monthly.reindex(months)["actual_event_rate"], width=0.36, label="Actual event rate", color="#F58518")
    ax.bar(months + 0.18, monthly.reindex(months)["predicted_event_rate"], width=0.36, label="Predicted event rate", color="#4C78A8")
    ax.set_xticks(months)
    ax.set_xlabel("Month")
    ax.set_ylabel("Event rate in test period")
    ax.set_title("Does the model capture monsoon seasonality? Test period: 2023–2025")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "08_monthly_actual_vs_predicted_event_rate.png", dpi=180)
    plt.close(fig)


# ------------------------------
# 5. Main workflow
# ------------------------------

def main() -> None:
    raw_df = download_nasa_power_data()
    df = add_hydrology_features(raw_df)
    df.to_csv(DATA_DIR / "processed_powai_hydroclimate_2010_2025.csv")

    bundle = make_lstm_dataset(df)
    print("\nDataset summary")
    print(f"Train: {bundle.X_train.shape}, event rate={bundle.y_train.mean():.3f}")
    print(f"Validation: {bundle.X_val.shape}, event rate={bundle.y_val.mean():.3f}")
    print(f"Test: {bundle.X_test.shape}, event rate={bundle.y_test.mean():.3f}")

    model = NumpyLSTMClassifier(
        input_size=bundle.X_train.shape[2], hidden_size=16, learning_rate=0.003, seed=42
    )
    history = model.fit(
        bundle.X_train,
        bundle.y_train,
        bundle.X_val,
        bundle.y_val,
        epochs=20,
        batch_size=64,
    )
    history.to_csv(RESULTS_DIR / "training_history.csv", index=False)

    val_prob = model.predict_proba(bundle.X_val)
    threshold, threshold_report = choose_threshold(bundle.y_val, val_prob)

    test_prob = model.predict_proba(bundle.X_test)
    test_pred = (test_prob >= threshold).astype(int)

    predictions = pd.DataFrame(
        {
            "actual_event": bundle.y_test,
            "lstm_probability": test_prob,
            "lstm_prediction": test_pred,
            "persistence_baseline": bundle.persistence_test,
            "seasonal_monsoon_baseline": bundle.seasonal_test,
        },
        index=bundle.dates_test,
    )
    predictions.index.name = "forecast_date"
    predictions.to_csv(RESULTS_DIR / "test_predictions_2023_2025.csv")

    metrics = {
        "project_location": "Powai / IIT Bombay, Mumbai, India",
        "data_source": "NASA POWER daily point data",
        "period": f"{START_DATE} to {END_DATE}",
        "lookback_days": LOOKBACK_DAYS,
        "target": f"Next-day runoff proxy >= {RUNOFF_EVENT_THRESHOLD_MM} mm",
        "features": FEATURE_COLUMNS,
        "train_period": "2010-2020",
        "validation_period": "2021-2022",
        "test_period": "2023-2025",
        "validation_threshold_choice": threshold_report,
        "selected_probability_threshold": threshold,
        "lstm_test_metrics": binary_metrics(bundle.y_test, test_pred, test_prob),
        "persistence_baseline_test_metrics": binary_metrics(bundle.y_test, bundle.persistence_test),
        "seasonal_monsoon_baseline_test_metrics": binary_metrics(bundle.y_test, bundle.seasonal_test),
    }
    save_metrics(metrics)

    # Plots
    plot_monthly_climatology(df)
    plot_annual_trend(df)
    plot_monsoon_sample(df)
    plot_training_loss(history)
    plot_test_probability(predictions, threshold)
    plot_roc_pr(bundle.y_test, test_prob)
    plot_confusion_matrix(bundle.y_test, test_pred)
    plot_monthly_test_summary(predictions)

    print("\nSelected threshold from validation set:", round(threshold, 3))
    print("LSTM test metrics:")
    for key, value in metrics["lstm_test_metrics"].items():
        print(f"  {key}: {value}")
    print(f"\nSaved results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
