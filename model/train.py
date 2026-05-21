"""
LSTM Stock Price Prediction - Training Script
Tech Challenge Fase 4 - Machine Learning Engineering
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "symbol": "PETR4.SA",        # Petrobras - ação brasileira
    "start_date": "2018-01-01",
    "end_date": "2024-12-31",
    "sequence_length": 60,        # Janela temporal: 60 dias de histórico
    "train_ratio": 0.8,
    "val_ratio": 0.1,             # 80% treino, 10% val, 10% teste
    "lstm_units": [128, 64],      # Neurônios por camada LSTM
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "patience": 15,               # EarlyStopping patience
    "features": ["Open", "High", "Low", "Close", "Volume"],
    "target": "Close",
    "model_dir": "model/artifacts",
}

os.makedirs(CONFIG["model_dir"], exist_ok=True)


# ─────────────────────────────────────────────
# 2. DATA COLLECTION
# ─────────────────────────────────────────────
def collect_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical stock data from Yahoo Finance."""
    print(f"\n📥 Coletando dados de {symbol} ({start_date} → {end_date})...")
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        raise ValueError(f"Nenhum dado encontrado para {symbol}.")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[CONFIG["features"]].dropna()
    print(f"✅ {len(df)} registros coletados | Colunas: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, seq_len: int, train_ratio: float, val_ratio: float):
    """Scale data and create sequences for LSTM."""
    print("\n🔧 Pré-processando dados...")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)

    # Save scaler for inference
    joblib.dump(scaler, os.path.join(CONFIG["model_dir"], "scaler.pkl"))

    # Save feature/target column info
    feature_cols = CONFIG["features"]
    target_idx = feature_cols.index(CONFIG["target"])
    meta = {
        "features": feature_cols,
        "target": CONFIG["target"],
        "target_idx": target_idx,
        "sequence_length": seq_len,
        "symbol": CONFIG["symbol"],
    }
    with open(os.path.join(CONFIG["model_dir"], "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Build sequences: X = seq_len days, y = next day Close
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])          # (seq_len, n_features)
        y.append(scaled[i, target_idx])           # scalar

    X, y = np.array(X), np.array(y)

    # Split
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx


# ─────────────────────────────────────────────
# 4. MODEL ARCHITECTURE
# ─────────────────────────────────────────────
def build_model(seq_len: int, n_features: int) -> tf.keras.Model:
    """Build a stacked LSTM model."""
    model = Sequential([
        Input(shape=(seq_len, n_features)),

        LSTM(CONFIG["lstm_units"][0], return_sequences=True),
        Dropout(CONFIG["dropout_rate"]),

        LSTM(CONFIG["lstm_units"][1], return_sequences=False),
        Dropout(CONFIG["dropout_rate"]),

        Dense(32, activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        loss="huber",          # Mais robusto a outliers do que MSE
        metrics=["mae"],
    )
    model.summary()
    return model


# ─────────────────────────────────────────────
# 5. TRAINING
# ─────────────────────────────────────────────
def train(model, X_train, y_train, X_val, y_val):
    """Train model with callbacks."""
    print("\n🏋️  Treinando modelo...")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=CONFIG["patience"],
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(CONFIG["model_dir"], "lstm_best.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                          min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, scaler, target_idx, n_features):
    """Compute MAE, RMSE, MAPE on test set."""
    print("\n📊 Avaliando modelo no conjunto de teste...")

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform — rebuild full array, replace target col
    def inverse(arr_scaled):
        dummy = np.zeros((len(arr_scaled), n_features))
        dummy[:, target_idx] = arr_scaled
        return scaler.inverse_transform(dummy)[:, target_idx]

    y_true = inverse(y_test)
    y_pred = inverse(y_pred_scaled)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
               "MAPE": round(mape, 4), "R2": round(r2, 4)}
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    with open(os.path.join(CONFIG["model_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return y_true, y_pred, metrics


# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────
def plot_results(history, y_true, y_pred, df):
    """Generate training loss and prediction plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"LSTM Stock Prediction — {CONFIG['symbol']}", fontsize=14, fontweight="bold")

    # Loss curve
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Predictions vs Actual
    axes[1].plot(y_true, label="Preço Real", alpha=0.8)
    axes[1].plot(y_pred, label="Previsão LSTM", alpha=0.8, linestyle="--")
    axes[1].set_title("Preço Real vs Previsão (Conjunto de Teste)")
    axes[1].set_xlabel("Dias")
    axes[1].set_ylabel("Preço (BRL)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["model_dir"], "results.png"), dpi=150)
    print(f"\n📈 Gráfico salvo em {CONFIG['model_dir']}/results.png")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    # Collect
    df = collect_data(CONFIG["symbol"], CONFIG["start_date"], CONFIG["end_date"])

    # Preprocess
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, target_idx = preprocess(
        df, CONFIG["sequence_length"], CONFIG["train_ratio"], CONFIG["val_ratio"]
    )

    n_features = len(CONFIG["features"])

    # Build & train
    model = build_model(CONFIG["sequence_length"], n_features)
    history = train(model, X_train, y_train, X_val, y_val)

    # Save last window for API fallback (used when yfinance is unavailable)
    last_window = X_test[-1]   # shape (seq_len, n_features)
    np.save(os.path.join(CONFIG["model_dir"], "last_window.npy"), last_window)
    print(f"💾 Janela de fallback salva.")

    # Save final model
    model.save(os.path.join(CONFIG["model_dir"], "lstm_final.keras"))
    print(f"\n💾 Modelo salvo em {CONFIG['model_dir']}/lstm_final.keras")

    # Evaluate
    y_true, y_pred, metrics = evaluate(model, X_test, y_test, scaler, target_idx, n_features)

    # Plot
    plot_results(history, y_true, y_pred, df)

    print("\n✅ Treinamento concluído!")
    print(f"   Métricas: {metrics}")


if __name__ == "__main__":
    main()
