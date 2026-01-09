# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# 2. 시퀀스 생성 함수
def create_sequences(X_data, y_data, seq_length):
    """
    X_data : (N, n_features) 형태의 numpy 배열 (스케일된 값)
    y_data : (N, 1) 형태의 numpy 배열 (스케일된 타깃)
    seq_length : 윈도우 크기 (예: 7이면 7일 보고 다음 날 예측)

    return
    X : (samples, seq_length, n_features)
    y : (samples,)
    """
    X, y = [], []
    for i in range(len(X_data) - seq_length):
        X.append(X_data[i : i + seq_length, :])
        y.append(y_data[i + seq_length, 0])  # 다음 시점의 타깃값
    return np.array(X), np.array(y)


# 3. 데이터 로드 및 전처리
def load_data(window_size, csv_path="weather_data.csv"):
    # 3-1. 데이터 읽기
    raw_data = pd.read_csv(csv_path)

    # === 여기만 나중에 수위/강우로 바꾸면 됨 ===
    features = ["Temp", "Humidity", "Rain", "Wind"]  # 입력으로 쓸 컬럼들
    target_col = "Temp"  # 예측할 컬럼
    # ==========================================

    # 3-2. 입력(X), 타깃(y) 분리
    X_df = raw_data[features].copy()
    y_df = raw_data[[target_col]].copy()  # (N, 1) 형태 유지

    # 3-3. 스케일링
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_df)  # (N, n_features)
    y_scaled = scaler_y.fit_transform(y_df)  # (N, 1)

    # 3-4. 시퀀스 생성
    X, y = create_sequences(X_scaled, y_scaled, window_size)

    # 3-5. train/test 분리 (시간 순서 유지)
    total_len = len(X)
    train_len = int(total_len * 0.7)

    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, features, target_col


# 4. 모델 정의
def build_lstm_model(window_size, n_features, hidden_units=64):
    model = Sequential()
    model.add(LSTM(hidden_units, activation="tanh", input_shape=(window_size, n_features)))
    model.add(Dense(1))  # 타깃 1개
    return model


def build_gru_model(window_size, n_features, hidden_units=64):
    model = Sequential()
    model.add(GRU(hidden_units, activation="tanh", input_shape=(window_size, n_features)))
    model.add(Dense(1))
    return model


# 5. 학습 + 평가 함수
def run_model(model, X_train, X_test, y_train, y_test, epochs=30, batch_size=32, model_name="model",):
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_split=0.2, callbacks=[early_stop],)

    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"[{model_name}] Test MSE (scaled): {test_loss:.6f}")

    # 예측 (스케일 상태)
    y_pred_scaled = model.predict(X_test)

    return test_loss, history, y_pred_scaled


# 6. 실제 스케일(원 단위)로 평가 지표 계산
def evaluate_on_original_scale(y_test_scaled, y_pred_scaled, scaler_y, model_name="model"):
    # (N,) -> (N, 1)로 reshape 후 inverse_transform
    y_test_scaled = y_test_scaled.reshape(-1, 1)
    y_pred_scaled = y_pred_scaled.reshape(-1, 1)

    y_test = scaler_y.inverse_transform(y_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n[{model_name}] Original scale 평가 지표")
    print(f"  MSE : {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  R2  : {r2:.4f}")

    # 앞부분 일부 출력
    print(f"\n[{model_name}] 실제 vs 예측 (앞 10개)")
    for i in range(min(10, len(y_test))):
        print(f"  실제: {y_test[i, 0]:.2f}, 예측: {y_pred[i, 0]:.2f}")

    return mse, rmse, mae, r2


# 7. main 함수
def main(window_size=7, csv_path="weather_data.csv"):
    tf.random.set_seed(2020)
    np.random.seed(2020)

    # 데이터 로드
    X_train, X_test, y_train, y_test, scaler_X, scaler_y, features, target_col = (load_data(window_size, csv_path))

    n_features = X_train.shape[2]

    # LSTM
    lstm_model = build_lstm_model(window_size, n_features, hidden_units=64)
    lstm_test_loss, lstm_history, lstm_y_pred_scaled = run_model(lstm_model, X_train,X_test, y_train, y_test, epochs=30, batch_size=32, model_name="LSTM",)
    lstm_mse, lstm_rmse, lstm_mae, lstm_r2 = evaluate_on_original_scale(y_test, lstm_y_pred_scaled, scaler_y, model_name="LSTM")

    # GRU
    gru_model = build_gru_model(window_size, n_features, hidden_units=64)
    gru_test_loss, gru_history, gru_y_pred_scaled = run_model(gru_model, X_train, X_test, y_train, y_test, epochs=30, batch_size=32, model_name="GRU",)
    gru_mse, gru_rmse, gru_mae, gru_r2 = evaluate_on_original_scale(y_test, gru_y_pred_scaled, scaler_y, model_name="GRU")

    print("\n" + "=" * 20, f"시계열 길이 {window_size}인 경우", "=" * 20)
    print(f"[LSTM] Test MSE (scaled) = {lstm_test_loss:.6f}")
    print(f"[GRU ] Test MSE (scaled) = {gru_test_loss:.6f}")

    return {
        "lstm": {
            "test_mse_scaled": lstm_test_loss,
            "mse": lstm_mse,
            "rmse": lstm_rmse,
            "mae": lstm_mae,
            "r2": lstm_r2,
        },
        "gru": {
            "test_mse_scaled": gru_test_loss,
            "mse": gru_mse,
            "rmse": gru_rmse,
            "mae": gru_mae,
            "r2": gru_r2,
        },
    }


# 8. 스크립트 실행
if __name__ == "__main__":
    result = main(window_size=7, csv_path="weather_data.csv")
