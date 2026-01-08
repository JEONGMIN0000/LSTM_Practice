import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import glob
import os
import re
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


# -------------------------
# 공통 유틸
# -------------------------
def load_data(pattern):
    file_list = glob.glob(pattern)
    data = []
    for path in sorted(
        file_list,
        key=lambda x: (
            int(re.search(r"(\d{4})", os.path.basename(x)).group(1)),
            int(re.search(r"(\d+)\s*번", os.path.basename(x)).group(1)),
        ),
    ):
        df = pd.read_csv(path)
        data.append(df)
        print(f"[load] load: {path}, shape={df.shape}")
    return data


def preprocess(df, target_col):
    df = df.copy()

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if target_col not in df.columns:
        raise ValueError(f"CSV에 target_col이 없습니다: {target_col}\n현재 컬럼: {list(df.columns)}")

    return df


def split_dataset(data):
    total_len = len(data)
    train_len = int(total_len * 0.7)
    valid_len = int(total_len * 0.2)

    train = data[:train_len]
    valid = data[train_len: train_len + valid_len]
    test  = data[train_len + valid_len:]

    print(f"[split events] total={total_len}, train={len(train)}, valid={len(valid)}, test={len(test)}")
    return train, valid, test


def data_scalers(train_events, feature_cols, target_col):
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_concat = pd.concat([df[feature_cols] for df in train_events], axis=0)
    y_concat = pd.concat([df[[target_col]] for df in train_events], axis=0)

    scaler_x.fit(X_concat.values)
    scaler_y.fit(y_concat.values)
    return scaler_x, scaler_y


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name)


# ============================================================
# ✅ 1-step 버전 (Dense(1) + rollout)
# ============================================================
def create_dataset_1step(df, feature_cols, target_col, scaler_x, scaler_y, seq_length=36):
    X_scaled = scaler_x.transform(df[feature_cols].values)
    y_scaled = scaler_y.transform(df[[target_col]].values).reshape(-1)

    X, y = [], []
    start_t = seq_length
    end_t = len(df)  # 1-step

    for i in range(start_t, end_t):
        X.append(X_scaled[i-seq_length:i, :])
        y.append([y_scaled[i]])  # (1,)

    return np.array(X, np.float32), np.array(y, np.float32)  # y:(N,1)


def build_xy_1step(events, feature_cols, target_col, scaler_x, scaler_y, seq_length=36):
    X_list, y_list = [], []
    for df in events:
        X, y = create_dataset_1step(df, feature_cols, target_col, scaler_x, scaler_y, seq_length)
        if len(X) == 0:
            continue
        X_list.append(X)
        y_list.append(y)

    if len(X_list) == 0:
        return (
            np.empty((0, seq_length, len(feature_cols)), np.float32),
            np.empty((0, 1), np.float32),
        )

    return np.concatenate(X_list), np.concatenate(y_list)


def make_lstm_model_1step(seq_length, n_features, learning_rate=3e-4, dropout=0.0, hidden_units=64):
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=(seq_length, n_features), return_sequences=False, dropout=dropout, recurrent_dropout=0.0))
    model.add(Dense(1))

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(loss="mean_squared_error", optimizer=opt)
    return model


def rollout_predict(model, df, feature_cols, target_col, scaler_x, scaler_y,
                    seq_length=36, step=36, start_idx=36):
    """
    앞 36행 입력으로 시작해서 뒤 36행을 순차 예측(rollout).
    미래 강우/티아이 등은 df에 있는 값을 그대로 쓰고,
    수위(target_col)만 예측값을 피드백함.
    """
    if target_col not in feature_cols:
        raise ValueError("rollout 하려면 feature_cols에 target_col(수위)이 포함되어야 합니다.")

    raw = df.reset_index(drop=True)
    if len(raw) < start_idx + step:
        return None

    y_true = raw.loc[start_idx:start_idx+step-1, target_col].to_numpy().astype(np.float32)

    src = raw.copy()
    preds = []

    for s in range(step):
        i = start_idx + s
        X_win = src.loc[i-seq_length:i-1, feature_cols].to_numpy()
        X_scaled = scaler_x.transform(X_win).astype(np.float32).reshape(1, seq_length, len(feature_cols))

        pred_scaled = model.predict(X_scaled, verbose=0).reshape(-1, 1)  # (1,1)
        pred = scaler_y.inverse_transform(pred_scaled).reshape(-1)[0]
        preds.append(float(pred))

        src.loc[i, target_col] = pred  # 피드백

    y_pred = np.array(preds, np.float32)
    return y_pred, y_true


# ============================================================
# ✅ multi-step 버전 (Dense(fcast_length) + one-shot)
# ============================================================
def create_dataset_36step(df, feature_cols, target_col, scaler_x, scaler_y, seq_length=36, fcast_length=36):
    X_scaled = scaler_x.transform(df[feature_cols].values)
    y_scaled = scaler_y.transform(df[[target_col]].values).reshape(-1)

    X, y = [], []
    start_t = seq_length
    end_t = len(df) - fcast_length + 1  # 미래 fcast_length 확보

    for i in range(start_t, end_t):
        X.append(X_scaled[i-seq_length:i, :])
        y.append(y_scaled[i:i+fcast_length])  # (fcast_length,)

    return np.array(X, np.float32), np.array(y, np.float32)  # y:(N,fcast_length)


def build_xy_36step(events, feature_cols, target_col, scaler_x, scaler_y, seq_length=36, fcast_length=36):
    X_list, y_list = [], []
    for df in events:
        X, y = create_dataset_36step(df, feature_cols, target_col, scaler_x, scaler_y, seq_length, fcast_length)
        if len(X) == 0:
            continue
        X_list.append(X)
        y_list.append(y)

    if len(X_list) == 0:
        return (
            np.empty((0, seq_length, len(feature_cols)), np.float32),
            np.empty((0, fcast_length), np.float32),
        )

    return np.concatenate(X_list), np.concatenate(y_list)


def make_lstm_model_36step(seq_length, n_features, fcast_length=36, learning_rate=3e-4, dropout=0.0, hidden_units=64):
    model = Sequential()
    model.add(LSTM(hidden_units, input_shape=(seq_length, n_features), return_sequences=False, dropout=dropout, recurrent_dropout=0.0))
    model.add(Dense(fcast_length))  # ✅ 한 번에 36개 출력

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(loss="mean_squared_error", optimizer=opt)
    return model


def predict_one_shot(model, df, feature_cols, target_col, scaler_x, scaler_y, seq_length=36, fcast_length=36, start_idx=36):
    """
    testdata가 72행이면:
    입력: 0~35 (seq_length)
    정답: 36~71 (fcast_length)
    """
    raw = df.reset_index(drop=True)
    if len(raw) < start_idx + fcast_length:
        return None

    X_win = raw.loc[start_idx-seq_length:start_idx-1, feature_cols].to_numpy()
    X_scaled = scaler_x.transform(X_win).astype(np.float32).reshape(1, seq_length, len(feature_cols))

    pred_scaled = model.predict(X_scaled, verbose=0)   # (1,fcast_length)
    y_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1).astype(np.float32)

    y_true = raw.loc[start_idx:start_idx+fcast_length-1, target_col].to_numpy().astype(np.float32)
    return y_pred, y_true


# -------------------------
# 실행 (모드 선택)
# -------------------------
if __name__ == "__main__":

    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    train_pattern = "./data/csv/*.csv"
    test_pattern  = "./testdata/*.csv"

    # ===== 설정 =====
    MODE = "1step"      # ✅ "1step" 또는 "36step"
    seq_length   = 36
    learning_rate = 3e-4
    dropout = 0.0
    target = "gn"  # "dg"
    # ===============
    fcast_length = 36    # ✅ 숫자 유지 (36step일 때 사용)


    if target == "gn":
        target_col = "성남시(궁내교)_WL"
    elif target == "dg":
        target_col = "서울시(대곡교)_WL"
    else:
        raise ValueError("target 설정 오류")

    feature_cols = [
        "성남시(한국학중앙연구원)","성남시(대장동)","성남시(구미초교)","서울시(대곡교)","성남시(성남북초교)","광주시(남한산초교)",
        "궁내교_Ti","대곡교_Ti",target_col,
    ]

    # 1) 학습 데이터 로드/전처리
    train_raw = load_data(train_pattern)
    train_events_all = [preprocess(df, target_col) for df in train_raw]
    train_events, valid_events, _ = split_dataset(train_events_all)

    # 2) scaler fit (train only)
    scaler_x, scaler_y = data_scalers(train_events, feature_cols, target_col)

    # 3) 윈도우 생성 + 모델 
    if MODE == "1step":  # 1step
        X_train, y_train = build_xy_1step(train_events, feature_cols, target_col, scaler_x, scaler_y, seq_length)
        X_valid, y_valid = build_xy_1step(valid_events, feature_cols, target_col, scaler_x, scaler_y, seq_length)
        model = make_lstm_model_1step(seq_length, X_train.shape[2], learning_rate, dropout, hidden_units=64)

    elif MODE == "36step":  # 36step
        X_train, y_train = build_xy_36step(train_events, feature_cols, target_col, scaler_x, scaler_y, seq_length, fcast_length)
        X_valid, y_valid = build_xy_36step(valid_events, feature_cols, target_col, scaler_x, scaler_y, seq_length, fcast_length)
        model = make_lstm_model_36step(seq_length, X_train.shape[2], fcast_length, learning_rate, dropout, hidden_units=64)

    else:
        raise ValueError("MODE는 '1step' 또는 '36step'")


    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)

    # 4) 모델 학습
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=100, batch_size=32, shuffle=False,
        callbacks=[es, rlr], verbose=1
    )

    # 학습 곡선 저장
    os.makedirs("./result_png", exist_ok=True)
    plt.figure(figsize=(18, 5))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["train", "val"])
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig("./result_png/train_loss.png")
    plt.close()

    # 5) 테스트 (파일별 평가)
    test_files = sorted(glob.glob(test_pattern))
    print(f"[test] files={len(test_files)}")

    results = []
    for fp in test_files:
        base = os.path.basename(fp)
        df = preprocess(pd.read_csv(fp), target_col)

        if MODE == "1step":
            out = rollout_predict(model, df, feature_cols, target_col, scaler_x, scaler_y,seq_length=seq_length, step=fcast_length, start_idx=seq_length)
        else:
            out = predict_one_shot(model, df, feature_cols, target_col, scaler_x, scaler_y,seq_length=seq_length, fcast_length=fcast_length, start_idx=seq_length)

        if out is None:
            print(f"[SKIP] {base} : 길이 부족")
            continue

        y_pred, y_true = out

        mse = float(np.mean((y_pred - y_true) ** 2))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        # 그래프 저장
        plt.figure(figsize=(14, 5))
        plt.plot(y_true, label="True")
        plt.plot(y_pred, label="Pred")
        plt.title(f"{MODE} 예측 (36-step)")
        plt.xlabel("step")
        plt.ylabel(target_col)
        plt.legend()
        plt.tight_layout()

        fig_path = f"./result_png/{safe_filename(os.path.splitext(base)[0])}_{MODE}.png"
        plt.savefig(fig_path)
        plt.close()

        print(f"\n[Test: {base}]  MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        results.append((base, mse, mae, rmse, r2))

    if results:
        arr = np.array([r[1:] for r in results], dtype=float)
        print("\n[SUMMARY] 파일별 평균")
        print(f" MSE={arr[:,0].mean():.4f}, MAE={arr[:,1].mean():.4f}, RMSE={arr[:,2].mean():.4f}, R2={arr[:,3].mean():.4f}")
