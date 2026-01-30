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


# 데이터 로드 및 전처리
def load_data(pattern):

    file_list = glob.glob(pattern)

    data = []

    # dataset 이벤트 순서대로
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


# 전처리 : time 정리 , 숫자 변환
def preprocess(df, target_col):

    df = df.copy()

    # time 컬럼 정리
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        print("⚠ DateTime 컬럼이 없음.")

    # 숫자형 변환(시간 제외 전부)
    for c in df.columns:
        if c != "time":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    missing = [c for c in [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다 : {missing}\n 현재 컬럼 : {list(df.columns)}")

    return df


# 데이터셋 분리
def split_dataset(data):

    total_len = len(data)
    train_len = int(total_len * 0.7)
    valid_len = int(total_len * 0.2)

    train = data[:train_len]
    valid = data[train_len : train_len + valid_len]
    test = data[train_len + valid_len :]

    print(f"[split events] total={total_len}, train={len(train)}, valid={len(valid)}, test={len(test)}")

    return train, valid, test


# MinMaxScaler 로 데이터 정규화
def data_scalers(train_events, feature_cols, target_col):

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_concat = pd.concat([df[feature_cols] for df in train_events], axis=0)
    y_concat = pd.concat([df[[target_col]] for df in train_events], axis=0)

    scaler_x.fit(X_concat.values)
    scaler_y.fit(y_concat.values)

    return scaler_x, scaler_y


# 데이터셋 생성
#    X: (N, seq_len, n_features)       y: (N, 1)
def create_dataset(df, feature_cols, target_col, scaler_x, scaler_y, seq_length):

    X_scaled = scaler_x.transform(df[feature_cols].values)  # (T, n_feat)
    y_scaled = scaler_y.transform(df[[target_col]].values).reshape(-1)  # (T,)

    X, y = [], []

    start_t = seq_length
    end_t = len(df)  # 1-step 이라 끝까지 가능

    for i in range(start_t, end_t):
        X.append(X_scaled[i - seq_length : i, :])  # (seq_len, n_feat)
        # y.append(y_scaled[i:i+seq_length])         #  ✅ 36step 미래 36-step 수위 (seq_len,)
        y.append([y_scaled[i]])  # ✅ 1step (1,)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  # (N,seq_length,n1) (N,seq_length)


# 이벤트 여러개 합치기
def build_xy(events, feature_cols, target_col, scaler_x, scaler_y, seq_length):

    X_list, y_list = [], []

    for df in events:
        X, y = create_dataset(df, feature_cols, target_col, scaler_x, scaler_y, seq_length)
        if len(X) == 0:
            continue
        X_list.append(X)
        y_list.append(y)

    if len(X_list) == 0:
        return np.empty((0, seq_length, len(feature_cols)), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all


# # 모델 구축
# def build_lstm_model(seq_length, n_features, hidden_units=64):
#     """
#     단일 입력(관측/피처) 기반 단일 LSTM 시계열 예측 모델
#     입력:  (batch, seq_length, n_features)
#     출력:  (batch, seq_length)  # 예: 미래 36스텝 수위
#     """

#     inp = tf.keras.Input(shape=(seq_length, n_features), name="ts_input")

#     x = tf.keras.layers.LSTM(hidden_units, return_sequences=True)(inp)
#     x = tf.keras.layers.LSTM(hidden_units, return_sequences=False)(x)

#     x = tf.keras.layers.Dense(64, activation="relu")(x)
#     out = tf.keras.layers.Dense(seq_length, name="y_seq")(x)

#     model = tf.keras.Model(inputs=inp, outputs=out)
#     model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
#     model.summary()

#     return model


# 모델 구축 Dense(1)
# def make_lstm_model(seq_length, n_features, hidden_units=64):

#     #모델 세팅
#     model = Sequential()
#     model.add(LSTM(hidden_units, input_shape=(seq_length, n_features), return_sequences=False))# 마지막 hidden state만 사용
#     model.add(Dense(1))
#     model.compile(loss='mean_squared_error', optimizer='adam')


#큰 수위 구간에 가중치 ( 가중치가 너무 크면 평상시 성능 저하 예상 - 2.0~5.0 범위에서 시작 추천 )
@tf.function
def weighted_mse(y_true, y_pred):
    # y_true, y_pred: (batch, 1), 
    # 예: y_true가 0.7 이상(상대적으로 높은 수위)이면 가중치 3배
    w = tf.where(y_true >= 0.7, 3.0, 1.0)
    return tf.reduce_mean(w * tf.square(y_true - y_pred))


#  모델 구축 Dense(1) + learning_rate, dropout
def make_lstm_model(seq_length, n_features, learning_rate, dropout, hidden_units=64):

    model = Sequential()
    model.add(LSTM(hidden_units,input_shape=(seq_length, n_features),return_sequences=False,dropout=dropout,recurrent_dropout=0.0))
    model.add(Dense(1)) # ✅ 1step

    # learning rate 조절 + gradient 폭주 방지 (rollout에서 특히 도움)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    # ✅ 기본 loss
    model.compile(loss="mean_squared_error", optimizer=opt)

    # ✅ Huber (delta는 스케일에 따라 조절)
    # loss_fn = tf.keras.losses.Huber(delta=0.1)  # (0.01~0.1)

    # model.compile(loss=loss_fn, optimizer=opt)

    # ✅ 큰 수위 구간에 가중치
    # model.compile(loss=weighted_mse, optimizer=opt)


    return model


# 36-step rollout 예측 (y_true 오염 방지)
def rollout_predict_36(model,df,feature_cols,target_col,scaler_x,scaler_y,seq_length=36,step=36,start_idx=None,):
    """
    1-step 모델(Dense(1))로 step(=36) 을 순차 예측.
    feature_cols에 target_col 포함 필수.
    """

    if target_col not in feature_cols:
        raise ValueError("rollout 하려면 feature_cols에 target_col(수위)이 포함되어야 합니다.")

    raw_data = df.reset_index(drop=True)
    total_length = len(raw_data)

    if start_idx is None:
        start_idx = seq_length

    if total_length < start_idx + step:
        return None

    # 실제값은 원본에서 먼저 고정 (오염 방지)
    y_true_36 = (raw_data.loc[start_idx : start_idx + step - 1, target_col].to_numpy().astype(np.float32))

    # 예측용 복사본
    src_copy = raw_data.copy()

    preds = []

    for s in range(step):
        i = start_idx + s

        X_win = src_copy.loc[i - seq_length : i - 1, feature_cols].to_numpy()
        X_scaled = (scaler_x.transform(X_win).astype(np.float32).reshape(1, seq_length, len(feature_cols)))

        pred_scaled = model.predict(X_scaled, verbose=0).reshape(-1, 1)  # (1,1)
        pred = scaler_y.inverse_transform(pred_scaled).reshape(-1)[0]

        preds.append(float(pred))

        # 다음 스텝 입력을 위해 예측 수위 피드백(예측용 pred_copy 에만)
        src_copy.loc[i, target_col] = pred

    y_pred_36 = np.array(preds, dtype=np.float32)

    return y_pred_36, y_true_36


# 예측
def predict(model,df,feature_cols,target_col,scaler_x,scaler_y,seq_length,out_dir="./test_outputs",show=False,):

    os.makedirs(out_dir, exist_ok=True)

    # 윈도우 생성
    X, y = create_dataset(df, feature_cols, target_col, scaler_x, scaler_y, seq_length)

    if len(X) == 0:
        return None  # 데이터 길이가 짧아서 샘플이 안 만들어짐

    # 예측
    pred_scaled = model.predict(X, verbose=0)  # (N, seq_length)

    # inverse
    y_pred = scaler_y.inverse_transform(pred_scaled) # (N, seq_length)
    y_true = scaler_y.inverse_transform(y)  # (N, seq_length)

    # 파일명 안전화
    return y_pred, y_true


#그래프
def drawCorrectionGraph(y_real, y_pred, pred_type, save_path):
    plt.figure(figsize=(15, 6))

    start = len(y_real) - len(y_pred)  # = 36

    plt.axvline(start - 0.5, ls="--", c="gray")
    plt.plot(y_real, c="blue")
    plt.plot(range(start, start + len(y_pred)), y_pred, c="orange")

    # 래프 스케일 고정
    if "궁내교" in pred_type:
        plt.ylim(0.7, 3.0)
    elif "대곡교" in pred_type:
        plt.ylim(1.5, 5.5)

    plt.title(f"Tancheon {pred_type} Prediction")
    plt.xlabel("step")
    plt.ylabel(pred_type)
    plt.legend(["pred start", "obs", "pred"])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# 결과 엑셀 저장
def makeResultChart(data,base, testname, y_pred, r2, mse, mae, rmse):
    # ------------------------------------------------------------------ 저장 경로
    # 파일명 안전 처리 (윈도우 특수문자 제거)
    safe_base = re.sub(r'[\\/:*?"<>|]+', "_", os.path.splitext(base)[0])

    out_path = f"./testdata/result/{safe_base}_{testname}.xlsx"
    
    ROW_START = 36   # 엑셀 38행 (0-based)
    ROW_END   = 72   # 엑셀 73행 (end exclusive)
    # ROW_START = 54   # 엑셀 38행 (0-based)
    # ROW_END   = 72   # 엑셀 73행 (end exclusive)
    N_WRITE   = ROW_END - ROW_START  # 36개
    METRIC_ROW = 0   # 엑셀 2행

    pred_col= {"gn": "궁내교 예측수위", "dg": "대곡교 예측수위"}
    metric_col = {
            "gn": ["궁내교 R2", "궁내교 MSE", "궁내교 MAE", "궁내교 RMSE"],
            "dg": ["대곡교 R2", "대곡교 MSE", "대곡교 MAE", "대곡교 RMSE"],
        }

    if os.path.exists(out_path):
        print(" 기존 파일 로드 : ", out_path)
        dfs = pd.read_excel(out_path)
    else:
        print(" 신규 파일 생성 : ", out_path)
        dfs = data.copy()

    pred_col = pred_col[target]

    # 컬럼 없으면 생성
    for col in ["궁내교 예측수위", "대곡교 예측수위"]:
        if col not in dfs.columns:
            dfs[col] = np.nan

    for cols in metric_col.values():
        for c in cols:
            if c not in dfs.columns:
                dfs[c] = np.nan

    # 예측수위 값 추가
    y_pred_36 = np.asarray(y_pred).reshape(-1)
    y_pred_36 = y_pred_36[:N_WRITE]
    
    col_idx = dfs.columns.get_indexer([pred_col])[0]  # 항상 정수 1개
    dfs.iloc[ROW_START:ROW_END, col_idx] = y_pred_36

    # 평가 메트릭 값 추가
    metric_cols = metric_col[target]
    dfs.loc[METRIC_ROW, metric_cols] = [r2, mse, mae, rmse]

    # 저장
    dfs.to_excel(out_path, index=False)
    
    print("✅ 저장 완료:", out_path)


# -------------------------
# 실행
# -------------------------
if __name__ == "__main__":

    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # ---------- 변수 설정 -----------------------------------------------------------------------------------

    testname = "누적강우"  # 누적강우 / 누적강우20
    train_pattern = f"./data/csv_{testname}/*.csv"
    test_pattern = "./testdata/누적강우/*.csv"

    seq_length = 36
    learning_rate = 3e-4  # 3e-4 = 0.0003 , 1e-4 = 0.0001
    dropout = 0.0
    target = "gn" # gn / dg

    #---------------------------------------------------------------------------------------------------------

    if target == "gn" :
        target_col = "성남시(궁내교)_WL"
        rain_col = '궁내교_누적강우'
    elif target == "dg" :
        target_col = "서울시(대곡교)_WL"
        rain_col = '대곡교_누적강우'
    else :
        print("target 설정 오류")

    feature_cols = [
        "성남시(한국학중앙연구원)","성남시(대장동)","성남시(구미초교)","서울시(대곡교)","성남시(성남북초교)","광주시(남한산초교)",
        "궁내교_Ti","대곡교_Ti", rain_col ,target_col
    ]

    # 1) 학습 데이터 로드/전처리
    train_raw = load_data(train_pattern)
    train_events_all = [preprocess(df, target_col) for df in train_raw]
    train_events, valid_events, _ = split_dataset(train_events_all)

    # 2) scaler fit (train only)
    scaler_x, scaler_y = data_scalers(train_events, feature_cols, target_col)

    # 3) 윈도우 생성 (train/valid)
    rng = np.random.default_rng(SEED)
    train_events_shuffled = train_events.copy()
    rng.shuffle(train_events_shuffled)

    X_train, y_train = build_xy(train_events_shuffled, feature_cols, target_col, scaler_x, scaler_y, seq_length)
    # X_train, y_train = build_xy(train_events, feature_cols, target_col, scaler_x, scaler_y, seq_length)
    X_valid, y_valid = build_xy(valid_events, feature_cols, target_col, scaler_x, scaler_y, seq_length)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)  # y_train: (N,1)
    print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)

    # 4) 모델 학습
    # model = build_lstm_model(seq_length=seq_length, n_features=X_train.shape[2], hidden_units=64)
    # model = make_lstm_model(seq_length=seq_length, n_features=X_train.shape[2], hidden_units=64)
    model = make_lstm_model(seq_length=seq_length,n_features=X_train.shape[2],learning_rate=learning_rate,dropout=dropout,hidden_units=64,)

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    history = model.fit(X_train,y_train,validation_data=(X_valid, y_valid),
                        epochs=100,batch_size=32,shuffle=False,callbacks=[es, rlr],verbose=1,)

    # 학습 과정 Plot 저장
    os.makedirs("./result_png", exist_ok=True)
    plt.figure(figsize=(25, 8))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    print("[ Train Chart ]")
    loss_min = min(history.history["val_loss"])
    # plt.ylim(0.0, loss_min * 10)
    plt.ylim(0.0, loss_min * 10 if loss_min > 0 else 1.0)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.tight_layout()
    plt.savefig(f"./result_png/train_forecast_{target}.png")
    plt.close()

    # 5) 테스트 (파일별 36-step rollout 평가)
    test_files = sorted(glob.glob(test_pattern))
    print(f"[test] files={len(test_files)}")

    results = []
    for fp in test_files:
        base = os.path.basename(fp)
        df_raw = pd.read_csv(fp)
        df = preprocess(df_raw, target_col)

        #     res = predict(model, df, feature_cols, target_col, scaler_x, scaler_y, seq_length=seq_length, out_dir="./result_png", show=False)

        #     if res is None:
        #         print(f"[SKIP] {base} : 길이가 짧아 seq_length={seq_length} 윈도우 생성 불가")
        #         continue

        out = rollout_predict_36(model,df,feature_cols,target_col,scaler_x,scaler_y,seq_length=seq_length,step=36,)

        if out is None:
            print(f" [ SKIP ] {base} : 길이가 짧아 rollout 36-step 불가 (need >= {seq_length+36}) ")
            continue

        y_pred_36, y_true_36 = out

        mse = float(np.mean((y_pred_36 - y_true_36) ** 2))
        mae = float(mean_absolute_error(y_true_36, y_pred_36))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true_36, y_pred_36))

        # # 그래프 저장
        # plt.figure(figsize=(14, 5))
        # plt.plot(y_true_36, label="True")
        # plt.plot(y_pred_36, label="Pred")
        # plt.title("36-step rollout 예측 그래프")
        # plt.xlabel("step")
        # plt.ylabel(target_col)
        # plt.legend()
        # plt.tight_layout()

        # safe_name = re.sub(r'[\\/:*?"<>|]+', "_", os.path.splitext(base)[0])
        # fig_path = f"./result_png/{safe_name}_{target}.png"
        # plt.savefig(fig_path)
        # plt.close()

        # 그래프 저장
        start_idx = seq_length

        y_real = np.concatenate([
            df.loc[start_idx-seq_length:start_idx-1, target_col].values,
            y_true_36
        ])

        safe_name = re.sub(r'[\\/:*?"<>|]+', "_", os.path.splitext(base)[0])
        fig_path = f"./result_png/{safe_name}_{target}_{testname}.png"

        drawCorrectionGraph(
            y_real=y_real,
            y_pred=y_pred_36,
            pred_type=target_col,
            save_path=fig_path,
        )

        print(f"\n[ Test : {base} ]")
        print(f" R2 = {r2:.4f} , MSE = {mse:.4f} , MAE = {mae:.4f} , RMSE = {rmse:.4f}")

        results.append((base, r2, mse, mae, rmse ))

        makeResultChart(df, base, testname, y_pred_36, r2, mse, mae, rmse)

    if results:
        arr = np.array([r[1:] for r in results], dtype=float)
        print("\n[ SUMMARY ] 파일별 지표 평균")
        print(f" R2 = {arr[:,3].mean():.4f} , MSE = {arr[:,0].mean():.4f} , MAE = {arr[:,1].mean():.4f} , RMSE = {arr[:,2].mean():.4f} ")
