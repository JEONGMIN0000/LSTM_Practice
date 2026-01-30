import numpy as np

def error_and_similarity(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    # 오차
    error = y_pred - y_true

    # 예측 유사도 (%)
    similarity = (1 - np.abs(error) / (np.abs(y_true) + eps)) * 100
    

    # |error| : 절대 오차
    # |error| / |y_true| : 상대오차
    # 1 - 상대오차 : 얼마나 원값에 가까운가
    # × 100 : 퍼센트화
    # 👉 관측값(y_true)을 기준으로 예측값이 얼마나 벗어났는지를 %로 표현한 지표

    # 표시용 반올림 (계산값은 유지)
    error_round = np.round(error, 2)        # 셋째 자리 반올림
    similarity_round = np.round(similarity, 1)  # 둘째 자리 반올림
    y_pred_disp = np.round(y_pred, 2)

    return error_round, similarity_round, y_pred_disp


#------------------------------------------------------------------------------------------------

print('-------------------------------관심수위 궁내교--------------------------------')

# 1시간
# 관심수위 gn
y_true = [1.3, 2.34, 2.22, 2.0, 1.29, 1.34]
y_pred = [1.257978, 2.501917, 1.893803, 1.593714, 1.035886, 1.383681]
# 예측값: [1.26 2.5  1.89 1.59 1.04 1.38]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------관심수위 대곡교--------------------------------')

# 관심수위 dg
y_true = [2.33, 4.55, 2.07, 2.35, 1.94, 2.93]
y_pred = [2.301992, 4.705499, 1.685189, 2.159919, 2.200546, 2.421900]
# 예측값: [2.3  4.71 1.69 2.16 2.2  2.42]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------관심수위 궁내교--------------------------------')

# 3시간
# 관심수위 gn
y_true = [2.3, 2.53, 1.75, 1.34, 1.7, 1.66]
y_pred = [2.425434, 2.829719 ,1.917738, 1.308139, 1.523724, 1.492799]
# 예측값: [2.43 2.83 1.92 1.31 1.52 1.49]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------관심수위 대곡교--------------------------------')

# 관심수위 dg
y_true = [3.62, 4.9, 4.49, 2.87, 3.79, 3.25]
y_pred = [3.139007, 5.192266, 4.574636, 2.825481, 3.548599, 3.142500]
# 예측값: [3.14 5.19 4.57 2.83 3.55 3.14]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

#------------------------------------------------------------------------------------------------

print('-------------------------------20년수위 궁내교--------------------------------')

# 1시간
# 20년수위gn
y_true = [1.3, 2.34, 2.22, 2.0, 1.29, 1.34]
y_pred = [1.261096, 2.545750, 1.939708, 1.609892, 1.044171, 1.421826]
# 예측값: [1.26 2.55 1.94 1.61 1.04 1.42]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------20년수위 대곡교--------------------------------')

# 20년수위dg
y_true = [2.33, 4.55, 2.07, 2.35, 1.94, 2.93]
y_pred = [2.308241, 4.810140, 1.908534, 2.287426, 2.258312, 2.515922]
# 예측값: [2.31 4.81 1.91 2.29 2.26 2.52]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------20년수위 궁내교--------------------------------')

# 3시간
# 20년수위gn
y_true_3h_gn = [2.3, 2.53, 1.75, 1.34, 1.7, 1.66]
y_pred_3h_gn = [ 2.450750, 2.836549, 2.069824, 1.283702, 1.595422, 1.484016]
# 예측값: [2.31 4.81 1.91 2.29 2.26 2.52]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------20년수위 대곡교--------------------------------')

# 20년수위dg
y_true = [3.62, 4.9, 4.49, 2.87, 3.79, 3.25]
y_pred = [3.443497, 5.201563, 4.672024, 2.760953, 3.683562, 3.213317]
# 예측값: [3.44 5.2  4.67 2.76 3.68 3.21]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

#------------------------------------------------------------------------------------------------

print('-------------------------------누적강우 궁내교--------------------------------')

# 1시간
# 누적강우gn
y_true = [1.3, 2.34, 2.22, 2.0, 1.34, 1.29]
y_pred = [1.272885, 2.487515, 1.881842, 1.609333, 1.036811, 1.374018]
# 예측값: [1.27 2.49 1.88 1.61 1.04 1.37]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------누적강우 대곡교--------------------------------')

# 누적강우dg
y_true = [2.33, 4.55, 2.07, 2.35, 2.93, 1.94]
y_pred = [2.289159, 4.720189, 1.699408, 2.170205, 2.187211, 2.419326]
# 예측값: [2.29 4.72 1.7  2.17 2.19 2.42]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)
print('-------------------------------누적강우 궁내교--------------------------------')

# 3시간
# 누적강우gn
y_true = [2.3, 2.53, 1.75, 1.34, 1.66, 1.7]
y_pred = [1.262381, 2.497848, 2.330591, 1.701506, 1.186446, 1.468794]
# 예측값: [1.26 2.5  2.33 1.7  1.19 1.47]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------누적강우 대곡교--------------------------------')

# 누적강우dg
y_true = [3.62, 4.9, 4.49, 2.87, 3.25, 3.79]
y_pred = [2.283837, 4.903264, 1.746548, 2.279450, 2.239884, 2.500868]
# 예측값: [2.28 4.9  1.75 2.28 2.24 2.5 ]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

#------------------------------------------------------------------------------------------------

print('-------------------------------20년이후 누적강우 궁내교--------------------------------')

# 1시간
# 누적강우gn
y_true = [1.3, 2.34, 2.22, 2.0, 1.34, 1.29]
y_pred = [1.188586, 2.589267, 1.986652, 1.574132, 1.418434, 1.097640]
# 예측값: [1.19 2.59 1.99 1.57 1.42 1.1 ]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)
print('-------------------------------20년이후 누적강우 대곡교--------------------------------')

# 누적강우dg
y_true = [2.33, 4.55, 2.07, 2.35, 2.93, 1.94]
y_pred = [2.277972, 4.809556, 1.900907, 2.290179, 2.477810, 2.242937]
# 예측값: [2.28 4.81 1.9  2.29 2.48 2.24]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------20년이후누적강우 궁내교--------------------------------')

# 3시간
# 누적강우gn
y_true = [2.3, 2.53, 1.75, 1.34, 1.66, 1.7]
y_pred = [2.438164, 2.848590, 2.180795, 1.294104, 1.476979, 1.666799]
# 예측값: [2.44 2.85 2.18 1.29 1.48 1.67]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------20년이후 누적강우 대곡교--------------------------------')

# 누적강우dg
y_true = [3.62, 4.9, 4.49, 2.87, 3.25, 3.79]
y_pred = [3.510123, 5.056725, 4.661264, 2.748068, 3.157265, 3.638675]

# 예측값: [3.51 5.06 4.66 2.75 3.16 3.64]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)
