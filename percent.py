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
y_true = [1.3, 2.31, 1.94, 1.67, 1.23, 1.24]
y_pred = [1.266423, 2.465924, 1.579616, 1.478475, 1.017331, 1.285511]
# 예측값: [1.27 2.47 1.58 1.48 1.02 1.29]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------관심수위 대곡교--------------------------------')

# 관심수위 dg
y_true = [2.33, 4.53, 2.06, 2.35, 1.91, 2.86]
y_pred = [2.307464, 4.61234, 1.752652, 2.156254, 2.156797, 2.421277]
# 예측값: [2.31 4.61 1.75 2.16 2.16 2.42]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------관심수위 궁내교--------------------------------')

# 3시간
# 관심수위 gn
y_true = [2.16, 2.51, 1.85, 1.36, 1.79, 1.57]
y_pred = [2.25404, 2.734961 ,2.03274 , 1.342173, 1.552897, 1.476442]
# 예측값: [2.25 2.73 2.03 1.34 1.55 1.48]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------관심수위 대곡교--------------------------------')

# 관심수위 dg
y_true = [3.6, 4.87, 4.62, 2.91, 3.75,2.97]
y_pred = [3.086084, 5.21119, 4.595579, 2.830576, 3.48754, 3.001413]
# 예측값: [3.09 5.21 4.6  2.83 3.49 3.  ]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

#------------------------------------------------------------------------------------------------

print('-------------------------------20년수위 궁내교--------------------------------')

# 1시간
# 20년수위gn
y_true = [1.3, 2.31, 1.94, 1.67, 1.23, 1.24]
y_pred = [1.270024, 2.504938, 1.616062, 1.508611, 1.025856, 1.344837]
# 예측값: [1.27 2.5  1.62 1.51 1.03 1.34]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------20년수위 대곡교--------------------------------')

# 20년수위dg
y_true = [2.33, 4.53, 2.06, 2.35, 1.91, 2.86]
y_pred = [2.314735, 4.689074, 1.902847, 2.249537, 2.216043, 2.496865]
# 예측값: [2.31 4.69 1.9  2.25 2.22 2.5 ]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------20년수위 궁내교--------------------------------')

# 3시간
# 20년수위gn
y_true = [2.16, 2.51, 1.85, 1.36, 1.79, 1.57]
y_pred = [ 2.319718, 2.731514, 2.197768, 1.315086, 1.627853, 1.463892]
# 예측값: [2.32 2.73 2.2  1.32 1.63 1.46]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------20년수위 대곡교--------------------------------')

# 20년수위dg
y_true = [3.6, 4.87, 4.62, 2.91, 3.75, 2.97]
y_pred = [3.345885, 5.223463, 4.68333, 2.774033, 3.62647, 3.042217]
# 예측값: [3.35 5.22 4.68 2.77 3.63 3.04]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

#------------------------------------------------------------------------------------------------

print('-------------------------------누적강우 궁내교--------------------------------')

# 1시간
# 누적강우gn
y_true = [1.3, 2.31, 1.94, 1.67, 1.24, 1.23]
y_pred = [1.278881, 2.450699, 1.576567, 1.497425, 1.017855, 1.283466]
# 예측값: [1.28 2.45 1.58 1.5  1.02 1.28]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------누적강우 대곡교--------------------------------')

# 누적강우dg
y_true = [2.33, 4.53, 2.06, 2.35, 2.86, 1.91]
y_pred = [2.295167, 4.614906, 1.784811, 2.172475, 2.14442, 2.412381]
# 예측값: [2.3  4.61 1.78 2.17 2.14 2.41]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)
print('-------------------------------누적강우 궁내교--------------------------------')

# 3시간
# 누적강우gn
y_true = [2.16, 2.51, 1.85, 1.36, 1.57, 1.79]
y_pred = [2.280504, 2.67595, 2.031045, 1.382422, 1.532802, 1.482555]
# 예측값: [2.28 2.68 2.03 1.38 1.53 1.48]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------누적강우 대곡교--------------------------------')

# 누적강우dg
y_true = [3.6, 4.87, 4.62, 2.91, 2.97, 3.75]
y_pred = [3.171292, 5.13309, 4.544338, 2.818239, 3.449378, 2.982898]
# 예측값: [3.17 5.13 4.54 2.82 3.45 2.98]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

#------------------------------------------------------------------------------------------------

print('-------------------------------20년이후 누적강우 궁내교--------------------------------')

# 1시간
# 누적강우gn
y_true = [1.3, 2.31, 1.94, 1.67, 1.23, 1.24]
y_pred = [1.200482, 2.555448, 1.669865, 1.489227, 1.065097, 1.357344]
# 예측값: [1.2  2.56 1.67 1.49 1.07 1.36]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)
print('-------------------------------20년이후 누적강우 대곡교--------------------------------')

# 누적강우dg
y_true = [2.33, 4.53, 2.06, 2.35, 1.91, 2.86]
y_pred = [2.285673, 4.677747, 1.910671, 2.248477, 2.204257, 2.455897]
# 예측값: [2.29 4.68 1.91 2.25 2.2  2.46]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------20년이후누적강우 궁내교--------------------------------')

# 3시간
# 누적강우gn
y_true = [2.16, 2.51, 1.85, 1.36, 1.79, 1.57]
y_pred = [2.3086, 2.760101, 2.30778, 1.322727, 1.702126, 1.477759]
# 예측값: [2.31 2.76 2.31 1.32 1.7  1.48]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)

print('-------------------------------20년이후 누적강우 대곡교--------------------------------')

# 누적강우dg
y_true = [3.6, 4.87, 4.62, 2.91, 3.75, 2.79]
y_pred = [3.403525, 5.084952, 4.673886, 2.761694, 3.579948, 3.010377]
# 예측값: [3.4  5.08 4.67 2.76 3.58 3.01]
error, similarity, y_pred_disp = error_and_similarity(y_true, y_pred)

print("관심수위 1시간 오차:", error)
print("예측 유사도(%):", similarity)
print("예측값:", y_pred_disp)
