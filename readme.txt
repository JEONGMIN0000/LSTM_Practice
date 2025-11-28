python-version : 3.8

# install _requirements
pip install -r requirements.txt

# run_python
python main.py

#1. 데이터 조회 (csv)

#2. 데이터 전처리 
# 2-1. 데이터 정규화 (스케일링) 
# 2-2. window-size 생성 (예시를 기준으로, window size를 24로 둔다면 24일치를 예측? 하는 모델)

#3. 데이터셋 분리
# train, validation, test 
# 보통 train 7: test 3 or train 7: validation 2: test 1

#4. 학습 및 검증(테스트)


#5. 평가 (회귀 모델의 평가 방법)
#평가 지표 (회귀 모델 평가 메트릭): R2 Score, MSE, MAE, RMSE 

#6. 모델 저장 -> file (.h5, checkpoint, save_model)

#7. 저장된 모델을 호출해서 예측수행한다.