import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

import matplotlib.pyplot as plt
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

test_array = np.array([1,2,3,4])
print(test_array.shape)
#1차원 배열로 크기는 4

test_array = test_array.reshape(2,2)
print(test_array.shape)
#1차원에서 2차원배열로 변경 크기는 2,2

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)
print(train_input.shape, test_input.shape)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
#k-최근접 이웃 회귀 모델을 훈련합니다.
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target))

from sklearn.metrics import mean_absolute_error
#테스트 세트에 대한 예측을 만든다.
test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
#결과에서 예측이 평균적으로 19g 정도의 타깃값과 다르다는 것을 알 수 있다

#앞에서 훈련한 모델을 사용해 훈련 세트의 R2 점수를 확인함
print(knr.score(train_input, train_target))

knr.n_neighbors = 3
#이웃의 개수를 3으로 설정한다.

knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
#모델을 다시 훈련한다.
print(knr.score(test_input, test_target))
