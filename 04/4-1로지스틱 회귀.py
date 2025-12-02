import pandas as pd
fish = pd.read_csv('http://bit.ly/fish_csv_data')
print(fish.head())
print(pd.unique(fish['Species']))

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']] #fish_input에 5개의 특성을 저장한다.
fish_input.head() # 5개의 행을 출력한다
fish_target = fish['Species']
#데이터를 훈련세트와 테스트 세트로 나눈다.

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
#StandardScaler 클래스를 사용해 훈련 세트와 테스트 세트를 표준화 전처리 함

from sklearn.preprocessing import StandardScaler 
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3) #n_neighbors=3 기본값을 3개로 지정한다.
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

#191p
print(kn.classes_)
#Bream이 첫번쨰 Parkki가 두번째가 되는 클래스이다.
# Predict()는 친절하게 타깃값을 얘측을 출력한다.
print(kn.predict(test_scaled[:5]))

#192p
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4)) #<-소수점 네번째 자리까지 표기합니다. 다섯번째에서 반올림합니다
#predict_proba() 출력 순서는 앞서 보았던 classes_속성과 같다. 첫번째 Bream에 대한 확률 Pakki에 대한 확률입니다.

distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target.iloc[indexes[0]])

#194p
import numpy as np
import matplotlib.pylab as plt
z = np.arange(-5,5,0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z,phi)
plt.xlabel('Z')
plt.ylabel('phi')
plt.show()
char_arr = np.array(['A','B','C','D','E'])
print(char_arr[True, False, True, False, False])

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[ bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
#두 번째 샘플을 제외하고는 모두 도미로 예측
print(lr.predict_proba(train_bream_smelt[:5]))
#샘플마다 2개의 확률이 출력됨 첫번째 열이 음성클래스 0에 대한 확률이고 
# 두번 째 열이 양성 클래스 1에 대한 확률입니다.
print(lr.classes_)
#방어가 양성 클래스 predict_proba 반환한 배열 값을 보면 두 번째 샘플 만 양성 클래스인 빙어의 확률이 높습니다. 나머지는 도미
print(lr.coef_, lr.intercept_)
#z = -0.45 * weight - 0.576 * Length - 0.662 * Diagonal - 1.013 * Height - 0.731 * Width - 2.162
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit
print(expit(decisions))
#decision_function은 양성 클래스에 대한 z값을 반환합니다.

lr=LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.predict(test_scaled[:5]))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
# 소수점 3자리 까지만
print(lr.classes_)
# 첫 번째 샘플은 perch를 가장 높은 확률로 예측했음,
# 두 번째 샘플은 smelt를 가장 높은 확률로 예측함
print(lr.coef_.shape, lr.intercept_.shape)
#5개의 특성을 사용하므로 code_ 배열은 5개입니다. 근데 행은 7개임
# 이진 분류에소 보았던 z를 7개나 계산한다는 의미
# 다중 분류는 클래스마다 z 값을 하나씩 계산합니다. 당연히 가장 높은 z 값을 출력하는 클래스가 예측 클래스가 됩니다.
# 이진 분류에서는 시그모이드 함수를 사용해 z를 0과 1사이의 값으로 변환했다 
# 다중 분류는 이와 달리 소프트맥스 함수를 사용하여 7개의 z값을 확률로 변환한다.
 
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision , decimals=2))

from scipy.special import softmax
proba = softmax(decision, axis=1) 
print(np.round(proba, decimals=3))


