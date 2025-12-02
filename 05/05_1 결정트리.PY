import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

print(wine.head())

#info() 메서드 : 데이터프레임의 각 열의 데이터 타입과 누락된 데이터가 있는지 확인
wine.info()
#non_null (누락은 없다), DateType : float 실수형

#describe() 메서드 : 열애 대한 간략한 통계를 출력해 줍니다. 최대 최소 평균값 구하기
print(wine.describe())
#mean(평균), std(표준편차), 25%(1사분위수), 50%(중간값/2사분위수), 75%(3사분위수), max(최대)

data = wine[['alcohol', 'sugar', 'pH']]
target = wine['class']
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
#train_test_split() 함수는 설정값을 지정하지 않으면 25%를 테스트 세트로 지정합니다.

#238p
print(train_input.shape, test_input.shape)
#훈련세트로 전처리한다.

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled=ss.transform(train_input)
test_scaled = ss.transform(test_input)
# 테스트 세트로 변환한다.

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
#로지스틱 회귀 모델을 훈련한다.

print(lr.coef_, lr.intercept_)

#241p
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
# 훈련세트에 대한 점수는 높지만 테스트 서버의 점수는 낮다.

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7)) # 도화지의 가로가 10, 세로가 7
plot_tree(dt)
plt.show()

plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

#246p
dt = DecisionTreeClassifier(max_depth=3, random_state=42)   #최대 3번쨰 가지까지 늘어난ㄷㅏ.
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
#훈련세트의 성능은 낮아졌지만 테스트에 성능은 그래도입니다.

plt.figure(figsize=(20,15))
plot_tree(dt, filled= True, feature_names=('alcohol', 'sugar', 'pH' ))
plt.show()

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

#248p
#트리를 그려보도록 하죠
plt.figure(figsize=(20,15))
plot_tree(dt, filled= True, feature_names=('alcohol', 'sugar', 'pH'))
plt.show()

print(dt.feature_importances_)