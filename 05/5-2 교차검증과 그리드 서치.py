import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data') #데이터를 가져온다
data = wine[['alcohol', 'sugar', 'pH']] #데이터에 와인 테이블을 만든다.
target = wine['class']  #타깃에 

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42) 
#train_input, test_input, train_target, test_target 네 개의 새로운 변수로 분할
# data는 입력 변수 target 정답변수를 담고 있다
# test_size=2.0 : {data}와 target의 20%를 테스트 세트에 넘긴다
# 나머지 80%는 훈련 세트에 넘긴다 
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
# sub_input, sub_target이라는 훈련세트와 sub_input, sub_target이라는 검증 세트를 만듭니다.
# 여기에서도 test_size=0.2를 통해 train_input의 20프로를 val_input으로 만든다.
print(sub_input.shape, val_input.shape)

from sklearn.tree import DecisionTreeClassifier
#sklearn.tree라는 패키지 또는 모듈을 지정하고, import 위치에 원하는 도구를 가져오는 역할
dt = DecisionTreeClassifier()
#dt라는 객체 생성
dt.fit(sub_input, sub_target)
#fit : 입력 데이터와 정답데이터 사이의 규칙을 학습해라
print(dt.score(sub_input, sub_target))
#score 정확도
print(dt.score(val_input, val_target))

#260
from sklearn.model_selection import cross_validate
#모듈 selection을 cross_validata라는 위치에 가져온다.
scores = cross_validate(dt, train_input, train_target)
#scores에 cross_validate함수 사용 결과를 저장한다.
print(scores)
# 이 함수는 fit_time, score_time, test_score키를 가진 딕셔너리를 반환합니다.
# 처음 두개의 키는 각각 모델을 훈련하는 시간과 검증하는 시간을 의미합니다. 

import numpy as np
print(np.mean(scores['test_score']))
#교차 검증의 최종 점수는 test_score키에 담긴 5개의 점수를 평균하여 얻을 수 있다.

from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
#scores에 cross_validate의 실행 결과를 저장한다.
print(np.mean(scores['test_score']))

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))

from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease' : [0.0001,0.0002,0.0003,0.0004,0.0005]}
#params = {'min_impurity_decrease' : [0.0001,0.0002,0.0003,0.0004,0.0005]} : 불순도 감소량을 제한하는 하이퍼파라미터에 대해 탐색할 값들을 딕셔너리 형태로 정의

gs = GridSearchCV(DecisionTreeClassifier(random_state=42),params, n_jobs=-1)
#그리드 서치 준비
# DecisionTreeClassifier(random_state=42) : 기반 모델로 결정 트리 사용한다.
# params : 위에서 정의한 탐색할 하이퍼파라미터 값 목록을 사용한다
# n_jobs = -1 : 모든 CPU코어를 사용하여 그리드 서치를 병렬로 빠르게 수행하도록 설정 

gs.fit(train_input, train_target)
#입력과 정답 사이를 알아라
dt = gs.best_estimator_
#교차중 검증 점수가 가장 높았던 바로 그 최적의 결정 트리 모델 객체를 dt에 저장한다.
print(dt.score(train_input,train_target))
#정답률 계산해서 출력해라
print(gs.best_params_)
#가장 높은 점수를 얻었을 때 사용했던 하이퍼파라미터 조합을 출력해라
print(gs.cv_results_['mean_test_score'])
# 모든 하이퍼파라미터 값에 대한 교차 검증의 평균 점수 목록을 출력
print(gs.cv_results_['params'][gs.best_index_])
# 딕셔너리 정보 담겨져 있다

#265
params = {'min_impurity_decrease' : np.arange(0.0001,0.001,0.0001), 'max_depth' : range(5,20,1), 'min_samples_split' : range(2,100,10)}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_params_)
print(np.max(gs.cv_results_['mean_test_score']))
from scipy.stats import uniform, randint
rgen = randint(0,10)
rgen.rvs(10)
np.unique(rgen.rvs(1000), return_counts=True)
ugen = uniform(0,1)
ugen.rvs(10)

params = {'min_impurity_decrease' : uniform(0.0001, 0.001),
          'max_depth' : randint(20, 50),
          'min_samples_split' : randint(2, 25),
          'min_samples_leaf' : randint(1,25),}

from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs =-1, random_state=42)
#랜덤 서치 준비하고 기반모델로 트리 설정, 위에서 정의한 하이퍼파라미터 값 복사, n_jobs 빠르게 수행하도록 설정, random_state 초기값 인위적으로 설정
rs.fit(train_input, train_target)
print(rs.best_params_)
print(np.max(rs.cv_results_['mean_test_score']))
dt = rs.best_estimator_
print(dt.score(test_input, test_target))

