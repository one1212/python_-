import matplotlib.pyplot as plt

#방어의 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]


plt.scatter(bream_length, bream_weight)
plt.xlabel('length') #x축은 길이
plt.ylabel('weight') # Y축은 무개
plt.show()           #show는 화면에 보여준다.

#이것은 빙어의 길이와 무개
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length') # X축은 길이
plt.ylabel('weight') # Y축은 무개
plt.show()

#길이 = 방어의 길이와 빙어의 길이를 더한다.
length = bream_length + smelt_length
#무개 = 방어의 무개와 빙어의 무개를 더한다.
weight = bream_weight + smelt_weight

#zip에서 나온 (1,w) 튜플을 하나씩 꺼내서 리스트 [1,w]로 만든다는 뜻
#[10,100] [20,200] [30,300]
#fish_data = 2차원 배열이 된다.
#두 방어와 빙어의 길이와 무개를 l1,w1이라고 하고 [[l1, w1] [l2,w2]] 형태로 만든코드
fish_data = [[l, w] for l, w in zip(length, weight)]

#터미널 창에 실행화면을 보여준다.
print(fish_data)

fish_target = [1] * 35 + [0] * 14
print(fish_target)

from sklearn.neighbors import KNeighborsClassifier
#from ~ import문 패키지나 모듈 존체를 임포트하지 않고 특정 클래스만 임포트 하려면 사용한다.

kn = KNeighborsClassifier()
#객체 생성한다
kn.fit(fish_data, fish_target)
#kn이라는 객체에 fit이라는 메서드를 사용한다.
# 1이라는 방어는 잘 출력하고 0이라는 빙어도 잘 출력한다

print(kn.score(fish_data, fish_target))
#score는 정확도이다. fish_data와 fish_target의 정확도는 1.0 정수이다. 완벽하게 분류했다

print(kn.predict([[10.5,7.5]]))
#kn.predict 는 x, y좌표에 있는 값이 0(빙어), 1(방어)인지 찾아주는것이다.
# 현재 좌표에서 10.5, 7.5에 있는건 방어가 아니고 빙어기 때문에 0을 찾아주는 모습이다

print(kn._fit_X)
print(kn._y)
# 메서드에 데이터를 저장하고 있다가 새로운 데이터가 오면 가장 가까운 5개의 데이터를 비교해 구분하는 메서드이다


kn49 = KNeighborsClassifier(n_neighbors=49)
#가장 가까운 데이터 49개를 사용해서 fish_data를 예측한다.
#49개 중에 도미가 35개로 다수 차지하므로 어떤 데이터를 넣어도 도미로 인식한다.

print(kn49.fit(fish_data, fish_target))
print(kn49.score(fish_data, fish_target))