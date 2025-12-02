from palmerpenguins import load_penguins
from sklearn.impute import SimpleImputer
import pandas as pd


dt = load_penguins().dropna()

# # 1. 숫자 컬럼과 범주형 컬럼 나누기
# num_cols = penguins.select_dtypes(include=['float64', 'int64']).columns
# cat_cols = penguins.select_dtypes(include=['object']).columns

# # 2. 숫자 컬럼: 평균으로 채우기
# num_imputer = SimpleImputer(strategy='mean')
# penguins[num_cols] = num_imputer.fit_transform(penguins[num_cols])

# # 3. 범주형 컬럼: 최빈값(mode)으로 채우기
# cat_imputer = SimpleImputer(strategy='most_frequent')
# penguins[cat_cols] = cat_imputer.fit_transform(penguins[cat_cols])

print(dt.info())

data = dt[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
target = dt['species'] #종족

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, random_state=42)

print(train_input.shape, test_input.shape)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.tree import DecisionTreeClassifier
lr = DecisionTreeClassifier(random_state=42)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled,test_target))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(lr)
plt.show()

plt.figure(figsize=(10,7))
plot_tree(lr, filled=True, feature_names=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
plt.show()

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input,train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input,test_target))

from scipy.stats import uniform, randint
import numpy as np


rgen = randint(0,25)

print(np.unique(rgen.rvs(333), return_counts=True))

from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

# 탐색할 하이퍼파라미터 범위
params = {
    'min_impurity_decrease': uniform(0.0001, 0.001),
    'max_depth': randint(20, 50),
    'min_samples_split': randint(2, 25),
    'min_samples_leaf': randint(1, 25)
}

# RandomizedSearchCV 객체 생성
rs = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),   # 기본 모델
    params,
    n_iter=333,
    n_jobs=-1,
    random_state=42
)

# 스케일된 데이터로 탐색
rs.fit(train_scaled, train_target)

# 최적 파라미터 출력
print(rs.best_params_)
print("Train score:", rs.score(train_scaled, train_target))
print("Test score:", rs.score(test_scaled, test_target))

print(np.max(rs.cv_results_['mean_test_score']))
dt = rs.best_estimator_
print(dt.score(test_scaled, test_target))
print(dt.score(train_scaled, train_target))

# --- 학습된 모델과 스케일러 저장 코드 ---
import pickle

# 1. 학습된 스케일러 저장
with open('ss.pkl', 'wb') as f:
    pickle.dump(ss, f)

# 2. 최적의 결정 트리 모델 저장
dt = rs.best_estimator_
with open('dt_best.pkl', 'wb') as f:
    pickle.dump(dt, f)

print("[SAVE COMPLETE] ss.pkl and dt_best.pkl saved.")

import socket
import threading
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --- 서버 설정 ---
# **주의:** 클라이언트(WinForm)가 접속할 수 있도록 HOST 주소를 설정해야 합니다.
# 로컬 테스트 시에는 '127.0.0.1' 또는 '0.0.0.0'을 사용하고, 
# 네트워크 테스트 시에는 서버 PC의 실제 IP 주소 ('220.90.180.80' 등)를 사용하세요.
HOST = '220.90.180.80' 
PORT = 8000         

# --- 모델 및 스케일러 로드 ---
try:
    # 1. 스케일러 로드
    with open('ss.pkl', 'rb') as f:
        ss = pickle.load(f)

    # 2. 최적의 결정 트리 모델 로드
    with open('dt_best.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("[MODEL LOADED] StandardScaler (ss.pkl) and Decision Tree Model (dt_best.pkl) loaded successfully.")

except FileNotFoundError:
    print("[FATAL ERROR] ss.pkl or dt_best.pkl not found. Run the training script first to save them.")
    exit() # 파일 없으면 서버 실행 중지
except Exception as e:
    print(f"[ERROR] Error loading model files: {e}")
    exit()

# --- 예측 함수 ---
def predict_penguin(bl, bd, fl, bm):
    """
    4가지 특성을 받아 로드된 모델로 펭귄의 종류를 예측합니다.
    (bl: 부리 길이, bd: 부리 깊이, fl: 날개 길이, bm: 몸무게)
    """
    # 1. 입력 데이터를 2차원 배열로 변환
    input_data = [[bl, bd, fl, bm]]
    
    # 2. 스케일러를 사용하여 데이터 변환 (학습 시와 동일한 전처리 적용)
    X_scaled = ss.transform(input_data)
    
    # 3. 예측 수행
    prediction = model.predict(X_scaled)[0] 
    
    return prediction

# --- 클라이언트 연결 처리 함수 ---
def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    
    try:
        while True:
            # 클라이언트로부터 데이터 수신 (최대 1024바이트)
            data = conn.recv(1024) 

            if not data:
                print("[CLIENT CLOSED]", addr)
                break
            
            # 수신된 데이터를 문자열로 디코딩하고 양쪽 공백 제거
            msg = data.decode().strip()
            print(f"[RECEIVED] {addr}: {msg}")

            # 1. 데이터 형식 검증 (콤마 개수 확인)
            parts = msg.split(',')
            if len(parts) != 4:
                conn.send(b"ERROR: Format must be 4 numbers separated by commas (bl,bd,fl,bm)\n")
                continue

            # 2. 데이터 값 검증 (숫자 변환 시도)
            try:
                # 4개의 부분을 float 형으로 변환
                bl, bd, fl, bm = map(float, parts)
            except ValueError:
                conn.send(b"ERROR: All values must be valid numbers.\n")
                continue

            # 3. 예측 수행 및 결과 전송
            result = predict_penguin(bl, bd, fl, bm)
            
            # 예측 결과를 문자열로 인코딩하여 전송 + 개행문자 추가
            response = str(result) + "\n"
            conn.send(response.encode())
            print(f"[SENT] {addr}: {result}")

    except Exception as e:
        print(f"[ERROR] {addr}: {e}")

    finally:
        # 클라이언트 소켓 연결 종료
        conn.close()
        print(f"[DISCONNECTED] {addr} disconnected.")

# --- 서버 시작 함수 ---
def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 포트 재사용 옵션 설정
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    
    try:
        server.bind((HOST, PORT))
        server.listen()
    except OSError as e:
        print(f"[FATAL ERROR] Cannot bind to {HOST}:{PORT}. Check address or port. Error: {e}")
        return

    print(f"[LISTENING] Server is listening on {HOST}:{PORT}")

    while True:
        try:
            conn, addr = server.accept()
            # 새로운 스레드에서 클라이언트 처리 함수 실행
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()
            print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}") 
        except Exception as e:
            print(f"[SERVER LOOP ERROR] {e}")
            break
            
    server.close()

if __name__ == "__main__":
    start_server()