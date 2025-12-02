import zmq
import time
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import sys

# --- 서버 설정 ---
# ZeroMQ는 기본적으로 TCP를 사용하며, 포트를 지정합니다.
HOST_BIND = "tcp://*:8000" # 모든 인터페이스의 포트 5555에 바인딩

# --- 모델 및 스케일러 로드 ---
try:
    # 1. StandardScaler 로드 (데이터 전처리용)
    with open('ss.pkl', 'rb') as f:
        ss = pickle.load(f)

    # 2. 최적 Decision Tree Model 로드 (예측 수행용)
    with open('dt_best.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("[MODEL LOADED] StandardScaler (ss.pkl) and Decision Tree Model (dt_best.pkl) loaded successfully.")

except FileNotFoundError:
    print("[FATAL ERROR] ss.pkl 또는 dt_best.pkl 파일이 없습니다. 학습 코드를 실행하여 파일을 먼저 저장하십시오.")
    sys.exit() 
except Exception as e:
    print(f"[ERROR] 모델 파일 로드 중 오류 발생: {e}")
    sys.exit()

# --- 예측 함수 ---
def predict_penguin(bl, bd, fl, bm):
    """
    4가지 특성을 받아 로드된 최적 모델로 펭귄의 종(species)을 예측합니다.
    (bl: 부리 길이, bd: 부리 깊이, fl: 날개 길이, bm: 몸무게)
    """
    # 입력 데이터를 2차원 배열로 변환
    input_data = np.array([[bl, bd, fl, bm]])
    
    # 학습 시와 동일하게 스케일러를 사용하여 데이터 변환
    X_scaled = ss.transform(input_data)
    
    # 예측 수행 (결과는 'Adelie', 'Gentoo', 'Chinstrap' 중 하나)
    prediction = model.predict(X_scaled)[0] 
    
    return prediction

# --- ZeroMQ 서버 시작 ---
def start_zmq_server():
    context = zmq.Context()
    # REP (Reply) 소켓은 요청(REQ)을 받으면 응답(REP)을 보내는 역할을 합니다.
    socket = context.socket(zmq.REP)
    
    try:
        socket.bind(HOST_BIND)
    except zmq.error.ZMQError as e:
        print(f"[FATAL ERROR] ZeroMQ bind error: {e}")
        print("포트 5555가 이미 사용 중일 수 있습니다. 다른 포트를 시도하거나 프로세스를 종료하십시오.")
        return

    print(f"[LISTENING] ZeroMQ Server is listening on {HOST_BIND}")

    while True:
        try:
            # 1. 요청 메시지 수신 (클라이언트 요청 대기)
            message = socket.recv_string()
            print(f"[REQUEST] {message}")
            
            # 2. 데이터 파싱 및 예측 수행
            parts = message.split(',')
            
            if len(parts) != 4:
                response = "ERROR: Format must be 4 numbers separated by commas."
            else:
                try:
                    # 데이터 float 변환
                    bl, bd, fl, bm = map(float, parts)
                    
                    # 실제 펭귄 예측 함수 호출
                    prediction = predict_penguin(bl, bd, fl, bm)
                    response = str(prediction)
                    
                except ValueError:
                    response = "ERROR: All values must be valid numbers (float/int)."
                except Exception as e:
                    response = f"ERROR: Prediction failed due to model error. {e}"

            print(f"[RESPONSE] {response}")

            # 3. 응답 전송
            socket.send_string(response)
            
        except zmq.error.ContextTerminated:
            print("[INFO] ZeroMQ Context terminated.")
            break
        except Exception as e:
            print(f"[SERVER ERROR] Unhandled exception: {e}")
            time.sleep(1) # 오류 발생 시 과부하 방지
            
    # 서버 종료 시 소켓 및 컨텍스트 정리
    socket.close()
    context.term()

if __name__ == "__main__":
    start_zmq_server()