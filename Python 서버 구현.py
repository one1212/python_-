import socket
import threading

# 서버 설정
HOST = '220.90.180.80'  # 로컬 루프백 주소
PORT = 8000         # 포트 번호 (임의 설정)

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            # 클라이언트로부터 받은 메시지 처리 (예: 게임 로직)
            received_message = data.decode('utf-8')
            print(f"[{addr}] {received_message}")
            
            # 클라이언트에게 응답 (예: 게임 상태 업데이트)
            response_message = f"Server received: {received_message}"
            conn.sendall(response_message.encode('utf-8'))
    print(f"[DISCONNECTED] {addr} disconnected.")

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[LISTENING] Server is listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.start()
            print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")

if __name__ == "__main__":
    start_server()
