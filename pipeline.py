from transformers import pipeline
import time

# 1. 파이프라인 초기화
# 'openai/whisper-large-v3' 모델을 로드하여 파이프라인을 생성합니다.
# device=0은 GPU 0번을 사용하겠다는 의미입니다. CPU만 사용하려면 device=-1로 설정합니다.
start_time = time.time()
print("모델 로드 및 초기화 중...")

# GPU가 사용 가능한지 확인하고 설정
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=device
)

print(f"모델 로드 완료! (소요 시간: {time.time() - start_time:.2f}초)")

# 2. 음성 파일 경로 지정
# 'your_audio_file.mp3'를 실제 음성 파일 경로로 변경하세요.
audio_file_path = "C:\\aaa\\KakaoTalk_Audio_20251128_1629_37_243.mp3"

# 3. 음성 인식 수행
print(f"음성 파일 인식 시작: {audio_file_path}")
try:
    recognition_start_time = time.time()
    result = pipe(audio_file_path, generate_kwargs={"language": "ko", "task": "transcribe"})
    recognition_end_time = time.time()

    # 4. 결과 출력
    print("\n--- 인식 결과 ---")
    print(f"인식된 텍스트: {result['text']}")
    print(f"인식 소요 시간: {recognition_end_time - recognition_start_time:.2f}초 (장비: {device})")

except Exception as e:
    print(f"오류가 발생했습니다: {e}")
    print("오디오 파일 경로를 확인하거나, ffmpeg 및 필수 라이브러리가 설치되었는지 확인해 주세요.")