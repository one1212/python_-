import torch

# 12행: 강제로 CPU로 설정하여 GPU 사용을 막습니다.
device = "cpu" 

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=device # 이제 "cpu"가 사용됩니다.
)
# ... 코드 계속 실행