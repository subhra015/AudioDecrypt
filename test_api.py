import requests

url = "http://localhost:8000/predict"

# Create a fake audio file for testing
import numpy as np
import soundfile as sf

# Generate 4 seconds of noise
audio = np.random.randn(16000 * 4) * 0.1
sf.write("test_audio.wav", audio, 16000)

# Send to backend
with open("test_audio.wav", "rb") as f:
    response = requests.post(url, files={"file": f})

print("Status:", response.status_code)
print("Response:", response.json())