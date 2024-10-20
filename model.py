import torch
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)
print(torchaudio.__version__)

torchaudio.set_audio_backend("sox")

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

model = bundle.get_model().to(device)

waveform, sample_rate = torchaudio.load("audiosamples/swipe_left.wav") # path to file??
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

# extracting acoustic features from audio
# ex. frequency, amplitude, duration, etc.

with torch.inference_mode():
    features, _ = model.extract_features(waveform)
    # comma means assign both new variables to the same right-hand side value?

# copied this code, go through and figure out what it all means later
fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu(), interpolation="nearest")
    ax[i].set_title(f"Feature from transformer layer {i+1}")
    ax[i].set_xlabel("Feature dimension")
    ax[i].set_ylabel("Frame (time-axis)")
fig.tight_layout()