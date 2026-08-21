# Talkbox

Translates what you say and speaks it back in your own voice, into a call.

Talkbox listens to your microphone, translates the sentence, and says it with a
copy of your voice. The audio goes to a virtual audio cable, so Discord, Zoom or
Teams pick it up as a microphone. Your real voice never reaches the call,
because the mic is connected to Talkbox and nothing else.

It can also type instead of speaking. Same recognition, but the text goes into
whatever window you have open.

Everything runs on your machine. No API keys, no accounts, no per-minute cost.

## Features

- Translated speech in your own voice, sent into a call as a microphone input.
- Dictation, typed into the focused window, with or without translation.
- 17 languages both ways: English, Spanish, French, German, Italian,
  Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese,
  Hungarian, Korean, Hindi.
- Voice cloning from about a minute of recording, or 58 built-in voices if you
  do not want to record.
- 620 to 700 ms from the end of your sentence to the first translated audio.
- Audio devices are found and opened automatically each time it starts.

## Requirements

Windows 10 or 11, and Python 3.10 to 3.12. `coqui-tts` does not install on 3.13.

For speaking into a call you need an NVIDIA GPU. VRAM use is 2.7 GB on the
lowest quality setting and 5.5 GB on the highest, so 8 GB is comfortable. You
also need a virtual audio cable. [VB-Cable](https://vb-audio.com/Cable/) is
free and the setup wizard can download it.

For dictation you need neither. It runs on the CPU. On an i7-14700F a 2.9
second sentence took 234 ms to transcribe with Whisper `base` and another
330 ms to translate. The recogniser is capped at `base` when there
is no GPU, because `small` took 1944 ms on the same sentence.

Talkbox is CUDA or CPU. There is no ROCm or DirectML path, so an AMD or Intel
GPU will not be used.

About 6 GB of disk for dependencies, plus 3 GB of models downloaded on first
run.

## Install

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This makes a virtual environment, installs PyTorch from the CUDA wheel index,
then installs everything else. No models are in this repository. The first
launch opens a setup wizard that downloads them, offers to fetch VB-Cable, and
records your voice if you want a clone.
