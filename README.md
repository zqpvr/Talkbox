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

## Usage

```powershell
.\run.ps1
```

Choose what happens when you talk, pick your languages, press Start.

For a call, set the call application's microphone to CABLE Output. Talkbox
writes to CABLE Input. If you want to hear the translation yourself, tick
"Play the translation to me as well" on the Audio tab.

For dictation, choose "Type it into whatever window I'm in", press Start, then
click into the window you want the text in. Nothing is typed while the Talkbox
window is in front, so it cannot type into itself. Tick "Translate it before
typing" to speak one language and have another appear.

To check an install:

```powershell
.\.venv\Scripts\python.exe -m talkbox --doctor
```

## How it works

Four models in a chain:

| Stage | Model | Job |
| --- | --- | --- |
| Segmentation | Silero VAD | Find where each sentence ends |
| Recognition | faster-whisper | Audio to text |
| Translation | NLLB-200 on CTranslate2 | Text to the target language |
| Speech | XTTS-v2 | Say it in your voice |

Dictation uses the first two, and the third only if you asked for translation.

One quality setting picks the recogniser and translator together. There is no
separate model choice.

Recognition and translation start partway into the end-of-sentence pause rather
than after it, and the result is kept only if no more speech arrived. The pause
ends at 240 ms when the transcript already looks like a finished sentence. It
runs to 700 ms when the transcript looks unfinished, which is what stops a long
sentence being cut in half when you take a breath.

It works a sentence at a time, so it cannot start before you finish a clause.
About 650 ms is the floor for this design.

[docs/performance.md](docs/performance.md) has the measurements and a list of
things that were tried and did not help.

## Troubleshooting

**The call hears nothing.** The call application's microphone should be CABLE
Output, not CABLE Input. VB-Cable has to be installed as administrator followed
by a reboot.

**Synthesis fails on Japanese, Chinese or Korean.** Install the extras:
`pip install "coqui-tts[ja,ko,zh]"`.

**A cuDNN error on startup.** `transformers` has to stay below 5.0, since 5.x
removes a function `coqui-tts` needs. Re-run `setup.ps1`.

## Layout

```
talkbox/
  pipeline.py   orchestration, threads, timing
  vad.py        segmentation and the pause rules
  asr.py        faster-whisper
  mt.py         NLLB-200, CTranslate2 with a HuggingFace fallback
  tts.py        XTTS-v2, cloned and built-in voices
  dictate.py    typing into the focused window
  audio.py      device selection and playback
  ui.py         the window
  wizard.py     first-run setup
tools/
  smoke_test.py     whole pipeline, no microphone needed
  test_segmenter.py the pause rules, no GPU needed
```

## Licence

Code is MIT, see [LICENSE](LICENSE). The models carry their own: NLLB-200 is
[CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) and XTTS-v2 is
under the [Coqui Public Model License](https://coqui.ai/cpml).
