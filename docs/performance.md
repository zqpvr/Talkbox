# Performance notes

Measurements and dead ends from building Talkbox, kept out of the README so it
stays readable. Everything here was measured on this machine: Windows 11, an
RTX 4060 with 8 GB, and an Intel Core i7-14700F.

## Latency

It works sentence by sentence, not word by word. It waits for you to finish a
clause, then translates and speaks it.

Measured on an RTX 4060 at Medium, clock started at the last frame of actual
speech:

    median 620-700 ms  from you stopping to the first translated audio
    range  556-812 ms  across runs

Two figures because it depends on what else is touching the GPU. A run with the
card idle gave 556/620/695 (min/median/max); a run started while the card was
at 41% gave 589/698/812. Down from 1274 ms either way. What is left is close to the floor: about 240 ms waiting to
be sure you have stopped, and about 200 ms for the voice model to produce its
first audio.

### What made it faster

**Transcribing during the pause.** Trailing silence does not change what
Whisper hears: measured identical text with 0, 120, 250 and 400 ms of silence
appended. So a fraction into a pause the audio is sent off to be transcribed
and translated, and the answer is usually waiting by the time the pause is
confirmed. A guess is only reused if no further speech arrived, checked by
comparing speech-frame counts, so a stale guess cannot be mistaken for a
current one.

**Ending the pause early when the sentence is clearly finished.** Whisper
punctuates. On partial cuts of a three-sentence passage, trailing punctuation
identified every real sentence end and rejected every mid-clause cut but one,
where it invented an ending at 0.77 confidence. So when a guess ends in
terminal punctuation and scores above 0.85, the pause is allowed to end at
240 ms instead of 400 ms. Otherwise it waits the full time. This splits at
sentence boundaries, which are safe places to split.

**A faster translator.** NLLB runs on CTranslate2 rather than HuggingFace
`generate`, which spends a Python loop per generated token. Same model, same
output, 1.9x faster (311 ms to 167 ms) and 0.49 GB less VRAM.

**A smaller streaming chunk.** Time to first audio is 19 ms of fixed setup plus
25 ms per token of lead-in, so the chunk size is nearly all of it. Simulating
playback on an idle GPU: chunk 4 runs the buffer dry by 0.32 s, chunk 6 keeps
0.23 s in hand, chunk 8 keeps 0.32 s. Six is the fastest safe setting.

### Things that were tried and did not help

Worth recording so nobody spends the afternoon again:

- **int8 quantisation for the recogniser.** `int8_float16` was *slower* than
  plain `float16` (297 ms against 242 ms on the same clip). `float16` is
  already the right choice.
- **Shrinking Whisper's encoder window** with `chunk_length`. No effect. The
  encoder already scales with input length: 114 ms for 2 s of audio, 204 ms for
  6 s. It is not padding short clips out to 30 seconds.
- **DeepSpeed for XTTS.** coqui-tts supports `use_deepspeed=True` and it is the
  documented way to speed up the token loop. The wheel installs, but the
  inference engine needs `CUDA_HOME`, meaning a full CUDA Toolkit rather than
  the pip CUDA wheels everything else here uses. It fails at model load with
  `MissingCUDAException`. Not worth making every user install a toolkit.
- **torch.compile on XTTS.** Needs Triton, which PyTorch does not ship for
  Windows; the community `triton-windows` build does work. With the default
  compile mode the result was 239 ms against 241 ms uncompiled, which is noise.
  `mode="reduce-overhead"` looked faster but that is CUDA graphs, and the same
  setting crashed the vocoder with a graph-tensor reuse error, so the apparent
  gain is not trustworthy. Removed again.

### Where the time actually goes

Instrumenting a live run, milliseconds after the speaker stops:

    speculation fired     100
    transcribe done       277
    pause confirmed       293
    translate done        377
    first audio out       571

The pause is no longer the constraint. Recognition, translation and synthesis
run one after another and together outlast it, so the chain is what sets the
latency. That is worth knowing before optimising: an earlier version of this
file claimed recognition and translation were hidden inside the pause, which
was true when the pause was a fixed 400 ms and stopped being true once early
commit shortened it. Making any of the three stages faster now helps directly.

Rough floor for this design: 100 ms before a guess starts, plus about 200 ms
recognising, 150 ms translating and 200 ms before the first audio. Around
650 ms, which is close to what it does.

### Smaller models

Recognition sits on the critical path, so a smaller recogniser buys latency
directly. Measured on twelve clips of one speaker's own voice, scored against
the prompt text they read:

| Model | Median | Word error | Against small |
|---|---|---|---|
| tiny | 73 ms | 22.5% | -118 ms, +4.0 points |
| base | 106 ms | 25.1% | -85 ms, +6.6 points |
| small *(default)* | 191 ms | 18.5% | - |
| medium | 401 ms | 16.7% | +210 ms, -1.8 points |

Going smaller costs more than it saves. In a translation pipeline a misheard
word does not stay a misheard word: it becomes a mistranslated sentence, spoken
talkbox in your voice. Four points of word error to save 118 ms is a bad trade.
Medium is the opposite trade and also marginal, 1.8 points for 210 ms, which is
why it sits behind the High preset rather than being the default.

`base` scored worse than `tiny` overall here, which is not what the published
benchmarks say. Looking per clip, `tiny` won six and `base` won two, but the
two `base` won were numbers and deliberately awkward phonetics, which are
exactly the cases that matter on a call. Twelve clips from one speaker is not
enough to act on, so Low keeps `base`.

### Under load

The numbers above are for an idle GPU. Anything else using the card changes
them completely. Measured while RPCS3 happened to be compiling shaders, which
pinned the GPU at 95%: the same sentence went from real-time factor 0.94 to
2.9, and playback would have run dry by many seconds.

That was a transient compile spike rather than steady gameplay, so treat it as
an illustration of what contention does, not a measurement of gaming. Either
way, no buffering strategy fixes a busy GPU. Use Low when something else needs
the card, and check `nvidia-smi` before trusting any timing you measure
yourself.

Long sentences get a small head start before playback, 10% of the expected
duration capped at 0.6 s, because they generate at real-time factor 0.84 to
0.94 and the thinnest measured margin was 0.11 s. Sentences under six seconds
get none.

Run `tools\smoke_test.py` for the numbers on your own machine. It makes its own
test audio and needs no microphone.


## Other models considered

The voice model is the one worth revisiting, so here is where that stands as of
August 2026.

**XTTS-v2** is what ships. Seventeen languages, zero-shot cloning, real
streaming, and it works. Its licence is the drawback: the Coqui Public Model
License is non-commercial, which constrains what anyone can do with the output
even though the code here is unrestricted.

**Qwen3-TTS** looked like the strongest alternative, so it was installed and
measured against XTTS on this machine with the same reference voice. It is
slower here by a wide margin:

| | XTTS-v2 | Qwen3-TTS 1.7B |
|---|---|---|
| First audio | 171 ms | 11,600 ms |
| Full utterance (2.6 s of speech) | 1,507 ms | 11,600 ms |
| Real-time factor | 0.55 | 4.53 |
| VRAM | 2.15 GB | 4.04 GB |
| Languages of our 17 | 17 | 10 |

The first-audio gap is the decisive one, and it is structural rather than a
matter of tuning. `generate_voice_clone` returns a finished array, not a
generator, even with `non_streaming_mode=False`. Nothing can be played until
the whole utterance is synthesised, so first audio equals full synthesis. XTTS
streams, so playback starts on its first chunk while the rest is still being
made.

Two caveats worth stating. The package warns that flash-attn is missing and
that it is therefore running "the manual PyTorch version", so this is not the
model at its best; flash-attn has no Windows wheels, same as DeepSpeed. And the
97 ms figure quoted for Qwen3-TTS elsewhere is presumably its serving stack on
server hardware, not this package on a consumer card.

Its own `get_supported_languages()` returns Chinese, English, French, German,
Italian, Japanese, Korean, Portuguese, Russian and Spanish, so adopting it
would drop Arabic, Czech, Dutch, Hindi, Hungarian, Polish and Turkish.

The 0.6B checkpoint exists and was tried, since a smaller model would be the
obvious answer to the speed problem. It loads and builds a cloning prompt, then
dies during generation with a CUDA device-side assertion, on the exact
`transformers` version the package pins. Size would not have fixed it anyway:
without a streaming API, first audio equals full synthesis whatever the model
weighs.

Worth re-measuring if it gains a streaming API. Until then it is slower,
heavier and narrower here, whatever the benchmarks say.

Also looked at: **CosyVoice2** (fast streaming, fewer languages),
**Fish Speech** (very broad language coverage), **F5-TTS** and
**Chatterbox**. **Kokoro** is excellent and permissively licensed but has fixed
voices and no cloning, so it cannot do the one thing this app exists for.

On the recogniser and translator there is nothing obviously better for this
job. Whisper is still the practical choice for multilingual speech, and NLLB
covers 200 languages in one model, which matters because the alternative is
loading a different checkpoint per direction.

