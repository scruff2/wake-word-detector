# Wake-Word Detector

Offline wake-word detection for laptop microphones using `sounddevice` and
`faster-whisper`.

This project is intentionally detector-only. It listens locally, transcribes
short overlapping microphone windows, checks transcripts for configured wake
phrases, and triggers a small local handler. The default handler writes a
console log and plays a chime.

The primary intent is to use this package as a reusable module inside other
local voice applications. The command-line app is included for standalone
testing, tuning, and troubleshooting before embedding the detector in a larger
assistant or automation project.

## Features

- Fully local microphone capture and transcription
- Configurable wake phrases, model, device, compute type, chunk size, overlap,
  VAD threshold, fuzzy matching, logging, and chime behavior
- Simple energy-based VAD to skip silent chunks
- Regex wake phrase matching with word boundaries
- Optional fuzzy matching for near misses
- Importable detector components for future projects
- CLI entry point for standalone testing

## Why Transcription-Based Wake Phrases

Many wake-word systems require a dedicated trained model for each supported
word or phrase. That works well for highly optimized fixed wake words, but it
can make custom phrases harder to support because every new phrase may need
recordings, training, validation, and model packaging.

This project takes a different approach: it transcribes short local audio
windows with Whisper, then searches the transcript for configured phrases. That
has several practical advantages:

- Wake phrases can be changed in config without training a new model.
- Multi-word phrases are supported the same way as single words.
- Multiple alternative phrases can run at the same time.
- Phrase matching can use normal text tools such as regex boundaries and
  optional fuzzy matching.
- The detector can log transcripts, which makes tuning and debugging easier.
- The same module can support many downstream apps with different wake phrases.

The tradeoff is that this approach is heavier than a tiny dedicated wake-word
model. It uses more CPU/GPU and usually has higher latency, but it is flexible,
private, and straightforward to adapt.

## Requirements

- Python 3.9+
- Working microphone
- Windows, macOS, or Linux
- Optional NVIDIA GPU with CUDA for faster transcription

## Setup

Create a virtual environment and install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .[dev]
```

If your shell has trouble with the optional dependency syntax on Windows, use:

```powershell
pip install -e ".[dev]"
```

## Configuration

The default configuration lives in `config.yaml`:

```yaml
model: tiny.en
compute_type: int8
device: cpu
language: en
wake_phrases:
  - "computer"
chunk_seconds: 2.0
overlap_seconds: 0.8
sample_rate: 16000
vad_energy_threshold: 0.01
fuzzy_threshold: 0
cooldown_seconds: 2.0
debug: false
log_file:
chime: true
```

Useful model choices:

- `tiny.en`: fastest default for quick local testing
- `base.en`: better accuracy with moderate CPU cost
- `small.en`: stronger accuracy, more latency on CPU
- `medium.en`: usually best reserved for GPU use

Useful compute choices:

- CPU: `int8`
- CUDA: `int8_float16` or `float16`

## Run

```powershell
wake-word-detector --config config.yaml
```

Or without installing the script entry point:

```powershell
python -m wake_word_detector --config config.yaml
```

Override the wake phrase from the command line:

```powershell
wake-word-detector --wake-phrase "computer" --wake-phrase "ok assistant"
```

Tune chunking and VAD from the command line:

```powershell
wake-word-detector --chunk-seconds 2.0 --overlap-seconds 1.0 --vad-energy-threshold 0.005
```

Use a larger model on CUDA when CUDA/cuDNN dependencies are installed:

```powershell
wake-word-detector --model small.en --whisper-device cuda --compute-type int8_float16
```

Disable the chime:

```powershell
wake-word-detector --no-chime
```

Stop with `Ctrl+C`.

## CUDA Notes

`faster-whisper` uses CTranslate2. CUDA acceleration requires compatible NVIDIA
drivers and CUDA/cuDNN libraries for your installed CTranslate2 version. If CUDA
startup fails, switch back to:

```yaml
device: cpu
compute_type: int8
```

For this detector, start with `tiny.en` or `base.en` on CPU before tuning GPU
settings. It makes microphone, VAD, and phrase detection issues easier to
separate from GPU setup issues.

## Module Usage

Future projects can import the parser directly:

```python
from wake_word_detector import WakePhraseDetector

detector = WakePhraseDetector(("computer", "ok assistant"))
event = detector.detect("Computer, start listening")
if event:
    print(event.phrase)
```

For full runtime integration, pass your own wake handler:

```python
from wake_word_detector.actions import WakeHandler

def on_wake_detected(event):
    print(f"Detected {event.phrase}: {event.transcript}")
```

The handler receives a `WakeDetectionEvent` with the matched phrase, transcript,
optional confidence value, detection timestamp, and optional fuzzy distance.

## Testing

Run unit tests:

```powershell
pytest
```

Current tests cover:

- Exact phrase detection
- Word boundary behavior
- Multi-word phrase matching
- Multiple phrase alternatives
- Fuzzy matching threshold behavior
- Config loading validation
- Energy VAD basics

Recorded-audio integration tests are not included yet because they require
sample audio files and model download time. Add them under `tests/fixtures/`
when you start collecting real microphone samples.

## Troubleshooting

If no microphone audio is captured:

- Confirm your OS granted microphone permission to the terminal.
- Try another microphone device.
- Run with `--debug` to see audio stream warnings.

If nothing is transcribed:

- Lower `vad_energy_threshold`, for example `0.005`.
- Increase `chunk_seconds` to `3.0`.
- Confirm the model downloaded successfully on first run.

If there are false positives:

- Keep `fuzzy_threshold: 0`.
- Increase `cooldown_seconds`.
- Use a less common wake phrase.

If wake detection feels slow:

- Use `tiny.en` or `base.en`.
- Keep `chunk_seconds` near `2.0`.
- Keep `overlap_seconds` between `0.5` and `0.8`.
- Use CUDA only after CPU behavior is confirmed.
