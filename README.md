# Queue Intelligence System

Tracks customer dwell time in retail video footage with YOLOv8, flags moments
where the tracker likely got confused (an ID switch, an occlusion, someone
leaving and returning), and sends only those short clips to a Gemini vision
agent to decide whether the tracked times need correcting.

## How it works

1. **Track** — `queue_tracker.py` runs YOLOv8 detection + tracking on every
   frame, recording how long each person is visible. Bounding boxes inside
   user-defined "worker zones" are excluded from customer dwell-time stats.
2. **Detect confusion** — the same pass watches for three heuristic signals:
   an ID that vanishes while a new one appears right where it left off
   (`id_switch`), a person reappearing after a brief gap (`occlusion`), or
   someone reappearing well after the disappear threshold (`return_after_leave`).
3. **AI correction** — for each flagged event, `tools/segment_extractor.py`
   cuts a short padded clip and `agent.py` sends it to Gemini 2.5 Flash with a
   confusion-type-specific prompt, asking it to look at the actual video and
   decide whether the tracker's IDs should be merged or kept separate.
4. **Report** — corrected dwell times and the AI's full reasoning are written
   to `results/`.

## Which script do I run?

There are exactly three ways to use this project — pick one:

| I want to... | Run |
|---|---|
| Click around a full UI: pick a video, draw zones, watch tracking live, browse results | `streamlit run streamlit_app.py` (or `run_streamlit.bat` / `run_streamlit.sh`) |
| Run the whole pipeline from a terminal (with or without arguments) | `python run_tracking.py [video_path] [fps_limit] [disappear_threshold]` |
| Check whether GPU acceleration is set up correctly | `python check_cuda_setup.py` |

`streamlit_app.py` itself launches `run_tracking.py` as a subprocess in its
own console window when you click "Start Processing" in Terminal mode — so
`run_tracking.py` is the one true processing engine either way.

## Project layout

| File | Purpose |
|---|---|
| `queue_tracker.py` | Core `DwellTimeTracker`: YOLO tracking + confusion heuristics |
| `agent.py` | Builds prompts and calls Gemini for confusion analysis |
| `tools/segment_extractor.py` | Extracts padded video clips around a confusion event |
| `pipeline.py` | Shared track → detect → analyze loop used by `run_tracking.py` |
| `run_tracking.py` | The one CLI entry point (see table above) |
| `streamlit_app.py` | Full operational web UI (live processing, zone editor, results browser) |
| `workzone.py` | Interactive tool to draw worker-exclusion polygons on a frame |
| `config.py` | Loads/persists settings from `.env` |
| `gpu_utils.py` | Shared CUDA/CPU device detection |
| `check_cuda_setup.py` | The one GPU diagnostic script |

## Setup

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the CUDA build of PyTorch for your platform
(see the comment in `requirements.txt`), then verify with
`python check_cuda_setup.py`.

Create a `.env` file (see variables read in `config.py`):

```
GOOGLE_API_KEY=your-gemini-api-key
VIDEO_PATH=data/input/your-video.mp4
```

Optional tuning knobs (defaults shown): `DISAPPEAR_THRESHOLD=10.0`,
`FPS_LIMIT=15`, `YOLO_CONF=0.4`, `YOLO_IOU=0.5`,
`CONFUSION_MATCH_DISTANCE_PX=50`, `CONFUSION_DEBOUNCE_FRAMES=5`.

## Running

```bash
# Draw worker zones once (optional but recommended)
python workzone.py data/input/your-video.mp4

# CLI run: uses .env defaults if you omit the arguments
python run_tracking.py data/input/your-video.mp4

# Headless (no OpenCV window, no end-of-run pause) - useful for scripting
python run_tracking.py data/input/your-video.mp4 --no-preview --no-pause

# Full web UI
streamlit run streamlit_app.py
```

## Hosted demo (no GPU, no API key)

`demo/demo_app.py` is a separate, lightweight Streamlit app that replays
precomputed results instead of running YOLO/Gemini live — meant for hosting
somewhere with no GPU (e.g. Streamlit Community Cloud). It only imports
`streamlit`, `pandas`, and `plotly` (see `demo/requirements.txt`).

It lives in its own `demo/` subdirectory - alongside its own lightweight
`requirements.txt` - specifically so Streamlit Community Cloud picks that one
up instead of the repo root's `requirements.txt` (torch/ultralytics/etc,
meant for the full operational app).

To (re)generate the data it reads from `demo_data/<slug>/`:

```bash
# 1. Run the real pipeline once per sample video (needs GPU + GOOGLE_API_KEY)
python run_tracking.py data/input/result3.mp4 --no-preview --no-pause

# 2. Package that run's results/, tracked video, and confusion clips into
#    demo_data/, re-encoding video to browser-friendly H.264 (needs ffmpeg)
python build_demo_data.py result3 "Store Camera 1" data/input/result3.mp4

# Repeat both steps for each additional sample video before running the next
# one - step 1 overwrites results/ and data/output/, so package before you
# move on.

# 3. Run the demo app locally
streamlit run demo/demo_app.py
```

### Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (already done if you're reading this from there).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click "New app", pick this repo and the `main` branch.
4. Set the main file path to `demo/demo_app.py`.
5. Deploy. It auto-redeploys on every push to `main`.

`demo_data/` is intentionally committed to git (it's what gets hosted) —
everything else generated by the real pipeline (`results/`, `data/output/`)
stays gitignored.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

Tests cover the confusion-detection heuristics (including a regression test
for a debounce-counter bug where `id_switch` events could never fire),
person-ID parsing, config validation, and GPU/CPU device resolution — all
without needing a GPU, YOLO weights, or a live API key.
