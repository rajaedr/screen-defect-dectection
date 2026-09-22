# AI Visual Screen Inspection

**AI-Based Visual Inspection System for Smartphone and Laptop Screen Damage**
— a Manufacturing Practice university project.

An automated quality-control prototype: you give it a **photo** of a
smartphone or laptop, and it detects the device type, locates the screen,
finds visible defects (scratches, cracks, dead pixels, stuck lines, etc.),
estimates severity, and produces a downloadable inspection report
(PASS / ATTENTION / FAIL).

> **This is a university prototype, not a certified industrial system.**
> See `MODEL_CARD.md` for exactly what it can and can't do, and why.

---

## 1. What this project does

```
IMAGE
  ↓
DEVICE DETECTION        → Smartphone / Laptop / Unknown
  ↓
SCREEN DETECTION        → locates the screen in the photo
  ↓
DAMAGE ANALYSIS         → classical CV baseline, or a trained YOLOv8 model
  ↓
DEFECT CLASSIFICATION   → scratch, crack, dead pixel, display line, ...
  ↓
LOCATION + CONFIDENCE   → bounding box + % confidence per defect
  ↓
SEVERITY ESTIMATION     → Low / Medium / High (engineering heuristic)
  ↓
INSPECTION REPORT       → PASS / ATTENTION / FAIL + downloadable JSON/PDF
```

**Important:** a **screenshot** cannot show physical scratches/cracks — it
only captures digital pixels. Use real **photos** of the physical device for
physical damage detection. The app has a separate "Screenshot mode" toggle
for digital-only anomaly checks (stuck lines, discoloration).

## 2. System architecture

```
screen-defect-detection/
├── app/                    Streamlit UI (app.py + reusable components)
├── src/
│   ├── detection/           screen localization
│   ├── preprocessing/       resize, quality checks
│   ├── device_recognition/  smartphone vs laptop
│   ├── damage_detection/    defect detection & classification
│   ├── severity/            severity heuristic
│   ├── reporting/           annotation drawing + JSON/PDF reports
│   ├── utils/                config loader, logger, image I/O
│   └── pipeline.py           ties every stage together
├── training/                train.py / validate.py / test.py / evaluation.py / predict.py
├── datasets/                 download_datasets.py / prepare_dataset.py / raw / processed
├── models/                   trained weights go here (git-ignored)
├── results/                  training/evaluation outputs (git-ignored)
├── tests/                    pytest suite
├── config/config.yaml        every tunable value — read this before editing code
├── sample_images/            what test photos to collect (see its README)
├── docs/MANUFACTURING_PRACTICE.md
├── DATASET_SETUP.md          real datasets, licenses, domain-gap discussion
├── MODEL_CARD.md              capabilities, limitations, no-invented-numbers policy
├── setup.sh / run.sh
└── requirements.txt
```

Everything is config-driven (`config/config.yaml`) — no thresholds or paths
are hardcoded inside the Python source.

## 3. Requirements

- **macOS** (also runs on Linux; not tested on Windows).
- **Python 3.10–3.12** (3.11 recommended). No NVIDIA GPU required — runs on
  CPU, and uses Apple Silicon MPS acceleration automatically when available.
- ~3–5 GB free disk space (mostly for `torch`/`ultralytics` and their
  dependencies).

## 4. Installation

```bash
cd screen-defect-detection
chmod +x setup.sh run.sh
./setup.sh
```

`setup.sh` creates an isolated virtual environment (`.venv/`) and installs
everything from `requirements.txt` into it — it does not touch your system
Python. First run takes a few minutes (downloading `torch` is the slow
part).

## 5. Running the application (the MVP works immediately — no dataset needed)

```bash
./run.sh
```

This opens the Streamlit app in your browser (usually
`http://localhost:8501`). The defect-detection baseline is a classical
computer-vision heuristic that needs **zero training data**, so you can use
the full pipeline right after `./setup.sh` — no dataset download or
training step is required to try it out.

### Simplest possible walkthrough

```
STEP 1   Open Terminal
STEP 2   cd into the screen-defect-detection folder
STEP 3   Run: ./setup.sh          (first time only)
STEP 4   Run: ./run.sh
STEP 5   Your browser opens the app — click "Upload Image" and choose a
         photo of a phone or laptop screen
STEP 6   See the annotated image + inspection report; click
         "Download Inspection Report" for a PDF/JSON copy
```

## 6. How to upload an image

- **Upload tab:** any `.jpg/.jpeg/.png/.bmp/.webp/.tiff` file of a real
  device.
- **Camera tab:** capture directly from your browser's webcam (falls back
  gracefully — if your browser blocks camera access, just use Upload).
- **Test-pattern tab:** for dead pixels / stuck lines / discoloration,
  display a solid color full-screen on the device first (see the tab for
  instructions), then photograph and upload that — much more reliable than
  a normal wallpaper/app screen for those specific defect types.

## 7. How to interpret results

- **Overall Result** — `PASS` (no defects found), `ATTENTION` (defects
  found, or only low-confidence "possible" defects — recommend manual
  check), `FAIL` (high-severity confirmed defect(s), e.g. a large crack or
  broken glass).
- **Confirmed defects** — shown in **red** on the annotated image, with a
  class label and confidence %.
- **Possible defects** — shown in **orange**, below the confidence
  threshold needed to "confirm" — the system is deliberately conservative
  here rather than making a false claim (see `MODEL_CARD.md` → false
  positive control).
- **Severity** — Low/Medium/High, an explicit **engineering heuristic**
  (`src/severity/severity_rules.yaml`), not a certified grading standard.
- The full JSON report (and an expandable view in the app) has every
  field machine-readable, for your own further processing if useful.

## 8. Dataset setup (optional — only needed to train a real model)

The app works out of the box using the classical-CV heuristic. If you want
to train the YOLOv8 model instead (`damage_detection.mode: "yolo"` in
`config/config.yaml`):

```bash
# 1. Get data — see DATASET_SETUP.md for exact dataset names/URLs/licenses
export ROBOFLOW_API_KEY="your_free_roboflow_api_key"   # optional, for automatic download
pip install roboflow                                     # optional extra, not in requirements.txt
python datasets/download_datasets.py
#   ...or download manually per DATASET_SETUP.md, or add your own team
#   photos per datasets/README.md — either/both works.

# 2. Merge everything into a unified train/val/test split
python datasets/prepare_dataset.py
```

Read `DATASET_SETUP.md` in full — it also explains the **domain-gap
problem** (a model trained on generic industrial scratches will not
reliably detect scratches on a glass phone screen) and exactly which
defect classes currently lack adequate public training data.

## 9. Training

```bash
python training/train.py
```

Reads every hyperparameter from `config/config.yaml -> training` (epochs,
batch size, image size, learning rate, etc.) — edit the config, not the
script, to change them. Trains YOLOv8n (nano) by default; auto-selects
Apple Silicon MPS / CUDA / CPU. The best checkpoint is copied to
`models/defect_yolo.pt` automatically.

Then:

```bash
python training/validate.py     # quick metrics on the 'val' split, during/after training
python training/test.py         # FINAL held-out metrics on the 'test' split — run once
python training/evaluation.py   # full report: precision/recall/F1/mAP/per-class AP/confusion matrix
                                 # -> results/evaluation_summary.json + plots
```

Once you have real metrics, switch the pipeline to use your trained model:

```yaml
# config/config.yaml
damage_detection:
  mode: "yolo"    # was "heuristic"
```

Then **update `MODEL_CARD.md`** with the actual numbers from
`results/evaluation_summary.json` — never invent or estimate accuracy
numbers, per the project's honesty requirements.

You can also run inference from the command line without the UI:

```bash
python training/predict.py sample_images/your_photo.jpg
```

## 10. How to add new defect classes

1. Add the class name to `config/config.yaml -> damage_detection.classes`.
2. Add a base severity weight for it in
   `src/severity/severity_rules.yaml -> defect_base_weight`.
3. Collect/label training images for the new class (YOLO format) under
   `datasets/raw/<your_dataset>/` (see `datasets/README.md`), including a
   `classes.txt`/`data.yaml` that lists the class.
4. If the raw dataset uses a different name for it, add a mapping entry in
   `CLASS_NAME_MAP` inside `datasets/prepare_dataset.py`.
5. Re-run `python datasets/prepare_dataset.py` then `python training/train.py`.

(The classical heuristic baseline does **not** auto-support new classes —
it only implements scratch/crack/display_line/dead_pixel/black_spot. A new
class needs the YOLO path.)

## 11. How to retrain the model

Same as section 9 — re-run `python datasets/prepare_dataset.py` (if data
changed) then `python training/train.py`. Each run overwrites
`models/defect_yolo.pt` with the new best checkpoint; previous run outputs
remain under `results/train_run/` from Ultralytics (rename/move that folder
first if you want to keep multiple experiments side by side).

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `./setup.sh: Permission denied` | Run `chmod +x setup.sh run.sh` first |
| No `python3` found | Install Python 3.11: `brew install python@3.11` (or from python.org) |
| `pip install` fails / very slow | Check your internet connection; `torch` is a large download. Re-run `./setup.sh` — it's safe to re-run |
| `streamlit: command not found` | `.venv` wasn't activated or setup didn't finish — re-run `./setup.sh`, then `./run.sh` |
| App says "Screen region could not be confidently localized" | Normal for cluttered backgrounds/extreme angles — the classical baseline works best face-on, screen filling most of the frame. Analysis still runs on the full image, just less precisely |
| Everything flagged as "possible defect" only | The image may be blurry, low-resolution, or have strong glare — check the yellow warning banners at the top of the results; retake the photo |
| `ultralytics`/`torch` import errors when switching to `damage_detection.mode: "yolo"` | Re-run `pip install -r requirements.txt` inside `.venv`; make sure `models/defect_yolo.pt` actually exists (train first) |
| Camera tab doesn't work | Browser/OS blocked camera permission — use the Upload tab instead; file upload always works |
| `FileNotFoundError: Config file not found` | Run commands from the project root directory, or check `config/config.yaml` wasn't moved/deleted |
| Tests fail to run: `No module named pytest` | `pip install -r requirements.txt` inside the activated `.venv` (pytest is included) |

## 13. Limitations

See `MODEL_CARD.md` for the full, honest list. In short: no deep-learning
model ships pretrained (see `DATASET_SETUP.md` for why); the classical CV
baseline is a reasonable starting demonstration but will miss subtle
defects and can be fooled by glare/fingerprints/dust/on-screen content;
specific device *model* recognition (e.g. "iPhone 14") is not implemented;
severity is a configurable heuristic, not a certified standard.

## 14. Future improvements

- Train YOLOv8 on a real, team-collected + public dataset blend and report
  measured metrics (`MODEL_CARD.md`).
- Segmentation model for pixel-level crack/scratch extent.
- Fixed camera rig + controlled lighting for production-realistic capture.
- Human-labeled severity dataset to replace the rule-based heuristic.
- Batch/continuous inspection for multiple units.

Full discussion: `docs/MANUFACTURING_PRACTICE.md`.

## Privacy & security

Images are processed **entirely locally** on your machine by this
application — nothing is uploaded to an external server by this app itself.
(If you separately choose to use the optional `roboflow` dataset-download
integration, that talks to Roboflow's servers for *dataset* downloads only,
never your inspection photos.)

## Running the tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Covers image loading/validation, preprocessing, device recognition, screen
detection, defect detection + confidence thresholds, severity calculation,
and report generation — using synthetic generated images (see
`tests/conftest.py`) so the suite runs deterministically without needing
real device photos.

## License

MIT for the source code in this repository — see `LICENSE`. Any datasets
you download and any pretrained/fine-tuned model weights you produce or use
carry their own separate licenses (see `DATASET_SETUP.md` and
`MODEL_CARD.md`).
