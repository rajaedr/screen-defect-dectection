# Model Card — AI Visual Screen Inspection

This follows the spirit of standard ML model cards (Mitchell et al., 2019)
to document what this system does and does not do, honestly.

## Overview

| | |
|---|---|
| Task | Device type recognition (smartphone/laptop) + screen localization + visible screen defect detection & classification |
| Intended use | University Manufacturing Practice course project / QC-workflow demonstrator |
| NOT intended for | Real manufacturing quality-control decisions, insurance claims, resale grading, or any decision with financial/safety consequences |

## Components & current mode

The system has three independently swappable stages, each with a
zero-training baseline and an optional trained-model upgrade path
(`config/config.yaml`):

| Stage | Default (MVP) mode | Trained-model mode | Status |
|---|---|---|---|
| Device recognition | `heuristic` — image aspect ratio | `yolo_cls` — fine-tuned classifier | Heuristic active by default; no model trained yet |
| Screen localization | `contour` — classical CV edge/contour analysis | `yolo` — fine-tuned detector | Contour active by default; no model trained yet |
| Defect detection | `heuristic` — edge/line/blob classical CV | `yolo` — fine-tuned YOLOv8 detector | Heuristic active by default; no model trained yet |

**As shipped, no deep-learning model has been trained on real defect
data** — see `DATASET_SETUP.md` for why (no adequately-sized, verified,
screen-specific public dataset was found; classical CV baseline used
instead). If your team trains a model with `training/train.py`, **update
this file with the actual metrics from `training/evaluation.py`'s output**
(`results/evaluation_summary.json`) before claiming any accuracy number —
this file must never contain invented numbers.

## Architecture (once trained)

- **Base model:** YOLOv8n (nano), via Ultralytics.
- **Why YOLOv8:** single-stage detector, fast enough for CPU/Apple-Silicon
  (MPS) inference with no NVIDIA GPU requirement (see `README.md` →
  Requirements), native bounding-box + confidence output, mature Python API
  with built-in train/val/test/metrics support so this project doesn't need
  a hand-rolled training loop.
- **Why not segmentation (YOLOv8-seg) for v1:** bounding boxes are
  sufficient to localize+classify a defect for a QC report, and detection
  needs far fewer pixel-perfect mask annotations than segmentation — which
  matters given the tiny available datasets (see `DATASET_SETUP.md`).
  Segmentation is a reasonable **future improvement** if per-pixel crack
  extent becomes important (e.g. for repair-cost estimation) and enough
  labeled masks exist.
- **Why not a pure classification model:** we need *location* (bounding
  box) for the annotated-image UI requirement, which classification alone
  cannot provide.

## Heuristic (classical CV) baseline — how it actually works

Documented in detail in `src/damage_detection/defect_detector.py`. Summary:
- Canny edges + probabilistic Hough transform find long straight line
  segments → clustered by proximity → classified as `scratch` (few, long,
  isolated segments) or `crack` (many intersecting/branching segments).
- Row/column median-intensity outlier analysis finds full-width/height
  anomalous bands → `display_line`.
- Small, compact, very dark contours → `dead_pixel` (tiny) / `black_spot`
  (larger).
- Bright, low-saturation regions are excluded from all of the above to
  reduce glare/reflection false positives.
- Every detection below `uncertain_confidence_threshold` (config.yaml) is
  marked **uncertain** ("possible defect — manual inspection required")
  rather than reported as confirmed.

**This is a rule-based baseline, not a learned model.** It has no concept
of `broken_glass`, `discoloration`, or `backlight_bleed` beyond what the
line/blob heuristics coincidentally catch — those three classes are
effectively unimplemented in heuristic mode and exist mainly as YOLO-mode
target classes once you have training data for them.

## Device / model recognition

- Smartphone vs. laptop: heuristic uses whole-image aspect ratio (wide →
  laptop, tall → smartphone) with a deliberately ambiguous middle band that
  reports `"unknown"` rather than guessing. This is intentionally simple
  and will be wrong for atypical crops/angles (e.g. a laptop screen
  photographed extremely close-up, or a phone held landscape).
- **Specific device model (e.g. "iPhone 14", "MacBook Air M2") is NOT
  implemented.** We do not have a legally-usable, adequately-labeled
  public dataset for per-model recognition, and a wrong specific-model
  guess is worse for a QC tool than an honest "Unknown". Every report
  shows `Specific model: Unknown` unless/until a team trains and documents
  such a classifier (update this card if you do).

## Known limitations (read before using this for anything beyond the course demo)

1. **Screenshots cannot show physical damage.** A screenshot is a capture
   of digital framebuffer pixels; a scratch or crack that exists on the
   physical glass will not appear in a screenshot unless it happens to
   distort what's rendered underneath (it usually doesn't). "Screenshot
   mode" in the app is explicitly limited to digital/display-anomaly
   classes (stuck lines, discoloration) — see `app/app.py`.
2. **False positives from glare, fingerprints, dust, and on-screen
   content** are a fundamental challenge for a photo-based system (see
   `DATASET_SETUP.md` "domain-gap" and the heuristic's glare-masking logic).
   The confidence threshold + "uncertain" state exist specifically to
   surface this honestly rather than hide it.
3. **Screen content vs. physical damage** cannot always be distinguished
   by either the heuristic or a small trained model — a high-contrast
   wallpaper edge can visually resemble a scratch. The "screen test-pattern
   mode" (solid color photographed on-screen) meaningfully reduces this for
   the digital-anomaly classes, but is an optional workflow step the user
   must do themselves.
4. **No dataset with human-annotated severity exists.** Severity (Low/
   Medium/High) is a configurable engineering heuristic
   (`src/severity/severity_rules.yaml`), not a validated grading standard —
   see the header comment in that file.
5. **Small/no training data** for `broken_glass`, `discoloration`,
   `backlight_bleed` as photographed defects (see `DATASET_SETUP.md`).
6. **Not evaluated on a held-out benchmark as shipped** — no numeric
   accuracy claim is made for defect detection in this card. If/when your
   team trains a model, this section must be replaced with real numbers
   from `results/evaluation_summary.json`, generated by
   `training/evaluation.py` — never invent or estimate metrics here.
7. **This is a university prototype**, not a certified, industrial, or
   commercially validated inspection system. See "Manufacturing readiness"
   in `docs/MANUFACTURING_PRACTICE.md`.

## Explicitly false claims we will not make

Per project requirements, this system and its documentation will never
claim: "100% accurate", "detects every scratch", "works on every phone", or
"industrial-grade", unless a specific, cited evaluation result actually
supports that specific claim (and even then, only with the exact numbers
and their evaluation conditions stated alongside).

## How to fill in real metrics once you train a model

```bash
python datasets/prepare_dataset.py
python training/train.py
python training/evaluation.py
cat results/evaluation_summary.json
```
Then paste the `precision`, `recall`, `f1_score`, `map50`, `map50_95`, and
`per_class_ap50` values from that JSON file into a new "Measured
performance" section here, along with the date, dataset used, and test-set
size — so anyone reading this card knows exactly what was measured and on
what.
