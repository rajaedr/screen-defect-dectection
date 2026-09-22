# Manufacturing Practice Documentation

## AI-Based Visual Inspection System for Smartphone and Laptop Screen Damage

This document frames the technical system (see `README.md` for the software
side) as a Manufacturing Practice case study: an automated visual
quality-control (QC) station for electronic-device screens.

---

## 1. The manufacturing inspection problem

In smartphone/laptop manufacturing, assembly, and refurbishment, every unit
must pass a **visual quality check** before shipping or resale: screens must
be free of scratches, cracks, dead pixels, and other visible defects that
affect either function or perceived product quality. This is a **100%
inspection** step in most consumer-electronics lines (every unit is checked,
unlike sampling-based inspection used for some other defect types), because
a visibly damaged screen reaching a customer is a high-impact failure (return,
refund, reputational damage).

## 2. Traditional / manual inspection

Historically this is done by a human inspector under controlled lighting,
visually scanning the screen (sometimes with the help of a jig or light
box), and manually recording pass/fail + defect type. This is:

- **Slow** relative to production line takt time.
- **Inconsistent** — fatigue, subjective judgment, and inter-inspector
  variability mean the same defect can be graded differently by different
  people or by the same person at different times of day.
- **Costly to scale** — throughput is bounded by the number of trained
  inspectors.
- **Not fully traceable** — a purely visual judgment call, unless
  photographed, leaves no reviewable record.

## 3. Proposed automated inspection

This project proposes replacing (or, more realistically for a prototype,
**augmenting**) manual inspection with a camera + AI pipeline that:

1. Captures a photo of the device (fixed camera on a line, or handheld for
   this prototype).
2. Automatically classifies device type and locates the screen.
3. Automatically detects and classifies visible screen defects.
4. Automatically estimates severity and an overall PASS/ATTENTION/FAIL
   result.
5. Produces a timestamped, storable digital inspection report (JSON/PDF).

## 4. Role of computer vision

Computer vision (classical, in the heuristic baseline: edge detection, Hough
line transform, contour/blob analysis) provides the first, zero-training
layer of the system: it can localize the screen and flag geometrically
distinctive damage patterns (long line segments = scratches/cracks; small
dark blobs = dead pixels/black spots; anomalous full-width bands = stuck
display lines) without needing any labeled training data. See
`src/damage_detection/defect_detector.py`.

## 5. Role of AI / deep learning

A trained convolutional detector (YOLOv8, see `MODEL_CARD.md`) is the
intended production-grade upgrade once sufficient labeled training data
exists: it can learn subtler, less geometrically-obvious defect patterns
(discoloration, backlight bleed, textured crack patterns) that hand-written
rules struggle to generalize across lighting/device variation. The
`training/` pipeline in this repository (`train.py`, `validate.py`,
`test.py`, `evaluation.py`) implements this path end-to-end; it is not yet
populated with a trained model because no adequately-sized, verified,
screen-specific public dataset was available at build time (see
`DATASET_SETUP.md`).

## 6. Quality-control workflow implemented

```
IMAGE
  ↓
DEVICE DETECTION        (smartphone vs laptop; "Unknown" if uncertain)
  ↓
SCREEN DETECTION         (localizes the screen region within the frame)
  ↓
SCREEN DAMAGE ANALYSIS   (classical CV or trained YOLO model)
  ↓
DEFECT CLASSIFICATION    (scratch, crack, dead pixel, display line, ...)
  ↓
LOCATION + CONFIDENCE    (bounding box + confidence per defect)
  ↓
SEVERITY ESTIMATION      (Low / Medium / High, engineering heuristic)
  ↓
FINAL QUALITY INSPECTION REPORT   (PASS / ATTENTION / FAIL + downloadable report)
```

This matches the standard "sense → analyze → classify → decide → record"
structure of an automated QC station.

## 7. Inspection criteria & defect classification

Defect classes (config.yaml → `damage_detection.classes`): no visible
defect, scratch, crack, broken/damaged glass, dead/stuck pixel, black spot,
display line (vertical/horizontal), discoloration, backlight bleed, other.
Each confirmed defect carries a class, a confidence score, and a bounding
box; low-confidence detections are explicitly marked **uncertain /
possible** rather than being silently treated as confirmed — deliberately
mirroring how a cautious human inspector would flag something for a second
look rather than reject or pass a unit on a marginal call.

## 8. System architecture

See `README.md` → "System architecture" for the software-level module
breakdown (`src/detection`, `src/device_recognition`,
`src/damage_detection`, `src/severity`, `src/reporting`, `app/`). At the
process level, it mirrors section 6 above 1:1 — each pipeline stage in code
corresponds to one QC-workflow stage in this document.

## 9. Data collection & dataset preparation

See `DATASET_SETUP.md` in full. Summary: public datasets are limited for
this specific domain (phone/laptop screen defects, as opposed to general
industrial surface defects); the recommended path for a genuinely reliable
model is team-collected photographs of the actual target devices, labeled
in YOLO format and merged via `datasets/prepare_dataset.py`.

## 10. Model training & evaluation

`training/train.py` fine-tunes YOLOv8n on the prepared dataset;
`training/validate.py` / `training/test.py` / `training/evaluation.py`
report precision, recall, F1, mAP@50, mAP@50:95, per-class AP, and a
confusion matrix — all computed from actual runs, never estimated (see
`MODEL_CARD.md` for the "no invented numbers" policy).

## 11. Inspection process (as a human user would run it)

1. Photograph the physical device screen (not a screenshot — see
   `README.md` "Screenshots vs. photos").
2. Upload it (or capture via webcam) in the Streamlit app.
3. Review the annotated image and inspection report.
4. For ATTENTION/FAIL results, or any "possible defect" finding, a human
   performs manual verification before any accept/reject decision — this
   prototype is a decision-support aid, not an autonomous
   accept/reject authority.
5. Download the JSON/PDF report for record-keeping.

## 12. Advantages of the proposed approach

- Consistent, repeatable application of the same criteria to every unit.
- Faster than manual inspection for the geometrically obvious defect
  classes handled by the heuristic baseline.
- Produces a timestamped, storable digital record automatically (audit
  trail), which manual-only inspection typically doesn't.
- Extensible: new defect classes can be added by relabeling data and
  retraining, without redesigning the workflow.

## 13. Limitations (Manufacturing-readiness specific)

- **Not validated against a certified QC standard or a labeled benchmark of
  real production units** — see `MODEL_CARD.md`.
- **Camera/lighting variability** on a real line would need a controlled
  capture rig (fixed distance, diffuse lighting) for consistent results;
  this prototype accepts arbitrary handheld photos, which is far more
  variable than a production inspection station would tolerate.
- **Throughput**: heuristic-mode inference is fast on a laptop CPU for
  single images, but this project has not been benchmarked against real
  production-line takt-time requirements, nor built for a continuous
  camera-feed / conveyor integration.
- **False positives/negatives are expected** — see `MODEL_CARD.md` →
  Limitations. This is why every result includes confidence scores and an
  explicit "uncertain" state rather than a bare pass/fail.
- This system is **not industrial-grade** and should not be described as
  such in any project report without a specific, cited validation result
  to back that claim.

## 14. Future improvements

- Train the YOLOv8 detector on a larger, team-collected + public dataset
  blend (see `DATASET_SETUP.md`) and report real evaluation metrics.
- Add segmentation (pixel-level crack/scratch extent) if repair-cost
  estimation becomes a project goal.
- Integrate a fixed-position camera rig + consistent lighting for a more
  production-realistic capture setup.
- Explore severity annotation (human-labeled defect severity) to replace
  the current rule-based heuristic with a learned severity estimator.
- Batch/continuous inspection mode for multiple units in sequence.

## 15. Application domains

- **Smartphone manufacturing**: end-of-line screen QC before packaging.
- **Laptop manufacturing**: similar end-of-line check, adapted for larger
  screens and keyboard/bezel context in frame.
- **Refurbishment**: grading used/returned devices for resale (e.g.
  "Grade A/B/C" workflows), where consistent automated grading criteria
  reduce disputes between refurbisher and buyer.
- **Repair / quality-control centers**: pre- and post-repair screen
  condition documentation (e.g. proving a screen was already damaged
  before a repair began, or verifying repair quality afterward).

In all four cases, this prototype demonstrates the workflow and
architecture; a real deployment would need the validation, camera rig, and
throughput work described in "Limitations" above before being trusted for
actual accept/reject decisions.
