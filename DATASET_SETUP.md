# Dataset Setup

This document is the complete, honest dataset strategy for the project. Read
section "IMPORTANT: the domain-gap problem" before you assume any of this
will "just work" — it explains why the MVP ships with a classical-CV
heuristic baseline (zero training data needed) rather than a pre-trained
model, and what real data you need to get a genuinely reliable YOLO model.

## Why the MVP doesn't ship a pretrained defect model

At the time this project was built, we could not find a public, adequately
sized, appropriately-licensed dataset of **smartphone/laptop screen**
defects (scratches, cracks, dead pixels, etc.) large and clean enough to
train a model we'd be comfortable calling "working" in a report. What
*does* exist on Roboflow Universe (listed below) is real, but small
(tens to a few hundred images), community-labeled (label quality/consistency
not verified by us), and license terms vary by project. We are not going to
pretend otherwise or invent accuracy numbers for models we haven't trained.

Because of that, `damage_detection.mode` defaults to `"heuristic"` in
`config/config.yaml` — a classical computer-vision pipeline (edge/line/blob
analysis, see `src/damage_detection/defect_detector.py`) that needs **no
training data** and gives the system something real to demonstrate on day
one. Training pipeline is fully implemented (`training/train.py` etc.) —
once you have a dataset prepared, switch `damage_detection.mode` to
`"yolo"` in config.yaml to use your trained model instead.

## Public datasets we identified (verify current status before use)

All of the datasets below were found on Roboflow Universe. **Roboflow
listings change over time (owners can edit/remove/relicense projects), so
re-check the license and image count on the page itself before relying on
it** — treat the numbers here as "as observed", not permanent guarantees.

| Dataset | Source | License (as listed) | Classes | Relevance |
|---|---|---|---|---|
| **crackedscreen** | https://universe.roboflow.com/20701s-workspace/crackedscreen | CC BY 4.0 | cracked screen (small dataset, ~65 images) | Direct match: phone screen crack detection |
| **Mobile Damage Diagnosis** | https://universe.roboflow.com/test2-cedxe/mobile-damage-diagnosis-14usw | CC BY 4.0 | scratch, dead pixel, screen crack | Direct match: closest to our class list (scratch/dead_pixel/crack) |
| **lens_defect_scratch** | https://universe.roboflow.com/lensdefect/lens_defect_scratch-vmydi | CC BY 4.0 | Scratch, Collapse, Digs, Glass Stick, Internal Crack | Glass-surface scratch/crack imagery (lens, not screen — see domain gap) |
| **classify screen damage** | https://universe.roboflow.com/classify-garar/classify-screen-damage | Public Domain | has_damage, no_damage | Binary damaged/undamaged classification, useful as a coarse pre-filter |
| **CRACK DETECTION objects** | https://universe.roboflow.com/raheel-ahmad-rs4jo/crack-detection-objects | CC BY 4.0 | general crack detection | General crack imagery (likely not glass-specific — check before use, domain gap risk) |
| **Scratch_labelled_2** | https://universe.roboflow.com/as-wxvig/scratch_labelled_2 | CC BY 4.0 | scratch/dent/pits | General surface scratch imagery (likely industrial, not glass — domain gap risk) |

**Required preprocessing for all of the above:** re-export in YOLOv8 format
from Roboflow (the platform does this for you), then run
`python datasets/prepare_dataset.py`, which remaps each dataset's own class
names onto our unified class list (`config.yaml -> damage_detection.classes`)
via the `CLASS_NAME_MAP` table in that script — edit that table if a
dataset's classes don't cleanly match ours.

**Commercial use:** CC BY 4.0 permits commercial use *with attribution* to
the original author (see each dataset's "Cite This Project" box on its
page). "Public Domain" has no restriction. This is an academic prototype, so
attribution is good practice regardless — cite the dataset(s) you actually
use in your project report.

**How to download:** Roboflow requires a free account + API key to export
even public datasets programmatically — that's a platform requirement, not
something this project works around. Either:
- **Automatic:** `export ROBOFLOW_API_KEY=your_key` then
  `python datasets/download_datasets.py` (needs `pip install roboflow`,
  not included in the default requirements.txt since it's optional), or
- **Manual:** open the dataset's URL above → "Download Dataset" → format
  "YOLOv8" → unzip into `datasets/raw/<dataset_name>/`.

## IMPORTANT: the domain-gap problem

**A model trained on one kind of surface defect does not reliably transfer
to another.** Concretely, in this project:

- A scratch on a **glass phone screen** looks different (specular
  highlights, sub-surface refraction, different scale) from a scratch on
  **brushed metal**, **painted plastic**, or a **camera lens** — even
  though all four might be labeled "scratch" in different datasets.
- Datasets like `lens_defect_scratch` (glass, but a lens, not a flat
  display) or `Scratch_labelled_2`/`CRACK DETECTION objects` (general,
  likely industrial-surface) are only *partially* transferable — useful for
  pretraining/augmenting a small screen-specific dataset, not a substitute
  for one.
- Laptop screens photographed with keyboard/bezel in frame introduce
  background clutter that phone-only datasets won't have examples of.

**We did not find sufficient public data for:** `black_spot`,
`display_line`, `discoloration`, and `backlight_bleed` as *photographed
physical-device* defects (some exist as pure digital/synthetic examples,
which don't teach a model what these look like in a real photo with glare
and reflections). This is exactly why we say so explicitly here instead of
pretending the YOLO model will handle every class well — see
`MODEL_CARD.md` → Limitations, and `config/config.yaml`'s heuristic
`display_line` detector, which handles some of these classes without
training data as an interim measure.

**Recommendation (and the most reliable path):** collect your own labeled
photos of real phone/laptop screens — with and without defects, varied
lighting/angles/backgrounds — using your team's own devices. See
`datasets/README.md` → "Adding your own photographs". Even 200-300 well
labeled real photos of your *actual* target devices will likely outperform
a larger but domain-mismatched public dataset for your demo.

## Extending the dataset with your own photos

See `datasets/README.md`. In short: put images in
`datasets/raw/own_photos/images/`, YOLO-format labels in
`datasets/raw/own_photos/labels/`, then re-run
`python datasets/prepare_dataset.py`.

## Citation template

If you use one of the datasets above in your project report, cite it using
the BibTeX shown on its Roboflow Universe page (each has a "Cite This
Project" box), e.g.:

```
@misc{ crackedscreen_dataset,
  title = { Crackedscreen Dataset },
  type = { Open Source Dataset },
  author = { 20701s Workspace },
  howpublished = { \url{ https://universe.roboflow.com/20701s-workspace/crackedscreen } },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
}
```
