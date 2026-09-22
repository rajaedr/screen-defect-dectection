# Sample Images

This folder intentionally ships **empty** of actual photos. We will not
fabricate images and label them as if they were real defective devices —
that would misrepresent test data, which the project spec explicitly
prohibits.

## What to collect (for your own testing / demo)

Take these photos yourselves with your team's own devices, in good, even
lighting, filling most of the frame with the screen:

| Filename (suggested) | What to photograph |
|---|---|
| `normal_screen.jpg` | A screen with no visible damage (your control/negative example) |
| `scratch_screen.jpg` | A screen with a real, visible surface scratch |
| `cracked_screen.jpg` | A screen with a visible crack |
| `dead_pixel_screen.jpg` | A screen displaying a solid color (see "test-pattern mode" in the app) showing a dead/stuck pixel |
| `screen_line.jpg` | A screen with a visible stuck horizontal/vertical line, ideally photographed in test-pattern mode |

## Tips for good test photos

- Fill most of the frame with the screen (helps screen localization).
- Avoid strong reflections/glare directly over the defect area — angle the
  shot slightly if needed.
- For dead pixels / lines / discoloration, use the app's **test-pattern
  mode** (display a solid white/black/red/green/blue screen, then
  photograph) — these classes are much easier to see reliably against a
  blank background than against normal wallpaper/app content.
- Take a few photos at slightly different angles/distances per defect —
  useful both for demoing the app and, if you label them, for training data
  (see `datasets/README.md`).

## Using your own photos

1. Drop your photos here (or anywhere convenient) and upload them through
   the app's "Upload Image" tab — that's all that's needed to *demo* the
   system.
2. If you also want to use them as **training data**, follow
   `datasets/README.md` → "Adding your own photographs" instead (they need
   YOLO-format labels and belong under `datasets/raw/own_photos/`, not
   here — this folder is for demo/manual-test images only).
