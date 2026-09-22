---
title: Gauss Elimination Simulator
emoji: 🧮
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Gauss Elimination / Gauss-Jordan simulator

Type a 3×3 system as an augmented matrix, for example `1 1 -1 -2; 2 -1 1 5; -1 2 2 1`,
and watch each row operation play out on the three planes, with the matrix updating beside them.

Each row operation is one short Manim clip. The server renders clip *k+1* while
the browser plays clip *k*, and the page swaps clips seamlessly, so it plays as
one continuous simulation.

| File | Role |
|---|---|
| `solver.py` | Exact (Fraction) elimination and Gauss-Jordan. Records every row op as a frame and classifies the system by rank |
| `test_solver.py` | Checks against `numpy.linalg.solve` / `matrix_rank`: unique, zero pivot, infinite, none |
| `geometry.py` | Clips each plane to the view cube (vertical planes included), handles `0 0 0 \| 0` and `0 0 0 \| k` rows, fixes the axis range once per run |
| `scenes.py` | `StepScene`: one clip per operation, with camera, matrix and highlights continuous across clips |
| `pipeline.py` | Generator that renders and yields clips; cache keyed by `hash(matrix, method, settings)` |
| `app.py` | Gradio UI plus the double-buffered `<video>` player and the side panel |

## Run locally

```bash
pip install -r requirements.txt
python -m pytest -q        # solver tests
python app.py              # http://127.0.0.1:7860
python app.py --share      # also prints a public https://....gradio.live link (valid 1 week)
```

No LaTeX needed: all on-screen text uses Manim `Text`. Clips are cached in
`$GAUSS_CACHE_DIR` (default: the system temp folder, `gauss_sim_clips`).

## Deploy to Hugging Face Spaces

Create a Gradio Space and push these files. `packages.txt` installs Cairo, Pango
and ffmpeg for Manim. Spaces sets `GRADIO_SERVER_NAME=0.0.0.0` itself.
