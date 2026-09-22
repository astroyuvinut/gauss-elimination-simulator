"""Phase 6: Gradio front end for the Gauss elimination / Gauss-Jordan simulator.

The Python handler is a generator: each time a clip finishes rendering it
yields an updated clip queue. A small script in the page watches that queue
and plays clips back to back on two stacked <video> elements (the next clip
preloads in the hidden one), so the result feels like one live simulation.
The side panel is updated when a clip *starts playing*, not when it renders,
so it always matches what is on screen.
"""
from __future__ import annotations

import html
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from pipeline import METHOD_LABEL, simulate
from scenes import ROW_COLORS
from solver import describe_solution, fmt, parse_matrix, var_names

CACHE_DIR = Path(os.environ.get("GAUSS_CACHE_DIR", Path(tempfile.gettempdir()) / "gauss_sim_clips")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
METHODS = {v: k for k, v in METHOD_LABEL.items()}  # label -> key

EXAMPLES = [
    ["1 1 -1 -2; 2 -1 1 5; -1 2 2 1", "Gauss Elimination"],
    ["0 2 1 7; 1 1 1 6; 2 1 -1 1", "Gauss-Jordan"],
    ["1 1 1 6; 2 2 2 12; 1 -1 2 5", "Gauss Elimination"],
    ["1 1 1 6; 2 2 2 13; 1 -1 2 5", "Gauss-Jordan"],
]


# ------------------------------------------------------------ validation

def validate(text: str):
    A = parse_matrix(text)
    if len(A) != 3 or any(len(r) != 4 for r in A):
        raise ValueError("Enter exactly 3 rows of 4 numbers (a b c | d), rows separated by ';'.")
    for r in A:
        for v in r:
            if abs(v) > 100 or v.denominator > 20:
                raise ValueError(f"Keep entries between -100 and 100 with small denominators (got {fmt(v)}).")
    return A


# ------------------------------------------------------------ HTML pieces

def file_url(path: Path) -> str:
    return "gradio_api/file=" + quote(str(Path(path).resolve()).replace("\\", "/"), safe="/:")


def matrix_table(M, highlight=()) -> str:
    rows = []
    for i, row in enumerate(M):
        cls = ' class="hl"' if i in highlight else ""
        cells = "".join(f"<td>{fmt(v)}</td>" for v in row[:-1])
        rows.append(f'<tr{cls} style="--rc:{ROW_COLORS[i % 3]}"><th>R{i + 1}</th>{cells}'
                    f'<td class="aug">{fmt(row[-1])}</td></tr>')
    names = var_names(len(M[0]) - 1)
    head = "<tr><th></th>" + "".join(f"<th>{n}</th>" for n in names) + '<th class="aug">b</th></tr>'
    return f'<table class="mat">{head}{"".join(rows)}</table>'


def panel_html(spec: dict, res, i: int, n: int) -> str:
    kind = spec["kind"]
    badge = {"swap": "Row swap", "scale": "Row scaling", "replace": "Row replacement",
             "intro": "Starting system", "result": "Result"}[kind]
    parts = [f'<div class="step">{html.escape(spec["step_label"])} · clip {i + 1} of {n}</div>',
             f'<div class="badge b-{kind}">{badge}</div>',
             f'<div class="op">{html.escape(spec["op_text"])}</div>',
             matrix_table(spec["after"], spec["rows"])]
    if kind == "result":
        sol = res.solution
        parts.append(f'<div class="sol s-{sol.kind}">{html.escape(describe_solution(sol))}</div>')
        parts.append(f'<div class="rank">rank(A) = {sol.rank_A}, rank([A|b]) = {sol.rank_Ab}, '
                     f'unknowns = {sol.nvars}</div>')
        if sol.back_sub:
            steps = "".join(f"<li>{html.escape(s)}</li>" for s in sol.back_sub)
            parts.append(f'<div class="bs"><b>Back-substitution</b><ol>{steps}</ol></div>')
    return "".join(parts)


def queue_html(run_id: str, items: list[dict], done: bool) -> str:
    data = html.escape(json.dumps(items), quote=True)
    return f'<div class="q" data-run="{run_id}" data-done="{int(done)}" data-items="{data}"></div>'


PLAYER = """
<div class="sim">
  <div class="stage">
    <video id="sim-v0" class="sim-v" muted playsinline preload="auto" disablepictureinpicture></video>
    <video id="sim-v1" class="sim-v" muted playsinline preload="auto" disablepictureinpicture></video>
    <div id="sim-msg" class="msg">Enter a 3×4 augmented matrix and press Run.</div>
    <button id="sim-replay" class="replay" type="button" title="Replay">↻ Replay</button>
  </div>
  <div id="sim-panel" class="panel"><div class="hint">Each row operation appears here as it plays.</div></div>
</div>
"""

HEAD = """
<script>
(() => {
  if (window.__gaussPlayer) return;       // Gradio may inject <head> content twice
  window.__gaussPlayer = true;
  const S = {run: null, items: [], next: 0, cur: null, busy: false, done: false};
  const $ = (id) => document.getElementById(id);
  const vids = () => [$('sim-v0'), $('sim-v1')];
  const idleVid = () => { const [a, b] = vids(); return S.cur === a ? b : a; };
  const msg = (t) => { const m = $('sim-msg'); if (m) { m.textContent = t; m.style.display = t ? '' : 'none'; } };

  function prep() {                       // preload the next clip in the hidden video
    const v = idleVid();
    if (!v || S.next >= S.items.length || v.dataset.idx === String(S.next)) return;
    v.dataset.idx = String(S.next);
    v.src = S.items[S.next].url;
    v.load();
  }
  function advance() {                    // start the next clip, swap it in when it plays
    if (S.next >= S.items.length) { S.busy = false; return; }
    prep();
    const v = idleVid(), old = S.cur, item = S.items[S.next];
    S.next += 1; S.busy = true;
    S.cur = v;                            // claim it now so prep() preloads into the other one
    v.addEventListener('playing', () => {
      v.classList.add('on');
      if (old && old !== v) { old.classList.remove('on'); old.pause(); }
      const p = $('sim-panel'); if (p) p.innerHTML = item.panel;
      msg('');
      prep();
    }, {once: true});
    v.currentTime = 0;
    v.play().catch(() => {});
  }
  function reset() {
    for (const v of vids()) { if (!v) continue; v.pause(); v.removeAttribute('src'); v.load();
                              v.classList.remove('on'); delete v.dataset.idx; }
    Object.assign(S, {items: [], next: 0, cur: null, busy: false, done: false});
  }
  function tick() {
    const q = document.querySelector('#clip-queue .q');
    if (!q || !$('sim-v0')) return;
    if (q.dataset.run !== S.run) { reset(); S.run = q.dataset.run; msg('Rendering the first step…'); }
    S.items = JSON.parse(q.dataset.items || '[]');
    S.done = q.dataset.done === '1';
    if (!S.busy && S.next < S.items.length) advance(); else prep();
    if (!S.busy && S.next >= S.items.length && !S.done && S.items.length) msg('Rendering next step…');
  }
  const onEnd = (e) => { if (e.target === S.cur) { S.busy = false; tick(); } };
  document.addEventListener('ended', onEnd, true);
  // a clip that fails to load is skipped (clearing src on reset also fires 'error': ignore that)
  document.addEventListener('error', (e) => { if (e.target === S.cur && e.target.getAttribute('src')) onEnd(e); }, true);
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#sim-replay') || !S.items.length) return;
    for (const v of vids()) delete v.dataset.idx;
    S.next = 0; S.busy = false; advance();
  });
  setInterval(tick, 150);
})();
</script>
"""

CSS = """
#clip-queue { display: none !important; }
.sim { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 14px; }
@media (max-width: 900px) { .sim { grid-template-columns: 1fr; } }
.stage { position: relative; aspect-ratio: 16 / 9; background: #101218; border-radius: 10px; overflow: hidden; }
.sim-v { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; }
.sim-v.on { opacity: 1; }
.msg { position: absolute; inset: auto 0 12px 0; text-align: center; color: #c9ccd6; font-size: 14px; }
.replay { position: absolute; right: 10px; bottom: 10px; background: #ffffff22; color: #fff; border: 0;
          border-radius: 6px; padding: 4px 10px; cursor: pointer; font-size: 13px; }
.panel { font-size: 14px; line-height: 1.45; }
.panel .step { color: #8a8f9c; font-size: 12px; }
.panel .badge { display: inline-block; margin: 6px 0 2px; padding: 1px 8px; border-radius: 99px;
                font-size: 12px; background: #8884; }
.panel .b-swap { background: #7c5cff33; } .panel .b-scale { background: #2bb3a033; }
.panel .b-replace { background: #f0a03033; } .panel .b-result { background: #e8d44d44; }
.panel .op { font-size: 20px; font-weight: 600; margin: 4px 0 10px; }
.mat { border-collapse: collapse; font-variant-numeric: tabular-nums; }
.mat th { color: #8a8f9c; font-weight: 500; padding: 2px 8px; }
.mat td { text-align: right; padding: 3px 10px; color: var(--rc); font-weight: 600; }
.mat td.aug, .mat th.aug { border-left: 2px solid #8884; }
.mat tr.hl td, .mat tr.hl th { background: #e8d44d2e; }
.panel .sol { margin-top: 12px; font-weight: 600; }
.panel .s-none { color: #e5484d; } .panel .s-unique { color: #30a46c; } .panel .s-infinite { color: #d6a300; }
.panel .rank { color: #8a8f9c; font-size: 12px; margin-top: 4px; }
.panel .bs ol { margin: 4px 0 0 18px; padding: 0; font-family: ui-monospace, monospace; font-size: 13px; }
"""


# ------------------------------------------------------------ handler

def run(matrix_text: str, method_label: str):
    try:
        A = validate(matrix_text)
    except ValueError as e:
        raise gr.Error(str(e))
    run_id = uuid.uuid4().hex
    items: list[dict] = []
    yield queue_html(run_id, items, False), "Rendering the first step…"
    for path, spec, i, n, res in simulate(A, METHODS[method_label], CACHE_DIR):
        items.append({"url": file_url(path), "panel": panel_html(spec, res, i, n)})
        done = i == n - 1
        status = (f"**{describe_solution(res.solution)}**" if done
                  else f"Rendered {i + 1} of {n} clips…")
        yield queue_html(run_id, items, done), status


with gr.Blocks(title="Gauss Elimination Simulator") as demo:
    gr.Markdown("## Gauss Elimination / Gauss-Jordan simulator\n"
                "Each row operation is one short Manim clip, rendered while the previous one plays. "
                "Row colours match the planes: **R1 blue, R2 green, R3 orange**.")
    with gr.Row():
        matrix_in = gr.Textbox(label="Augmented matrix [A | b]  (3 rows of a b c d, separated by ';')",
                               value=EXAMPLES[0][0], scale=4)
        method_in = gr.Dropdown(list(METHODS), value="Gauss Elimination", label="Method", scale=1)
        run_btn = gr.Button("Run", variant="primary", scale=1)
    gr.HTML(PLAYER)
    status = gr.Markdown()
    queue = gr.HTML(elem_id="clip-queue")
    gr.Examples(EXAMPLES, [matrix_in, method_in],
                label="Examples: unique · zero pivot (swap) · infinite · no solution")
    run_btn.click(run, [matrix_in, method_in], [queue, status])
    matrix_in.submit(run, [matrix_in, method_in], [queue, status])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        head=HEAD, css=CSS, allowed_paths=[str(CACHE_DIR)],
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        share="--share" in sys.argv,  # python app.py --share -> public *.gradio.live link
    )
