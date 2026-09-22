"""Phase 4: one short Manim clip per row operation.

A clip is described by a plain dict ("spec") so it can be cached and rendered
anywhere. Continuity between clips is what sells the "live" illusion:
  * camera: clip k starts at THETA0 + k*DTHETA and ends at THETA0 + (k+1)*DTHETA,
    exactly where clip k+1 starts;
  * matrix: the layout (column widths) is fixed for the whole run, so the
    brackets never move and only changed entries morph;
  * transient effects (row highlight box, dimming other planes) finish inside
    the clip, so the last frame of clip k equals the first frame of clip k+1.
"""
from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
from manim import (
    BLUE_C, DEGREES, DL, DOWN, GREEN_C, GREY_B, LEFT, ORANGE, ORIGIN, UL, UP, UR, WHITE, YELLOW,
    Create, Dot3D, FadeIn, FadeOut, Line, Line3D, Polygon, ReplacementTransform,
    SurroundingRectangle, Text, ThreeDAxes, ThreeDScene, VGroup, config, squish_rate_func,
    tempconfig, there_and_back,
)

from manim.animation.animation import prepare_animation

from geometry import clip_line, plane_polygon, row_caption, tick_step
from solver import fmt

ROW_COLORS = [BLUE_C, GREEN_C, ORANGE]
BG = "#101218"
PHI = 68 * DEGREES
THETA0 = -115 * DEGREES
DTHETA = 9 * DEGREES
AXIS_LEN = 5.2
PLANE_OPACITY = 0.42
CELL_FONT = 26
MATRIX_SCALE = 0.85

_render_lock = threading.Lock()  # manim's config is global: one render at a time


MAIN_WINDOW = (0.15, 0.75)  # fraction of the clip in which the row operation animates


def _pulse(t: float) -> float:
    """0 -> 1 (early), hold, -> 0 (late): for highlights that must vanish by the end."""
    if t < 0.05 or t > 0.97:
        return 0.0
    if t < 0.15:
        return (t - 0.05) / 0.10
    if t > 0.85:
        return (0.97 - t) / 0.12
    return 1.0


def txt(text: str, size: float, color=WHITE) -> Text:
    """Text rendered at 2x then scaled down: Pango drops spaces at small sizes."""
    return Text(text, font_size=size * 2, color=color).scale(0.5)


def row_color(i: int) -> str:
    return ROW_COLORS[i % len(ROW_COLORS)]


class MatrixLayout:
    """Fixed slot positions for an augmented matrix, shared by every clip of a
    run. `col_text` holds the widest string ever seen in each column."""

    def __init__(self, n_rows: int, col_text: list[str]):
        widths = [txt(t, CELL_FONT).width for t in col_text]
        self.row_h, gap = 0.5, 0.42
        self.right_edges, x = [], 0.0
        for j, w in enumerate(widths):
            if j == len(widths) - 1:
                x += gap * 0.8  # extra room for the augmentation bar
            self.right_edges.append(x + w)
            x += w + gap
        top, bottom = self.row_h * 0.55, -(n_rows - 1) * self.row_h - self.row_h * 0.55
        left, right = -gap * 0.6, self.right_edges[-1] + gap * 0.6
        bar_x = self.right_edges[-2] + gap * 0.9
        t = 0.12
        self.frame = VGroup(
            Line([left, top, 0], [left, bottom, 0]), Line([left, top, 0], [left + t, top, 0]),
            Line([left, bottom, 0], [left + t, bottom, 0]),
            Line([right, top, 0], [right, bottom, 0]), Line([right, top, 0], [right - t, top, 0]),
            Line([right, bottom, 0], [right - t, bottom, 0]),
            Line([bar_x, top - 0.05, 0], [bar_x, bottom + 0.05, 0], stroke_width=2, color=GREY_B),
        )
        # place the frame once; remember the transform so cells can follow it
        self.frame.scale(MATRIX_SCALE, about_point=ORIGIN)
        before = self.frame.get_corner(UL)
        self.frame.to_corner(UL, buff=0.45)
        self.shift = self.frame.get_corner(UL) - before

    def cell(self, value, i: int, j: int, color: str | None = None) -> Text:
        c = txt(fmt(value), CELL_FONT, color or row_color(i))
        c.move_to([self.right_edges[j] - c.width / 2, -i * self.row_h, 0])
        return c.scale(MATRIX_SCALE, about_point=ORIGIN).shift(self.shift)

    def cells(self, M) -> list[list[Text]]:
        return [[self.cell(v, i, j) for j, v in enumerate(row)] for i, row in enumerate(M)]


class StepScene(ThreeDScene):
    def __init__(self, spec: dict, **kwargs):
        self.spec = spec
        super().__init__(**kwargs)

    # ---- helpers
    def _plane(self, row, i):
        pts = plane_polygon(row, self.R)
        if pts is None:
            return None
        return Polygon(*[self.axes.c2p(*p) for p in pts], color=row_color(i),
                       fill_color=row_color(i), fill_opacity=PLANE_OPACITY, stroke_width=1.5)

    def _hud(self, *mobs):
        self.add_fixed_in_frame_mobjects(*mobs)
        return mobs[0] if len(mobs) == 1 else mobs

    def _hud_target(self, mob):
        """Register as fixed-in-frame without showing it yet (animation target)."""
        self.add_fixed_in_frame_mobjects(mob)
        self.remove(mob)
        return mob

    def _notes(self, M):
        lines = [c for i, r in enumerate(M) if (c := row_caption(i, r))]
        if not lines:
            return None
        g = VGroup(*[txt(t, 22, YELLOW) for t in lines]).arrange(DOWN, aligned_edge=LEFT)
        return g.to_corner(DL, buff=0.4).shift(UP * 0.6)

    # ---- scene
    def construct(self):
        s = self.spec
        self.R = R = s["R"]
        k, kind = s["index"], s["kind"]
        before, after, rows = s["before"], s["after"], s["rows"]
        self.set_camera_orientation(phi=PHI, theta=THETA0 + k * DTHETA, zoom=0.95)

        # axes (range fixed for the whole run, computed in geometry.view_range)
        step = tick_step(R)
        self.axes = axes = ThreeDAxes(
            x_range=[-R, R, step], y_range=[-R, R, step], z_range=[-R, R, step],
            x_length=AXIS_LEN, y_length=AXIS_LEN, z_length=AXIS_LEN,
            axis_config={"stroke_width": 1.5, "tip_width": 0.15, "tip_height": 0.15},
        ).set_color(GREY_B)
        labels = [txt(n, 26, GREY_B).move_to(axes.c2p(*(p * R * 1.12)))
                  for n, p in zip("xyz", np.eye(3))]
        self.add(axes, *labels)
        self.add_fixed_orientation_mobjects(*labels)

        # planes first, HUD on top; the same order in every clip so nothing
        # changes stacking at a clip boundary
        planes = [self._plane(r, i) for i, r in enumerate(before)]
        self.add(*[p for p in planes if p])  # intro: Create() starts them from empty

        layout = MatrixLayout(len(before), s["col_text"])
        old_cells = layout.cells(before)
        self._hud(layout.frame, *[c for r in old_cells for c in r])
        self._hud(txt(s["op_text"], 30, WHITE).to_edge(DOWN, buff=0.35))
        self._hud(txt(s["step_label"], 22, GREY_B).to_corner(UR, buff=0.4))
        notes = self._notes(before)
        if notes:
            self._hud(notes)

        # Every animation runs for the whole clip alongside the camera move; a
        # squished rate function gives each one its time window. (No Succession:
        # it adds all of its mobjects at frame 0 and re-stacks them.)
        main: list = []      # the operation itself, in the MAIN window
        extra: list = []     # animations with their own timing
        if kind == "intro":
            main = [Create(p) for p in planes if p]

        elif kind in ("swap", "scale", "replace"):
            # row mapping: old row src ends up in row dst
            mapping = {rows[0]: rows[1], rows[1]: rows[0]} if kind == "swap" else {rows[0]: rows[0]}
            for src, dst in mapping.items():
                for j, old in enumerate(old_cells[src]):
                    if src == dst and before[src][j] == after[dst][j]:
                        continue  # unchanged entry: leave it alone
                    new = self._hud_target(layout.cell(after[dst][j], dst, j))
                    if kind == "swap":
                        # same text, new place: glyph counts match, so morphing is clean
                        main.append(ReplacementTransform(old, new))
                    else:
                        # different text: roll the old value out and the new one in.
                        # (Morphing unequal glyph counts spawns helper pieces that are
                        # not fixed-in-frame and get drawn in 3D space.)
                        main += [FadeOut(old, shift=UP * 0.18), FadeIn(new, shift=UP * 0.18)]
                # highlight box: fades in, holds, fades out, all inside this clip
                box = self._hud(SurroundingRectangle(VGroup(*old_cells[src]), color=YELLOW,
                                                     buff=0.08, stroke_width=2.5).set_stroke(opacity=0))
                extra.append(box.animate(rate_func=_pulse).set_stroke(opacity=1))
            if notes:
                main.append(FadeOut(notes))
            new_notes = self._notes(after)
            if new_notes:
                main.append(FadeIn(self._hud_target(new_notes)))

            if kind == "swap":
                # the plane that was row i is now row j (and vice versa): swap colours
                for src, dst in mapping.items():
                    if planes[src]:
                        main.append(planes[src].animate.set_color(row_color(dst))
                                    .set_fill(row_color(dst), opacity=PLANE_OPACITY))
            elif kind == "replace":
                i = rows[0]
                old, new = planes[i], self._plane(after[i], i)
                if old and new:
                    main.append(ReplacementTransform(old, new))
                elif old:
                    main.append(FadeOut(old))
                elif new:
                    main.append(FadeIn(new))
                for j, p in enumerate(planes):
                    if p and j != i:  # dim the others, back to normal by the end
                        main.append(p.animate(rate_func=there_and_back).set_fill(opacity=0.15))
            # scale: the plane is the same set of points, so it stays still;
            # only the matrix row changes

        elif kind == "result":
            main = self._result_anims(s["result"])

        main = [prepare_animation(a) for a in main]  # turns .animate builders into Animations
        for a in main:
            a.rate_func = squish_rate_func(a.rate_func, *MAIN_WINDOW)
        self.move_camera(theta=THETA0 + (k + 1) * DTHETA, run_time=s["duration"],
                         added_anims=main + extra)
        # the last frame of play() is drawn just before alpha=1; hold one frame at
        # the exact end state so it matches the first frame of the next clip
        self.wait(1 / config.frame_rate)

    def _result_anims(self, res):
        caption = txt(res["caption"], 24, YELLOW).to_corner(UR, buff=0.4).shift(DOWN * 0.55)
        anims = [FadeIn(self._hud_target(caption))]
        if res["kind"] == "unique":
            anims.append(FadeIn(Dot3D(self.axes.c2p(*res["point"]), radius=0.1, color=YELLOW), scale=3))
        elif res["kind"] == "line":
            seg = clip_line(res["point"], res["direction"], self.R)
            if seg:
                anims.append(Create(Line3D(self.axes.c2p(*seg[0]), self.axes.c2p(*seg[1]),
                                           color=YELLOW, thickness=0.03)))
        return anims


def render_clip(spec: dict, out_path: Path, quality: str = "low_quality") -> Path:
    """Render one clip to out_path. Renders into a private temp folder (so
    simultaneous users never collide) and moves the file into place atomically."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="gauss_"))
    try:
        with _render_lock, tempconfig({
            "quality": quality, "media_dir": str(work), "output_file": "clip",
            "disable_caching": True, "progress_bar": "none", "verbosity": "ERROR",
            "background_color": BG, "write_to_movie": True,
        }):
            scene = StepScene(spec)
            scene.render()
            src = Path(scene.renderer.file_writer.movie_file_path)
        tmp = out_path.with_suffix(".part.mp4")
        shutil.move(str(src), tmp)
        tmp.replace(out_path)
        return out_path
    finally:
        shutil.rmtree(work, ignore_errors=True)
