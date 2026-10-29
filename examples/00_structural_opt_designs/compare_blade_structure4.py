#!/usr/bin/env python3
"""
Compare the structural design of wind turbine blades stored in windIO (v2) YAML files,
e.g. the WISDEM input geometry and the optimised blade_out.yaml.

Works directly from the YAML files, so it does not need WISDEM, OpenMDAO or the
modeling/analysis options.  Merged from compare_blade_structure3.py (base) with the
section geometry, layup drawing, width-based anchors and cap/web plot of
compare_blade_structure2.py.

Usage
-----
    python compare_blade_structure4.py Q3v5_windio_v2.yaml blade_out.yaml \
        --labels Initial Optimised --spans 0.25 0.5 0.75 --outdir blade_compare

Produces (in --outdir, figure extension set by --format):
    1_outer_shape         chord, twist, relative & absolute thickness, prebend, sweep
    2_layer_thickness     thickness of every shell layer vs span
    3_web_layers          thickness of shear-web layers vs span
    4_arc_positions       nd-arc extent of every partial shell layer, webs and LE
    5_caps_and_webs       spar-cap width and chordwise centre, web chordwise positions
    6_sections            cross-sections with every layer stacked inward, coloured by material
    7_elastic_props       EA, EI_flap, EI_edge, GJ, mass/length (+ % change vs first file)
    8_layer_mass          estimated mass of each layer (bar chart)
    layer_mass.csv        the same mass table, for spreadsheets
    summary.txt           everything printed to the console

Conventions (windIO v2)
-----------------------
* span grids are normalised by the curvilinear length of the reference axis
* nd arc: 0 = TE (suction side) -> LE -> pressure side -> 1 = TE.  A layer whose
  start_nd_arc > end_nd_arc wraps across the trailing edge.
* a layer exists only where its thickness grid is defined (no extrapolation), and
  additionally only inside start_nd_grid..end_nd_grid when those are given.
* layers are listed outer -> inner, which is the order they are stacked in the sections.
* anchors may give start/end_nd_arc directly, or a width [m] together with
  midpoint_nd_arc, start_nd_arc or end_nd_arc.  plane_intersection anchors are not
  evaluated (run the file through WISDEM first, its output has the arcs resolved).
"""
# =========================================================================== SETTINGS
# Used when the script is started without command-line arguments, e.g. with the
# "Run Python File" button in VS Code.  Relative paths are taken from the folder
# this script is in.  Command-line arguments, when given, override all of these.
SETTINGS = {
    "files":  ["Q3v5_windio_v2.yaml", "cap_both.yaml", "cap_PS.yaml", "cap_SS.yaml"],   # first file is the baseline
    "labels": ["Initial", "cap_both", "cap_PS", "cap_SS"],                     # one per file, or None
    "spans":  [0.25, 0.5, 0.75],                            # r/R of the cross-sections
    "scale":  1.0,                                          # thickness magnification in sections
    "ngrid":  200,                                          # spanwise points
    "format": "png",                                        # png, pdf or svg
    "dpi":    150,
    "outdir": "blade_compare",
}
# ===================================================================================

import argparse
import csv
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

NGRID = 200          # default spanwise resolution
AF_N = 121           # points per airfoil surface after resampling
_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz

SAVE = {"dpi": 150, "fmt": "png"}   # set from the command line in main()

# Layer categories highlighted in the plots (matched case-insensitively).
CATEGORIES = [
    ("Spar cap", re.compile(r"spar.?cap", re.I), "tab:red"),
    ("TE reinf", re.compile(r"(^|[^a-z])(te|trailing).*reinf", re.I), "tab:green"),
    ("LE reinf", re.compile(r"(^|[^a-z])(le|leading).*reinf", re.I), "tab:purple"),
]
MATERIAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                   "#008300", "#6250d6", "#e34948", "#7f7f7f", "#17becf"]

_warned = set()
REPORT = []


def warn(msg):
    if msg not in _warned:
        _warned.add(msg)
        print("warning:", msg, file=sys.stderr)
        REPORT.append("warning: " + msg)


def say(line=""):
    print(line)
    REPORT.append(line)


def category(name):
    return next(((lab, col) for lab, rx, col in CATEGORIES if rx.search(name)), (None, None))


def arc_width(s0, s1):
    """Non-dimensional arc covered from s0 to s1, wrapping across the TE when s0 > s1."""
    return np.where(s1 >= s0, s1 - s0, 1.0 - s0 + s1)


def arc_mid(s0, s1):
    """Midpoint of an arc range, wrap-aware."""
    return np.mod(s0 + 0.5 * arc_width(s0, s1), 1.0)


def arc_pieces(s0, s1):
    """Split an arc range into non-wrapping [a, b] pieces."""
    return [(s0, s1)] if s1 >= s0 else [(s0, 1.0), (0.0, s1)]


def in_arc(s, s0, s1):
    """Boolean mask of contour points s that lie inside the (possibly wrapping) range."""
    return (s >= s0) & (s <= s1) if s1 >= s0 else (s >= s0) | (s <= s1)


# --------------------------------------------------------------------------- loading
class Design:
    def __init__(self, path, label, ngrid=NGRID):
        with open(path) as f:
            self.d = yaml.safe_load(f)
        self.path, self.label = path, label
        self.grid = np.linspace(0.0, 1.0, ngrid)
        try:
            self.blade = self.d["components"]["blade"]
            self.os = self.blade["outer_shape"]
            self.st = self.blade["structure"]
            ra = self.blade["reference_axis"]
        except (KeyError, TypeError) as e:
            raise SystemExit(f"{path}: missing windIO v2 section {e}")
        self.materials = {m["name"]: m for m in self.d.get("materials") or []}
        self.airfoils = {a["name"]: a for a in self.d.get("airfoils") or []}
        self.layers = {l["name"]: l for l in self.st.get("layers") or []}
        self.webs = {w["name"]: w for w in self.st.get("webs") or []}

        afs = self.os.get("airfoils")
        if not isinstance(afs, list) or not afs:
            raise SystemExit(f"{path}: outer_shape.airfoils must be a list (windIO v2)")
        self.af_stations = sorted(((float(a["spanwise_position"]), a["name"]) for a in afs),
                                  key=lambda p: p[0])
        missing = {n for _, n in self.af_stations if n not in self.airfoils}
        if missing:
            raise SystemExit(f"{path}: airfoils not defined: {', '.join(sorted(missing))}")
        if self.outer_block("chord") is None:
            raise SystemExit(f"{path}: outer_shape has no chord")

        # every anchor that a layer may refer to (shell anchors + web anchors)
        self.anchors = {a["name"]: a for a in self.st.get("anchors") or []}
        for w in self.webs.values():
            for a in w.get("anchors") or []:
                self.anchors[a["name"]] = a

        # curvilinear coordinate along the reference axis [m] (windIO grids are normalised by it)
        xyz = [self.gv(ra[k]) if k in ra else np.zeros_like(self.grid) for k in "xyz"]
        seg = np.sqrt(sum(np.diff(c) ** 2 for c in xyz))
        self.s_curve = np.concatenate([[0.0], np.cumsum(seg)])
        self.span_len = float(self.s_curve[-1])

        self.elastic = self.st.get("elastic_properties") or self.blade.get("elastic_properties")
        self._sections = {}
        self._af_cache = {}
        self._arc_cache = {}
        self._perims = None

    # -- interpolation of a {grid, values} block ----------------------------------
    def gv(self, block, grid=None, left=None, right=None):
        grid = self.grid if grid is None else np.asarray(grid, float)
        g = np.asarray(block["grid"], float)
        v = np.asarray(block["values"], float)
        if v.ndim > 1:  # 2D (chordwise-tapered) values: use the chordwise mean
            if v.shape[0] == g.size and v.shape[1] != g.size:
                v = v.mean(axis=1)
            elif v.shape[1] == g.size and v.shape[0] != g.size:
                v = v.mean(axis=0)
            else:
                warn(f"{self.label}: ambiguous 2D grid/values block, assuming rows follow the span")
                v = v.mean(axis=1)
        order = np.argsort(g, kind="stable")
        return np.interp(grid, g[order], v[order], left=left, right=right)

    def outer_block(self, key):
        aliases = {"rthick": ("rthick", "relative_thickness")}
        for k in aliases.get(key, (key,)):
            if k in self.os:
                return self.os[k]
        return None

    def outer(self, key, grid=None):
        b = self.outer_block(key)
        if b is None:
            return None
        return self.gv(b, grid)

    # -- arcs ------------------------------------------------------------------------
    def resolve_arc(self, spec, depth=0):
        """Return nd_arc values on the grid for a start/end_nd_arc spec, or None."""
        if spec is None or depth > 10:
            return None
        if isinstance(spec, (int, float)):
            return np.full_like(self.grid, float(spec))
        if not isinstance(spec, dict):
            warn(f"{self.label}: unsupported arc definition {spec!r}")
            return None
        if "grid" in spec and "values" in spec:
            return self.gv(spec)
        if "anchor" in spec:
            ref = spec["anchor"]
            name, handle = ref.get("name"), ref.get("handle", ref.get("handles"))
            anc = self.anchors.get(name)
            if anc is None:
                warn(f"{self.label}: anchor '{name}' is not defined")
                return None
            return self.anchor_arc(anc, handle, depth + 1)
        warn(f"{self.label}: unsupported arc definition {list(spec)} (resolve it with WISDEM first)")
        return None

    def anchor_arc(self, anc, handle, depth):
        """Evaluate one handle of an anchor, using its width [m] when the handle is not given."""
        if handle in anc:
            return self.resolve_arc(anc[handle], depth + 1)
        if "width" in anc and handle in ("start_nd_arc", "end_nd_arc", "midpoint_nd_arc"):
            w = self.gv(anc["width"]) / self.perimeters          # width in metres -> nd arc
            for key, start, end in (("midpoint_nd_arc", -0.5, 0.5),
                                    ("start_nd_arc", 0.0, 1.0),
                                    ("end_nd_arc", -1.0, 0.0)):
                if key in anc:
                    ref = self.resolve_arc(anc[key], depth + 1)
                    if ref is None:
                        return None
                    s0, s1 = np.mod(ref + start * w, 1.0), np.mod(ref + end * w, 1.0)
                    if handle == "midpoint_nd_arc":
                        return arc_mid(s0, s1)
                    return s0 if handle == "start_nd_arc" else s1
        how = "plane_intersection" if "plane_intersection" in anc else f"handle '{handle}'"
        warn(f"{self.label}: cannot evaluate anchor '{anc.get('name')}' ({how}) – "
             "run the file through WISDEM first")
        return None

    def layer_arcs(self, layer):
        key = ("layer", layer["name"])
        if key not in self._arc_cache:
            self._arc_cache[key] = (self.resolve_arc(layer.get("start_nd_arc")),
                                    self.resolve_arc(layer.get("end_nd_arc")))
        return self._arc_cache[key]

    def web_arcs(self, web):
        key = ("web", web["name"])
        if key not in self._arc_cache:
            self._arc_cache[key] = (self.resolve_arc(web.get("start_nd_arc")),
                                    self.resolve_arc(web.get("end_nd_arc")))
        return self._arc_cache[key]

    def shell_layers(self):
        return [l for l in self.layers.values() if "web" not in l]

    def web_layers(self, web=None):
        return [l for l in self.layers.values() if "web" in l and (web is None or l["web"] == web)]

    def layer_thickness(self, layer, grid=None):
        """Thickness [m]; zero outside the layer's own thickness grid and start/end_nd_grid."""
        g = self.grid if grid is None else np.atleast_1d(np.asarray(grid, float))
        t = np.maximum(self.gv(layer["thickness"], g, left=0.0, right=0.0), 0.0)
        s0 = layer.get("start_nd_grid", 0.0)
        s1 = layer.get("end_nd_grid", 1.0)
        t[(g < s0) | (g > s1)] = 0.0
        return t

    def web_active(self, web, grid=None):
        g = self.grid if grid is None else np.atleast_1d(np.asarray(grid, float))
        return (g >= web.get("start_nd_grid", 0.0)) & (g <= web.get("end_nd_grid", 1.0))

    def lerp(self, arr, r):
        return float(np.interp(r, self.grid, arr))

    # -- section geometry --------------------------------------------------------
    def _airfoil_surfaces(self, name):
        """Resample an airfoil to common cosine-spaced x: returns (x, y_suction, y_pressure)."""
        if name in self._af_cache:
            return self._af_cache[name]
        c = self.airfoils[name]["coordinates"]
        x, y = np.asarray(c["x"], float), np.asarray(c["y"], float)
        ext = x.max() - x.min()
        x, y = (x - x.min()) / ext, y / ext
        i_le = int(np.argmin(x))
        a = (x[:i_le + 1][::-1], y[:i_le + 1][::-1])   # LE -> TE along the first half
        b = (x[i_le:], y[i_le:])                         # LE -> TE along the second half
        xc = 0.5 * (1 - np.cos(np.linspace(0, np.pi, AF_N)))

        def resample(xs, ys):
            o = np.argsort(xs, kind="stable")
            return np.interp(xc, xs[o], ys[o])

        ya, yb = resample(*a), resample(*b)
        if ya.mean() < yb.mean():
            ya, yb = yb, ya
        self._af_cache[name] = (xc, ya, yb)
        return self._af_cache[name]

    def airfoil_at(self, r):
        """Airfoil at span r, normalised by chord, ordered TE -> suction -> LE -> pressure -> TE.

        The two airfoils bracketing r are blended with a weight set by the target relative
        thickness (span position when both have the same thickness), then thickness is
        scaled about the camber line so that max(t/c) equals rthick exactly."""
        st = self.af_stations
        pos = np.array([p for p, _ in st])
        if len(st) == 1:
            i = j = 0
        else:
            i = int(np.clip(np.searchsorted(pos, r, side="right") - 1, 0, len(st) - 2))
            j = i + 1
        xc, ua, la = self._airfoil_surfaces(st[i][1])
        _, ub, lb = self._airfoil_surfaces(st[j][1])
        ta, tb = float(np.max(ua - la)), float(np.max(ub - lb))
        rt = self.outer("rthick", [r])
        rt = None if rt is None else float(rt[0])
        if i == j:
            w = 0.0
        elif rt is not None and abs(tb - ta) > 1e-6:
            w = float(np.clip((rt - ta) / (tb - ta), 0.0, 1.0))
        elif pos[j] > pos[i]:
            w = float(np.clip((r - pos[i]) / (pos[j] - pos[i]), 0.0, 1.0))
        else:
            w = 0.0
        yu, yl = (1 - w) * ua + w * ub, (1 - w) * la + w * lb
        t_now = float(np.max(yu - yl))
        if rt is not None and t_now > 1e-9:
            camber = 0.5 * (yu + yl)
            k = rt / t_now
            yu, yl = camber + (yu - camber) * k, camber + (yl - camber) * k
        x = np.concatenate([xc[::-1], xc[1:]])
        y = np.concatenate([yu[::-1], yl[1:]])
        return x, y

    def section(self, r):
        """Dimensional, untwisted section at span r (cached): X, Y [m from LE], nd arc s,
        chord, perimeter and the inward unit normals nx, ny."""
        key = round(float(r), 9)
        if key not in self._sections:
            c = float(self.outer("chord", [r])[0])
            x, y = self.airfoil_at(r)
            X, Y = x * c, y * c
            s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
            perim = float(s[-1])
            # orientation from the signed area: CCW (>0) means the left normal points inward
            sign = 1.0 if np.sum(X[:-1] * Y[1:] - X[1:] * Y[:-1]) > 0 else -1.0
            tx, ty = np.gradient(X), np.gradient(Y)
            n = np.hypot(tx, ty) + 1e-12
            self._sections[key] = SimpleNamespace(X=X, Y=Y, s=s / perim, chord=c, perim=perim,
                                                  nx=-sign * ty / n, ny=sign * tx / n)
        return self._sections[key]

    def point(self, sec, s_query, depth=None):
        """Point on the contour at nd arc s_query, moved inward by depth [m] if given
        (depth is a scalar or an array matching s_query)."""
        px, py = np.interp(s_query, sec.s, sec.X), np.interp(s_query, sec.s, sec.Y)
        if depth is None:
            return px, py
        nx, ny = np.interp(s_query, sec.s, sec.nx), np.interp(s_query, sec.s, sec.ny)
        n = np.hypot(nx, ny) + 1e-12
        return px + nx / n * depth, py + ny / n * depth

    @property
    def perimeters(self):
        if self._perims is None:
            self._perims = np.array([self.section(r).perim for r in self.grid])
        return self._perims

    def web_height(self, web):
        w0, w1 = self.web_arcs(web)
        if w0 is None or w1 is None:
            return None
        h = np.zeros_like(self.grid)
        for k, r in enumerate(self.grid):
            sec = self.section(r)
            xa, ya = self.point(sec, w0[k])
            xb, yb = self.point(sec, w1[k])
            h[k] = np.hypot(xa - xb, ya - yb)
        return h

    def is_full_wrap(self, layer):
        s0, s1 = self.layer_arcs(layer)
        if s0 is None or s1 is None:
            return False
        m = self.layer_thickness(layer) > 0
        return bool(m.any() and np.all(arc_width(s0, s1)[m] > 1 - 1e-6))


# --------------------------------------------------------------------------- layup
def layup(D, r, scale=1.0):
    """Polygons of every layer at span r, shell layers stacked inward in list order.

    Returns (section, shells, webs):
      shells: [(layer, t [m], [(xs, ys), ...])]   one polygon per non-wrapping piece
      webs:   [(web, p1, p2, [(layer, t, (xs, ys))])]   p1/p2 = web ends on the inner skin
    Thickness is drawn multiplied by `scale`."""
    sec = D.section(r)
    depth = np.zeros_like(sec.s)
    shells = []
    for lay in D.shell_layers():
        t = float(D.layer_thickness(lay, [r])[0])
        s0, s1 = D.layer_arcs(lay)
        if t <= 0 or s0 is None or s1 is None:
            continue
        a0, a1 = D.lerp(s0, r), D.lerp(s1, r)
        polys = []
        for a, b in arc_pieces(a0, a1):
            if b - a < 1e-6:
                continue
            sq = np.linspace(a, b, max(12, int(400 * (b - a))))
            d0 = np.interp(sq, sec.s, depth)
            ox, oy = D.point(sec, sq, d0)
            ix, iy = D.point(sec, sq, d0 + t * scale)
            polys.append((np.concatenate([ox, ix[::-1]]), np.concatenate([oy, iy[::-1]])))
        depth[in_arc(sec.s, a0, a1)] += t * scale
        shells.append((lay, t, polys))

    webs = []
    for w in D.webs.values():
        w0, w1 = D.web_arcs(w)
        if w0 is None or w1 is None or not D.web_active(w, [r])[0]:
            continue
        sa, sb = D.lerp(w0, r), D.lerp(w1, r)
        p1 = np.array(D.point(sec, sa, float(np.interp(sa, sec.s, depth))))
        p2 = np.array(D.point(sec, sb, float(np.interp(sb, sec.s, depth))))
        u = (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-12)
        nrm = np.array([-u[1], u[0]])
        wl = [(L, float(D.layer_thickness(L, [r])[0])) for L in D.web_layers(w["name"])]
        wl = [(L, t) for L, t in wl if t > 0]
        off = -0.5 * sum(t for _, t in wl) * scale
        quads = []
        for L, t in wl:
            q = np.array([p1 + nrm * off, p2 + nrm * off,
                          p2 + nrm * (off + t * scale), p1 + nrm * (off + t * scale)])
            quads.append((L, t, (q[:, 0], q[:, 1])))
            off += t * scale
        webs.append((w, p1, p2, quads))
    return sec, shells, webs


# --------------------------------------------------------------------------- helpers
def styles(n):
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ls = ["-", "--", "-.", ":"]
    return [dict(color=cols[i % len(cols)], ls=ls[i % len(ls)], lw=1.6) for i in range(n)]


def save(fig, outdir, name, tight=True):
    p = os.path.join(outdir, f"{name}.{SAVE['fmt']}")
    if tight:
        fig.tight_layout()
    fig.savefig(p, dpi=SAVE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def material_colors(designs):
    mats = list(dict.fromkeys(l["material"] for D in designs for l in D.layers.values()))
    return {m: MATERIAL_COLORS[i % len(MATERIAL_COLORS)] for i, m in enumerate(mats)}


# --------------------------------------------------------------------------- plots
def plot_outer_shape(designs, outdir):
    st = styles(len(designs))
    panels = [("Chord", "m", lambda D: D.outer("chord")),
              ("Twist", "deg", lambda D: D.outer("twist")),
              ("Relative thickness", "-", lambda D: D.outer("rthick")),
              ("Absolute thickness", "m",
               lambda D: None if D.outer("rthick") is None else D.outer("chord") * D.outer("rthick")),
              ("Prebend (reference axis x)", "m",
               lambda D: D.gv(D.blade["reference_axis"]["x"]) if "x" in D.blade["reference_axis"] else None),
              ("Sweep (reference axis y)", "m",
               lambda D: D.gv(D.blade["reference_axis"]["y"]) if "y" in D.blade["reference_axis"] else None)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for a, (title, unit, fn) in zip(ax.flat, panels):
        for D, s in zip(designs, st):
            v = fn(D)
            if v is not None:
                a.plot(D.grid, v, label=D.label, **s)
        a.set_title(title); a.set_xlabel("r/R"); a.set_ylabel(unit); a.grid(alpha=.3)
    ax[0, 0].legend()
    save(fig, outdir, "1_outer_shape")


def plot_layer_thickness(designs, outdir, web=False):
    names = list(dict.fromkeys(l["name"] for D in designs
                               for l in (D.web_layers() if web else D.shell_layers())))
    if not names:
        return
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    st = styles(len(designs))
    for a, name in zip(ax.flat, names):
        mats = []
        for D, s in zip(designs, st):
            lay = D.layers.get(name)
            if lay is None:
                continue
            a.plot(D.grid, D.layer_thickness(lay) * 1e3, label=D.label, **s)
            if lay["material"] not in mats:
                mats.append(lay["material"])
        a.set_title(f"{name}\n({' → '.join(mats)})", fontsize=9)
        a.set_xlabel("r/R"); a.set_ylabel("t [mm]"); a.grid(alpha=.3)
    for a in list(ax.flat)[len(names):]:
        a.axis("off")
    handles = [Line2D([], [], **s, label=D.label) for D, s in zip(designs, st)]
    fig.legend(handles=handles, loc="upper center", ncol=len(designs), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (3.0 * nrow)))
    save(fig, outdir, "3_web_layers" if web else "2_layer_thickness", tight=False)


def plot_arcs(designs, outdir):
    """nd-arc extent of every shell layer that does not cover the whole perimeter."""
    partial = list(dict.fromkeys(l["name"] for D in designs for l in D.shell_layers()
                                 if not D.is_full_wrap(l)))
    cols = plt.cm.tab10.colors + plt.cm.Dark2.colors
    cmap = {n: cols[i % len(cols)] for i, n in enumerate(partial)}
    fig, ax = plt.subplots(1, len(designs), figsize=(6 * len(designs), 5), squeeze=False, sharey=True)
    for a, D in zip(ax.flat, designs):
        for lay in D.shell_layers():
            if lay["name"] not in cmap:
                continue
            s0, s1 = D.layer_arcs(lay)
            if s0 is None or s1 is None:
                continue
            col = cmap[lay["name"]]
            mask = D.layer_thickness(lay) > 0
            wrap = s0 > s1
            # piece 1: s0 -> (s1, or 1 where wrapping); piece 2: 0 -> s1 where wrapping
            hi1 = np.where(wrap, 1.0, s1)
            a.fill_between(D.grid, np.where(mask, s0, np.nan), np.where(mask, hi1, np.nan),
                           color=col, alpha=.3, lw=0, label=lay["name"])
            m2 = mask & wrap
            if m2.any():
                a.fill_between(D.grid, np.where(m2, 0.0, np.nan), np.where(m2, s1, np.nan),
                               color=col, alpha=.3, lw=0)
        for j, w in enumerate(D.webs.values()):
            w0, w1 = D.web_arcs(w)
            if w0 is None or w1 is None:
                continue
            on = D.web_active(w)
            ls = ["-", "--", "-.", ":"][j % 4]
            a.plot(D.grid, np.where(on, w0, np.nan), color="k", ls=ls, lw=1.2,
                   label=f"{w['name']} (ss / ps)")
            a.plot(D.grid, np.where(on, w1, np.nan), color="k", ls=ls, lw=1.2)
        if "LE" in D.anchors:
            le = D.anchor_arc(D.anchors["LE"], "start_nd_arc", 0)
            if le is not None:
                a.plot(D.grid, le, color="grey", ls=":", label="LE")
        a.set_title(D.label); a.set_xlabel("r/R"); a.grid(alpha=.3)
        a.set_ylim(0, 1)
        a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.flat[0].set_ylabel("nd arc  (0 = TE, suction side → LE ≈ 0.5 → pressure side → 1 = TE)")
    save(fig, outdir, "4_arc_positions")


def caps_and_webs(D):
    """Spar-cap width [m] and chordwise centre x/c, web attachment x/c, on D.grid (NaN where absent)."""
    caps = {}
    for lay in D.shell_layers():
        if category(lay["name"])[0] != "Spar cap":
            continue
        s0, s1 = D.layer_arcs(lay)
        if s0 is None or s1 is None:
            continue
        on = D.layer_thickness(lay) > 0
        width = np.where(on, arc_width(s0, s1) * D.perimeters, np.nan)
        mid = arc_mid(s0, s1)
        xc = np.array([D.point(D.section(r), m)[0] / D.section(r).chord for r, m in zip(D.grid, mid)])
        caps[lay["name"]] = (width, np.where(on, xc, np.nan))
    webs = {}
    for w in D.webs.values():
        w0, w1 = D.web_arcs(w)
        if w0 is None or w1 is None:
            continue
        on = D.web_active(w)
        wl = D.web_layers(w["name"])
        if wl:
            on &= np.any([D.layer_thickness(L) > 0 for L in wl], axis=0)
        xa = np.array([D.point(D.section(r), s)[0] / D.section(r).chord for r, s in zip(D.grid, w0)])
        xb = np.array([D.point(D.section(r), s)[0] / D.section(r).chord for r, s in zip(D.grid, w1)])
        webs[w["name"]] = (np.where(on, xa, np.nan), np.where(on, xb, np.nan))
    return caps, webs


def plot_caps_webs(designs, outdir):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    st = styles(len(designs))
    for D, s in zip(designs, st):
        caps, webs = caps_and_webs(D)
        for k, (name, (w, c)) in enumerate(caps.items()):
            mk = dict(marker=["o", "s", "^", "v"][k % 4], ms=3.5, markevery=max(1, len(D.grid) // 12))
            axs[0].plot(D.grid, w, label=f"{D.label}: {name}", **s, **mk)
            axs[1].plot(D.grid, c, label=f"{D.label}: {name}", **s, **mk)
        for k, (name, (xa, xb)) in enumerate(webs.items()):
            mk = dict(marker=["^", "v", "D", "x"][k % 4], ms=3.5, markevery=max(1, len(D.grid) // 12))
            axs[2].plot(D.grid, xa, label=f"{D.label}: {name} ss", **s, **mk)
            axs[2].plot(D.grid, xb, label=f"{D.label}: {name} ps", alpha=.55, **s, **mk)
    for a, title, lab in zip(axs, ["Spar-cap width", "Spar-cap chordwise centre", "Web chordwise position"],
                             ["width along the surface [m]", "x/c [-]", "attachment x/c [-]"]):
        a.set_title(title); a.set_ylabel(lab); a.set_xlabel("r/R"); a.grid(alpha=.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, outdir, "5_caps_and_webs")


def plot_sections(designs, outdir, spans, scale):
    mcol = material_colors(designs)
    fig, ax = plt.subplots(len(spans), len(designs), figsize=(6 * len(designs), 2.8 * len(spans)),
                           squeeze=False)
    used = []
    for i, r in enumerate(spans):
        for j, D in enumerate(designs):
            a = ax[i, j]
            sec, shells, webs = layup(D, r, scale)
            a.plot(sec.X, sec.Y, color="k", lw=.7)
            for lay, t, polys in shells:
                for px, py in polys:
                    a.fill(px, py, color=mcol[lay["material"]], ec="white", lw=.2)
                used.append(lay["material"])
                if category(lay["name"])[0] and polys:
                    px, py = max(polys, key=lambda p: p[0].size)
                    a.annotate(f"{t * 1e3:.1f}", (px.mean(), py.mean()), fontsize=6.5,
                               ha="center", va="center")
            for w, p1, p2, quads in webs:
                if not quads:
                    a.plot(*zip(p1, p2), color="grey", lw=.8, ls=":")
                for L, t, (qx, qy) in quads:
                    a.fill(qx, qy, color=mcol[L["material"]], ec="white", lw=.2)
                    used.append(L["material"])
            a.set_aspect("equal"); a.grid(alpha=.2)
            a.set_title(f"{D.label}  –  r/R = {r:.2f}  (chord {sec.chord:.2f} m)", fontsize=9)
        # same limits across a row so the designs can be compared by eye
        xl = [ax[i, j].get_xlim() for j in range(len(designs))]
        yl = [ax[i, j].get_ylim() for j in range(len(designs))]
        for j in range(len(designs)):
            ax[i, j].set_xlim(min(x[0] for x in xl), max(x[1] for x in xl))
            ax[i, j].set_ylim(min(y[0] for y in yl), max(y[1] for y in yl))
    handles = [Patch(color=mcol[m], label=m) for m in dict.fromkeys(used)]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), fontsize=8)
    note = "to scale" if scale == 1 else f"thickness drawn ×{scale:g}"
    fig.suptitle(f"Cross-sections (untwisted, x from LE; layers stacked outer → inner, {note}; "
                 "numbers = cap/reinforcement thickness in mm)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save(fig, outdir, "6_sections", tight=False)


def _elastic_series(D):
    """Return {quantity: (grid, values)} from the elastic properties, or {}."""
    ep = D.elastic
    if not ep:
        return {}
    out = {}
    # windIO 6x6 stiffness: K33 = EA, K44 = EI_edge, K55 = EI_flap, K66 = GJ (reference-axis frame)
    K = ep.get("stiffness_matrix") or {}
    for k, name in [("K33", "EA"), ("K55", "EI_flap"), ("K44", "EI_edge"), ("K66", "GJ")]:
        if k in K and "grid" in K:
            out[name] = (np.asarray(K["grid"], float), np.asarray(K[k], float))
    im = ep.get("inertia_matrix") or {}
    if "mass" in im and "grid" in im:
        out["mass"] = (np.asarray(im["grid"], float), np.asarray(im["mass"], float))
    return out


def plot_elastic(designs, outdir):
    series = [_elastic_series(D) for D in designs]
    if not any(series):
        print("no elastic_properties in any file – skipping 7_elastic_props")
        return
    keys = [("EA", "EA [N]"), ("EI_flap", "EI flap [N·m²]"), ("EI_edge", "EI edge [N·m²]"),
            ("GJ", "GJ [N·m²]"), ("mass", "Mass per length [kg/m]")]
    base = next(i for i, s in enumerate(series) if s)
    others = [i for i, s in enumerate(series) if s and i != base]
    nrow = 2 if others else 1
    fig, ax = plt.subplots(nrow, 5, figsize=(22, 4 * nrow), squeeze=False, sharex=True)
    st = styles(len(designs))
    for i, (D, ser, s) in enumerate(zip(designs, series, st)):
        for k, (key, title) in enumerate(keys):
            if key not in ser:
                continue
            g, v = ser[key]
            a = ax[0, k]
            a.plot(g, v, label=D.label, **s)
            if key != "mass" and np.all(v > 0):
                a.set_yscale("log")
            a.set_title(title)
            if i in others and key in series[base]:
                gb, vb = series[base][key]
                o = np.argsort(gb)
                ref = np.interp(g, gb[o], vb[o])
                with np.errstate(divide="ignore", invalid="ignore"):
                    pct = np.where(np.abs(ref) > 0, 100 * (v - ref) / ref, np.nan)
                ax[1, k].plot(g, pct, label=f"{D.label} vs {designs[base].label}", **s)
                ax[1, k].axhline(0, color="grey", lw=.8)
                ax[1, k].set_ylabel("change [%]")
    for a in ax.flat:
        a.grid(alpha=.3, which="both")
    for a in ax[-1]:
        a.set_xlabel("r/R")
    ax[0, 0].legend()
    if others:
        ax[1, 0].legend(fontsize=8)
    missing = [D.label for D, s in zip(designs, series) if not s]
    if missing:
        fig.suptitle("No elastic properties stored for: " + ", ".join(missing), fontsize=9)
    save(fig, outdir, "7_elastic_props")


def layer_masses(D):
    """Estimated mass of each layer [kg]: ∫ t · width · rho ds along the reference axis.
    Shell layers use arc width × outer perimeter (stacking ignored); web layers use the
    web height between the outer-surface attachment points."""
    out = {}
    for lay in D.layers.values():
        mat = D.materials.get(lay["material"])
        if mat is None or "rho" not in mat:
            warn(f"{D.label}: no density for material '{lay['material']}' (layer {lay['name']}) – mass set to 0")
        rho = float((mat or {}).get("rho", 0.0))
        t = D.layer_thickness(lay)
        if "web" in lay:
            web = D.webs.get(lay["web"])
            if web is None:
                warn(f"{D.label}: layer {lay['name']} refers to undefined web '{lay['web']}'")
            width = None if web is None else D.web_height(web)
            if width is not None:
                width = np.where(D.web_active(web), width, 0.0)
        else:
            s0, s1 = D.layer_arcs(lay)
            width = None if s0 is None or s1 is None else arc_width(s0, s1) * D.perimeters
        if width is None:
            warn(f"{D.label}: could not resolve extent of layer {lay['name']} – skipped in mass table")
            continue
        out[lay["name"]] = float(_trapz(t * width * rho, D.s_curve))
    return out


def plot_masses(designs, outdir):
    masses = [layer_masses(D) for D in designs]
    names = list(dict.fromkeys(n for m in masses for n in m))
    x = np.arange(len(names)); w = 0.8 / len(designs)
    fig, ax = plt.subplots(figsize=(max(8, .8 * len(names)), 4.5))
    for i, (D, m) in enumerate(zip(designs, masses)):
        ax.bar(x + i * w, [m.get(n, 0) for n in names], w, label=f"{D.label} (Σ {sum(m.values()):.0f} kg)")
    ax.set_xticks(x + w * (len(designs) - 1) / 2); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Estimated mass per blade [kg]"); ax.grid(alpha=.3, axis="y"); ax.legend()
    ax.set_title("Laminate mass by layer (geometric estimate, excludes adhesive and root hardware)")
    save(fig, outdir, "8_layer_mass")

    two = len(designs) == 2
    rows = [[n] + [m.get(n, 0.0) for m in masses] for n in names]
    rows.append(["TOTAL"] + [sum(m.values()) for m in masses])
    say("\nEstimated layer mass per blade [kg]")
    say(f"{'layer':<22}" + "".join(f"{D.label:>14}" for D in designs) +
        (f"{'change':>12}{'%':>8}" if two else ""))
    for r in rows:
        line = f"{r[0]:<22}" + "".join(f"{v:14.1f}" for v in r[1:])
        if two:
            d = r[2] - r[1]
            line += f"{d:+12.1f}" + (f"{100 * d / r[1]:+8.1f}" if r[1] else f"{'–':>8}")
        say(line)

    p = os.path.join(outdir, "layer_mass.csv")
    with open(p, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["layer"] + [D.label for D in designs] + (["change"] if two else []))
        for r in rows:
            wr.writerow([r[0]] + [f"{v:.2f}" for v in r[1:]] + ([f"{r[2] - r[1]:.2f}"] if two else []))
    print("saved", p)
    return masses


def report_changes(designs, masses):
    """Print what changed between the first two designs, plus consistency checks."""
    if len(designs) >= 2:
        A, B = designs[:2]
        g = A.grid
        added = [n for n in B.layers if n not in A.layers]
        removed = [n for n in A.layers if n not in B.layers]
        if added:
            say(f"\nLayers only in {B.label}: " + ", ".join(added))
        if removed:
            say(f"Layers only in {A.label}: " + ", ".join(removed))

        say(f"\nChanges {A.label} -> {B.label} (max over span, location in r/R):")
        say(f"  {'layer':<22}{'|Δt| [mm]':>12}{'at r/R':>8}{'|Δ start arc|':>15}{'|Δ end arc|':>13}")
        for n, lb in B.layers.items():
            la = A.layers.get(n)
            if la is None:
                continue
            dt = np.abs(B.layer_thickness(lb) - A.layer_thickness(la))
            k = int(np.argmax(dt))
            cols = [f"{dt[k] * 1e3:12.3f}", f"{g[k]:8.2f}" if dt[k] > 1e-9 else f"{'–':>8}"]
            changed = dt[k] > 1e-9
            if "web" in la and "web" in lb:   # web layers: compare the web's own arcs
                wa, wb = A.webs.get(la["web"]), B.webs.get(lb["web"])
                arcs_a = A.web_arcs(wa) if wa else (None, None)
                arcs_b = B.web_arcs(wb) if wb else (None, None)
            else:
                arcs_a, arcs_b = A.layer_arcs(la), B.layer_arcs(lb)
            for ea, eb, width in zip(arcs_a, arcs_b, (15, 13)):
                if ea is None or eb is None:
                    cols.append(f"{'n/a':>{width}}")
                    continue
                d = float(np.max(np.abs(eb - ea)))
                changed |= d > 1e-6
                cols.append(f"{d:{width}.4f}")
            say(f"  {n:<22}" + "".join(cols) + ("" if changed else "   (unchanged)"))

    k = len(designs[0].grid) // 2
    say(f"\nSpar caps and webs at r/R = {designs[0].grid[k]:.2f}:")
    for D in designs:
        caps, webs = caps_and_webs(D)
        for n, (w, c) in caps.items():
            say(f"  {D.label:<12} {n:<16} width {w[k]:.3f} m, centre x/c {c[k]:.3f}")
        for w in D.webs.values():
            w0, w1 = D.web_arcs(w)
            if w0 is None or w1 is None:
                say(f"  {D.label:<12} {w['name']:<16} (unresolved)")
                continue
            xa, xb = webs[w["name"]]
            say(f"  {D.label:<12} {w['name']:<16} nd arc ss {w0[k]:.3f} ps {w1[k]:.3f}, "
                f"x/c ss {xa[k]:.3f} ps {xb[k]:.3f}")

    say("\nChecks")
    n_issues = 0
    for D in designs:
        arcs = [(n, D.web_arcs(w)) for n, w in D.webs.items()]
        arcs = [(n, a) for n, a in arcs if a[0] is not None and a[1] is not None]
        for i in range(len(arcs)):
            for j in range(i + 1, len(arcs)):
                (na, (a0, a1)), (nb, (b0, b1)) = arcs[i], arcs[j]
                if np.allclose(a0, b0) and np.allclose(a1, b1):
                    n_issues += 1
                    say(f"  WARNING {D.label}: webs '{na}' and '{nb}' sit at the same position – "
                        "check the web anchors / design variables.")
        for lay in D.layers.values():
            if lay["material"] not in D.materials:
                n_issues += 1
                say(f"  WARNING {D.label}: material '{lay['material']}' of layer '{lay['name']}' is not defined.")
            if "web" in lay and lay["web"] not in D.webs:
                n_issues += 1
                say(f"  WARNING {D.label}: layer '{lay['name']}' refers to undefined web '{lay['web']}'.")
    if not n_issues:
        say("  no overlapping webs, undefined materials or undefined webs found")

    say("\nBlade length along reference axis: " +
        ", ".join(f"{D.label} {D.span_len:.2f} m" for D in designs))
    for D, m in zip(designs, masses):
        ser = _elastic_series(D).get("mass")
        if ser is None:
            continue
        g, v = ser
        beam = _trapz(v, np.interp(g, D.grid, D.s_curve))
        lam = sum(m.values())
        say(f"  {D.label}: mass from inertia_matrix {beam:.0f} kg, laminate estimate {lam:.0f} kg "
            f"({100 * lam / beam:.0f} %)" if beam else f"  {D.label}: inertia_matrix mass is zero")


# --------------------------------------------------------------------------- main
def settings_argv():
    """Turn SETTINGS into command-line arguments, with paths relative to this script."""
    here = os.path.dirname(os.path.abspath(__file__))
    where = lambda p: p if os.path.isabs(p) else os.path.join(here, p)
    argv = [where(f) for f in SETTINGS["files"]]
    if SETTINGS.get("labels"):
        argv += ["--labels", *map(str, SETTINGS["labels"])]
    argv += ["--spans", *map(str, SETTINGS["spans"]), "--scale", str(SETTINGS["scale"]),
             "--ngrid", str(SETTINGS["ngrid"]), "--format", SETTINGS["format"],
             "--dpi", str(SETTINGS["dpi"]), "--outdir", where(SETTINGS["outdir"])]
    return argv


def main(argv=None):
    if argv is None and len(sys.argv) == 1:        # no arguments: use SETTINGS
        argv = settings_argv()
        print("No command-line arguments, using SETTINGS at the top of the script.\n")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="windIO v2 yaml files to compare")
    p.add_argument("--labels", nargs="+", help="labels for the files (one per file)")
    p.add_argument("--spans", nargs="+", type=float, default=[0.25, 0.5, 0.75],
                   help="r/R stations for the cross-section plot")
    p.add_argument("--scale", type=float, default=1.0,
                   help="thickness magnification in the section plot (1 = to scale)")
    p.add_argument("--ngrid", type=int, default=NGRID, help="number of spanwise evaluation points")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="figure file format")
    p.add_argument("--dpi", type=int, default=150, help="resolution for raster figures")
    p.add_argument("--outdir", default="blade_compare")
    args = p.parse_args(argv)

    if args.labels and len(args.labels) != len(args.files):
        p.error(f"got {len(args.files)} files but {len(args.labels)} labels")
    for f in args.files:
        if not os.path.isfile(f):
            p.error(f"file not found: {f}")
    bad = [r for r in args.spans if not 0.0 <= r <= 1.0]
    if bad:
        p.error(f"--spans must lie in [0, 1], got {bad}")
    if args.ngrid < 10:
        p.error("--ngrid must be at least 10")
    SAVE.update(dpi=args.dpi, fmt=args.format)
    REPORT.clear()

    labels = args.labels or [os.path.splitext(os.path.basename(f))[0] for f in args.files]
    designs = [Design(f, l, args.ngrid) for f, l in zip(args.files, labels)]
    os.makedirs(args.outdir, exist_ok=True)
    say("Blade structural comparison")
    for D in designs:
        say(f"  {D.label}: {D.path}  ({len(D.layers)} layers, {len(D.webs)} webs)")

    plot_outer_shape(designs, args.outdir)
    plot_layer_thickness(designs, args.outdir, web=False)
    plot_layer_thickness(designs, args.outdir, web=True)
    plot_arcs(designs, args.outdir)
    plot_caps_webs(designs, args.outdir)
    plot_sections(designs, args.outdir, args.spans, args.scale)
    plot_elastic(designs, args.outdir)
    masses = plot_masses(designs, args.outdir)
    report_changes(designs, masses)

    p_sum = os.path.join(args.outdir, "summary.txt")
    with open(p_sum, "w") as f:
        f.write("\n".join(REPORT) + "\n")
    print("saved", p_sum)


if __name__ == "__main__":
    main()
