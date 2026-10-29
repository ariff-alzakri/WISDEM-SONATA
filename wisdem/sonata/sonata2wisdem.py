#!/usr/bin/env python3
"""
sonata2wisdem.py
================

Convert a SONATA blade input file (windIO v1.0 layout plus SONATA extensions)
into a windIO v2 turbine file that WISDEM 4.x can read for structural and
hydro-structural (tidal) optimisation.

It follows the rules of the official windIO v1 -> v2 converter
(windIO/converters/windIO2windIO.py) and also handles the items that converter
does not:

  * layers defined only by a mid-point + width (e.g. LE reinforcement with
    no start/end arc), whose arcs and the LE anchor are computed by lofting
    the actual airfoil shapes instead of using dummy values;
  * layers that wrap through the trailing edge (start_nd_arc > end_nd_arc),
    which are split into a suction-side and a pressure-side layer;
  * airfoil polars mixing the v1 (c_l, radians) and v2 (re_sets, degrees)
    formats in the same file;
  * airfoils without relative_thickness (computed from coordinates);
  * the SONATA-only `cavity_filler`, which WISDEM/PreComp cannot model
    (the missing foam mass is estimated and reported);
  * the windIO v2 move of environment and costs into the WISDEM modeling
    options (written to a separate snippet, with seawater properties).

Non-blade components (hub, drivetrain, tower, control, assembly) are taken
from an existing WISDEM v2 file given with --template when one is available,
because the values in the SONATA file are often leftovers from the reference
turbine it was derived from.

Usage
-----
    python sonata2wisdem.py Q3v5_web.yaml -o Q3v5_wisdem.yaml \
        --template stall_margin_12p0.yaml

    python sonata2wisdem.py Q3v5_web.yaml -o Q3v5_wisdem.yaml \
        --template stall_margin_12p0.yaml --arc-mode parametric

Arc modes
---------
  explicit   (default) Every layer and web keeps the start/end arc positions
             written in the SONATA file. WISDEM then analyses exactly the
             section SONATA meshed. Best for structural optimisation
             (thickness design variables) on a fixed outer shape.
  parametric Spar caps, webs, LE/TE reinforcements also carry their width /
             plane-offset definitions, so WISDEM re-places them when chord
             or twist change. Use for combined hydro-structural optimisation,
             but read the consistency warnings first: if the SONATA widths or
             offsets disagree with the SONATA arcs, the section will change.

Requirements: Python >= 3.8, numpy, scipy, pyyaml.
"""

import argparse
import copy
import re
import sys
from datetime import date

import numpy as np
import yaml

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:  # pragma: no cover
    PchipInterpolator = None


# --------------------------------------------------------------------------
# YAML I/O
# --------------------------------------------------------------------------
class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads YAML 1.2 floats such as 140e6 or 1e-5."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _represent_list(dumper, data):
    flow = all(isinstance(x, (int, float, str, bool)) or x is None for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)


def to_builtin(obj):
    """Recursively turn numpy types into plain Python types for YAML output."""
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_builtin(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def write_yaml(data, path, header=""):
    text = yaml.dump(to_builtin(data), Dumper=_Dumper, sort_keys=False,
                     default_flow_style=False, width=1e9, allow_unicode=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            f.write(header)
        f.write(text)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
class Log:
    def __init__(self):
        self.warnings = []
        self.notes = []

    def warn(self, msg):
        self.warnings.append(msg)
        print("  WARNING: " + msg)

    def note(self, msg):
        self.notes.append(msg)
        print("  note: " + msg)


LOG = Log()


def gv(d):
    return np.asarray(d["grid"], dtype=float), np.asarray(d["values"], dtype=float)


def gvdict(grid, values):
    return {"grid": [float(g) for g in grid], "values": [float(v) for v in values]}


def interp(x, d):
    g, v = gv(d)
    return np.interp(x, g, v)


def pchip_or_linear(x_new, x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if PchipInterpolator is not None and np.all(np.diff(x) > 0):
        return PchipInterpolator(x, y)(x_new)
    return np.interp(x_new, x, y)


# --------------------------------------------------------------------------
# Airfoils
# --------------------------------------------------------------------------
def split_surfaces(x, y):
    """Split windIO coordinates (TE -> suction -> LE -> pressure -> TE)
    into upper and lower surfaces, each as (x, y) sorted from LE to TE."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ile = int(np.argmin(x))
    xu, yu = x[: ile + 1][::-1], y[: ile + 1][::-1]
    xl, yl = x[ile:], y[ile:]
    iu = np.argsort(xu, kind="stable")
    il = np.argsort(xl, kind="stable")
    return (xu[iu], yu[iu]), (xl[il], yl[il])


def rthick_from_coords(coords):
    (xu, yu), (xl, yl) = split_surfaces(coords["x"], coords["y"])
    xs = np.linspace(0.0, 1.0, 801)
    return float(np.max(np.abs(np.interp(xs, xu, yu) - np.interp(xs, xl, yl))))


def _aoa_to_deg(grid):
    g = np.asarray(grid, float).copy()
    if np.max(np.abs(g)) <= np.pi + 1e-3:  # radians
        g = np.rad2deg(g)
    if abs(g[0] + 180.0) < 0.5:
        g[0] = -180.0
    if abs(g[-1] - 180.0) < 0.5:
        g[-1] = 180.0
    return g


def _coef(block):
    return {"grid": _aoa_to_deg(block["grid"]).tolist(),
            "values": [float(v) for v in block["values"]]}


def convert_airfoil(af_v1, rthick_override=None):
    af = {"name": af_v1["name"], "coordinates": copy.deepcopy(af_v1["coordinates"])}
    if "aerodynamic_center" in af_v1:
        af["aerodynamic_center"] = float(af_v1["aerodynamic_center"])

    configs = {}  # configuration name -> list of re_sets
    for p in af_v1.get("polars", []):
        cfg = str(p.get("configuration", "default")).strip()
        cfg = "default" if cfg.lower() == "default" else cfg
        sets = []
        if "re_sets" in p:  # already v2 style
            for rs in p["re_sets"]:
                new = {k: copy.deepcopy(v) for k, v in rs.items() if k not in ("cl", "cd", "cm")}
                for k in ("cl", "cd", "cm"):
                    if k in rs:
                        new[k] = _coef(rs[k])
                sets.append(new)
        else:  # v1 style: c_l / c_d / c_m on a radian grid
            new = {"re": float(p.get("re", 1.0e6))}
            for k1, k2 in (("c_l", "cl"), ("c_d", "cd"), ("c_m", "cm")):
                if k1 in p:
                    new[k2] = _coef(p[k1])
            sets.append(new)
        configs.setdefault(cfg, []).extend(sets)

    af["polars"] = []
    for cfg, sets in configs.items():
        sets = sorted(sets, key=lambda s: s["re"])
        af["polars"].append({"configuration": cfg, "re_sets": sets})
    if "default" not in configs:
        LOG.warn(f"airfoil {af['name']}: no 'default' polar configuration; "
                 "the blade airfoil list references 'default'.")

    geo_t = rthick_from_coords(af["coordinates"])
    if rthick_override and af["name"] in rthick_override:
        af["rthick"] = float(rthick_override[af["name"]])
        LOG.note(f"airfoil {af['name']}: rthick set to {af['rthick']} by --rthick "
                 f"(coordinates are {geo_t:.4f} thick)")
    elif "relative_thickness" in af_v1:
        af["rthick"] = float(af_v1["relative_thickness"])
    elif "rthick" in af_v1:
        af["rthick"] = float(af_v1["rthick"])
    else:
        af["rthick"] = round(geo_t, 4)
        LOG.note(f"airfoil {af['name']}: relative thickness missing, computed "
                 f"from coordinates = {af['rthick']:.4f}")
    if abs(af["rthick"] - geo_t) > 0.01 and af["rthick"] < 0.99:
        LOG.warn(f"airfoil {af['name']}: rthick {af['rthick']:.3f} but the coordinates are "
                 f"{geo_t:.3f} thick.")
    m = re.fullmatch(r"(?i)naca\s*-?\s*(\d)(\d)(\d\d)", af["name"].strip())
    if m and not (rthick_override and af["name"] in rthick_override) and abs(int(m.group(3)) / 100.0 - geo_t) > 0.01:
        LOG.warn(f"airfoil {af['name']}: the name implies {int(m.group(3))}% thickness but the "
                 f"coordinates are {100*geo_t:.1f}% thick (the polars may belong to a different "
                 f"shape). Use --rthick {af['name']}=<value> to force a value.")
    return af


# --------------------------------------------------------------------------
# Section lofting (used for LE anchor, width-defined arcs and diagnostics)
# --------------------------------------------------------------------------
class Lofter:
    """Approximate blade cross sections from the airfoil stack.

    Shapes are blended between the spanwise-neighbouring airfoils according
    to the local relative thickness, which is close to what WISDEM does.
    Only used for arc estimates and checks, never for the aero model."""

    def __init__(self, af_db, af_list, rthick, chord, pitch_axis, n=181):
        self.xc = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
        self.pos = np.array([a["spanwise_position"] for a in af_list], float)
        self.names = [a["name"] for a in af_list]
        self.t_af = np.array([af_db[nm]["rthick"] for nm in self.names], float)
        self.rthick = rthick
        self.chord = chord
        self.pitch_axis = pitch_axis
        self.shapes = {}
        for nm in set(self.names):
            (xu, yu), (xl, yl) = split_surfaces(af_db[nm]["coordinates"]["x"],
                                                af_db[nm]["coordinates"]["y"])
            yu_i, yl_i = np.interp(self.xc, xu, yu), np.interp(self.xc, xl, yl)
            if np.trapezoid(yu_i - yl_i, self.xc) < 0:  # coordinates given clockwise
                yu_i, yl_i = yl_i, yu_i
            self.shapes[nm] = (yu_i, yl_i)
        self._cache = {}

    def _blend(self, r):
        p = self.pos
        if r <= p[0]:
            return self.shapes[self.names[0]]
        if r >= p[-1]:
            return self.shapes[self.names[-1]]
        j = int(np.searchsorted(p, r, side="right"))
        i = j - 1
        t = float(interp(r, self.rthick))
        ti, tj = self.t_af[i], self.t_af[j]
        if abs(tj - ti) > 1e-6:
            w = (t - ti) / (tj - ti)
        else:
            w = (r - p[i]) / max(p[j] - p[i], 1e-12)
        w = min(max(w, 0.0), 1.0)
        (ui, li), (uj, lj) = self.shapes[self.names[i]], self.shapes[self.names[j]]
        return (1 - w) * ui + w * uj, (1 - w) * li + w * lj

    def section(self, r):
        key = round(float(r), 10)
        if key in self._cache:
            return self._cache[key]
        yu, yl = self._blend(r)
        c = float(interp(r, self.chord))
        x = np.concatenate([self.xc[::-1], self.xc[1:]]) * c
        y = np.concatenate([yu[::-1], yl[1:]]) * c
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        out = {
            "x": x, "y": y, "s": s, "P": s[-1], "c": c, "yu": yu * c, "yl": yl * c,
            "le_nd": s[len(self.xc) - 1] / s[-1],
            "area": 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)),
        }
        self._cache[key] = out
        return out

    def x_at_arc(self, r, nd):
        sec = self.section(r)
        return float(np.interp(nd * sec["P"], sec["s"], sec["x"]))

    def height_at_x(self, r, x_abs):
        sec = self.section(r)
        xx = self.xc * sec["c"]
        return float(np.interp(x_abs, xx, sec["yu"]) - np.interp(x_abs, xx, sec["yl"]))

    def arcs_at_x(self, r, x_abs):
        """nd arc of the suction- and pressure-side points at chordwise x."""
        sec = self.section(r)
        n = len(self.xc)
        s_ss = sec["s"][:n][::-1]          # LE -> TE on suction side
        x_ss = sec["x"][:n][::-1]
        s_ps = sec["s"][n - 1:]            # LE -> TE on pressure side
        x_ps = sec["x"][n - 1:]
        return (float(np.interp(x_abs, x_ss, s_ss)) / sec["P"],
                float(np.interp(x_abs, x_ps, s_ps)) / sec["P"])


# --------------------------------------------------------------------------
# Blade outer shape
# --------------------------------------------------------------------------
def convert_outer_shape(os_v1, af_db):
    out = {}
    cg, cv = gv(os_v1["chord"])
    tg, tv = gv(os_v1["twist"])
    if np.max(np.abs(tv)) < 2.0 * np.pi:
        tv = np.rad2deg(tv)  # windIO v1 twist is in radians
    else:
        LOG.note("twist already looks like degrees, kept as is")

    pa_on_chord = np.interp(cg, *gv(os_v1["pitch_axis"]))
    out["chord"] = gvdict(cg, cv)
    out["twist"] = gvdict(tg, tv)
    out["section_offset_y"] = gvdict(cg, pa_on_chord * cv)

    afp = os_v1["airfoil_position"]
    out["airfoils"] = []
    for g, nm in zip(afp["grid"], afp["labels"]):
        if nm not in af_db:
            raise KeyError(f"airfoil '{nm}' used on the blade is not in the airfoil list")
        out["airfoils"].append({"name": nm, "spanwise_position": float(g),
                                "configuration": ["default"], "weight": [1.0]})

    if "rthick" in os_v1:
        out["rthick"] = copy.deepcopy(os_v1["rthick"])
    else:
        t_af = [af_db[nm]["rthick"] for nm in afp["labels"]]
        rt = np.minimum(pchip_or_linear(cg, afp["grid"], t_af), 1.0)
        out["rthick"] = gvdict(cg, rt)

    # Diagnostic: steep thickness change between neighbouring airfoils
    g = np.asarray(afp["grid"], float)
    t = np.asarray([af_db[nm]["rthick"] for nm in afp["labels"]])
    for i in range(len(g) - 1):
        if g[i + 1] > g[i]:
            slope = abs(t[i + 1] - t[i]) / (g[i + 1] - g[i])
            if slope > 4.0:
                LOG.warn(f"relative thickness drops from {t[i]:.2f} to {t[i+1]:.2f} "
                         f"between r/R={g[i]:.3f} and {g[i+1]:.3f}; WISDEM will "
                         "interpolate a very short transition.")
    return out


# --------------------------------------------------------------------------
# Blade internal structure
# --------------------------------------------------------------------------
PLANE_ROT0 = {"anchor_curve": "reference_axis", "anchors_nd_grid": [0.0, 1.0], "rotation": 0.0}


def _has_values(d):
    return isinstance(d, dict) and "values" in d and "grid" in d


def _fiber(layer, grid_default, to_deg):
    if "fiber_orientation" in layer:
        g, v = gv(layer["fiber_orientation"])
        if to_deg:
            v = np.rad2deg(v)
        return gvdict(g, v)
    return gvdict([grid_default[0], grid_default[-1]], [0.0, 0.0])


def _lam_mass_props(materials):
    return {m["name"]: float(m.get("rho", 0.0)) for m in materials}


def convert_structure(st_v1, lofter, ref_z, mode, fiber_to_deg, te_names, materials_v1):
    st = {"anchors": [], "webs": [], "layers": []}
    anchors = st["anchors"]
    anchor_names = set()

    def add_anchor(a):
        if a["name"] in anchor_names:
            raise ValueError(f"duplicate anchor name {a['name']}")
        anchor_names.add(a["name"])
        anchors.append(a)

    # Reference grid for computed quantities: first layer thickness grid
    sgrid = np.asarray(st_v1["layers"][0]["thickness"]["grid"], float)

    add_anchor({"name": "TE",
                "start_nd_arc": gvdict([0.0, 1.0], [0.0, 0.0]),
                "end_nd_arc": gvdict([0.0, 1.0], [1.0, 1.0])})
    le_nd = np.array([lofter.section(r)["le_nd"] for r in sgrid])
    add_anchor({"name": "LE", "start_nd_arc": gvdict(sgrid, le_nd)})

    perim = np.array([lofter.section(r)["P"] for r in sgrid])
    pa_abs = lambda r: float(interp(r, lofter.pitch_axis)) * float(interp(r, lofter.chord))

    # ---------------- webs ----------------
    web_arcs = {}
    for w in st_v1.get("webs", []):
        name = w["name"]
        a = {"name": name}
        has_arcs = _has_values(w.get("start_nd_arc")) and _has_values(w.get("end_nd_arc"))
        has_off = "offset_y_pa" in w
        if has_arcs:
            a["start_nd_arc"] = gvdict(*gv(w["start_nd_arc"]))
            a["end_nd_arc"] = gvdict(*gv(w["end_nd_arc"]))
        if (mode == "parametric" and has_off) or not has_arcs:
            if not has_off:
                raise ValueError(f"web {name}: needs start/end_nd_arc or offset_y_pa")
            a["plane_intersection"] = {"side": "both",
                                       "defines": ["start_nd_arc", "end_nd_arc"],
                                       "plane_type1": copy.deepcopy(PLANE_ROT0),
                                       "offset": gvdict(*gv(w["offset_y_pa"]))}
        # Arcs implied by the offset (for checks / missing values)
        if has_off:
            og, ov = gv(w["offset_y_pa"])
            implied = np.array([lofter.arcs_at_x(r, pa_abs(r) + np.interp(r, og, ov))
                                for r in sgrid])
            if not has_arcs:
                a["start_nd_arc"] = gvdict(sgrid, implied[:, 0])
                a["end_nd_arc"] = gvdict(sgrid, implied[:, 1])
            else:
                sa = np.interp(sgrid, *gv(w["start_nd_arc"]))
                ea = np.interp(sgrid, *gv(w["end_nd_arc"]))
                inner = (sgrid > 0.12) & (sgrid < 0.95)
                dmax = np.max(np.abs(np.r_[sa - implied[:, 0], ea - implied[:, 1]][np.r_[inner, inner]]))
                if dmax > 0.02:
                    msg = (f"web {name}: SONATA start/end_nd_arc and offset_y_pa disagree "
                           f"(up to {dmax:.3f} in nd arc, about {dmax*perim.mean():.2f} m). ")
                    msg += ("In parametric mode WISDEM will place the web from the offset, "
                            "i.e. not where SONATA meshed it." if mode == "parametric"
                            else "Explicit mode keeps the SONATA arcs.")
                    LOG.warn(msg)
        web_arcs[name] = (np.interp(sgrid, *gv(a["start_nd_arc"])),
                          np.interp(sgrid, *gv(a["end_nd_arc"])))
        g0, g1 = float(a["start_nd_arc"]["grid"][0]), float(a["start_nd_arc"]["grid"][-1])
        add_anchor(a)
        st["webs"].append({
            "name": name, "start_nd_grid": g0, "end_nd_grid": g1,
            "start_nd_arc": {"anchor": {"name": name, "handle": "start_nd_arc"}},
            "end_nd_arc": {"anchor": {"name": name, "handle": "end_nd_arc"}},
            "anchors": [{"name": f"{name}_shell_attachment",
                         "start_nd_arc": gvdict([g0, g1], [0.0, 0.0]),
                         "end_nd_arc": gvdict([g0, g1], [1.0, 1.0])}],
        })

    # ---------------- layers ----------------
    rho = _lam_mass_props(materials_v1)
    lam_area = np.zeros_like(sgrid)     # shell laminate area [m2]
    lam_mass = np.zeros_like(sgrid)     # shell + web laminate mass [kg/m]
    web_area = np.zeros_like(sgrid)
    usage = {}                          # material -> set of roles (for manufacturing_id)

    def base_layer(L, name):
        tg = L["thickness"]["grid"]
        out = {"name": name, "start_nd_grid": float(tg[0]), "end_nd_grid": float(tg[-1]),
               "material": L["material"], "thickness": gvdict(*gv(L["thickness"])),
               "fiber_orientation": _fiber(L, tg, fiber_to_deg)}
        if "n_plies" in L:
            out["n_plies"] = copy.deepcopy(L["n_plies"])
        return out

    def ref(name, handle):
        return {"anchor": {"name": name, "handle": handle}}

    def width_m(L):
        return np.interp(sgrid, *gv(L["width"]))

    for L in st_v1["layers"]:
        name = L["name"]
        tL = np.interp(sgrid, *gv(L["thickness"]))
        mat = L["material"]

        # ---- shear-web layers
        if "web" in L:
            web = L["web"]
            if web not in web_arcs:
                raise ValueError(f"layer {name} references unknown web {web}")
            lay = base_layer(L, name)
            lay["start_nd_arc"] = ref(f"{web}_shell_attachment", "start_nd_arc")
            lay["end_nd_arc"] = ref(f"{web}_shell_attachment", "end_nd_arc")
            lay["web"] = web
            st["layers"].append(lay)
            sa, _ = web_arcs[web]
            h = np.array([lofter.height_at_x(r, lofter.x_at_arc(r, s)) for r, s in zip(sgrid, sa)])
            web_area += tL * h
            lam_mass += rho.get(mat, 0.0) * tL * h
            role = "filler" if rho.get(mat, 0) > 3000 or "filler" in name.lower() else "web"
            usage.setdefault(mat, set()).add(role)
            continue

        nl = name.lower()
        if "uv" in nl or "gelcoat" in nl or "coat" in nl:
            role = "coating"
        elif "spar" in nl or "cap" in nl:
            role = "spar"
        elif "te" in nl.split("_") or "te_" in nl or "trail" in nl or "le_" in nl or "le" == nl[:2]:
            role = "te"
        elif "filler" in nl or "core" in nl or "foam" in nl:
            role = "filler"
        else:
            role = "shell"
        usage.setdefault(mat, set()).add(role)

        s_d, e_d = L.get("start_nd_arc"), L.get("end_nd_arc")
        s_fixed = s_d.get("fixed") if isinstance(s_d, dict) else None
        e_fixed = e_d.get("fixed") if isinstance(e_d, dict) else None
        s_vals = np.interp(sgrid, *gv(s_d)) if _has_values(s_d) else None
        e_vals = np.interp(sgrid, *gv(e_d)) if _has_values(e_d) else None
        has_w = "width" in L
        has_mid = isinstance(L.get("midpoint_nd_arc"), dict) and "fixed" in L["midpoint_nd_arc"]
        mid = L["midpoint_nd_arc"]["fixed"] if has_mid else None
        has_off = "offset_y_pa" in L and "side" in L
        explicit_ok = s_vals is not None and e_vals is not None

        # Fill in missing arcs from width definitions
        if not explicit_ok:
            if has_mid and has_w:
                half = width_m(L) / (2.0 * perim)
                centre = le_nd if mid == "LE" else np.zeros_like(sgrid)
                if mid == "TE":
                    s_vals, e_vals = 1.0 - half, half
                else:
                    s_vals, e_vals = centre - half, centre + half
            elif has_off and has_w:
                og, ov = gv(L["offset_y_pa"])
                mids = []
                for r in sgrid:
                    ss, ps = lofter.arcs_at_x(r, pa_abs(r) + np.interp(r, og, ov))
                    mids.append(ss if L["side"].lower().startswith("s") else ps)
                half = width_m(L) / (2.0 * perim)
                s_vals, e_vals = np.array(mids) - half, np.array(mids) + half
            elif s_fixed is None and e_fixed is None:
                raise ValueError(f"layer {name}: cannot determine arc extent")

        use_def = (mode == "parametric" and (has_w or has_mid or has_off)) or not explicit_ok

        # Diagnostic: width field vs explicit arcs
        if explicit_ok and has_w:
            span = np.where(e_vals >= s_vals, e_vals - s_vals, 1.0 - s_vals + e_vals)
            w_arc = span * perim
            w_def = width_m(L)
            act = (tL > 0) & (w_def > 0) & (sgrid < 0.95)
            if np.any(act):
                rel = np.max(np.abs(w_arc[act] - w_def[act]) / w_def[act])
                if rel > 0.15:
                    LOG.warn(f"layer {name}: the 'width' field ({w_def[act].min():.2f}-"
                             f"{w_def[act].max():.2f} m) and the start/end arcs "
                             f"({w_arc[act].min():.2f}-{w_arc[act].max():.2f} m) disagree "
                             f"by up to {100*rel:.0f}%. "
                             + ("Parametric mode uses the width field." if mode == "parametric"
                                else "Explicit mode uses the arcs."))

        wraps = s_vals is not None and e_vals is not None and np.any(s_vals > e_vals + 1e-9)

        if wraps:
            if not np.all((s_vals > e_vals) | (tL <= 0)):
                LOG.warn(f"layer {name}: wraps through the TE at some stations only; "
                         "split applied everywhere, please check.")
            n_ss, n_ps = te_names if mid == "TE" or "te" in nl else (f"{name}_ss", f"{name}_ps")
            LOG.note(f"layer {name} wraps through the trailing edge -> split into "
                     f"{n_ss} (suction) and {n_ps} (pressure)")
            # suction-side half: TE(start) -> end
            a_ss = {"name": n_ss, "start_nd_arc": ref("TE", "start_nd_arc"),
                    "end_nd_arc": gvdict(sgrid, e_vals)}
            a_ps = {"name": n_ps, "start_nd_arc": gvdict(sgrid, s_vals),
                    "end_nd_arc": ref("TE", "end_nd_arc")}
            if use_def and has_w:
                wg, wv = gv(L["width"])
                a_ss["width"] = {"grid": list(map(float, wg)), "values": list(map(float, wv / 2)),
                                 "defines": ["end_nd_arc"]}
                a_ps["width"] = {"grid": list(map(float, wg)), "values": list(map(float, wv / 2)),
                                 "defines": ["start_nd_arc"]}
            add_anchor(a_ss)
            add_anchor(a_ps)
            for nm, a in ((n_ss, a_ss), (n_ps, a_ps)):
                lay = base_layer(L, nm)
                lay["start_nd_arc"] = ref(nm, "start_nd_arc") if nm == n_ps else ref("TE", "start_nd_arc")
                lay["end_nd_arc"] = ref(nm, "end_nd_arc") if nm == n_ss else ref("TE", "end_nd_arc")
                st["layers"].append(lay)
            span = 1.0 - s_vals + e_vals
        else:
            a = {"name": name}
            lay = base_layer(L, name)
            # start
            if s_fixed is not None:
                lay["start_nd_arc"] = ref(s_fixed, "start_nd_arc" if s_fixed == "TE" else "end_nd_arc")
            else:
                a["start_nd_arc"] = gvdict(sgrid, s_vals)
                lay["start_nd_arc"] = ref(name, "start_nd_arc")
            # end
            if e_fixed is not None:
                lay["end_nd_arc"] = ref(e_fixed, "end_nd_arc" if e_fixed == "TE" else "start_nd_arc")
            else:
                a["end_nd_arc"] = gvdict(sgrid, e_vals)
                lay["end_nd_arc"] = ref(name, "end_nd_arc")
            if use_def:
                if has_mid:
                    a["midpoint_nd_arc"] = ref(mid, "start_nd_arc")
                if has_w:
                    wg, wv = gv(L["width"])
                    defines = ["start_nd_arc", "end_nd_arc"]
                    if s_fixed:
                        defines = ["end_nd_arc"]
                    if e_fixed:
                        defines = ["start_nd_arc"]
                    a["width"] = {"grid": list(map(float, wg)), "values": list(map(float, wv)),
                                  "defines": defines}
                if has_off and not has_mid:
                    a["plane_intersection"] = {"side": L["side"], "defines": ["midpoint_nd_arc"],
                                               "plane_type1": copy.deepcopy(PLANE_ROT0),
                                               "offset": gvdict(*gv(L["offset_y_pa"]))}
            if len(a) > 1:
                add_anchor(a)
            st["layers"].append(lay)
            if s_vals is None or e_vals is None:
                span = np.ones_like(sgrid)
            else:
                span = e_vals - s_vals

        lam_area += tL * span * perim
        lam_mass += rho.get(mat, 0.0) * tL * span * perim

    # ---------------- cavity filler (SONATA only) ----------------
    filler = st_v1.get("cavity_filler")
    report = {}
    z = np.interp(sgrid, *gv(ref_z))
    report["laminate_mass"] = float(np.trapezoid(lam_mass, z))
    if filler:
        fmat = filler.get("material")
        usage.setdefault(fmat, set()).add("filler")
        enclosed = np.array([lofter.section(r)["area"] for r in sgrid])
        cav = np.maximum(enclosed - lam_area - web_area, 0.0)
        report["cavity_mass"] = float(np.trapezoid(rho.get(fmat, 0.0) * cav, z))
        report["cavity_material"] = fmat
        LOG.warn(f"SONATA cavity_filler '{fmat}' cannot be represented in WISDEM "
                 f"(PreComp models laminates only). Estimated missing mass per blade: "
                 f"~{report['cavity_mass']:.0f} kg, against ~{report['laminate_mass']:.0f} kg "
                 "of laminate. WISDEM blade mass, inertia and frequencies will differ from SONATA.")
    return st, usage, report


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------
MANUF_ID = {"coating": 0, "filler": 1, "shell": 2, "web": 3, "spar": 4, "te": 5}
ROLE_PRIORITY = ["spar", "te", "web", "shell", "coating", "filler"]

COMPOSITE_DEFAULTS = {"ply_t": 0.001, "waste": 0.05, "unit_cost": 2.0,
                      "fvf": 0.57, "fwf": 0.745, "fiber_density": 2535.5}

STEEL_DEFAULT = {"name": "steel", "description": "Steel ASTM A572 Grade 50 (default added by converter)",
                 "orth": 0, "rho": 7800.0, "alpha": 0.0, "E": 2.0e11, "nu": 0.3, "G": 7.93e10,
                 "GIc": 0, "GIIc": 0, "alp0": 0.0, "Xt": 4.5e8, "Xc": 4.5e8, "S": 0, "Xy": 3.45e8,
                 "m": 3, "unit_cost": 0.7}

RESIN_DEFAULT = {"name": "resin", "description": "epoxy", "orth": 0.0, "rho": 1150.0,
                 "E": 1.0e6, "nu": 0.3, "G": 312500.0, "alpha": 0.0, "GIc": 0.0,
                 "GIIc": 0.0, "alp0": 0.0, "Xt": 0.0, "Xc": 0.0, "S": 0.0, "unit_cost": 3.63}


def convert_materials(mats_v1, usage, template_mats):
    tmpl = {m["name"].lower(): m for m in (template_mats or [])}
    out = []
    for m0 in mats_v1:
        m = copy.deepcopy(m0)
        nm = m["name"]
        if "component_id" in m:
            m["manufacturing_id"] = int(m.pop("component_id"))
        if "alp0" in m and m["alp0"] is not None and m["alp0"] < np.pi:
            m["alp0"] = float(np.rad2deg(m["alp0"]))
        # isotropic G missing
        if m.get("G") is None and not isinstance(m.get("E"), list):
            m["G"] = float(m["E"]) / (2.0 * (1.0 + float(m.get("nu", 0.3))))
            LOG.note(f"material {nm}: G was Null, set to E/(2(1+nu)) = {m['G']:.3e} Pa")
        roles = usage.get(nm, set())
        if roles and "manufacturing_id" not in m:
            role = next(r for r in ROLE_PRIORITY if r in roles)
            m["manufacturing_id"] = MANUF_ID[role]
            LOG.note(f"material {nm}: manufacturing_id set to {MANUF_ID[role]} ({role}) from its use")
        # strengths entered in MPa instead of Pa
        for key in ("Xt", "Xc", "S"):
            val = m.get(key)
            vals = val if isinstance(val, list) else [val]
            if any(isinstance(v, (int, float)) and 0 < v < 1.0e4 for v in vals):
                LOG.warn(f"material {nm}: {key} = {val} looks like MPa; windIO expects Pa.")
        orth = float(m.get("orth", 0)) == 1.0
        if roles and orth:
            src = tmpl.get(nm.lower(), {})
            missing = []
            for k, dflt in COMPOSITE_DEFAULTS.items():
                if k not in m:
                    m[k] = src.get(k, dflt)
                    missing.append(k)
            for k in ("Xt", "Xc", "S"):
                if k not in m and k in src:
                    m[k] = copy.deepcopy(src[k])
                    missing.append(k)
            if "area_density_dry" not in m:
                m["area_density_dry"] = float(m["ply_t"]) * float(m["rho"]) * float(m["fwf"])
                missing.append("area_density_dry")
            if missing:
                LOG.warn(f"material {nm}: added missing {', '.join(missing)} "
                         f"({'from template' if src else 'generic defaults'}); check values, "
                         "they feed the blade cost model.")
        elif roles:
            src = tmpl.get(nm.lower(), {})
            for k, dflt in (("waste", 0.05), ("unit_cost", 1.0)):
                if k not in m:
                    m[k] = src.get(k, dflt)
                    LOG.warn(f"material {nm}: added missing {k} = {m[k]} "
                             f"({'from template' if k in src else 'generic default'})")
        out.append(m)
    return out


def collect_material_refs(node, acc=None):
    acc = set() if acc is None else acc
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("material", "hub_material", "spinner_material") and isinstance(v, str):
                acc.add(v)
            else:
                collect_material_refs(v, acc)
    elif isinstance(node, list):
        for v in node:
            collect_material_refs(v, acc)
    return acc


# --------------------------------------------------------------------------
# Non-blade components when no template is given
# --------------------------------------------------------------------------
def convert_hub_v1(hub):
    out = {}
    osb = hub.get("outer_shape_bem", hub)
    out["diameter"] = float(osb["diameter"])
    out["cone_angle"] = float(np.rad2deg(osb["cone_angle"]))
    out["cd"] = float(osb.get("drag_coefficient", osb.get("cd", 0.5)))
    epm = hub.get("elastic_properties_mb")
    if epm:
        ep = {}
        if "system_mass" in epm:
            ep["mass"] = float(epm["system_mass"])
        if "system_inertia" in epm:
            I = list(epm["system_inertia"]) + [0.0] * 6
            ep["inertia"] = I[:6]
        ep["location"] = list(epm.get("system_center_mass", [0.0, 0.0, 0.0]))
        out["elastic_properties"] = ep
    return out


def convert_nacelle_v1(nac):
    dt = nac.get("drivetrain", {})
    osb = nac.get("outer_shape_bem", {})
    os_ = {}
    upt = osb.get("uptilt_angle", dt.get("uptilt"))
    if upt is not None:
        os_["uptilt"] = float(np.rad2deg(upt))
    for k_out, keys in (("distance_tt_hub", ("distance_tt_hub",)), ("overhang", ("overhang",)),
                        ("distance_hub_mb", ("distance_hub2mb",)), ("distance_mb_mb", ("distance_mb2mb",))):
        for k in keys:
            v = osb.get(k, dt.get(k))
            if v is not None:
                os_[k_out] = float(v)
    out = {"outer_shape": os_, "gearbox": {}}
    gb = out["gearbox"]
    if "gear_ratio" in dt:
        gb["gear_ratio"] = float(dt["gear_ratio"])
    eff = dt.get("gearbox_efficiency", dt.get("efficiency"))
    if eff is not None:
        gb["efficiency"] = float(eff)
    if "planet_numbers" in dt:
        gb["planet_numbers"] = list(dt["planet_numbers"])
    epm = nac.get("elastic_properties_mb", {})
    yaw = None
    if epm:
        ep = {}
        m = epm.get("above_yaw_mass", epm.get("system_mass"))
        if m is not None:
            ep["mass"] = float(m)
        if "inertia" in epm or "system_inertia" in epm:
            I = list(epm.get("inertia", epm.get("system_inertia"))) + [0.0] * 6
            ep["inertia"] = I[:6]
        ep["location"] = list(epm.get("center_mass", epm.get("system_center_mass", [0, 0, 0])))
        out["elastic_properties"] = ep
        if "yaw_mass" in epm:
            yaw = {"elastic_properties": {"mass": float(epm["yaw_mass"]),
                                          "inertia": [0.0, 0.0, 0.0], "location": [0.0, 0.0, 0.0]}}
    return out, yaw


def convert_tower_v1(tw):
    out = {"outer_shape": copy.deepcopy(tw["outer_shape_bem"]),
           "structure": copy.deepcopy(tw["internal_structure_2d_fem"])}
    out["reference_axis"] = out["outer_shape"].pop("reference_axis")
    out["structure"].pop("reference_axis", None)
    if "drag_coefficient" in out["outer_shape"]:
        out["outer_shape"]["cd"] = out["outer_shape"].pop("drag_coefficient")
    return out


def convert_control_v1(ctrl):
    """Old flat windIO v1 control block (Vin, Vout, maxOmega, tsr, ...) -> v2."""
    rpm = 30.0 / np.pi
    c = {}
    if "minOmega" in ctrl:
        c["min_rotor_speed"] = float(ctrl["minOmega"]) * rpm
    if "maxOmega" in ctrl:
        c["max_rotor_speed"] = float(ctrl["maxOmega"]) * rpm
        c["rated_rotor_speed"] = float(ctrl["maxOmega"]) * rpm
    if "tsr" in ctrl:
        c["optimal_tsr"] = float(ctrl["tsr"])
    if "min_pitch" in ctrl:
        c["min_pitch_limit"] = float(np.rad2deg(ctrl["min_pitch"]))
    if "max_pitch" in ctrl:
        c["max_pitch_limit"] = float(np.rad2deg(ctrl["max_pitch"]))
    if "max_pitch_rate" in ctrl:
        c["max_pitch_rate"] = float(np.rad2deg(ctrl["max_pitch_rate"]))
    if "ps_percent" in ctrl:
        c["peak_thrust_shaving"] = float(ctrl["ps_percent"])
    if "maxTS" in ctrl:
        c["max_allowable_blade_tip_speed"] = float(ctrl["maxTS"])
    if "max_torque_rate" in ctrl:
        c["max_torque_rate"] = float(ctrl["max_torque_rate"])
    return c


# --------------------------------------------------------------------------
# Main conversion
# --------------------------------------------------------------------------
def convert(sonata, template, args):
    t = template
    out = {"windIO_version": str(t.get("windIO_version", "2.0")) if t else "2.0",
           "name": sonata.get("name", "blade")}
    if sonata.get("description"):
        out["comments"] = sonata["description"]

    # ---- airfoils
    print("\n[airfoils]")
    overrides = {}
    for item in args.rthick or []:
        k, v = item.split("=")
        overrides[k.strip()] = float(v)
    af_list_v2 = [convert_airfoil(a, overrides) for a in sonata["airfoils"]]
    af_db = {a["name"]: a for a in af_list_v2}

    blade_v1 = sonata["components"]["blade"]
    os_v1 = blade_v1["outer_shape_bem"]
    st_v1 = blade_v1["internal_structure_2d_fem"]

    # ---- outer shape
    print("\n[blade outer shape]")
    ref_axis = copy.deepcopy(os_v1["reference_axis"])
    outer = convert_outer_shape(os_v1, af_db)
    used = sorted(set(a["name"] for a in outer["airfoils"]))
    if not args.keep_unused_airfoils:
        dropped = [a["name"] for a in af_list_v2 if a["name"] not in used]
        af_list_v2 = [a for a in af_list_v2 if a["name"] in used]
        if dropped:
            LOG.note(f"unused airfoils dropped: {', '.join(dropped)} (use --keep-unused-airfoils to keep)")

    lofter = Lofter(af_db, outer["airfoils"], outer["rthick"], outer["chord"],
                    os_v1["pitch_axis"])

    # ---- structure
    print(f"\n[blade structure, arc mode = {args.arc_mode}]")
    structure, usage, mass_report = convert_structure(
        st_v1, lofter, ref_axis["z"], args.arc_mode, not args.keep_fiber_units,
        tuple(args.te_split_names), sonata["materials"])
    if not args.keep_fiber_units:
        nz = [L["name"] for L in st_v1["layers"]
              if "fiber_orientation" in L and np.any(np.abs(gv(L["fiber_orientation"])[1]) > 0)]
        if nz:
            LOG.note("fiber_orientation converted rad -> deg (windIO v2 uses degrees). "
                     f"Non-zero in: {', '.join(nz)}. Use --keep-fiber-units if your "
                     "SONATA values were already degrees.")
    if t:
        tbs = t.get("components", {}).get("blade", {}).get("structure", {})
        if "root" in tbs:
            structure["root"] = copy.deepcopy(tbs["root"])
            LOG.note("blade root fastener data (structure.root) copied from template")
        if "joint" in tbs:
            LOG.note("template blade has a spanwise joint; not copied (the SONATA layup has no joint layers)")

    blade = {"reference_axis": ref_axis, "outer_shape": outer, "structure": structure}

    # ---- other components
    print("\n[turbine-level data]")
    comps = {"blade": blade}
    if t:
        for k, v in t["components"].items():
            if k != "blade":
                comps[k] = copy.deepcopy(v)
        assembly = copy.deepcopy(t["assembly"])
        control = copy.deepcopy(t.get("control", {}))
        LOG.note("hub, drivetrain, tower, yaw, RNA, control and assembly taken from template")
        # Show assembly differences
        a_s = sonata.get("assembly", {})
        for k in ("number_of_blades", "turbine_class", "turbulence_class", "rotor_orientation"):
            if k in a_s and k in assembly and a_s[k] != assembly[k]:
                LOG.note(f"assembly.{k}: SONATA = {a_s[k]}, template = {assembly[k]} (template kept)")
    else:
        LOG.warn("no --template given: hub, nacelle, tower and control converted from the "
                 "SONATA file. Check them, they are often leftovers from a reference turbine.")
        comps["hub"] = convert_hub_v1(sonata["components"]["hub"])
        dt, yaw = convert_nacelle_v1(sonata["components"]["nacelle"])
        comps["drivetrain"] = dt
        if yaw:
            comps["yaw"] = yaw
        comps["tower"] = convert_tower_v1(sonata["components"]["tower"])
        a_s = sonata.get("assembly", {})
        assembly = {k: a_s[k] for k in ("turbine_class", "turbulence_class", "drivetrain",
                                        "rotor_orientation", "number_of_blades") if k in a_s}
        c_s = sonata.get("control", {})
        if "rated_power" in c_s:
            assembly["rated_power"] = float(c_s["rated_power"])
        if "Vin" in c_s:
            assembly["cut_in_wind_speed"] = float(c_s["Vin"])
        if "Vout" in c_s:
            assembly["cut_out_wind_speed"] = float(c_s["Vout"])
        tz = comps["tower"]["reference_axis"]["z"]["values"][-1]
        assembly["hub_height"] = float(tz) + float(dt["outer_shape"].get("distance_tt_hub", 0.0))
        control = convert_control_v1(c_s)
        if "foundation" in sonata["components"]:
            LOG.note("components.foundation dropped (not part of the windIO v2 turbine file)")

    # Rotor diameter: WISDEM rescales the blade to match assembly.rotor_diameter
    hub_d = float(comps.get("hub", {}).get("diameter", 0.0))
    blade_len = float(ref_axis["z"]["values"][-1])
    d_new = 2.0 * (0.5 * hub_d + blade_len) if args.rotor_diameter is None else args.rotor_diameter
    d_old = assembly.get("rotor_diameter")
    assembly["rotor_diameter"] = float(d_new)
    if d_old is not None and abs(d_old - d_new) > 1e-3:
        LOG.warn(f"assembly.rotor_diameter changed {d_old:.3f} -> {d_new:.3f} m to match the "
                 f"SONATA blade ({blade_len:.3f} m) and hub diameter ({hub_d:.2f} m). WISDEM "
                 "scales the blade to this diameter, so a wrong value silently resizes the blade.")
    else:
        LOG.note(f"assembly.rotor_diameter = {d_new:.3f} m")
    if hub_d > 0.5 * blade_len:
        LOG.warn(f"hub diameter {hub_d:.2f} m is large for a {blade_len:.2f} m blade; "
                 "check it is not a reference-turbine leftover.")

    out["assembly"] = assembly
    out["components"] = comps
    out["airfoils"] = af_list_v2

    # ---- materials
    print("\n[materials]")
    mats = convert_materials(sonata["materials"], usage, t.get("materials") if t else None)
    have = {m["name"] for m in mats}
    needed = collect_material_refs(comps)
    tmpl_mats = {m["name"]: m for m in (t.get("materials", []) if t else [])}
    for nm in sorted(needed - have):
        if nm in tmpl_mats:
            mats.append(copy.deepcopy(tmpl_mats[nm]))
            LOG.note(f"material '{nm}' (used by non-blade components) copied from template")
        elif nm == "steel":
            mats.append(copy.deepcopy(STEEL_DEFAULT))
            LOG.warn("material 'steel' is used by the tower but not defined; a default "
                     "structural steel (rho 7800, E 200 GPa) was added")
        else:
            LOG.warn(f"material '{nm}' is referenced but not defined anywhere")
    if not any(m["name"].lower() == "resin" for m in mats):
        mats.append(copy.deepcopy(tmpl_mats.get("resin", RESIN_DEFAULT)))
        LOG.note("material 'resin' added (WISDEM's blade cost model uses it)")
    out["materials"] = mats
    out["control"] = control
    return out, mass_report


def modeling_snippet(sonata, out, args):
    env = sonata.get("environment", {})
    costs = sonata.get("costs", {})
    rho_w = float(env.get("water_density", 1025.0))
    mu_w = float(env.get("water_dyn_viscosity", 1.3351e-3))
    names = [L["name"] for L in out["components"]["blade"]["structure"]["layers"]]

    def pick(*cands):
        for c in cands:
            if c in names:
                return c
        return "none"

    snippet = {
        "WISDEM": {
            "RotorSE": {
                "spar_cap_ss": pick("Spar_cap_ss", "spar_cap_ss"),
                "spar_cap_ps": pick("Spar_cap_ps", "spar_cap_ps"),
                "te_ss": pick(args.te_split_names[0]),
                "te_ps": pick(args.te_split_names[1]),
            },
            "LCOE": {k: costs[k] for k in ("wake_loss_factor", "fixed_charge_rate", "bos_per_kW",
                                           "opex_per_kW", "turbine_number") if k in costs},
        },
        "Environment": {
            "air_density": rho_w if args.tidal else float(env.get("air_density", 1.225)),
            "air_dyn_viscosity": mu_w if args.tidal else float(env.get("air_dyn_viscosity", 1.81e-5)),
            "water_density": rho_w,
            "water_dyn_viscosity": mu_w,
            "weib_shape_parameter": float(env.get("weib_shape_parameter", 2.0)),
            "shear_exp": float(env.get("shear_exp", 0.2)),
        },
    }
    if not snippet["WISDEM"]["LCOE"]:
        snippet["WISDEM"].pop("LCOE")
    header = (
        "# Merge these entries into your WISDEM modeling options file.\n"
        "# Since WISDEM 4.0 / windIO 2.0, Environment and costs live here,\n"
        "# not in the turbine file.\n#\n"
        "# TIDAL: CCBlade takes the fluid properties from Environment.air_density\n"
        "# and air_dyn_viscosity, so they are set to seawater values below.\n"
        "# WISDEM's modeling schema limits air_density to <= 1.5 and\n"
        "# air_dyn_viscosity to <= 2e-5, so validation will reject these values\n"
        "# until you raise those two maxima in wisdem/inputs/modeling_schema.yaml\n"
        "# (or keep whatever approach you used for your previous tidal runs).\n"
        "# Re-run the converter with --no-tidal to write air values instead.\n\n"
    )
    return snippet, header


def main(argv=None):
    p = argparse.ArgumentParser(description="SONATA (windIO v1) blade yaml -> WISDEM windIO v2 turbine yaml")
    p.add_argument("sonata", help="SONATA / windIO v1 yaml")
    p.add_argument("-o", "--output", required=True, help="output WISDEM turbine yaml")
    p.add_argument("-t", "--template", help="existing WISDEM windIO v2 file; non-blade data are taken from it")
    p.add_argument("--arc-mode", choices=["explicit", "parametric"], default="explicit")
    p.add_argument("--modeling-snippet", help="where to write the modeling-options snippet "
                   "(default: <output>_modeling_snippet.yaml)")
    p.add_argument("--rotor-diameter", type=float, help="override assembly.rotor_diameter [m]")
    p.add_argument("--te-split-names", nargs=2, default=["TE_reinf_ss", "TE_reinf_ps"],
                   metavar=("SS", "PS"), help="names for the halves of a layer that wraps through the TE")
    p.add_argument("--keep-fiber-units", action="store_true",
                   help="do not convert fiber_orientation from rad to deg")
    p.add_argument("--keep-unused-airfoils", action="store_true")
    p.add_argument("--rthick", action="append", metavar="NAME=VALUE",
                   help="force an airfoil relative thickness, e.g. --rthick NACA4418=0.18 (repeatable)")
    p.add_argument("--no-tidal", dest="tidal", action="store_false",
                   help="write air (not seawater) properties in the modeling snippet")
    args = p.parse_args(argv)

    print(f"Reading {args.sonata}")
    sonata = load_yaml(args.sonata)
    template = load_yaml(args.template) if args.template else None
    if template:
        print(f"Template {args.template} (windIO {template.get('windIO_version', '?')})")

    out, mass_report = convert(sonata, template, args)

    header = (f"# Generated by sonata2wisdem.py on {date.today().isoformat()}\n"
              f"# source: {args.sonata}\n"
              f"# template: {args.template or 'none'}\n"
              f"# arc mode: {args.arc_mode}\n#\n")
    if LOG.warnings:
        header += "# Conversion warnings:\n" + "".join(
            "#  - " + w.replace("\n", " ") + "\n" for w in LOG.warnings) + "#\n"
    write_yaml(out, args.output, header + "\n")
    print(f"\nWrote turbine file:   {args.output}")

    snip_path = args.modeling_snippet or re.sub(r"\.ya?ml$", "", args.output) + "_modeling_snippet.yaml"
    snippet, snip_header = modeling_snippet(sonata, out, args)
    write_yaml(snippet, snip_path, snip_header)
    print(f"Wrote modeling snippet: {snip_path}")
    print(f"\n{len(LOG.warnings)} warning(s), {len(LOG.notes)} note(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
