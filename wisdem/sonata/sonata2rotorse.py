#!/usr/bin/env python3
"""
sonata2rotorse.py
=================

Turn the results of a SONATA (ANBA4) run into the cross-sectional quantities
that WISDEM's RotorSE structural chain consumes, i.e. the output set of
RunPreComp (wisdem/rotorse/rotor_elasticity.py), in WISDEM's conventions:

  * K, I            6x6 stiffness / inertia at the reference (pitch) axis, in
                    the generate_KI convention (BeamDyn axes, chord aligned)
  * EA, EIxx, EIyy, EIxy, GJ and the coupling terms, about the tension centre
  * rhoA, rhoJ, flap_iner, edge_iner, Tw_iner, x/y_tc, x/y_sc, x/y_cg
  * A               exact material area from the SONATA mesh
  * xu/xl/yu/yl_spar, xu/xl/yu/yl_te
                    strain points for ComputeStrains, taken from the SONATA
                    mesh (KI_to_Elastic leaves these at zero)
  * sc_ss/ps_mats, te_ss/ps_mats
                    material flags of the laminate at those points
  * blade mass, spanwise centre of mass, root moment of inertia

The K/I -> elastic conversion is a port of KI_to_Elastic.compute, so the
numbers are what WISDEM itself would derive from the same matrices. It is
ported rather than imported because importing the wisdem package pulls in
the whole glue code, which is not available in the SONATA environment.

Usage
-----
In memory, at the end of a SONATA run (see SONATA_working.py):

    from sonata2rotorse import from_job, extract, write_outputs
    write_outputs(extract(from_job(job)), run_dir + "Q3v5_rotorse")

From the files SONATA wrote (csv_export/ and stress-map/):

    python sonata2rotorse.py Q3v5_web.yaml -o Q3v5_rotorse --n-span 30

Conventions
-----------
SONATA works in [x1 axial, x2 towards the LE, x3 towards the suction side].
BeamDyn / WISDEM K and I use [x flapwise towards the suction side,
y chordwise towards the TE, z axial], so x = x3, y = -x2, z = x1. This is the
same rotation beam_struct_eval applies before writing csv_export/ and the
BeamDyn files. Both frames are chord aligned (rotated by the twist) as long as
SONATA ran with flag_output_zero_twist = False.

The strain points follow PreComp: x is the flapwise and y the chordwise
distance of the point from the tension centre, in metres.

Requirements: Python >= 3.8, numpy, scipy, pyyaml.
"""

import argparse
import csv
import glob
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from sonata2wisdem import Log, gv, load_yaml, pchip_or_linear, write_yaml  # noqa: E402

LOG = Log()

_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------
# Same matrix as utl/beam_struct_eval.py (SONATA/VABS -> BeamDyn)
T_SONATA_TO_BD = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def sonata_to_beamdyn(M):
    """Rotate a 6x6 stiffness or mass matrix from the SONATA/VABS frame to the
    BeamDyn frame (same operation as SONATA.cbm.cbm_utl.trsf_sixbysix)."""
    T = np.linalg.inv(T_SONATA_TO_BD)
    R = np.zeros((6, 6))
    R[:3, :3] = T
    R[3:, 3:] = T
    return R.T @ np.asarray(M, float) @ R


def translation_matrix(x, y):
    """As TransformCrossSectionMatrix.CrossSectionTranslationMatrix
    (wisdem/precomp/precomp_to_beamdyn.py)."""
    T = np.eye(6)
    T[0, 5] = y
    T[1, 5] = -x
    T[2, 3] = -y
    T[2, 4] = x
    return T


def rotation_matrix(alpha):
    """As TransformCrossSectionMatrix.CrossSectionRotationMatrix."""
    c, s = np.cos(alpha), np.sin(alpha)
    R1 = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    R = np.zeros((6, 6))
    R[:3, :3] = R1
    R[3:, 3:] = R1
    return R


# --------------------------------------------------------------------------
# SONATA results
# --------------------------------------------------------------------------
@dataclass
class SonataResult:
    """SONATA results at the analysed stations, already in BeamDyn axes.

    r       station positions [r/R]
    K, I    (n, 6, 6) stiffness and mass matrices at the reference axis
    meshes  per station dict(area, x, y, mat) of the mesh elements (x flapwise,
            y chordwise to the TE, mat = material name), or None if unknown
    yml     the SONATA input yaml (windIO v1 layout)
    """

    r: np.ndarray
    K: np.ndarray
    I: np.ndarray
    meshes: list
    yml: dict


def _mesh_dict(area, cxy, mat):
    cxy = np.asarray(cxy, float)
    return {"area": np.asarray(area, float), "x": cxy[:, 1], "y": -cxy[:, 0],
            "mat": np.asarray(mat, dtype=str)}


def from_job(job, yml=None):
    """Collect the results of an in-memory SONATA Blade job (after
    beam_struct_eval / blade_run_anbax)."""
    bp = job.beam_properties
    r = np.array([float(row[0]) for row in bp])
    K = np.array([sonata_to_beamdyn(row[1].TS) for row in bp])
    I = np.array([sonata_to_beamdyn(row[1].MM) for row in bp])

    sections = {round(float(x), 10): cs for x, cs in job.sections}
    meshes = []
    for x in r:
        cs = sections.get(round(float(x), 10))
        if cs is None or not getattr(cs, "mesh", None):
            meshes.append(None)
            continue
        cells = cs.mesh
        meshes.append(_mesh_dict([c.area for c in cells],
                                 [np.asarray(c.center, float)[:2] for c in cells],
                                 [cs.materials[c.MatID].name for c in cells]))
    return SonataResult(r, K, I, meshes, yml if yml is not None else job.yml)


def _find_csv(csv_dir, stem, suffix):
    path = os.path.join(csv_dir, f"{stem}_anbax_beam_properties_{suffix}.csv")
    if os.path.isfile(path):
        return path
    hits = glob.glob(os.path.join(csv_dir, f"*_anbax_beam_properties_{suffix}.csv"))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"cannot find {path}" + (f" (candidates: {hits})" if hits else ""))


def read_matrix_csv(path):
    """Read a SONATA csv_export stiffness or mass file: per station a
    'section in r/R, <r>' row followed by the 6 rows of the matrix."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.reader(f) if any(c.strip() for c in row)]
    stations, mats = [], []
    i = 0
    while i < len(rows):
        if rows[i][0].strip() == "section in r/R":
            stations.append(float(rows[i][1]))
            mats.append(np.array([[float(v) for v in rows[i + 1 + j]] for j in range(6)]))
            i += 7
        else:
            i += 1
    return np.array(stations), np.array(mats)


def read_coordsys(path):
    with open(path, newline="", encoding="utf-8") as f:
        first = next(csv.reader(f))
    return first[1].strip() if len(first) > 1 and first[0].startswith("Coordinate system") else ""


def read_stress_maps(stress_map_dir, r):
    """Mesh data of each station from SONATA's stress-strain map files."""
    meshes = [None] * len(r)
    for path in sorted(glob.glob(os.path.join(stress_map_dir, "blade_station*_stress_strain_map.npz"))):
        with np.load(path) as d:
            pos = float(np.ravel(d["station_pos"])[0])
            j = np.flatnonzero(np.isclose(r, pos, atol=1e-6))
            if j.size == 0:
                LOG.warn(f"{os.path.basename(path)}: station r/R={pos} is not in the csv files, ignored")
                continue
            names = d["material_names"].astype(str)
            mat = names[d["elem_materials"].astype(int)]
            meshes[j[0]] = _mesh_dict(d["elem_areas"], d["elem_cxy"], mat)
    return meshes


def from_files(yaml_path, csv_dir=None, stress_map_dir=None):
    """Collect SONATA results from the files written by beam_struct_eval
    (csv_export/) and blade_exp_stress_strain_map (stress-map/)."""
    yml = load_yaml(yaml_path)
    run_dir = os.path.dirname(os.path.abspath(yaml_path))
    csv_dir = csv_dir or os.path.join(run_dir, "csv_export")
    stem = os.path.splitext(os.path.basename(yaml_path))[0]

    r, K = read_matrix_csv(_find_csv(csv_dir, stem, "stiff_matrices"))
    r_m, I = read_matrix_csv(_find_csv(csv_dir, stem, "mass_matrices"))
    if r.shape != r_m.shape or not np.allclose(r, r_m):
        raise ValueError("stiffness and mass csv files have different stations")

    coordsys = read_coordsys(_find_csv(csv_dir, stem, "general"))
    if "beamdyn" not in coordsys.lower():
        LOG.note(f"csv matrices are in '{coordsys or 'unknown'}' coordinates, rotated to BeamDyn")
        K = np.array([sonata_to_beamdyn(k) for k in K])
        I = np.array([sonata_to_beamdyn(m) for m in I])

    if stress_map_dir is None and os.path.isdir(os.path.join(run_dir, "stress-map")):
        stress_map_dir = os.path.join(run_dir, "stress-map")
    if stress_map_dir:
        meshes = read_stress_maps(stress_map_dir, r)
    else:
        LOG.warn("no stress-map folder: A falls back to an estimate and the strain points "
                 "and *_mats stay zero")
        meshes = [None] * len(r)
    return SonataResult(r, K, I, meshes, yml)


# --------------------------------------------------------------------------
# K, I -> elastic properties (port of KI_to_Elastic.compute)
# --------------------------------------------------------------------------
ELASTIC_KEYS = ["EA", "EIxx", "EIyy", "EIxy", "GJ", "EA_EIxx", "EA_EIyy", "EA_GJ",
                "EIxx_GJ", "EIyy_GJ", "rhoA", "rhoJ", "Tw_iner", "x_tc", "y_tc",
                "x_sc", "y_sc", "x_cg", "y_cg", "flap_iner", "edge_iner", "A_est"]


def elastic_from_KI(K, I, theta):
    """PreComp-style properties from K and I at the reference axis.

    theta is the aerodynamic twist [deg]. Follows KI_to_Elastic.compute
    (wisdem/rotorse/rotor_elasticity.py) except for the principal inertia
    angle, which is taken in closed form so that it is the exact inverse of
    pc2bd_I. The principal axes are labelled so that edge_iner >= flap_iner."""
    n = K.shape[0]
    out = {k: np.zeros(n) for k in ELASTIC_KEYS}
    for i in range(n):
        K_ref, I_ref = K[i], I[i]

        # Shear and tension centres (as SONATA's anbax_utl)
        try:
            Y = np.linalg.solve(K_ref[:3, :3], -K_ref[:3, 3:])
            x_sc, y_sc = -Y[1, 2], Y[0, 2]
            x_tc, y_tc = Y[2, 1], -Y[2, 0]
        except np.linalg.LinAlgError:
            LOG.warn(f"station {i}: singular shear block, tension and shear centres set to 0")
            x_sc = y_sc = x_tc = y_tc = 0.0

        T = translation_matrix(x_tc, y_tc)
        K_tc = T.T @ K_ref @ T
        out["EA"][i] = K_tc[2, 2]
        out["EA_EIxx"][i] = K_tc[2, 3]
        out["EA_EIyy"][i] = K_tc[2, 4]
        out["EA_GJ"][i] = K_tc[2, 5]
        out["EIxx"][i] = K_tc[3, 3]
        out["EIyy"][i] = K_tc[4, 4]
        out["EIxy"][i] = K_tc[3, 4]
        out["EIxx_GJ"][i] = K_tc[3, 5]
        out["EIyy_GJ"][i] = K_tc[4, 5]
        out["GJ"][i] = K_tc[5, 5]
        out["x_tc"][i], out["y_tc"][i] = x_tc, y_tc
        out["x_sc"][i], out["y_sc"][i] = x_sc, y_sc

        # Mass matrix at the centre of mass, then to the principal axes
        m = I_ref[0, 0]
        x_cg, y_cg = I_ref[1, 5] / m, -I_ref[0, 5] / m
        T = translation_matrix(x_cg, y_cg)
        I_cg = T.T @ I_ref @ T
        if np.hypot(2.0 * I_cg[3, 4], I_cg[3, 3] - I_cg[4, 4]) > 1e-6 * (I_cg[3, 3] + I_cg[4, 4]):
            angle = 0.5 * np.arctan2(2.0 * I_cg[3, 4], I_cg[3, 3] - I_cg[4, 4])
        else:  # axisymmetric (e.g. circular root): any axes are principal, keep the chord ones
            angle = 0.0
        R = rotation_matrix(-angle)
        I_p = R.T @ I_cg @ R
        out["rhoA"][i] = I_p[0, 0]
        out["rhoJ"][i] = I_p[5, 5]
        out["edge_iner"][i] = I_p[3, 3]
        out["flap_iner"][i] = I_p[4, 4]
        out["Tw_iner"][i] = theta[i] - np.rad2deg(angle)
        out["x_cg"][i], out["y_cg"][i] = x_cg, y_cg

        # KI_to_Elastic's area estimate, only used when there is no mesh
        with np.errstate(divide="ignore", invalid="ignore"):
            a_xx = out["EA"][i] * out["edge_iner"][i] / out["EIxx"][i]
            a_yy = out["EA"][i] * out["flap_iner"][i] / out["EIyy"][i]
        out["A_est"][i] = 0.5 * (a_xx + a_yy) if np.isfinite(a_xx + a_yy) else 0.0
    return out


# --------------------------------------------------------------------------
# Quantities that need the section mesh
# --------------------------------------------------------------------------
POINT_KEYS = ["xu_spar", "xl_spar", "yu_spar", "yl_spar", "xu_te", "xl_te", "yu_te", "yl_te"]
GROUPS = {  # group -> (strain point keys, side, aft of the tension centre only)
    "sc_ss": (("xu_spar", "yu_spar"), +1, False),
    "sc_ps": (("xl_spar", "yl_spar"), -1, False),
    "te_ss": (("xu_te", "yu_te"), +1, True),
    "te_ps": (("xl_te", "yl_te"), -1, True),
}


def _surface_side(x, y, n_bins=100):
    """+1 for elements above the local mid-line of the section (suction side),
    -1 below it. The mid-line is halfway between the highest and lowest
    element in each chordwise bin, so it follows the camber up to the TE."""
    edges = np.linspace(y.min(), y.max(), n_bins + 1)
    idx = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    hi = np.full(n_bins, -np.inf)
    lo = np.full(n_bins, np.inf)
    np.maximum.at(hi, idx, x)
    np.minimum.at(lo, idx, x)
    return np.where(x >= 0.5 * (hi[idx] + lo[idx]), 1, -1)


def _layer_materials(st, names):
    """Materials of the named layers of internal_structure_2d_fem."""
    by_name = {L["name"]: L["material"] for L in st["layers"]}
    mats = set()
    for nm in names:
        if nm in by_name:
            mats.add(by_name[nm])
        else:
            LOG.warn(f"layer '{nm}' not found in the yaml; it is ignored for the strain points")
    return mats


def _shell_only_materials(st):
    """Materials that only appear in webs or the cavity filler (excluded from
    the *_mats laminate flags)."""
    shell = {L["material"] for L in st["layers"] if "web" not in L}
    other = {L["material"] for L in st["layers"] if "web" in L}
    if st.get("cavity_filler"):
        other.add(st["cavity_filler"]["material"])
    return other - shell


def mesh_quantities(res, el, roles, chord):
    """Area, strain points and material flags from the SONATA meshes.

    roles maps sc_ss, sc_ps, te_ss, te_ps to lists of layer names. A group is
    made of the elements of those layers' materials on its side of the
    section mid-line (TE groups: also aft of the tension centre, which
    separates them from LE layers of the same material). Its strain point sits at the
    area-weighted chordwise position of the group, at its outermost element."""
    n = len(res.r)
    yml = res.yml
    st = yml["components"]["blade"]["internal_structure_2d_fem"]
    mat_names = [m["name"] for m in yml["materials"]]
    rho = {m["name"]: float(m.get("rho", 0.0)) for m in yml["materials"]}
    group_mats = {g: _layer_materials(st, roles[g]) for g in GROUPS}
    not_laminate = _shell_only_materials(st)

    out = {"A": np.array(el["A_est"]), "mat_names": np.array(mat_names)}
    out.update({k: np.full(n, np.nan) for k in POINT_KEYS})
    out.update({f"{g}_mats": np.zeros((n, len(mat_names))) for g in GROUPS})

    for i, mesh in enumerate(res.meshes):
        if mesh is None:
            continue
        a, x, y, mat = mesh["area"], mesh["x"], mesh["y"], mesh["mat"]
        out["A"][i] = a.sum()

        # Consistency of the mesh with the mass matrix (frames and signs)
        dens = np.array([rho.get(m, 0.0) for m in mat])
        mass = np.sum(dens * a)
        xm, ym = np.sum(dens * a * x) / mass, np.sum(dens * a * y) / mass
        d = np.hypot(xm - el["x_cg"][i], ym - el["y_cg"][i])
        if d > 0.01 * chord[i]:
            LOG.warn(f"r/R={res.r[i]:.4f}: mesh centre of mass ({xm:.4f}, {ym:.4f}) is {d:.4f} m "
                     f"from the one in the mass matrix ({el['x_cg'][i]:.4f}, {el['y_cg'][i]:.4f})")
        if abs(mass - el["rhoA"][i]) > 0.01 * el["rhoA"][i]:
            LOG.warn(f"r/R={res.r[i]:.4f}: mesh mass {mass:.3f} kg/m vs {el['rhoA'][i]:.3f} kg/m "
                     "in the mass matrix")

        x_tc, y_tc = el["x_tc"][i], el["y_tc"][i]
        surf = _surface_side(x, y)
        for g, ((kx, ky), side, aft) in GROUPS.items():
            sel = np.isin(mat, list(group_mats[g])) & (surf == side)
            if aft:
                sel &= y > y_tc
            if not np.any(sel):
                continue
            yg = np.sum(a[sel] * y[sel]) / np.sum(a[sel])
            y_lo, y_hi = y[sel].min(), y[sel].max()
            band = sel & (np.abs(y - yg) <= 0.25 * (y_hi - y_lo) + 1e-12)
            xg = side * np.max(side * x[band])
            out[kx][i] = xg - x_tc
            out[ky][i] = yg - y_tc
            # Laminate at the point: shell elements on this side within the chordwise extent
            lam = (surf == side) & (y >= y_lo) & (y <= y_hi) & ~np.isin(mat, list(not_laminate))
            present = set(mat[lam])
            out[f"{g}_mats"][i] = [1.0 if m in present else 0.0 for m in mat_names]

    # Stations without the reinforcement (e.g. circular root) or without mesh
    for k in POINT_KEYS:
        v = np.isfinite(out[k])
        if np.all(v):
            continue
        if np.any(v):
            out[k][~v] = np.interp(res.r[~v], res.r[v], out[k][v])
            LOG.warn(f"{k}: no elements found at r/R = {np.round(res.r[~v], 4).tolist()}; "
                     "interpolated from the neighbouring stations")
        else:
            out[k][:] = 0.0
            if any(m is not None for m in res.meshes):
                LOG.warn(f"{k}: no elements found at any station, set to 0 (check the layer names)")
    if all(m is None for m in res.meshes):
        LOG.warn("no mesh data: A is KI_to_Elastic's estimate, strain points and *_mats are 0")
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
DEFAULT_ROLES = {"sc_ss": ["Spar_cap_ss"], "sc_ps": ["Spar_cap_ps"],
                 "te_ss": ["TE_reinforcement"], "te_ps": ["TE_reinforcement"]}


def _blade_axes(yml, s):
    """Twist [deg], chord [m] and reference-axis z [m] at r/R = s."""
    os_ = yml["components"]["blade"]["outer_shape_bem"]
    tg, tv = gv(os_["twist"])
    if np.max(np.abs(tv)) < 2.0 * np.pi:  # windIO v1 twist is in radians
        tv = np.rad2deg(tv)
    theta = np.interp(s, tg, tv)
    chord = np.interp(s, *gv(os_["chord"]))
    z = np.interp(s, *gv(os_["reference_axis"]["z"]))
    return theta, chord, z


def _interp_rows(s_new, s, arr):
    """PCHIP along the first axis of arr, entry by entry."""
    flat = arr.reshape(len(s), -1)
    out = np.column_stack([pchip_or_linear(s_new, s, flat[:, j]) for j in range(flat.shape[1])])
    return out.reshape((len(s_new),) + arr.shape[1:])


def extract(res, roles=None, s_new=None):
    """All RotorSE quantities from a SonataResult.

    s_new: optional r/R grid (e.g. WISDEM's n_span grid). K, I, A and the
    strain points are PCHIP-interpolated onto it (the *_mats flags are taken
    from the nearest station) and the elastic properties are recomputed there."""
    roles = {**DEFAULT_ROLES, **(roles or {})}
    theta, chord, _ = _blade_axes(res.yml, res.r)
    el = elastic_from_KI(res.K, res.I, theta)
    geo = mesh_quantities(res, el, roles, chord)

    s, K, I = res.r, res.K, res.I
    if s_new is not None:
        s_new = np.asarray(s_new, float)
        if s_new.min() < s.min() - 1e-9 or s_new.max() > s.max() + 1e-9:
            LOG.warn(f"interpolation grid [{s_new.min()}, {s_new.max()}] goes beyond the SONATA "
                     f"stations [{s.min()}, {s.max()}]; values are extrapolated")
        K = _interp_rows(s_new, s, K)
        I = _interp_rows(s_new, s, I)
        nearest = np.abs(s_new[:, None] - s[None, :]).argmin(axis=1)
        for k, v in geo.items():
            if k.endswith("_mats"):
                geo[k] = v[nearest]
            elif k != "mat_names":
                geo[k] = pchip_or_linear(s_new, s, v)
        s = s_new
        theta, chord, _ = _blade_axes(res.yml, s)
        el = elastic_from_KI(K, I, theta)

    _, _, z = _blade_axes(res.yml, s)
    props = {"r": s, "z": z, "theta": theta, "chord": chord, "K": K, "I": I}
    props.update({k: v for k, v in el.items() if k != "A_est"})
    props.update(geo)

    mass = _trapz(props["rhoA"], z)
    props["blade_mass"] = mass
    props["blade_span_cg"] = _trapz(props["rhoA"] * z, z) / mass
    props["blade_root_inertia"] = _trapz(props["rhoA"] * (z - z[0]) ** 2, z)
    if len(s) < 15:
        LOG.note(f"blade mass and inertia integrated over only {len(s)} stations; "
                 "use --n-span for a finer grid")
    return props


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
TABLE_KEYS = ["r", "z", "theta", "chord", "EA", "EIxx", "EIyy", "EIxy", "GJ", "EA_EIxx",
              "EA_EIyy", "EA_GJ", "EIxx_GJ", "EIyy_GJ", "rhoA", "rhoJ", "flap_iner",
              "edge_iner", "Tw_iner", "A", "x_tc", "y_tc", "x_sc", "y_sc", "x_cg", "y_cg"] + POINT_KEYS
STIFF_TERMS = [11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 33, 34, 35, 36, 44, 45, 46, 55, 56, 66]


def elastic_properties_block(props):
    """windIO v2 components.blade.structure.elastic_properties.

    The inertia terms are those assemble_KI reads back into I, so
    i_cp = -I[3,4] (gc_LoadInputs writes +I[3,4], which assemble_KI would
    turn into the opposite sign)."""
    K, I = props["K"], props["I"]
    grid = props["r"].tolist()
    stiff = {"grid": grid}
    for t in STIFF_TERMS:
        stiff[f"K{t}"] = K[:, t // 10 - 1, t % 10 - 1].tolist()
    m = I[:, 0, 0]
    inertia = {"grid": grid, "mass": m.tolist(),
               "cm_x": (I[:, 1, 5] / m).tolist(), "cm_y": (-I[:, 0, 5] / m).tolist(),
               "i_edge": I[:, 3, 3].tolist(), "i_flap": I[:, 4, 4].tolist(),
               "i_plr": I[:, 5, 5].tolist(), "i_cp": (-I[:, 3, 4]).tolist()}
    return {"elastic_properties": {"stiffness_matrix": stiff, "inertia_matrix": inertia,
                                   "structural_damping": {"mu": [0.0] * 6}}}


def write_outputs(props, prefix):
    """Write <prefix>.npz (everything), <prefix>.csv (per-station table) and
    <prefix>_elastic_properties.yaml (windIO v2 block)."""
    np.savez(prefix + ".npz", **props)

    with open(prefix + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TABLE_KEYS)
        for i in range(len(props["r"])):
            w.writerow([f"{props[k][i]:.10g}" for k in TABLE_KEYS])

    header = ("# SONATA cross-sectional properties in windIO v2 format.\n"
              "# Paste under components.blade.structure of the WISDEM turbine yaml.\n"
              "# K and I are at the reference axis, BeamDyn axes, chord aligned.\n\n")
    write_yaml(elastic_properties_block(props), prefix + "_elastic_properties.yaml", header)
    print(f"Wrote {prefix}.npz, {prefix}.csv and {prefix}_elastic_properties.yaml")


def print_summary(props):
    print(f"\n{'r/R':>7} {'EA [N]':>11} {'EIflap':>11} {'EIedge':>11} {'GJ':>11} "
          f"{'rhoA':>9} {'xu_spar':>8} {'xl_spar':>8}")
    for i in range(len(props["r"])):
        print(f"{props['r'][i]:7.4f} {props['EA'][i]:11.4e} {props['EIyy'][i]:11.4e} "
              f"{props['EIxx'][i]:11.4e} {props['GJ'][i]:11.4e} {props['rhoA'][i]:9.3f} "
              f"{props['xu_spar'][i]:8.4f} {props['xl_spar'][i]:8.4f}")
    print(f"\nblade mass {props['blade_mass']:.2f} kg, spanwise cg {props['blade_span_cg']:.3f} m")


def main(argv=None):
    p = argparse.ArgumentParser(description="SONATA results -> WISDEM RotorSE cross-sectional quantities")
    p.add_argument("sonata", help="SONATA input yaml used for the run")
    p.add_argument("--csv-dir", help="SONATA csv_export folder (default: <yaml dir>/csv_export)")
    p.add_argument("--stress-map-dir", help="SONATA stress-map folder (default: <yaml dir>/stress-map if present)")
    p.add_argument("-o", "--output", help="output prefix (default: <yaml name>_rotorse)")
    p.add_argument("--n-span", type=int, help="also interpolate onto linspace(0, 1, n_span)")
    p.add_argument("--spar-cap-ss", nargs="+", default=DEFAULT_ROLES["sc_ss"], metavar="LAYER")
    p.add_argument("--spar-cap-ps", nargs="+", default=DEFAULT_ROLES["sc_ps"], metavar="LAYER")
    p.add_argument("--te-ss", nargs="+", default=DEFAULT_ROLES["te_ss"], metavar="LAYER")
    p.add_argument("--te-ps", nargs="+", default=DEFAULT_ROLES["te_ps"], metavar="LAYER")
    args = p.parse_args(argv)

    res = from_files(args.sonata, args.csv_dir, args.stress_map_dir)
    roles = {"sc_ss": args.spar_cap_ss, "sc_ps": args.spar_cap_ps,
             "te_ss": args.te_ss, "te_ps": args.te_ps}
    s_new = np.linspace(0.0, 1.0, args.n_span) if args.n_span else None
    props = extract(res, roles, s_new)
    print_summary(props)
    write_outputs(props, args.output or os.path.splitext(args.sonata)[0] + "_rotorse")
    print(f"\n{len(LOG.warnings)} warning(s), {len(LOG.notes)} note(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
