"""Tests for sonata2rotorse.py that do not need SONATA.

Run with:  python wisdem/sonata/test_sonata2rotorse.py
or, once WISDEM is built (pytest imports wisdem/__init__.py first):
           pytest wisdem/sonata/test_sonata2rotorse.py
"""

import csv
import importlib.util
import inspect
import os
import pathlib
import sys
import tempfile
import traceback

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import sonata2rotorse as s2r  # noqa: E402


def _load_precomp_to_beamdyn():
    # Loaded by path: importing the wisdem package would pull in the whole glue code
    path = os.path.join(HERE, os.pardir, "precomp", "precomp_to_beamdyn.py")
    spec = importlib.util.spec_from_file_location("precomp_to_beamdyn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p2b = _load_precomp_to_beamdyn()


def _random_section(rng):
    edge = rng.uniform(5.0, 50.0)
    return {
        "EA": rng.uniform(1e9, 5e9), "EIxx": rng.uniform(1e8, 1e9), "EIyy": rng.uniform(1e7, 1e8),
        "EIxy": rng.uniform(-1e6, 1e6), "EA_EIxx": 0.0, "EA_EIyy": 0.0,  # zero about the tension centre
        "EIxx_GJ": rng.uniform(-1e5, 1e5), "EIyy_GJ": rng.uniform(-1e5, 1e5),
        "EA_GJ": rng.uniform(-1e5, 1e5), "GJ": rng.uniform(1e6, 1e7),
        "rhoA": rng.uniform(50.0, 500.0), "edge_iner": edge, "flap_iner": edge * rng.uniform(0.05, 0.5),
        "x_tc": rng.uniform(-0.05, 0.05), "y_tc": rng.uniform(-0.2, 0.2),
        "x_cg": rng.uniform(-0.05, 0.05), "y_cg": rng.uniform(-0.2, 0.2),
        "theta": rng.uniform(-5.0, 20.0), "Tw_iner": rng.uniform(-10.0, 25.0),
    }


def _KI(p):
    rhoJ = p["edge_iner"] + p["flap_iner"]
    K = p2b.pc2bd_K(p["EA"], p["EIxx"], p["EIyy"], p["EIxy"], p["EA_EIxx"], p["EA_EIyy"],
                    p["EIxx_GJ"], p["EIyy_GJ"], p["EA_GJ"], p["GJ"], rhoJ, p["edge_iner"],
                    p["flap_iner"], p["x_tc"], p["y_tc"])
    I = p2b.pc2bd_I(p["rhoA"], p["edge_iner"], p["flap_iner"], rhoJ, p["x_cg"], p["y_cg"],
                    np.deg2rad(p["Tw_iner"]), np.deg2rad(p["theta"]))
    return K, I


def test_elastic_from_KI_inverts_pc2bd():
    rng = np.random.default_rng(0)
    secs = [_random_section(rng) for _ in range(20)]
    KI = [_KI(p) for p in secs]
    K = np.array([k for k, _ in KI])
    I = np.array([i for _, i in KI])
    el = s2r.elastic_from_KI(K, I, np.array([p["theta"] for p in secs]))
    for key in ["EA", "EIxx", "EIyy", "EIxy", "GJ", "EA_GJ", "EIxx_GJ", "EIyy_GJ", "rhoA",
                "edge_iner", "flap_iner", "x_tc", "y_tc", "x_cg", "y_cg", "Tw_iner"]:
        ref = np.array([p[key] for p in secs])
        np.testing.assert_allclose(el[key], ref, rtol=1e-9, atol=1e-9 * np.max(np.abs(ref)), err_msg=key)
    np.testing.assert_allclose(el["EA_EIxx"], 0.0, atol=1e-3)
    np.testing.assert_allclose(el["EA_EIyy"], 0.0, atol=1e-3)
    np.testing.assert_allclose(el["rhoJ"], [p["edge_iner"] + p["flap_iner"] for p in secs], rtol=1e-9)


def test_sonata_to_beamdyn_index_mapping():
    # SONATA order [F1 axial, F2 (to LE), F3 (up), M1 torsion, M2 flap, M3 edge]
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Kbd = s2r.sonata_to_beamdyn(np.diag(d))
    # BeamDyn [Fx (=F3), Fy (=-F2), Fz (=F1), Mx (=M3, edge), My (=-M2, flap), Mz (=M1, torsion)]
    np.testing.assert_allclose(np.diag(Kbd), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0])
    # Sign flip on y: an F1-M2 coupling becomes a negative Fz-My coupling
    K = np.eye(6)
    K[0, 4] = K[4, 0] = 0.3
    np.testing.assert_allclose(s2r.sonata_to_beamdyn(K)[2, 4], -0.3)


def test_read_matrix_csv_exporter_format(tmp_path):
    r = [0.0, 0.5]
    mats = np.arange(72, dtype=float).reshape(2, 6, 6) * 1.5e3
    path = tmp_path / "x_anbax_beam_properties_stiff_matrices.csv"
    with open(path, "w", newline="") as f:  # same calls as anbax_export_beam_struct_properties
        w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(2):
            w.writerow(" ")
            w.writerow(["section in r/R", str(r[i])])
            for j in range(6):
                w.writerow(mats[i, j, :])
    s, M = s2r.read_matrix_csv(str(path))
    np.testing.assert_allclose(s, r)
    np.testing.assert_allclose(M, mats)


def _box_mesh(h=0.2, w=0.5, t=0.02, n=40):
    """Thin box section: caps (carbon) at x = +-h over |y| <= w/2, skin
    (glass) elsewhere, plus an aft TE strip (uni) at y in [0.8, 1.0]. The TE
    is thin and drooped so that both of its surfaces lie below the tension
    centre (x_tc = 0.01), as on a cambered airfoil."""
    xs, ys, mats = [], [], []
    for side in (+1, -1):
        for yy in np.linspace(-0.6, 1.0, n):
            te = yy >= 0.8
            outer = side * 0.02 - 0.05 if te else side * (h + t / 2)
            inner = side * 0.01 - 0.05 if te else side * (h - t / 2)
            xs.append(outer)
            ys.append(yy)
            mats.append("glass")
            if abs(yy) <= w / 2:
                xs.append(inner)
                ys.append(yy)
                mats.append("carbon")
            if te:
                xs.append(inner)
                ys.append(yy)
                mats.append("uni")
            if yy <= -0.5:  # LE strip, same material as the TE strip
                xs.append(inner)
                ys.append(yy)
                mats.append("uni")
    x, y = np.array(xs), np.array(ys)
    return {"area": np.full(len(x), 1e-3), "x": x, "y": y, "mat": np.array(mats)}


def _box_result():
    mesh = _box_mesh()
    rho = {"glass": 1900.0, "carbon": 1600.0, "uni": 2000.0, "foam": 100.0}
    dens = np.array([rho[m] for m in mesh["mat"]])
    m = np.sum(dens * mesh["area"])
    x_cg = np.sum(dens * mesh["area"] * mesh["x"]) / m
    y_cg = np.sum(dens * mesh["area"] * mesh["y"]) / m
    K = p2b.pc2bd_K(1e9, 1e8, 1e7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e6, 3.0, 2.0, 1.0, 0.01, 0.05)
    I = p2b.pc2bd_I(m, 2.0, 1.0, 3.0, x_cg, y_cg, 0.0, 0.0)
    yml = {
        "components": {"blade": {
            "outer_shape_bem": {
                "twist": {"grid": [0.0, 1.0], "values": [0.0, 0.0]},
                "chord": {"grid": [0.0, 1.0], "values": [2.0, 2.0]},
                "reference_axis": {"z": {"grid": [0.0, 1.0], "values": [0.0, 10.0]}},
            },
            "internal_structure_2d_fem": {
                "layers": [{"name": "skin", "material": "glass"},
                           {"name": "Spar_cap_ss", "material": "carbon"},
                           {"name": "Spar_cap_ps", "material": "carbon"},
                           {"name": "TE_reinforcement", "material": "uni"},
                           {"name": "LE_reinf", "material": "uni"}],
                "cavity_filler": {"material": "foam"},
            },
        }},
        "materials": [{"name": k, "rho": v} for k, v in rho.items()],
    }
    r = np.array([0.0, 1.0])
    return s2r.SonataResult(r, np.array([K, K]), np.array([I, I]), [mesh, mesh], yml)


def test_strain_points_and_mats_from_mesh():
    props = s2r.extract(_box_result())
    x_tc, y_tc = 0.01, 0.05
    np.testing.assert_allclose(props["x_tc"], x_tc)
    np.testing.assert_allclose(props["y_tc"], y_tc)
    # Caps: outermost cap element at x = +-(h - t/2), centred at y = 0
    np.testing.assert_allclose(props["xu_spar"], 0.19 - x_tc)
    np.testing.assert_allclose(props["xl_spar"], -0.19 - x_tc)
    np.testing.assert_allclose(props["yu_spar"], 0.0 - y_tc, atol=0.02)
    # TE: only the aft "uni" strip counts, not the LE one
    np.testing.assert_allclose(props["yu_te"], 0.9 - y_tc, atol=0.02)
    np.testing.assert_allclose(props["yl_te"], 0.9 - y_tc, atol=0.02)
    # Drooped TE: both points are below the tension centre, split by the mid-line
    np.testing.assert_allclose(props["xu_te"], -0.04 - x_tc)
    np.testing.assert_allclose(props["xl_te"], -0.06 - x_tc)
    np.testing.assert_allclose(props["A"], _box_mesh()["area"].sum())
    names = list(props["mat_names"])
    sc = props["sc_ss_mats"][0]
    assert sc[names.index("carbon")] == 1 and sc[names.index("glass")] == 1
    assert sc[names.index("foam")] == 0
    np.testing.assert_allclose(props["blade_mass"], props["rhoA"][0] * 10.0)


def test_interpolation_keeps_constant_section():
    props = s2r.extract(_box_result(), s_new=np.linspace(0.0, 1.0, 5))
    assert props["K"].shape == (5, 6, 6)
    np.testing.assert_allclose(props["EA"], 1e9)
    np.testing.assert_allclose(props["xu_spar"], 0.19 - 0.01)
    assert props["sc_ps_mats"].shape[0] == 5


def test_write_outputs(tmp_path):
    props = s2r.extract(_box_result())
    prefix = str(tmp_path / "box")
    s2r.write_outputs(props, prefix)
    d = np.load(prefix + ".npz")
    np.testing.assert_allclose(d["EA"], props["EA"])
    blk = s2r.load_yaml(prefix + "_elastic_properties.yaml")["elastic_properties"]
    np.testing.assert_allclose(blk["stiffness_matrix"]["K33"], props["K"][:, 2, 2])
    np.testing.assert_allclose(blk["inertia_matrix"]["cm_y"], props["y_cg"])


def _run_all():
    failed = 0
    for name, fn in list(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            with tempfile.TemporaryDirectory() as d:
                kwargs = {"tmp_path": pathlib.Path(d)} if "tmp_path" in inspect.signature(fn).parameters else {}
                fn(**kwargs)
            print(f"PASSED  {name}")
        except Exception:
            failed += 1
            print(f"FAILED  {name}")
            traceback.print_exc()
    print(f"\n{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
