"""Lapis 3-4 — uji model dan perilaku."""
import json, joblib, numpy as np, pandas as pd, pytest, yaml
from src import data as D, features as F

CFG = yaml.safe_load(open("conf/params.yaml", encoding="utf-8"))
GATES = yaml.safe_load(open("conf/gates.yaml", encoding="utf-8"))


@pytest.fixture(scope="module")
def bundel():
    model = joblib.load("artifacts/model.joblib")
    art = F.ArtefakFitur.muat("artifacts/artefak_fitur.json")
    meta = json.load(open("artifacts/metadata.json", encoding="utf-8"))
    return model, art, meta


def test_gerbang_auc(bundel):
    _, _, meta = bundel
    g = GATES[GATES["tingkat_risiko"]]
    assert meta["metrik_uji_oot"]["auc"] >= g["auc_min"]


def test_gerbang_segmen(bundel):
    _, _, meta = bundel
    g = GATES[GATES["tingkat_risiko"]]
    for r in meta["metrik_segmen"]:
        if r["n"] >= g["segmen_min_n"] and r["auc"] == r["auc"]:
            assert r["auc"] >= g["auc_min_per_segmen"], f"segmen {r['segmen']} gagal"


def test_tidak_ada_fitur_terlarang(bundel):
    _, _, meta = bundel
    g = GATES[GATES["tingkat_risiko"]]
    for larangan in g["fitur_terlarang"]:
        assert not any(larangan in k for k in meta["kolom_fitur"])


def test_arah_pengaruh_tunggakan(bundel):
    """Menaikkan jumlah tunggakan tidak boleh MENURUNKAN skor risiko rata-rata."""
    model, art, _ = bundel
    df = D.muat(CFG["data"]["mentah"]).sample(400, random_state=0)
    dasar = df.copy(); dasar["jumlah_tunggakan_12b"] = 0
    naik = df.copy(); naik["jumlah_tunggakan_12b"] = 6
    s0 = model.predict_proba(F.terapkan(dasar, art))[:, 1].mean()
    s1 = model.predict_proba(F.terapkan(naik, art))[:, 1].mean()
    assert s1 >= s0, f"arah terbalik: {s0:.4f} -> {s1:.4f}"
