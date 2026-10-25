"""evaluate.py — metrik, analisis segmen, biaya, dan gerbang mutu."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def ks(y, skor) -> float:
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(skor)}).sort_values("s")
    pos = (d.y == 1).cumsum() / max((d.y == 1).sum(), 1)
    neg = (d.y == 0).cumsum() / max((d.y == 0).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


def gini(y, skor) -> float:
    return 2 * roc_auc_score(y, skor) - 1


def biaya_total(y, skor, ambang: float, biaya_fn: float, biaya_fp: float) -> float:
    pred = (np.asarray(skor) >= ambang).astype(int)
    y = np.asarray(y)
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    return fn * biaya_fn + fp * biaya_fp


def ambang_optimal(y, skor, biaya_fn: float, biaya_fp: float) -> tuple[float, float]:
    kandidat = np.arange(0.02, 0.81, 0.005)
    nilai = [biaya_total(y, skor, a, biaya_fn, biaya_fp) for a in kandidat]
    i = int(np.argmin(nilai))
    return float(kandidat[i]), float(nilai[i])


def metrik_segmen(df: pd.DataFrame, y, skor, kolom="segmen", min_n=100) -> pd.DataFrame:
    baris = []
    s = pd.Series(np.asarray(skor), index=df.index)
    yy = pd.Series(np.asarray(y), index=df.index)
    for nama, g in df.groupby(kolom):
        if len(g) < min_n or yy.loc[g.index].nunique() < 2:
            baris.append({kolom: nama, "n": len(g), "auc": np.nan, "tingkat_setuju": np.nan})
            continue
        baris.append({kolom: nama, "n": len(g),
                      "auc": float(roc_auc_score(yy.loc[g.index], s.loc[g.index])),
                      "laju_target": float(yy.loc[g.index].mean())})
    return pd.DataFrame(baris).sort_values("n", ascending=False)


def evaluasi(df: pd.DataFrame, y, skor, biaya_fn, biaya_fp) -> dict:
    a_opt, biaya = ambang_optimal(y, skor, biaya_fn, biaya_fp)
    return {
        "auc": float(roc_auc_score(y, skor)),
        "ks": ks(y, skor),
        "gini": gini(y, skor),
        "ambang_optimal": a_opt,
        "biaya_total": biaya,
        "n": int(len(y)),
    }


def periksa_gerbang(hasil: dict, seg: pd.DataFrame, gates: dict) -> tuple[bool, list[str]]:
    """Mengeksekusi kebijakan gates.yaml. Mengembalikan (lulus, daftar pelanggaran)."""
    g = gates[gates["tingkat_risiko"]] if "tingkat_risiko" in gates else gates
    pelanggaran = []
    if hasil["auc"] < g["auc_min"]:
        pelanggaran.append(f"AUC {hasil['auc']:.4f} < ambang {g['auc_min']}")
    if g.get("wajib_laporan_segmen") and "auc_min_per_segmen" in g:
        layak = seg[seg["n"] >= g.get("segmen_min_n", 0)]
        buruk = layak[layak["auc"] < g["auc_min_per_segmen"]]
        for _, r in buruk.iterrows():
            pelanggaran.append(
                f"segmen {r.iloc[0]} (n={int(r['n'])}) AUC {r['auc']:.4f} < {g['auc_min_per_segmen']}")
    return (len(pelanggaran) == 0), pelanggaran
