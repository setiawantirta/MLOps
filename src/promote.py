"""promote.py — champion/challenger: memutuskan apakah penantang layak dipromosikan."""
from __future__ import annotations
import argparse, json, sys, os
import joblib, numpy as np, pandas as pd, yaml
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data as D, features as F, evaluate as E

MARGIN_MIN = 0.008        # dua kali rentang ketidakpastian tipikal
TURUN_MAKS_SEGMEN = 0.02  # penurunan AUC segmen yang masih dapat diterima
KS_MAKS_SKOR = 0.08       # batas laju perubahan perilaku


def skor(dirartefak: str, df: pd.DataFrame) -> np.ndarray:
    model = joblib.load(f"{dirartefak}/model.joblib")
    art = F.ArtefakFitur.muat(f"{dirartefak}/artefak_fitur.json")
    return model.predict_proba(F.terapkan(df, art))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--juara", required=True)
    ap.add_argument("--penantang", required=True)
    ap.add_argument("--config", default="conf/params.yaml")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    df = D.muat(cfg["data"]["mentah"])
    _, _, uji = D.bagi_waktu(df, cfg["data"]["batas_latih"], cfg["data"]["batas_validasi"])
    y = uji["gagal_bayar"].values

    sj, sp = skor(a.juara, uji), skor(a.penantang, uji)
    b = cfg["biaya"]
    hj = E.evaluasi(uji, y, sj, b["fn"], b["fp"])
    hp = E.evaluasi(uji, y, sp, b["fn"], b["fp"])
    segj = E.metrik_segmen(uji, y, sj).set_index("segmen")
    segp = E.metrik_segmen(uji, y, sp).set_index("segmen")

    syarat = []
    syarat.append(("AUC unggul secara bermakna",
                   hp["auc"] - hj["auc"] >= MARGIN_MIN,
                   f"selisih {hp['auc'] - hj['auc']:+.4f} (minimal +{MARGIN_MIN})"))

    turun = []
    for s in segp.index.intersection(segj.index):
        if pd.notna(segp.loc[s, "auc"]) and pd.notna(segj.loc[s, "auc"]):
            d = segp.loc[s, "auc"] - segj.loc[s, "auc"]
            if d < -TURUN_MAKS_SEGMEN:
                turun.append(f"{s} ({d:+.4f})")
    syarat.append(("Tidak ada segmen memburuk melewati batas", not turun,
                   "aman" if not turun else "memburuk: " + ", ".join(turun)))

    stat_ks, _ = ks_2samp(sj, sp)
    syarat.append(("Perubahan perilaku dalam batas wajar", stat_ks <= KS_MAKS_SKOR,
                   f"KS distribusi skor {stat_ks:.4f} (maksimal {KS_MAKS_SKOR})"))

    syarat.append(("Biaya simulasi tidak lebih besar", hp["biaya_total"] <= hj["biaya_total"],
                   f"Rp {hp['biaya_total']:,.0f} vs Rp {hj['biaya_total']:,.0f}"))

    print(f"{'Syarat promosi':<45} {'Status':<8} Keterangan")
    print("-" * 100)
    for nama, lulus, ket in syarat:
        print(f"{nama:<45} {'LULUS' if lulus else 'GAGAL':<8} {ket}")

    semua = all(s[1] for s in syarat)
    print("\nKEPUTUSAN:", "PROMOSIKAN penantang (setelah persetujuan manusia)"
          if semua else "PERTAHANKAN juara — penantang tidak lulus seluruh syarat")
    sys.exit(0 if semua else 1)


if __name__ == "__main__":
    main()
