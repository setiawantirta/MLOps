"""rilis.py — simulasi strategi rilis: shadow, canary berstrata, dan kriteria pembatalan."""
from __future__ import annotations
import argparse, sys, os
import joblib, numpy as np, pandas as pd, yaml
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data as D, features as F

KRITERIA = {
    "ks_skor_maks": 0.10,              # pembatalan bila distribusi skor terlalu bergeser
    "turun_setuju_segmen_maks": 0.10,  # pembatalan bila satu segmen turun > 10 poin persentase
    "latensi_p95_maks_ms": 300.0,
}


def muat(dirartefak):
    return (joblib.load(f"{dirartefak}/model.joblib"),
            F.ArtefakFitur.muat(f"{dirartefak}/artefak_fitur.json"))


def alokasi_canary(df: pd.DataFrame, persen: float, kolom="segmen", seed=42) -> np.ndarray:
    """Alokasi ACAK BERSTRATA: proporsi tiap segmen pada trafik canary
    mencerminkan proporsinya pada populasi. Alokasi per lokasi tunggal akan
    menyembunyikan kegagalan pada segmen yang tidak terwakili."""
    rng = np.random.default_rng(seed)
    pilih = np.zeros(len(df), dtype=bool)
    for _, g in df.groupby(kolom):
        n = max(1, int(round(len(g) * persen)))
        idx = rng.choice(g.index.values, size=min(n, len(g)), replace=False)
        pilih[df.index.get_indexer(idx)] = True
    return pilih


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--juara", default="artifacts_juara")
    ap.add_argument("--penantang", default="artifacts_penantang")
    ap.add_argument("--persen", type=float, default=0.05)
    ap.add_argument("--ambang", type=float, default=0.085)
    ap.add_argument("--config", default="conf/params.yaml")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    df = D.muat(cfg["data"]["mentah"])
    _, _, uji = D.bagi_waktu(df, cfg["data"]["batas_latih"], cfg["data"]["batas_validasi"])
    uji = uji.reset_index(drop=True)

    mj, aj = muat(a.juara)
    mp, ap_ = muat(a.penantang)
    sj = mj.predict_proba(F.terapkan(uji, aj))[:, 1]
    sp = mp.predict_proba(F.terapkan(uji, ap_))[:, 1]

    # ---------------- TAHAP 1: SHADOW ----------------
    print("=== TAHAP SHADOW (seluruh trafik disalin, keluaran TIDAK dipakai) ===")
    stat, _ = ks_2samp(sj, sp)
    setuju_j = (sj < a.ambang).mean()
    setuju_p = (sp < a.ambang).mean()
    print(f"KS distribusi skor           : {stat:.4f} (batas {KRITERIA['ks_skor_maks']})")
    print(f"Tingkat setuju juara/penantang: {setuju_j:.3f} / {setuju_p:.3f}")
    lolos_shadow = stat <= KRITERIA["ks_skor_maks"]
    print("Keputusan shadow             :", "LANJUT ke canary" if lolos_shadow else "BATALKAN")

    # ---------------- TAHAP 2: CANARY ----------------
    print(f"\n=== TAHAP CANARY {a.persen:.0%} (acak berstrata menurut segmen) ===")
    pilih = alokasi_canary(uji, a.persen)
    print(f"Trafik canary: {pilih.sum()} dari {len(uji)} permintaan")
    baris = []
    for seg, g in uji.groupby("segmen"):
        idx = g.index.values
        m = pilih[idx]
        if m.sum() < 5:
            continue
        s_j = (sj[idx][m] < a.ambang).mean()
        s_p = (sp[idx][m] < a.ambang).mean()
        baris.append({"segmen": seg, "n_canary": int(m.sum()),
                      "setuju_juara": round(float(s_j), 4),
                      "setuju_penantang": round(float(s_p), 4),
                      "selisih": round(float(s_p - s_j), 4)})
    t = pd.DataFrame(baris).sort_values("selisih")
    print(t.to_string(index=False))

    pelanggaran = t[t["selisih"] < -KRITERIA["turun_setuju_segmen_maks"]]
    print("\n--- Evaluasi kriteria pembatalan ---")
    if len(pelanggaran):
        for _, r in pelanggaran.iterrows():
            print(f"  PEMBATALAN: segmen {r['segmen']} turun {abs(r['selisih']):.1%} "
                  f"(batas {KRITERIA['turun_setuju_segmen_maks']:.0%})")
        print("\nTINDAKAN: hentikan canary, kembalikan seluruh trafik ke juara.")
    else:
        print("  Tidak ada segmen melampaui batas pembatalan.")
        print("\nTINDAKAN: lanjut ke canary 25%, lalu rilis penuh setelah persetujuan.")


if __name__ == "__main__":
    main()
