"""monitor.py — deteksi drift (uji univariat + domain classifier) dan keadilan."""
from __future__ import annotations
import argparse, json, sys, os
import numpy as np, pandas as pd, yaml
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data as D, features as F

KATEGORIK = {"segmen", "wilayah", "kanal"}


def skor_drift(ref: pd.DataFrame, baru: pd.DataFrame, kolom: list[str]) -> pd.DataFrame:
    baris = []
    for k in kolom:
        if k in KATEGORIK:
            tabel = pd.crosstab(
                pd.concat([ref[k].astype(str), baru[k].astype(str)], ignore_index=True),
                np.r_[["ref"] * len(ref), ["baru"] * len(baru)])
            stat, p, _, _ = chi2_contingency(tabel)
            uji = "chi2"
        else:
            a, b = ref[k].dropna(), baru[k].dropna()
            stat, p = ks_2samp(a, b)
            uji = "ks"
        baris.append({"fitur": k, "uji": uji, "statistik": float(stat), "p_value": float(p)})
    h = pd.DataFrame(baris)
    # Koreksi Bonferroni: tanpa ini, menguji 15 fitur pada alfa 0,05 memicu
    # sekitar satu alarm palsu setiap pemeriksaan.
    h["p_terkoreksi"] = (h["p_value"] * len(h)).clip(upper=1.0)
    h["bergeser"] = (h["p_terkoreksi"] < 0.05) & (h["statistik"] > 0.10)
    return h.sort_values("statistik", ascending=False).reset_index(drop=True)


def domain_classifier(ref: pd.DataFrame, baru: pd.DataFrame, art, seed=42):
    Xr, Xb = F.terapkan(ref, art), F.terapkan(baru, art)
    X = pd.concat([Xr, Xb], ignore_index=True)
    y = np.r_[np.zeros(len(Xr)), np.ones(len(Xb))]
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=-1, random_state=seed)
    auc = float(cross_val_score(clf, X, y, cv=4, scoring="roc_auc").mean())
    clf.fit(X, y)
    penting = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return auc, penting.head(8)


def keadilan(df: pd.DataFrame, skor, ambang: float, kolom="segmen") -> pd.DataFrame:
    s = pd.Series(np.asarray(skor), index=df.index)
    baris = []
    for nama, g in df.groupby(kolom):
        pred_tolak = (s.loc[g.index] >= ambang)
        d = {kolom: nama, "n": len(g), "tingkat_setuju": float(1 - pred_tolak.mean())}
        if "gagal_bayar" in g and g["gagal_bayar"].nunique() > 1:
            y = g["gagal_bayar"]
            # equal opportunity: TPR (proporsi gagal bayar yang berhasil ditolak)
            d["tpr"] = float(pred_tolak[y == 1].mean())
            d["fpr"] = float(pred_tolak[y == 0].mean())
        baris.append(d)
    t = pd.DataFrame(baris)
    t["selisih_setuju_vs_rerata"] = (t["tingkat_setuju"] - t["tingkat_setuju"].mean()).round(4)
    return t.sort_values("n", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/params.yaml")
    ap.add_argument("--artefak", default="artifacts")
    ap.add_argument("--keluaran", default="laporan")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    os.makedirs(a.keluaran, exist_ok=True)
    df = D.muat(cfg["data"]["mentah"])
    latih, _, uji = D.bagi_waktu(df, cfg["data"]["batas_latih"], cfg["data"]["batas_validasi"])

    art = F.ArtefakFitur.muat(os.path.join(a.artefak, "artefak_fitur.json"))
    kolom = ["usia", "penghasilan_bulanan", "rasio_angsuran", "ragam_penghasilan",
             "rata_saldo_90h", "jumlah_tunggakan_12b", "tenor_bulan", "lama_usaha_tahun",
             "segmen", "wilayah", "kanal"]
    ref = F.turunkan(latih)
    baru = F.turunkan(uji)

    h = skor_drift(ref, baru, kolom)
    auc_dc, penting = domain_classifier(latih, uji, art)

    import joblib
    model = joblib.load(os.path.join(a.artefak, "model.joblib"))
    meta = json.load(open(os.path.join(a.artefak, "metadata.json"), encoding="utf-8"))
    skor = model.predict_proba(F.terapkan(uji, art))[:, 1]
    adil = keadilan(uji, skor, meta["metrik_uji_oot"]["ambang_optimal"])

    h.to_csv(f"{a.keluaran}/drift.csv", index=False)
    adil.to_csv(f"{a.keluaran}/keadilan.csv", index=False)

    print("=== Skor drift per fitur ===")
    print(h.to_string(index=False))
    print(f"\n=== Domain classifier: AUC = {auc_dc:.4f} ===")
    print("(0,5 = tidak dapat dibedakan; mendekati 1 = drift kuat)")
    print(penting.round(4).to_string())
    print("\n=== Analisis keadilan per segmen ===")
    print(adil.round(4).to_string(index=False))

    bergeser = h[h["bergeser"]]["fitur"].tolist()
    print("\n--- Kesimpulan otomatis ---")
    if not bergeser:
        print("Tidak ada fitur yang bergeser secara bermakna.")
    elif len(bergeser) == 1:
        print(f"Hanya SATU fitur bergeser: {bergeser[0]}.")
        print("Pola satu fitur sendirian lebih konsisten dengan CACAT PIPELINE")
        print("daripada perubahan populasi. Periksa skema sumber sebelum melatih ulang.")
    else:
        print(f"{len(bergeser)} fitur bergeser bersamaan: {', '.join(bergeser)}.")
        print("Pola ini konsisten dengan PERUBAHAN POPULASI nyata.")
        print("Periksa segmen mana yang tumbuh sebelum memutuskan pelatihan ulang.")


if __name__ == "__main__":
    main()
