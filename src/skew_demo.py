"""skew_demo.py — memperagakan training-serving skew dan biayanya.

Dijalankan pada Praktikum 2. Tujuannya membuat mahasiswa MELIHAT bahwa kesalahan
ini tidak menimbulkan galat apa pun: layanan tetap menjawab, hanya jawabannya salah.
"""
from __future__ import annotations
import sys, os
import numpy as np, pandas as pd, yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data as D

CFG = yaml.safe_load(open("conf/params.yaml", encoding="utf-8"))
KOLOM = ["usia", "tanggungan", "penghasilan_bulanan", "ragam_penghasilan",
         "plafon_diajukan", "tenor_bulan", "rata_saldo_90h",
         "jumlah_tunggakan_12b", "rasio_angsuran"]


def fitur_pelatihan(df: pd.DataFrame) -> pd.DataFrame:
    """Versi notebook: nilai hilang diisi MEDIAN seluruh dataset."""
    d = df.copy()
    d["rasio_angsuran"] = d["angsuran_bulanan"] / d["penghasilan_bulanan"]
    return d[KOLOM].fillna(d[KOLOM].median())


def fitur_penyajian_salah(df: pd.DataFrame) -> pd.DataFrame:
    """Versi layanan yang ditulis ulang oleh tim lain: nilai hilang diisi NOL.
    Perbedaan sekecil ini tidak menimbulkan galat apa pun."""
    d = df.copy()
    d["rasio_angsuran"] = d["angsuran_bulanan"] / d["penghasilan_bulanan"]
    return d[KOLOM].fillna(0)


def fitur_penyajian_benar(df: pd.DataFrame, median: pd.Series) -> pd.DataFrame:
    """Versi benar: memakai artefak median yang DIPELAJARI saat pelatihan."""
    d = df.copy()
    d["rasio_angsuran"] = d["angsuran_bulanan"] / d["penghasilan_bulanan"]
    return d[KOLOM].fillna(median)


def main():
    df = D.muat(CFG["data"]["mentah"])
    latih, _, uji = D.bagi_waktu(df, CFG["data"]["batas_latih"], CFG["data"]["batas_validasi"])

    Xl = fitur_pelatihan(latih)
    median = Xl.median()
    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06,
                                           max_iter=250, random_state=42)
    model.fit(Xl, latih["gagal_bayar"])

    y = uji["gagal_bayar"].values
    auc_benar = roc_auc_score(y, model.predict_proba(fitur_penyajian_benar(uji, median))[:, 1])
    auc_skew = roc_auc_score(y, model.predict_proba(fitur_penyajian_salah(uji))[:, 1])

    n_hilang = int(fitur_penyajian_salah(uji).isna().any(axis=1).sum()) + int(uji[["rata_saldo_90h", "lama_usaha_tahun"]].isna().any(axis=1).sum())
    print(f"Baris uji dengan setidaknya satu nilai hilang : {n_hilang} dari {len(uji)}"
          f" ({n_hilang / len(uji):.1%})")
    print(f"AUC dengan definisi fitur KONSISTEN           : {auc_benar:.4f}")
    print(f"AUC dengan definisi fitur MENYIMPANG          : {auc_skew:.4f}")
    print(f"Selisih                                       : {auc_benar - auc_skew:+.4f}")
    print("\nPerhatikan: TIDAK ADA satu pun galat, peringatan, atau pengecualian.")
    print("Layanan tetap mengembalikan angka. Hanya angkanya yang salah.")

    # Dampak yang sesungguhnya terasa: berapa KEPUTUSAN yang berubah
    s_benar = model.predict_proba(fitur_penyajian_benar(uji, median))[:, 1]
    s_skew = model.predict_proba(fitur_penyajian_salah(uji))[:, 1]
    ambang = 0.085
    balik = ((s_benar >= ambang) != (s_skew >= ambang))
    kena = uji[KOLOM[:-1]].isna().any(axis=1).values
    print(f"\nKeputusan yang BERUBAH akibat skew          : {balik.sum()} "
          f"({balik.mean():.2%} dari seluruh permohonan)")
    if kena.sum():
        print(f"AUC pada baris yang terdampak — konsisten    : "
              f"{roc_auc_score(y[kena], s_benar[kena]):.4f}")
        print(f"AUC pada baris yang terdampak — menyimpang   : "
              f"{roc_auc_score(y[kena], s_skew[kena]):.4f}")

    beda = np.abs(fitur_penyajian_benar(uji, median).to_numpy()
                  - fitur_penyajian_salah(uji).to_numpy()).max()
    print(f"\nSelisih maksimum vektor fitur: {beda:,.2f}")
    print("Uji kesetaraan (tests/test_equivalence.py) akan menangkap ini seketika,")
    print("sedangkan uji unit biasa tidak akan pernah menangkapnya.")


if __name__ == "__main__":
    main()
