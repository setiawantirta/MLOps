"""
buat_dataset.py — Pembangkit dataset sintetis "Kredit Sumatera".

Dataset ini dirancang khusus untuk praktikum MLOps: deterministik, dapat dibuat
ulang tanpa internet, memiliki dimensi waktu, segmen nasabah, dan pergeseran
populasi yang disengaja pada periode akhir sehingga deteksi drift dapat dilatih.

Pemakaian:
    python buat_dataset.py --keluaran data/raw --seed 42
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

SEGMEN = ["pegawai_tetap", "wiraswasta", "pedagang_musiman", "petani", "nelayan"]
# proporsi segmen pada periode awal dan periode akhir (pergeseran disengaja)
PROP_AWAL = np.array([0.42, 0.24, 0.14, 0.13, 0.07])
PROP_AKHIR = np.array([0.28, 0.22, 0.20, 0.18, 0.12])
WILAYAH = ["Bandar Lampung", "Metro", "Pringsewu", "Pesawaran", "Lampung Barat"]


def _proporsi(bulan_ke: int, total_bulan: int) -> np.ndarray:
    """Interpolasi linear proporsi segmen; pergeseran mulai dari 60% periode."""
    mulai = int(total_bulan * 0.6)
    if bulan_ke <= mulai:
        return PROP_AWAL
    t = (bulan_ke - mulai) / max(1, total_bulan - mulai)
    p = (1 - t) * PROP_AWAL + t * PROP_AKHIR
    return p / p.sum()


def buat(n_per_bulan: int = 900, bulan_awal: str = "2022-01", total_bulan: int = 36,
         seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periode = pd.period_range(bulan_awal, periods=total_bulan, freq="M")
    baris = []

    for b, per in enumerate(periode):
        prop = _proporsi(b, total_bulan)
        n = n_per_bulan
        seg = rng.choice(SEGMEN, size=n, p=prop)

        # usia: pergeseran ke lebih muda pada periode akhir (kanal digital)
        geser_usia = 0.0 if b < total_bulan * 0.6 else -6.0 * (b - total_bulan * 0.6) / (total_bulan * 0.4)
        usia = np.clip(rng.normal(39 + geser_usia, 9.5, n), 19, 70).round().astype(int)

        # penghasilan bulanan (rupiah), bergantung segmen
        basis = {"pegawai_tetap": 6.2e6, "wiraswasta": 7.8e6, "pedagang_musiman": 4.6e6,
                 "petani": 3.6e6, "nelayan": 3.4e6}
        mu = np.array([basis[s] for s in seg])
        penghasilan = np.clip(rng.lognormal(np.log(mu), 0.42), 1.5e6, 9e7).round(-4)

        # ragam penghasilan: tinggi untuk segmen musiman
        ragam_basis = {"pegawai_tetap": 0.04, "wiraswasta": 0.22, "pedagang_musiman": 0.41,
                       "petani": 0.46, "nelayan": 0.48}
        ragam = np.clip(rng.normal([ragam_basis[s] for s in seg], 0.06), 0.0, 0.95).round(3)

        lama_usaha = np.clip(rng.gamma(2.2, 2.6, n), 0, 35).round(1)
        tanggungan = rng.poisson(1.9, n).clip(0, 8)

        plafon = np.clip(rng.lognormal(np.log(penghasilan * 4.0), 0.5), 2e6, 4e8).round(-4)
        tenor = rng.choice([6, 12, 18, 24, 36, 48], n, p=[.09, .23, .17, .27, .16, .08])
        bunga = np.clip(rng.normal(0.16, 0.022, n), 0.09, 0.28).round(4)
        angsuran = (plafon * (1 + bunga * tenor / 12) / tenor).round(-3)
        rasio_angsuran = (angsuran / penghasilan).round(4)

        rata_saldo_90h = np.clip(rng.lognormal(np.log(penghasilan * 0.55), 0.85), 0, 5e8).round(-3)
        jumlah_tunggakan_12b = rng.poisson(
            np.where(np.isin(seg, ["pedagang_musiman", "petani", "nelayan"]), 0.85, 0.35)).clip(0, 12)
        punya_riwayat_biro = rng.random(n) > np.where(seg == "pegawai_tetap", 0.08, 0.30)
        jumlah_tunggakan_12b = np.where(punya_riwayat_biro, jumlah_tunggakan_12b, -1)

        wilayah = rng.choice(WILAYAH, n, p=[.34, .17, .18, .17, .14])
        kanal = rng.choice(["cabang", "agen", "daring"], n,
                           p=[.62, .26, .12] if b < total_bulan * 0.6 else [.38, .24, .38])

        # ---------------- pembentukan target ----------------
        z = (
            -3.85
            + 3.10 * np.clip(rasio_angsuran, 0, 1.4)
            + 0.155 * np.maximum(jumlah_tunggakan_12b, 0)
            + 0.62 * (jumlah_tunggakan_12b < 0)
            + 1.42 * ragam
            - 0.031 * (usia - 39) / 5
            - 0.30 * np.log1p(lama_usaha)
            + 0.085 * tanggungan
            - 0.28 * np.log1p(rata_saldo_90h / np.maximum(angsuran, 1))
            + 0.19 * (tenor >= 36)
            + 0.22 * (kanal == "daring")
        )
        # concept drift ringan: tekanan makro pada 20% periode terakhir
        if b >= total_bulan * 0.8:
            z += 0.55 * np.clip(rasio_angsuran, 0, 1.4) + 0.12
        p_gagal = 1 / (1 + np.exp(-z))
        gagal_bayar = (rng.random(n) < p_gagal).astype(int)

        baris.append(pd.DataFrame({
            "pengajuan_id": [f"{per}-{i:05d}" for i in range(n)],
            "tanggal_pengajuan": rng.choice(pd.date_range(per.start_time, per.end_time), n),
            "periode": str(per),
            "segmen": seg,
            "wilayah": wilayah,
            "kanal": kanal,
            "usia": usia,
            "tanggungan": tanggungan,
            "lama_usaha_tahun": lama_usaha,
            "penghasilan_bulanan": penghasilan,
            "ragam_penghasilan": ragam,
            "plafon_diajukan": plafon,
            "tenor_bulan": tenor,
            "bunga_tahunan": bunga,
            "angsuran_bulanan": angsuran,
            "rata_saldo_90h": rata_saldo_90h,
            "jumlah_tunggakan_12b": jumlah_tunggakan_12b,
            "gagal_bayar": gagal_bayar,
        }))

    df = pd.concat(baris, ignore_index=True)
    df = df.sort_values("tanggal_pengajuan").reset_index(drop=True)

    # nilai hilang yang realistis (bukan acak sempurna)
    hilang = (df["segmen"].isin(["petani", "nelayan"])) & (rng.random(len(df)) < 0.11)
    df.loc[hilang, "rata_saldo_90h"] = np.nan
    hilang2 = rng.random(len(df)) < 0.03
    df.loc[hilang2, "lama_usaha_tahun"] = np.nan

    # kolom jebakan kebocoran: hanya terisi bagi yang gagal bayar
    df["status_restrukturisasi"] = np.where(
        (df["gagal_bayar"] == 1) & (rng.random(len(df)) < 0.62), "ya", None)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keluaran", default="data/raw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-per-bulan", type=int, default=900)
    ap.add_argument("--total-bulan", type=int, default=36)
    a = ap.parse_args()

    os.makedirs(a.keluaran, exist_ok=True)
    df = buat(n_per_bulan=a.n_per_bulan, total_bulan=a.total_bulan, seed=a.seed)
    jalur = os.path.join(a.keluaran, "kredit_sumatera.csv")
    df.to_csv(jalur, index=False)

    print(f"Tersimpan: {jalur}")
    print(f"Baris      : {len(df):,}")
    print(f"Periode    : {df['periode'].min()} s.d. {df['periode'].max()}")
    print(f"Gagal bayar: {df['gagal_bayar'].mean():.2%}")
    print("\nProporsi segmen periode awal vs akhir:")
    awal = df[df.periode <= df.periode.unique()[5]]["segmen"].value_counts(normalize=True)
    akhir = df[df.periode >= df.periode.unique()[-6]]["segmen"].value_counts(normalize=True)
    print(pd.DataFrame({"awal": awal, "akhir": akhir}).round(3).to_string())


if __name__ == "__main__":
    main()
