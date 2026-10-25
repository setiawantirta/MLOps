"""
features.py — SATU-SATUNYA definisi transformasi fitur.

Modul ini diimpor oleh pipeline pelatihan (src/train.py) DAN oleh layanan
penyajian (src/api.py). Inilah pencegahan training-serving skew: tidak ada
definisi fitur yang ditulis dua kali.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

NUMERIK = [
    "usia", "tanggungan", "lama_usaha_tahun", "penghasilan_bulanan",
    "ragam_penghasilan", "plafon_diajukan", "tenor_bulan", "bunga_tahunan",
    "angsuran_bulanan", "rata_saldo_90h", "jumlah_tunggakan_12b",
    "rasio_angsuran", "saldo_per_angsuran", "log_penghasilan", "tenor_panjang",
    "tanpa_riwayat_biro",
]
KATEGORIK = ["segmen", "wilayah", "kanal"]

# Fitur yang DILARANG dipakai: kebocoran target / proksi atribut terlindungi.
TERLARANG = ["status_restrukturisasi", "gagal_bayar", "pengajuan_id",
             "tanggal_pengajuan", "periode"]

BATAS = {
    "usia": (17, 75),
    "rasio_angsuran": (0.0, 3.0),
    "penghasilan_bulanan": (1_000_000, 200_000_000),
    "tenor_bulan": (3, 60),
}


@dataclass
class ArtefakFitur:
    """Nilai yang DIPELAJARI dari data latih dan wajib dipakai ulang saat penyajian."""
    median: dict
    kategori: dict
    versi: str = "1.0.0"

    def simpan(self, jalur: str) -> None:
        with open(jalur, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def muat(jalur: str) -> "ArtefakFitur":
        with open(jalur, encoding="utf-8") as f:
            return ArtefakFitur(**json.load(f))


def turunkan(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur turunan yang TIDAK mempelajari apa pun dari data (aman, stateless)."""
    d = df.copy()
    d["rasio_angsuran"] = d["angsuran_bulanan"] / d["penghasilan_bulanan"]
    d["saldo_per_angsuran"] = d["rata_saldo_90h"] / d["angsuran_bulanan"]
    d["log_penghasilan"] = np.log1p(d["penghasilan_bulanan"])
    d["tenor_panjang"] = (d["tenor_bulan"] >= 36).astype(int)
    d["tanpa_riwayat_biro"] = (d["jumlah_tunggakan_12b"] < 0).astype(int)
    return d


def pelajari(df: pd.DataFrame) -> ArtefakFitur:
    """Mempelajari artefak HANYA dari data latih. Jangan pernah dari data uji."""
    d = turunkan(df)
    median = {k: float(d[k].median()) for k in NUMERIK if k in d}
    kategori = {k: sorted(map(str, d[k].dropna().unique())) for k in KATEGORIK}
    return ArtefakFitur(median=median, kategori=kategori)


def terapkan(df: pd.DataFrame, art: ArtefakFitur) -> pd.DataFrame:
    """Membentuk matriks fitur akhir. Urutan kolom dijamin stabil."""
    d = turunkan(df)
    X = pd.DataFrame(index=d.index)
    for k in NUMERIK:
        kolom = pd.to_numeric(d[k], errors="coerce") if k in d else pd.Series(np.nan, index=d.index)
        X[k] = kolom.fillna(art.median.get(k, 0.0)).astype(float)
    for k in KATEGORIK:
        nilai = d[k].astype(str) if k in d else pd.Series("", index=d.index)
        for kat in art.kategori[k]:
            X[f"{k}__{kat}"] = (nilai == kat).astype(float)
    return X[kolom_fitur(art)]


def kolom_fitur(art: ArtefakFitur) -> list[str]:
    return NUMERIK + [f"{k}__{kat}" for k in KATEGORIK for kat in art.kategori[k]]


def periksa_batas(baris: dict) -> list[str]:
    """Menolak lebih aman daripada menebak. Kosong berarti aman."""
    masalah = []
    for nama, (bawah, atas) in BATAS.items():
        nilai = baris.get(nama)
        if nilai is None:
            masalah.append(f"{nama}: nilai tidak tersedia")
        elif not (bawah <= float(nilai) <= atas):
            masalah.append(f"{nama}={nilai} di luar rentang [{bawah}, {atas}]")
    return masalah
