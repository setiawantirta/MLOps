"""Lapis 6 — uji kesetaraan fitur pelatihan vs penyajian.

Inilah uji dengan pengembalian terbesar per satuan usaha: ia menangkap
training-serving skew, kelas kesalahan yang tidak menimbulkan galat apa pun.
"""
import numpy as np, pandas as pd, pytest, yaml
from src import data as D, features as F

CFG = yaml.safe_load(open("conf/params.yaml", encoding="utf-8"))
TOLERANSI = 1e-9


@pytest.fixture(scope="module")
def bahan():
    df = D.muat(CFG["data"]["mentah"]).sample(1000, random_state=7).reset_index(drop=True)
    art = F.ArtefakFitur.muat("artifacts/artefak_fitur.json")
    return df, art


def test_batch_sama_dengan_satu_per_satu(bahan):
    """Jalur pelatihan memproses batch; jalur penyajian memproses satu baris."""
    df, art = bahan
    batch = F.terapkan(df, art).to_numpy()
    satuan = np.vstack([F.terapkan(df.iloc[[i]], art).to_numpy() for i in range(len(df))])
    selisih = np.nanmax(np.abs(batch - satuan))
    assert selisih < TOLERANSI, f"selisih maksimum {selisih}"


def test_urutan_kolom_stabil(bahan):
    df, art = bahan
    a = list(F.terapkan(df, art).columns)
    b = list(F.terapkan(df.iloc[::-1], art).columns)
    assert a == b == F.kolom_fitur(art)
