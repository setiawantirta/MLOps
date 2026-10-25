"""Lapis 1 — uji unit fungsi transformasi."""
import pandas as pd
from src import features as F


def test_rasio_angsuran():
    df = pd.DataFrame([{"angsuran_bulanan": 1_000_000, "penghasilan_bulanan": 5_000_000,
                        "rata_saldo_90h": 2_000_000, "tenor_bulan": 24,
                        "jumlah_tunggakan_12b": 0}])
    assert F.turunkan(df)["rasio_angsuran"].iloc[0] == 0.2


def test_tenor_panjang():
    df = pd.DataFrame([{"angsuran_bulanan": 1, "penghasilan_bulanan": 1,
                        "rata_saldo_90h": 1, "tenor_bulan": 36,
                        "jumlah_tunggakan_12b": 0}])
    assert F.turunkan(df)["tenor_panjang"].iloc[0] == 1


def test_periksa_batas_menolak_usia_mustahil():
    masalah = F.periksa_batas({"usia": 120, "rasio_angsuran": 0.3,
                               "penghasilan_bulanan": 5e6, "tenor_bulan": 12})
    assert any("usia" in m for m in masalah)


def test_periksa_batas_lolos():
    assert F.periksa_batas({"usia": 30, "rasio_angsuran": 0.3,
                            "penghasilan_bulanan": 5e6, "tenor_bulan": 12}) == []
