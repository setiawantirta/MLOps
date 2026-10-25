"""Lapis 2 — uji kontrak data."""
import pandas as pd, pytest, yaml
from src import data as D

CFG = yaml.safe_load(open("conf/params.yaml", encoding="utf-8"))


@pytest.fixture(scope="module")
def df():
    return D.muat(CFG["data"]["mentah"])


def test_skema_dan_rentang_lulus(df):
    assert D.validasi(df) == []


def test_validasi_menangkap_kolom_hilang(df):
    rusak = df.drop(columns=["usia"])
    assert any("usia" in m for m in D.validasi(rusak))


def test_validasi_menangkap_nilai_di_luar_rentang(df):
    rusak = df.copy()
    rusak.loc[rusak.index[:50], "usia"] = 200
    assert any("usia" in m for m in D.validasi(rusak))


def test_pembagian_tidak_tumpang_tindih(df):
    a, b, c = D.bagi_waktu(df, CFG["data"]["batas_latih"], CFG["data"]["batas_validasi"])
    assert len(a) + len(b) + len(c) == len(df)
    assert a["periode"].max() < b["periode"].min() < c["periode"].min()
