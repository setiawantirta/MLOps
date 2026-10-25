"""model_card.py — membangkitkan Dokumen Model dari metadata, bukan diketik ulang."""
from __future__ import annotations
import argparse, json, os, datetime as dt
import pandas as pd, yaml

TEMPLAT = """# Dokumen Model (Model Card)

## 1. Identitas
- **Nama model**: Skoring Risiko Kredit Konsumen — Sumatera
- **Versi**: {versi}
- **Tanggal dibangkitkan**: {tanggal}
- **Git commit**: `{commit}`
- **Tingkat risiko**: {risiko}
- **Status**: Staging

## 2. Tujuan dan cakupan
Model memberi estimasi probabilitas gagal bayar dalam 12 bulan bagi pemohon kredit
tanpa agunan. Keluaran dipakai untuk keputusan setujui / tolak / tinjau manual.

**Penggunaan yang TIDAK dimaksudkan**: penetapan harga (bunga), penilaian ulang
debitur berjalan, dan keputusan pada segmen yang tidak terwakili dalam data latih.

## 3. Data
- Sumber: `{sumber}`
- Pembagian: latih ≤ {batas_latih} · validasi ≤ {batas_validasi} · uji out-of-time setelahnya
- Jumlah baris uji out-of-time: {n_uji}

## 4. Fitur
Jumlah kolom fitur: {n_fitur}. Daftar lengkap tersedia pada `artifacts/metadata.json`.
Fitur terlarang yang secara eksplisit dikeluarkan: {terlarang}

## 5. Metode
- Algoritma: `{algoritma}`
- Parameter: {parameter}

## 6. Evaluasi (uji out-of-time)
| Metrik | Nilai |
|---|---|
| AUC | {auc:.4f} |
| KS | {ks:.4f} |
| Gini | {gini:.4f} |
| Ambang optimal biaya | {ambang:.4f} |
| Biaya simulasi | Rp {biaya:,.0f} |

## 7. Kinerja per segmen
{tabel_segmen}

## 8. Keterjelasan
Laporan kontribusi fitur wajib dilampirkan pada berkas terpisah sebelum promosi ke
Production. Pemeriksaan proksi atribut terlindungi: belum / sudah (coret salah satu).

## 9. Keterbatasan yang diketahui
- Kinerja pada segmen dengan n kecil tidak stabil dan tidak dijamin.
- Pemohon tanpa riwayat biro kredit ditandai `tanpa_riwayat_biro`; kinerja pada
  kelompok ini perlu dipantau tersendiri.
- Ground truth baru matang setelah sekurang-kurangnya 12 bulan.

## 10. Operasi
- Artefak: `model.joblib`, `artefak_fitur.json`, `metadata.json`
- Kontrak antarmuka: `POST /predict` (lihat `src/api.py`)
- Prosedur pengembalian: ganti tag citra ke versi sebelumnya; target < 5 menit.

## 11. Pemantauan
Metrik, ambang, dan pemilik tercantum pada `laporan/` dan dasbor pemantauan.

## 12. Persetujuan
| Peran | Nama | Tanggal | Tanda tangan |
|---|---|---|---|
| Pengembang model |  |  |  |
| Validator independen |  |  |  |
| Pemilik proses bisnis |  |  |  |
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefak", default="artifacts")
    ap.add_argument("--gates", default="conf/gates.yaml")
    ap.add_argument("--keluaran", default="laporan/model_card.md")
    a = ap.parse_args()

    meta = json.load(open(f"{a.artefak}/metadata.json", encoding="utf-8"))
    gates = yaml.safe_load(open(a.gates, encoding="utf-8"))
    risiko = gates["tingkat_risiko"]
    seg = pd.DataFrame(meta["metrik_segmen"])
    m = meta["metrik_uji_oot"]

    teks = TEMPLAT.format(
        versi=meta["versi_model"], tanggal=dt.date.today().isoformat(),
        commit=meta["git_commit"], risiko=risiko,
        sumber=meta["pembagian"]["mentah"],
        batas_latih=meta["pembagian"]["batas_latih"],
        batas_validasi=meta["pembagian"]["batas_validasi"],
        n_uji=m["n"], n_fitur=len(meta["kolom_fitur"]),
        terlarang=", ".join(f"`{x}`" for x in gates[risiko]["fitur_terlarang"]),
        algoritma=meta["algoritma"], parameter=meta["parameter"],
        auc=m["auc"], ks=m["ks"], gini=m["gini"],
        ambang=m["ambang_optimal"], biaya=m["biaya_total"],
        tabel_segmen=seg.round(4).to_markdown(index=False),
    )
    os.makedirs(os.path.dirname(a.keluaran), exist_ok=True)
    open(a.keluaran, "w", encoding="utf-8").write(teks)
    print("Tersimpan:", a.keluaran)


if __name__ == "__main__":
    main()
