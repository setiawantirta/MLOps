# Proyek Praktikum MLOps — Skoring Risiko Kredit Sumatera

Repositori awal (*starter*) untuk praktikum mata kuliah **Machine Learning Operations
(SD25-40001)**, Program Studi Sarjana Sains Data, Institut Teknologi Sumatera.

Repositori ini bukan contoh jadi yang tinggal dijalankan. Ia adalah **kerangka yang
dibangun bertahap selama dua belas praktikum**. Berkas yang sudah tersedia menjadi
acuan mutu; berkas yang belum ada Anda tulis sendiri mengikuti Modul Praktikum.

## Menyiapkan lingkungan

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.lock
```

## Alur singkat

```bash
make data      # bangkitkan dataset sintetis (deterministik, tanpa internet)
make latih     # jalankan pipeline pelatihan
make uji       # jalankan enam lapis pengujian
make drift     # laporan drift, domain classifier, dan keadilan
make kartu     # bangkitkan Dokumen Model
make api       # jalankan layanan penskoran di http://localhost:8000/docs
```

## Struktur

| Jalur | Isi | Mulai dipakai |
|---|---|---|
| `buat_dataset.py` | Pembangkit dataset sintetis Kredit Sumatera | Praktikum 1 |
| `src/data.py` | Pemuatan, validasi skema, pembagian *out-of-time* | Praktikum 3 |
| `src/features.py` | **Satu-satunya** definisi fitur — dipakai pelatihan dan penyajian | Praktikum 5 |
| `src/train.py` | Pipeline pelatihan yang dapat dijalankan ulang | Praktikum 4 |
| `src/evaluate.py` | Metrik, analisis segmen, biaya, gerbang mutu | Praktikum 6 |
| `src/api.py` | Layanan penskoran FastAPI dengan log terstruktur | Praktikum 8 |
| `src/monitor.py` | Deteksi *drift* dan analisis keadilan | Praktikum 10 |
| `src/promote.py` | Champion/challenger dan aturan promosi | Praktikum 11 |
| `src/rilis.py` | Simulasi *shadow* dan *canary* berstrata | Praktikum 9 |
| `src/model_card.py` | Pembangkit Dokumen Model | Praktikum 12 |
| `src/skew_demo.py` | Peragaan *training–serving skew* | Praktikum 2 |
| `src/federated_demo.py` | Simulasi *federated learning* | Praktikum 9 |
| `conf/params.yaml` | Konfigurasi pipeline | Praktikum 4 |
| `conf/gates.yaml` | Kebijakan tata kelola yang **dieksekusi** | Praktikum 7 & 12 |
| `tests/` | Enam lapis pengujian | Praktikum 7 |
| `dvc.yaml`, `Makefile` | Definisi pipeline | Praktikum 3–4 |
| `Dockerfile`, `.github/workflows/ci.yml` | Kontainerisasi dan CI | Praktikum 8 |

## Aturan yang berlaku sepanjang praktikum

1. **Data besar dan artefak tidak masuk Git.** Keduanya dikelola DVC atau dibangkitkan ulang.
2. **Definisi fitur hanya ditulis sekali**, di `src/features.py`, dan diimpor kedua pipeline.
3. **Artefak yang dipelajari dari data latih** (median, daftar kategori) wajib disimpan
   dan dipakai ulang saat penyajian — tidak pernah dihitung ulang di produksi.
4. **Pembagian data selalu menurut waktu**, tidak pernah acak.
5. **Setiap keputusan dicatat** — konfigurasi, versi, dan alasannya.
