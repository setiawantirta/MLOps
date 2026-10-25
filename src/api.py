"""api.py — layanan penskoran. Mengimpor features.py yang SAMA dengan pelatihan."""
from __future__ import annotations
import json, os, time, uuid, logging
import joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src import features as F

DIR_ARTEFAK = os.getenv("DIR_ARTEFAK", "artifacts")
AMBANG = float(os.getenv("AMBANG", "0.085"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("prediksi")

model = joblib.load(os.path.join(DIR_ARTEFAK, "model.joblib"))
artefak = F.ArtefakFitur.muat(os.path.join(DIR_ARTEFAK, "artefak_fitur.json"))
meta = json.load(open(os.path.join(DIR_ARTEFAK, "metadata.json"), encoding="utf-8"))

app = FastAPI(title="Skoring Risiko Kredit", version=meta["versi_model"])


class Permohonan(BaseModel):
    usia: int = Field(ge=17, le=75)
    tanggungan: int = 0
    lama_usaha_tahun: float | None = None
    penghasilan_bulanan: float
    ragam_penghasilan: float = 0.2
    plafon_diajukan: float
    tenor_bulan: int
    bunga_tahunan: float = 0.16
    angsuran_bulanan: float
    rata_saldo_90h: float | None = None
    jumlah_tunggakan_12b: int = -1
    segmen: str
    wilayah: str
    kanal: str = "cabang"


@app.get("/health")
def health():
    return {"status": "sehat", "versi_model": meta["versi_model"],
            "git_commit": meta["git_commit"]}


@app.post("/predict")
def predict(p: Permohonan):
    t0 = time.perf_counter()
    baris = p.model_dump()
    turunan = F.turunkan(pd.DataFrame([baris])).iloc[0].to_dict()

    masalah = F.periksa_batas(turunan)
    if masalah:
        return {"keputusan": "tinjau_manual", "alasan": masalah,
                "versi_model": meta["versi_model"]}

    X = F.terapkan(pd.DataFrame([baris]), artefak)
    skor = float(model.predict_proba(X)[0, 1])
    keputusan = "tolak" if skor >= AMBANG else "setujui"
    latensi = (time.perf_counter() - t0) * 1000

    log.info(json.dumps({
        "request_id": str(uuid.uuid4()), "waktu": pd.Timestamp.now().isoformat(),
        "versi_model": meta["versi_model"], "versi_artefak_fitur": artefak.versi,
        "fitur": baris, "skor": round(skor, 6), "ambang": AMBANG,
        "keputusan": keputusan, "latensi_ms": round(latensi, 2),
        "segmen": baris["segmen"],
    }, ensure_ascii=False))

    return {"skor": round(skor, 6), "ambang": AMBANG, "keputusan": keputusan,
            "versi_model": meta["versi_model"], "latensi_ms": round(latensi, 2)}
