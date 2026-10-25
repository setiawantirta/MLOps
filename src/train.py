"""train.py — pipeline pelatihan yang dapat dijalankan ulang dan dicatat."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import joblib, numpy as np, pandas as pd, yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data as D, features as F, evaluate as E


def komit_git() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "tidak-ada-git"


def bangun_model(p: dict):
    algo = p["algoritma"]
    if algo == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_depth=p["max_depth"], learning_rate=p["learning_rate"],
            max_iter=p["max_iter"], random_state=p["random_state"])
    if algo == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, max_depth=p["max_depth"],
                                      min_samples_leaf=20, n_jobs=-1,
                                      random_state=p["random_state"])
    if algo == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000,
                                                random_state=p["random_state"]))
    raise ValueError(f"algoritma tidak dikenal: {algo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/params.yaml")
    ap.add_argument("--keluaran", default="artifacts")
    ap.add_argument("--mlflow", action="store_true", help="catat ke MLflow bila tersedia")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    os.makedirs(a.keluaran, exist_ok=True)

    df = D.muat(cfg["data"]["mentah"])
    masalah = D.validasi(df)
    if masalah:
        print("VALIDASI DATA GAGAL:")
        for m in masalah:
            print("  -", m)
        sys.exit(2)

    latih, val, uji = D.bagi_waktu(df, cfg["data"]["batas_latih"], cfg["data"]["batas_validasi"])
    print(f"latih={len(latih)}  validasi={len(val)}  uji_oot={len(uji)}")

    art = F.pelajari(latih)                       # artefak DIPELAJARI HANYA dari data latih
    Xl, Xv, Xu = (F.terapkan(x, art) for x in (latih, val, uji))
    yl, yv, yu = (x["gagal_bayar"].values for x in (latih, val, uji))

    model = bangun_model(cfg["model"])
    model.fit(Xl, yl)

    skor_v = model.predict_proba(Xv)[:, 1]
    skor_u = model.predict_proba(Xu)[:, 1]
    b = cfg["biaya"]
    hasil_v = E.evaluasi(val, yv, skor_v, b["fn"], b["fp"])
    hasil_u = E.evaluasi(uji, yu, skor_u, b["fn"], b["fp"])
    seg = E.metrik_segmen(uji, yu, skor_u, "segmen", min_n=100)

    joblib.dump(model, os.path.join(a.keluaran, "model.joblib"))
    art.simpan(os.path.join(a.keluaran, "artefak_fitur.json"))
    metadata = {
        "versi_model": "2.0.0",
        "git_commit": komit_git(),
        "algoritma": cfg["model"]["algoritma"],
        "parameter": cfg["model"],
        "pembagian": cfg["data"],
        "kolom_fitur": F.kolom_fitur(art),
        "metrik_validasi": hasil_v,
        "metrik_uji_oot": hasil_u,
        "metrik_segmen": seg.to_dict(orient="records"),
    }
    with open(os.path.join(a.keluaran, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    seg.to_csv(os.path.join(a.keluaran, "metrik_segmen.csv"), index=False)

    print("\n--- Uji out-of-time ---")
    for k, v in hasil_u.items():
        print(f"  {k:16s}: {v:,.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")
    print("\n--- Metrik per segmen ---")
    print(seg.to_string(index=False))

    if a.mlflow:
        try:
            import mlflow
            mlflow.set_experiment("kredit-sumatera")
            with mlflow.start_run(run_name=f"{cfg['model']['algoritma']}-oot"):
                mlflow.log_params(cfg["model"])
                mlflow.log_params({"batas_latih": cfg["data"]["batas_latih"]})
                mlflow.log_metrics({f"oot_{k}": v for k, v in hasil_u.items()
                                    if isinstance(v, (int, float))})
                for r in seg.dropna(subset=["auc"]).itertuples():
                    mlflow.log_metric(f"auc_segmen_{r.segmen}", r.auc)
                mlflow.set_tags({"git_commit": komit_git()})
                mlflow.log_artifacts(a.keluaran)
            print("\nTercatat ke MLflow.")
        except ImportError:
            print("\nMLflow tidak terpasang; pencatatan dilewati.")


if __name__ == "__main__":
    main()
