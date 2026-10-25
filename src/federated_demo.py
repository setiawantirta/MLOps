"""federated_demo.py — simulasi federated learning (FedAvg) tanpa pustaka tambahan.

Setiap "koperasi" melatih model logistik lokal dengan datanya sendiri; hanya
PARAMETER yang dikirim ke server, lalu dirata-ratakan berbobot jumlah data.
Data mentah tidak pernah berpindah.
"""
from __future__ import annotations
import sys, os
import numpy as np, pandas as pd, yaml
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data as D, features as F

CFG = yaml.safe_load(open("conf/params.yaml", encoding="utf-8"))
PUTARAN = 15
EPOCH_LOKAL = 3


def buat_peserta(df: pd.DataFrame, n: int = 5, seed: int = 0):
    """Partisi NON-IID: tiap peserta didominasi satu segmen, seperti koperasi nyata."""
    segmen = sorted(df["segmen"].unique())[:n]
    rng = np.random.default_rng(seed)
    peserta = []
    for i, s in enumerate(segmen):
        inti = df[df.segmen == s]
        lain = df[df.segmen != s].sample(frac=0.08, random_state=seed + i)
        peserta.append(pd.concat([inti, lain]).sample(frac=1.0, random_state=seed + i))
    return dict(zip([f"koperasi_{s}" for s in segmen], peserta))


def main():
    df = D.muat(CFG["data"]["mentah"])
    latih, _, uji = D.bagi_waktu(df, CFG["data"]["batas_latih"], CFG["data"]["batas_validasi"])
    art = F.pelajari(latih)
    scaler = StandardScaler().fit(F.terapkan(latih, art))

    Xu = scaler.transform(F.terapkan(uji, art)); yu = uji["gagal_bayar"].values
    peserta = buat_peserta(latih)

    # ---- pembanding 1: model terpusat (seluruh data disatukan) ----
    # Pembanding yang jujur: model terbaik yang DAPAT dicapai bila data disatukan.
    pusat = LogisticRegression(max_iter=3000, random_state=42)
    pusat.fit(scaler.transform(F.terapkan(latih, art)), latih["gagal_bayar"])
    auc_pusat = roc_auc_score(yu, pusat.decision_function(Xu))

    # ---- pembanding 2: model lokal masing-masing ----
    auc_lokal = {}
    for nama, d in peserta.items():
        m = LogisticRegression(max_iter=3000, random_state=42)
        m.fit(scaler.transform(F.terapkan(d, art)), d["gagal_bayar"])
        auc_lokal[nama] = roc_auc_score(yu, m.decision_function(Xu))

    # ---- federated averaging ----
    n_fitur = Xu.shape[1]
    w_global = np.zeros(n_fitur); b_global = np.zeros(1)
    bobot = np.array([len(d) for d in peserta.values()], dtype=float)
    bobot /= bobot.sum()
    riwayat = []

    for putaran in range(1, PUTARAN + 1):
        Ws, Bs = [], []
        for nama, d in peserta.items():
            m = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=EPOCH_LOKAL,
                              learning_rate="constant", eta0=0.01,
                              warm_start=True, random_state=42)
            m.coef_ = w_global.reshape(1, -1).copy()
            m.intercept_ = b_global.copy()
            m.classes_ = np.array([0, 1])
            X = scaler.transform(F.terapkan(d, art)); y = d["gagal_bayar"].values
            m.partial_fit(X, y, classes=np.array([0, 1]))
            Ws.append(m.coef_.ravel()); Bs.append(m.intercept_)
        # yang dipertukarkan HANYA parameter ini — bukan satu baris data pun
        w_global = np.average(np.vstack(Ws), axis=0, weights=bobot)
        b_global = np.average(np.vstack(Bs), axis=0, weights=bobot)
        auc = roc_auc_score(yu, Xu @ w_global + b_global)
        riwayat.append({"putaran": putaran, "auc_global": round(float(auc), 4)})

    print("=== Federated averaging: AUC model global per putaran ===")
    print(pd.DataFrame(riwayat).to_string(index=False))
    print(f"\nAUC model TERPUSAT (data disatukan)     : {auc_pusat:.4f}")
    print(f"AUC model FEDERATED (putaran terakhir)  : {riwayat[-1]['auc_global']:.4f}")
    print("\nAUC model LOKAL masing-masing peserta:")
    for nama, a in sorted(auc_lokal.items(), key=lambda x: -x[1]):
        print(f"  {nama:<28s} {a:.4f}")
    print("\nBacalah angka-angka ini sebagai PERTUKARAN NILAI, bukan sebagai kemenangan:")
    print("federated mengorbankan sebagian mutu demi tidak memindahkan data sama sekali.")
    print(f"Ukuran yang dipertukarkan per putaran: {n_fitur + 1} bilangan per peserta —")
    print("bandingkan dengan memindahkan seluruh catatan pembiayaan.")


if __name__ == "__main__":
    main()
