#
# ZAVRŠNI NEOVISNI TEST na BCN20000 (Barcelona) – skup koji nikad prije nije gledan
# Isti model i ISTI prag (prag.json, izabran na HAM validaciji). Jedna slika po leziji (popis iz bilježnice).
#
import json                                   # čitanje praga
from pathlib import Path                      # putanje

import matplotlib.pyplot as plt               # grafovi
import pandas as pd                           # tablice
import torch                                  # PyTorch
from torch.utils.data import DataLoader       # batchevi slika
from sklearn.metrics import roc_curve, roc_auc_score   # ROC i AUC

from evaluate_melanom import load_model, get_eval_transform, izracunaj_metrike, nacrtaj_matricu, MELANOM, PRAG_PATH
from evaluate_hiba import HibaDataset      # isti Dataset: radi za svaku tablicu sa stupcima isic_id i oznaka


#
# GLOBALNE POSTAVKE
#
BCN_DIR = Path("data") / "bcn20000"                                        # BCN slike
BCN_POPIS = Path("rezultati") / "eda" / "bcn_test_popis.csv"               # popis iz ćelije 10b
HAM_TEST_PRED = Path("rezultati") / "evaluacija" / "predikcije_test.csv"   # za usporedni ROC
HIBA_PRED = Path("rezultati") / "hiba" / "predikcije_hiba.csv"             # za usporedni ROC
REZ_DIR = Path("rezultati") / "bcn"                                        # kamo spremamo rezultate
BATCH_SIZE = 32
NUM_WORKERS = 4


@torch.no_grad()                                                           # bez računanja gradijenata
def predict_bcn(model, bcn, device):
    putanje = {p.stem: p for p in BCN_DIR.rglob("*.jpg")}                  # kazalo: isic_id -> putanja
    ds = HibaDataset(bcn, putanje, get_eval_transform())                   # skup slika
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    vjerojatnosti = []
    for x, _ in loader:                                                    # batch po batch
        logits = model(x.to(device))                                       # 2 broja po slici
        vjerojatnosti.extend(torch.softmax(logits, dim=1)[:, MELANOM].cpu().tolist())   # skor za melanom

    bcn["p_melanom"] = vjerojatnosti                                       # dopiši skor u tablicu
    return bcn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REZ_DIR.mkdir(parents=True, exist_ok=True)

    with open(PRAG_PATH, encoding="utf-8") as f:
        prag = json.load(f)["prag"]                                        # ZAKLJUČANI prag s HAM validacije
    print(f"Prag iz {PRAG_PATH}: {prag:.4f}")

    model = load_model(device)                                             # isti model
    bcn = predict_bcn(model, pd.read_csv(BCN_POPIS), device)               # skorovi za sve BCN lezije
    bcn["odluka"] = (bcn["p_melanom"] >= prag).astype(int)                 # odluka uz zaključani prag
    print(f"BCN lezija: {len(bcn)} | melanoma: {bcn['oznaka'].sum()}")

    # 1) Ukupno: prag 0,5 i zaključani prag
    redovi = []
    for p in [0.5, prag]:
        m = izracunaj_metrike(bcn, p)                                      # iste formule kao za HAM
        m["skup"] = "BCN20000"
        redovi.append(m)
    ukupno = pd.DataFrame(redovi)
    print("\nUKUPNO:\n" + ukupno.to_string(index=False))
    ukupno.to_csv(REZ_DIR / "metrike_bcn.csv", index=False)

    # 2) Po dijagnozi
    po_dijagnozi = pd.crosstab(bcn["dg"], bcn["odluka"], margins=True, margins_name="ukupno")
    print("\nPO DIJAGNOZI (0 = ostalo, 1 = model kaže melanom):\n", po_dijagnozi)
    po_dijagnozi.to_csv(REZ_DIR / "bcn_po_dijagnozi.csv")

    # 3) ROC: sva tri skupa
    plt.figure(figsize=(6, 6))
    for ime, df in [("HAM10000 test", pd.read_csv(HAM_TEST_PRED)),
                    ("HIBA (Buenos Aires)", pd.read_csv(HIBA_PRED)),
                    ("BCN20000 (Barcelona)", bcn)]:
        fpr, tpr, _ = roc_curve(df["oznaka"], df["p_melanom"])            # točke krivulje
        auc = roc_auc_score(df["oznaka"], df["p_melanom"])                # AUC
        plt.plot(fpr, tpr, label=f"{ime} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="slučajni model")
    plt.xlabel("1 − specifičnost")
    plt.ylabel("osjetljivost")
    plt.title("ROC – jedan model, tri bolnice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REZ_DIR / "roc_tri_skupa.png", dpi=150)
    plt.close()

    # 4) Matrica zabune BCN
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    nacrtaj_matricu(bcn, prag, f"BCN20000 – prag {prag:.3f}", ax)
    plt.tight_layout()
    plt.savefig(REZ_DIR / "matrica_zabune_bcn.png", dpi=150)
    plt.close()

    # 5) Predikcije po leziji
    bcn.to_csv(REZ_DIR / "predikcije_bcn.csv", index=False)               # spremi tablicu
    print(f"\nSpremljeno u {REZ_DIR}")


if __name__ == "__main__":     # pokreće se samo kad skriptu pokrenemo izravno
    main()
