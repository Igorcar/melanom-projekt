#
# Vanjska provjera generalizacije na HIBA (samo dermatoskopske slike)
# Isti model i ISTI prag kao za test (prag.json, izabran na HAM validaciji) – ništa se ne podešava na HIBA-i.
# Rezultati: ukupno, po tipu kože (Fitzpatrick), po dijagnozi; ROC HAM test nasuprot HIBA.
#
import json                                   # čitanje praga
from pathlib import Path                      # putanje

import matplotlib.pyplot as plt               # grafovi
import pandas as pd                           # tablice
import torch                                  # PyTorch
from PIL import Image                         # otvaranje slika
from torch.utils.data import Dataset, DataLoader   # vlastiti skup slika i batchevi
from sklearn.metrics import roc_curve, roc_auc_score   # ROC i AUC

# funkcije i postavke koje smo već napisali u evaluate_melanom.py (ista obrada i iste formule)
from evaluate_melanom import load_model, get_eval_transform, izracunaj_metrike, nacrtaj_matricu, MELANOM, PRAG_PATH


#
# GLOBALNE POSTAVKE
#
HIBA_DIR = Path("data") / "hiba"                                          # HIBA slike i tablica
HAM_TEST_PRED = Path("rezultati") / "evaluacija" / "predikcije_test.csv"  # HAM test – za usporedni ROC
REZ_DIR = Path("rezultati") / "hiba"                                      # kamo spremamo rezultate
BATCH_SIZE = 32
NUM_WORKERS = 4
SKUPINE_KOZE = {"I": "I–II", "II": "I–II", "III": "III–IV", "IV": "III–IV"}   # spajamo tipove kože u skupine


class HibaDataset(Dataset):
    # HIBA nije složena u mape po klasama, pa ImageFolder ne radi – vlastiti Dataset (DL20)
    def __init__(self, tablica, putanje, transform):
        self.tablica = tablica          # tablica s isic_id i oznakom
        self.putanje = putanje          # isic_id -> datoteka
        self.transform = transform      # obrada slike (kao na testu)

    def __len__(self):
        return len(self.tablica)        # broj slika u skupu

    def __getitem__(self, i):
        red = self.tablica.iloc[i]                                         # i-ti redak tablice
        slika = Image.open(self.putanje[red["isic_id"]]).convert("RGB")   # uvijek 3 kanala boje
        return self.transform(slika), int(red["oznaka"])                  # (obrađena slika, oznaka)


def ucitaj_hiba():
    hiba = pd.read_csv(HIBA_DIR / "metadata.csv")                              # HIBA tablica
    hiba = hiba[hiba["image_type"] == "dermoscopic"].reset_index(drop=True)   # samo dermatoskopske
    hiba["oznaka"] = (hiba["diagnosis"] == "melanoma").astype(int)           # 1 = melanom
    hiba["skupina_koze"] = hiba["fitzpatrick_skin_type"].map(SKUPINE_KOZE).fillna("nema podatka")   # I–II / III–IV / nema podatka
    return hiba


@torch.no_grad()                                                              # bez računanja gradijenata
def predict_hiba(model, hiba, device):
    putanje = {p.stem: p for p in HIBA_DIR.rglob("*.jpg")}                    # kazalo: isic_id -> putanja
    ds = HibaDataset(hiba, putanje, get_eval_transform())                     # skup slika
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)   # stalan redoslijed

    vjerojatnosti = []
    for x, _ in loader:                                                       # batch po batch
        logits = model(x.to(device))                                          # 2 broja po slici
        vjerojatnosti.extend(torch.softmax(logits, dim=1)[:, MELANOM].cpu().tolist())   # skor za melanom

    hiba["p_melanom"] = vjerojatnosti                                         # dopiši skor u tablicu
    return hiba


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # GPU ili CPU
    REZ_DIR.mkdir(parents=True, exist_ok=True)                               # mapa za rezultate

    with open(PRAG_PATH, encoding="utf-8") as f:
        prag = json.load(f)["prag"]                       # ZAKLJUČANI prag s HAM validacije
    print(f"Prag iz {PRAG_PATH}: {prag:.4f}")

    model = load_model(device)                                               # isti model kao za test
    hiba = predict_hiba(model, ucitaj_hiba(), device)                        # skorovi za sve HIBA slike
    hiba["odluka"] = (hiba["p_melanom"] >= prag).astype(int)                 # odluka uz zaključani prag
    print(f"HIBA dermatoskopskih slika: {len(hiba)} | melanoma: {hiba['oznaka'].sum()}")

    # 1) Ukupno: prag 0,5 i zaključani prag
    redovi = []
    for p in [0.5, prag]:                                                    # oba praga
        m = izracunaj_metrike(hiba, p)                                       # iste formule kao za HAM
        m["skup"] = "HIBA"
        redovi.append(m)
    ukupno = pd.DataFrame(redovi)
    print("\nUKUPNO:\n" + ukupno.to_string(index=False))
    ukupno.to_csv(REZ_DIR / "metrike_hiba.csv", index=False)

    # 2) Po tipu kože (zaključani prag)
    redovi = []
    for skupina in ["I–II", "III–IV", "nema podatka"]:                       # svaka skupina kože
        dio = hiba[hiba["skupina_koze"] == skupina]                          # samo slike te skupine
        tp = ((dio["odluka"] == 1) & (dio["oznaka"] == 1)).sum()             # pronađeni melanomi
        fn = ((dio["odluka"] == 0) & (dio["oznaka"] == 1)).sum()             # propušteni melanomi
        tn = ((dio["odluka"] == 0) & (dio["oznaka"] == 0)).sum()             # točno "ostalo"
        fp = ((dio["odluka"] == 1) & (dio["oznaka"] == 0)).sum()             # lažne uzbune
        redovi.append({"skupina_koze": skupina, "slike": len(dio), "melanomi": int(tp + fn),
                       "TP": int(tp), "FN": int(fn),
                       "osjetljivost": round(tp / (tp + fn), 3),
                       "specificnost": round(tn / (tn + fp), 3)})
    po_kozi = pd.DataFrame(redovi)
    print("\nPO TIPU KOŽE (zaključani prag):\n" + po_kozi.to_string(index=False))
    po_kozi.to_csv(REZ_DIR / "hiba_po_tipu_koze.csv", index=False)

    # 3) Po dijagnozi: koliko slika je model označio kao melanom (1) ili ostalo (0)
    po_dijagnozi = pd.crosstab(hiba["diagnosis"], hiba["odluka"], margins=True, margins_name="ukupno")
    print("\nPO DIJAGNOZI (0 = ostalo, 1 = model kaže melanom):\n", po_dijagnozi)
    po_dijagnozi.to_csv(REZ_DIR / "hiba_po_dijagnozi.csv")

    # 4) ROC: HAM test nasuprot HIBA
    ham_test = pd.read_csv(HAM_TEST_PRED)                                    # spremljene HAM predikcije
    plt.figure(figsize=(6, 6))
    for ime, df in [("HAM10000 test", ham_test), ("HIBA (vanjski)", hiba)]:
        fpr, tpr, _ = roc_curve(df["oznaka"], df["p_melanom"])               # točke krivulje
        auc = roc_auc_score(df["oznaka"], df["p_melanom"])                   # AUC
        plt.plot(fpr, tpr, label=f"{ime} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="slučajni model")
    plt.xlabel("1 − specifičnost")
    plt.ylabel("osjetljivost")
    plt.title("ROC – HAM10000 test i vanjska provjera HIBA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REZ_DIR / "roc_ham_hiba.png", dpi=150)
    plt.close()

    # 5) Matrica zabune HIBA (zaključani prag)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    nacrtaj_matricu(hiba, prag, f"HIBA – prag {prag:.3f}", ax)
    plt.tight_layout()
    plt.savefig(REZ_DIR / "matrica_zabune_hiba.png", dpi=150)
    plt.close()

    # 6) Predikcije po slici (za prilagodbu praga i analizu pogrešaka)
    hiba[["isic_id", "lesion_id", "patient_id", "diagnosis", "fitzpatrick_skin_type", "skupina_koze",
          "oznaka", "p_melanom", "odluka"]].to_csv(REZ_DIR / "predikcije_hiba.csv", index=False)
    print(f"\nSpremljeno u {REZ_DIR}")


if __name__ == "__main__":     # pokreće se samo kad skriptu pokrenemo izravno
    main()
