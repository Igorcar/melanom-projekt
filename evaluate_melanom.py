#
# Evaluacija modela za melanom
# 1) VALIDACIJA: biram prag tako da osjetljivost za melanom bude >= 95 % (rezerva) -> prag se ZAKLJUČA (prag.json)
# 2) TEST: jedna provjera, s pragom 0,5 i s odabranim pragom
# 3) ROC krivulje, matrice zabune, tablica metrika i predikcije po slici (za analizu pogrešaka)
#
import json                                   # spremanje praga u datoteku
from pathlib import Path                      # putanje do mapa i datoteka

import matplotlib.pyplot as plt               # grafovi
import pandas as pd                           # tablice
import torch                                  # PyTorch
import torch.nn as nn                         # slojevi mreže
from torch.utils.data import DataLoader       # batchevi slika
from torchvision import transforms            # obrada slika
from torchvision.datasets import ImageFolder  # slike iz mapa po klasama
from torchvision.models import resnet18, ResNet18_Weights   # ResNet-18
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay   # metrike


#
# GLOBALNE POSTAVKE
#
MODEL_PATH = Path("runs") / "melanom_resnet18" / "best_model.pt"   # najbolji model iz treninga
VAL_DIR = Path("data_split") / "val"                               # validacijske slike
TEST_DIR = Path("data_split") / "test"                             # testne slike
REZ_DIR = Path("rezultati") / "evaluacija"                         # kamo spremamo rezultate
PRAG_PATH = REZ_DIR / "prag.json"                                  # datoteka sa zaključanim pragom

CILJNA_OSJETLJIVOST = 0.95    # cilj na validaciji s rezervom (v1: 0.90 dao je 84 % na testu)
BATCH_SIZE = 32               # slika po batchu
NUM_WORKERS = 4               # pomoćni procesi za učitavanje
MELANOM = 1                   # indeks klase 1_melanom


def get_eval_transform():
    # ista obrada kao validacija u treningu: bez augmentacije, čuva omjer stranica
    weights = ResNet18_Weights.DEFAULT
    return transforms.Compose([
        transforms.Resize(256),                # kraća stranica -> 256 piksela
        transforms.CenterCrop(224),            # središnji izrez 224 × 224
        transforms.ToTensor(),                 # slika -> tenzor
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),   # ImageNet normalizacija
    ])


def load_model(device):
    ckpt = torch.load(MODEL_PATH, map_location=device)            # rječnik koji je spremio train_melanom.py
    model = resnet18(weights=None)                                # prazan ResNet-18 (težine dolaze iz datoteke)
    model.fc = nn.Linear(model.fc.in_features, ckpt["num_classes"])   # isti zadnji sloj kao u treningu (512 -> 2)
    model.load_state_dict(ckpt["model_state_dict"])               # naše naučene težine
    model.to(device)                                              # model na uređaj (CPU)
    model.eval()                                                  # način rada za procjenu
    print(f"Učitan model iz epohe {ckpt['epoch']} (val_AUC={ckpt['val_auc']:.4f})")
    return model


@torch.no_grad()                                                  # bez računanja gradijenata
def predict(model, data_dir, device):
    # vraća tablicu: putanja slike, točna oznaka, vjerojatnost melanoma
    ds = ImageFolder(root=str(data_dir), transform=get_eval_transform())                       # slike iz mapa
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)     # stalan redoslijed

    vjerojatnosti = []                                            # skorovi za melanom
    for x, _ in loader:                                           # batch po batch
        logits = model(x.to(device))                              # 2 broja po slici
        vjerojatnosti.extend(torch.softmax(logits, dim=1)[:, MELANOM].cpu().tolist())   # skor za melanom

    putanje = [putanja for putanja, _ in ds.samples]     # shuffle=False -> isti redoslijed kao predikcije
    oznake = [oznaka for _, oznaka in ds.samples]        # prave oznake (0 ili 1)
    return pd.DataFrame({"putanja": putanje, "oznaka": oznake, "p_melanom": vjerojatnosti})


def odaberi_prag(df):
    # roc_curve daje pragove od najvišeg prema najnižem; osjetljivost (tpr) pri tome raste.
    # Uzimam PRVI (najviši) prag pri kojem osjetljivost dosegne cilj -> uz to je specifičnost najveća.
    fpr, tpr, pragovi = roc_curve(df["oznaka"], df["p_melanom"])   # sve točke ROC krivulje
    for i in range(len(tpr)):                                      # od najvišeg praga prema najnižem
        if tpr[i] >= CILJNA_OSJETLJIVOST:                          # prvi put kad osjetljivost ≥ 95 %
            return float(pragovi[i])                               # vrati taj prag


def izracunaj_metrike(df, prag):
    pred = (df["p_melanom"] >= prag).astype(int)           # 1 = model kaže melanom
    tn, fp, fn, tp = confusion_matrix(df["oznaka"], pred, labels=[0, 1]).ravel()   # četiri polja matrice zabune

    osjetljivost = tp / (tp + fn)                    # od svih melanoma, koliko je pronađeno
    specificnost = tn / (tn + fp)                    # od svih ne-melanoma, koliko je točno "ostalo"
    preciznost = tp / (tp + fp)                      # od svih "melanom" nalaza, koliko je stvarno melanom
    f1 = 2 * preciznost * osjetljivost / (preciznost + osjetljivost)   # sklad preciznosti i osjetljivosti

    return {
        "prag": round(prag, 4),
        "TP": int(tp), "FN": int(fn), "TN": int(tn), "FP": int(fp),
        "osjetljivost": round(osjetljivost, 4),
        "specificnost": round(specificnost, 4),
        "preciznost": round(preciznost, 4),
        "F1": round(f1, 4),
        "balanced_acc": round((osjetljivost + specificnost) / 2, 4),   # prosjek osjetljivosti i specifičnosti
        "tocnost": round((tp + tn) / len(df), 4),            # samo informativno
        "AUC": round(roc_auc_score(df["oznaka"], df["p_melanom"]), 4),   # ne ovisi o pragu
    }


def nacrtaj_matricu(df, prag, naslov, ax):
    pred = (df["p_melanom"] >= prag).astype(int)                       # odluka uz zadani prag
    cm = confusion_matrix(df["oznaka"], pred, labels=[0, 1])           # matrica zabune 2 × 2
    ConfusionMatrixDisplay(cm, display_labels=["ostalo", "melanom"]).plot(ax=ax, colorbar=False, cmap="Blues")   # nacrtaj
    ax.set_title(naslov)                                               # naslov
    ax.set_xlabel("predviđeno")                                        # x-os: odluka modela
    ax.set_ylabel("stvarno")                                           # y-os: prava dijagnoza


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # GPU ako postoji, inače CPU
    REZ_DIR.mkdir(parents=True, exist_ok=True)                              # mapa za rezultate
    model = load_model(device)                                              # učitaj najbolji model

    # 1) Predikcije na validaciji i testu
    df_val = predict(model, VAL_DIR, device)                                # skorovi za validaciju
    df_test = predict(model, TEST_DIR, device)                              # skorovi za test
    print(f"Validacija: {len(df_val)} slika | Test: {len(df_test)} slika")

    # 2) Prag biram SAMO na validaciji i zaključavam ga
    prag = odaberi_prag(df_val)                                             # prag uz osjetljivost ≥ 95 %
    with open(PRAG_PATH, "w", encoding="utf-8") as f:                       # zapiši prag u datoteku
        json.dump({"prag": prag, "ciljna_osjetljivost": CILJNA_OSJETLJIVOST, "izabran_na": "validacija"}, f, indent=2)
    print(f"Odabrani prag (validacija, osjetljivost >= {CILJNA_OSJETLJIVOST:.0%}): {prag:.4f} -> {PRAG_PATH}")

    # 3) Metrike: validacija i test, prag 0,5 i odabrani prag
    redovi = []
    for ime_skupa, df in [("validacija", df_val), ("test", df_test)]:       # oba skupa
        for p in [0.5, prag]:                                               # oba praga
            m = izracunaj_metrike(df, p)                                    # sve metrike
            m["skup"] = ime_skupa                                           # dopiši ime skupa
            redovi.append(m)
    tablica = pd.DataFrame(redovi)                                          # 4 retka: 2 skupa × 2 praga
    print("\n" + tablica.to_string(index=False))
    tablica.to_csv(REZ_DIR / "metrike.csv", index=False)                    # spremi tablicu

    # 4) ROC krivulje (validacija i test)
    plt.figure(figsize=(6, 6))
    for ime_skupa, df in [("validacija", df_val), ("test", df_test)]:
        fpr, tpr, _ = roc_curve(df["oznaka"], df["p_melanom"])              # točke krivulje
        auc = roc_auc_score(df["oznaka"], df["p_melanom"])                  # površina ispod krivulje
        plt.plot(fpr, tpr, label=f"{ime_skupa} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="slučajni model")   # dijagonala = pogađanje
    plt.xlabel("1 − specifičnost (lažno pozitivni udio)")
    plt.ylabel("osjetljivost")
    plt.title("ROC krivulja – melanom nasuprot ostalom")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REZ_DIR / "roc_val_test.png", dpi=150)                     # spremi graf
    plt.close()                                                             # zatvori (skripta ne prikazuje)

    # 5) Matrice zabune na TESTU: prag 0,5 i odabrani prag
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))                       # dvije matrice jedna uz drugu
    nacrtaj_matricu(df_test, 0.5, "Test – prag 0,5", axes[0])
    nacrtaj_matricu(df_test, prag, f"Test – odabrani prag {prag:.3f}", axes[1])
    plt.tight_layout()
    plt.savefig(REZ_DIR / "matrice_zabune_test.png", dpi=150)
    plt.close()

    # 6) Predikcije po slici (za analizu pogrešaka)
    for ime_skupa, df in [("val", df_val), ("test", df_test)]:
        df["odluka"] = (df["p_melanom"] >= prag).astype(int)                # odluka uz zaključani prag
        df.to_csv(REZ_DIR / f"predikcije_{ime_skupa}.csv", index=False)     # spremi tablicu

    print(f"\nSpremljeno u {REZ_DIR}")


if __name__ == "__main__":     # pokreće se samo kad skriptu pokrenemo izravno
    main()
