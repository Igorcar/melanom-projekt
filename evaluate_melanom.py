#
# Evaluacija modela za melanom
# 1) VALIDACIJA: biram prag tako da osjetljivost za melanom bude >= 95 % (rezerva) -> prag se ZAKLJUČA (prag.json)
# 2) TEST: jedna provjera, s pragom 0,5 i s odabranim pragom
# 3) ROC krivulje, matrice zabune, tablica metrika i predikcije po slici (za analizu pogrešaka)
#
import json                                   # spremanje praga u datoteku
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


#
# GLOBALNE POSTAVKE
#
MODEL_PATH = Path("runs") / "melanom_resnet18" / "best_model.pt"
VAL_DIR = Path("data_split") / "val"
TEST_DIR = Path("data_split") / "test"
REZ_DIR = Path("rezultati") / "evaluacija"
PRAG_PATH = REZ_DIR / "prag.json"

CILJNA_OSJETLJIVOST = 0.95    # cilj na validaciji s rezervom (v1: 0.90 dao je 84 % na testu)
BATCH_SIZE = 32
NUM_WORKERS = 4
MELANOM = 1                   # indeks klase 1_melanom


def get_eval_transform():
    # ista obrada kao validacija u treningu: bez augmentacije, čuva omjer stranica
    weights = ResNet18_Weights.DEFAULT
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])


def load_model(device):
    ckpt = torch.load(MODEL_PATH, map_location=device)            # rječnik koji je spremio train_melanom.py
    model = resnet18(weights=None)                                # prazan ResNet-18 (težine dolaze iz datoteke)
    model.fc = nn.Linear(model.fc.in_features, ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])               # naše naučene težine
    model.to(device)
    model.eval()
    print(f"Učitan model iz epohe {ckpt['epoch']} (val_AUC={ckpt['val_auc']:.4f})")
    return model


@torch.no_grad()
def predict(model, data_dir, device):
    # vraća tablicu: putanja slike, točna oznaka, vjerojatnost melanoma
    ds = ImageFolder(root=str(data_dir), transform=get_eval_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    vjerojatnosti = []
    for x, _ in loader:
        logits = model(x.to(device))
        vjerojatnosti.extend(torch.softmax(logits, dim=1)[:, MELANOM].cpu().tolist())

    putanje = [putanja for putanja, _ in ds.samples]     # shuffle=False -> isti redoslijed kao predikcije
    oznake = [oznaka for _, oznaka in ds.samples]
    return pd.DataFrame({"putanja": putanje, "oznaka": oznake, "p_melanom": vjerojatnosti})


def odaberi_prag(df):
    # roc_curve daje pragove od najvišeg prema najnižem; osjetljivost (tpr) pri tome raste.
    # Uzimam PRVI (najviši) prag pri kojem osjetljivost dosegne cilj -> uz to je specifičnost najveća.
    fpr, tpr, pragovi = roc_curve(df["oznaka"], df["p_melanom"])
    for i in range(len(tpr)):
        if tpr[i] >= CILJNA_OSJETLJIVOST:
            return float(pragovi[i])


def izracunaj_metrike(df, prag):
    pred = (df["p_melanom"] >= prag).astype(int)           # 1 = model kaže melanom
    tn, fp, fn, tp = confusion_matrix(df["oznaka"], pred, labels=[0, 1]).ravel()

    osjetljivost = tp / (tp + fn)                    # od svih melanoma, koliko je pronađeno
    specificnost = tn / (tn + fp)                    # od svih ne-melanoma, koliko je točno "ostalo"
    preciznost = tp / (tp + fp)                      # od svih "melanom" nalaza, koliko je stvarno melanom
    f1 = 2 * preciznost * osjetljivost / (preciznost + osjetljivost)

    return {
        "prag": round(prag, 4),
        "TP": int(tp), "FN": int(fn), "TN": int(tn), "FP": int(fp),
        "osjetljivost": round(osjetljivost, 4),
        "specificnost": round(specificnost, 4),
        "preciznost": round(preciznost, 4),
        "F1": round(f1, 4),
        "balanced_acc": round((osjetljivost + specificnost) / 2, 4),
        "tocnost": round((tp + tn) / len(df), 4),            # samo informativno
        "AUC": round(roc_auc_score(df["oznaka"], df["p_melanom"]), 4),
    }


def nacrtaj_matricu(df, prag, naslov, ax):
    pred = (df["p_melanom"] >= prag).astype(int)
    cm = confusion_matrix(df["oznaka"], pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["ostalo", "melanom"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(naslov)
    ax.set_xlabel("predviđeno")
    ax.set_ylabel("stvarno")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REZ_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(device)

    # 1) Predikcije na validaciji i testu
    df_val = predict(model, VAL_DIR, device)
    df_test = predict(model, TEST_DIR, device)
    print(f"Validacija: {len(df_val)} slika | Test: {len(df_test)} slika")

    # 2) Prag biram SAMO na validaciji i zaključavam ga
    prag = odaberi_prag(df_val)
    with open(PRAG_PATH, "w", encoding="utf-8") as f:
        json.dump({"prag": prag, "ciljna_osjetljivost": CILJNA_OSJETLJIVOST, "izabran_na": "validacija"}, f, indent=2)
    print(f"Odabrani prag (validacija, osjetljivost >= {CILJNA_OSJETLJIVOST:.0%}): {prag:.4f} -> {PRAG_PATH}")

    # 3) Metrike: validacija i test, prag 0,5 i odabrani prag
    redovi = []
    for ime_skupa, df in [("validacija", df_val), ("test", df_test)]:
        for p in [0.5, prag]:
            m = izracunaj_metrike(df, p)
            m["skup"] = ime_skupa
            redovi.append(m)
    tablica = pd.DataFrame(redovi)
    print("\n" + tablica.to_string(index=False))
    tablica.to_csv(REZ_DIR / "metrike.csv", index=False)

    # 4) ROC krivulje (validacija i test)
    plt.figure(figsize=(6, 6))
    for ime_skupa, df in [("validacija", df_val), ("test", df_test)]:
        fpr, tpr, _ = roc_curve(df["oznaka"], df["p_melanom"])
        auc = roc_auc_score(df["oznaka"], df["p_melanom"])
        plt.plot(fpr, tpr, label=f"{ime_skupa} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="slučajni model")
    plt.xlabel("1 − specifičnost (lažno pozitivni udio)")
    plt.ylabel("osjetljivost")
    plt.title("ROC krivulja – melanom nasuprot ostalom")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REZ_DIR / "roc_val_test.png", dpi=150)
    plt.close()

    # 5) Matrice zabune na TESTU: prag 0,5 i odabrani prag
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    nacrtaj_matricu(df_test, 0.5, "Test – prag 0,5", axes[0])
    nacrtaj_matricu(df_test, prag, f"Test – odabrani prag {prag:.3f}", axes[1])
    plt.tight_layout()
    plt.savefig(REZ_DIR / "matrice_zabune_test.png", dpi=150)
    plt.close()

    # 6) Predikcije po slici (za analizu pogrešaka)
    for ime_skupa, df in [("val", df_val), ("test", df_test)]:
        df["odluka"] = (df["p_melanom"] >= prag).astype(int)
        df.to_csv(REZ_DIR / f"predikcije_{ime_skupa}.csv", index=False)

    print(f"\nSpremljeno u {REZ_DIR}")


if __name__ == "__main__":
    main()
