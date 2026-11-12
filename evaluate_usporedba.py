#
# Evaluacija usporednih modela – ISTO PRAVILO kao glavni model:
# prag biran na HAM validaciji (najviši prag uz osjetljivost ≥ 95 %), zatim primijenjen na test, HIBA i BCN.
# Pokretanje: python evaluate_usporedba.py glavni | zamrznuti | seed_123 | seed_2024
# Rezultati: rezultati/usporedba/metrike_<varijanta>.csv  (glavni rezultati i prag.json se NE diraju)
#
import sys                                     # ime varijante iz naredbenog retka
from pathlib import Path                       # putanje

import pandas as pd                            # tablice
import torch                                   # PyTorch
import torch.nn as nn                          # zadnji sloj
from torchvision.models import resnet18        # ResNet-18

from evaluate_melanom import predict, odaberi_prag, izracunaj_metrike, VAL_DIR, TEST_DIR   # isti postupak
from evaluate_hiba import predict_hiba, ucitaj_hiba                                        # HIBA
from evaluate_bcn import predict_bcn, BCN_POPIS                                            # BCN (1 slika po leziji)

REZ_DIR = Path("rezultati") / "usporedba"      # zasebna mapa za usporedbe


def ucitaj_model(putanja, device):
    ckpt = torch.load(putanja, map_location=device)                        # spremljeni rječnik
    model = resnet18(weights=None)                                         # prazan ResNet-18
    model.fc = nn.Linear(model.fc.in_features, ckpt["num_classes"])        # zadnji sloj 512 -> 2
    model.load_state_dict(ckpt["model_state_dict"])                        # naučene težine
    model.to(device)
    model.eval()
    print(f"Učitan {putanja} (epoha {ckpt['epoch']}, val_AUC={ckpt['val_auc']:.4f})")
    return model


def main():
    varijanta = sys.argv[1]                                                # npr. "zamrznuti"
    if varijanta == "glavni":
        putanja = Path("runs") / "melanom_resnet18" / "best_model.pt"      # glavni model (seed 42)
    else:
        putanja = Path("runs") / ("usporedba_" + varijanta) / "best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REZ_DIR.mkdir(parents=True, exist_ok=True)

    model = ucitaj_model(putanja, device)
    df_val = predict(model, VAL_DIR, device)                               # skorovi na validaciji
    prag = odaberi_prag(df_val)                                            # isto pravilo: osjetljivost ≥ 95 %
    print(f"Prag (validacija): {prag:.4f}")

    skupovi = [("HAM validacija", df_val),
               ("HAM test", predict(model, TEST_DIR, device)),
               ("HIBA", predict_hiba(model, ucitaj_hiba(), device)),
               ("BCN20000", predict_bcn(model, pd.read_csv(BCN_POPIS), device))]

    redovi = []
    for ime, df in skupovi:
        m = izracunaj_metrike(df, prag)                                    # iste formule kao za glavni model
        m["skup"] = ime
        m["model"] = varijanta
        redovi.append(m)
    tablica = pd.DataFrame(redovi)
    print("\n" + tablica.to_string(index=False))
    tablica.to_csv(REZ_DIR / f"metrike_{varijanta}.csv", index=False)
    print(f"\nSpremljeno: {REZ_DIR / f'metrike_{varijanta}.csv'}")


if __name__ == "__main__":
    main()
