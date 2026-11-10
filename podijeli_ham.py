#
# PODJELA HAM10000 NA TRENING (70 %), VALIDACIJU (15 %) I TEST (15 %) – PO LEZIJI
# Sve slike iste lezije idu u isti skup (inače bi model na testu "prepoznao" lezije iz treninga).
# Stratificirano po melanomu: u svakom skupu podjednak udio melanoma.
# Slike se kopiraju u data_split/<skup>/<klasa>, da ImageFolder radi kao u prošlom projektu.
#
import shutil                                          # kopiranje datoteka
from pathlib import Path                               # putanje do mapa i datoteka

import pandas as pd                                    # tablice
from sklearn.model_selection import train_test_split  # stratificirana podjela (DL6, ML)


#                             POSTAVKE
HAM_DIR = Path("data") / "ham10000"                    # izvorne HAM10000 slike
HAM_CSV = HAM_DIR / "HAM10000_metadata.csv"            # tablica metapodataka
SPLIT_DIR = Path("data_split")                         # ovdje nastaju mape za ImageFolder
PODJELA_CSV = Path("rezultati") / "podjela_ham.csv"    # popis: koja slika je u kojem skupu
VAL_RATIO = 0.15                                       # 15 % lezija za validaciju
TEST_RATIO = 0.15                                      # 15 % lezija za test
SEED = 42                                              # ista podjela pri svakom pokretanju
MAPE_KLASA = {0: "0_ostalo", 1: "1_melanom"}           # broj ispred imena -> ImageFolder daje melanomu indeks 1


def main():
    if SPLIT_DIR.exists():                                # zaštita: ne dijeli dvaput
        print("Mapa data_split već postoji – podjela je već napravljena.")
        return                                            # izlaz iz funkcije, ništa se ne mijenja

    # 1) učitavanje i oznaka melanoma
    ham = pd.read_csv(HAM_CSV)                            # jedna slika = jedan redak
    ham["melanom"] = (ham["dx"] == "mel").astype(int)     # 1 = melanom, 0 = ostalo

    # 2) tablica LEZIJA: jedan redak = jedna lezija (sve njezine slike imaju istu dijagnozu)
    lezije = ham.groupby("lesion_id")["melanom"].first().reset_index()

    # 3) dijelimo lezije: prvo 15 % za test, pa iz ostatka 15 % svih lezija za validaciju
    ostatak, test = train_test_split(lezije, test_size=TEST_RATIO,
                                     stratify=lezije["melanom"], random_state=SEED)   # odvoji test
    val_udio = VAL_RATIO / (1 - TEST_RATIO)            # 0.15 / 0.85 ≈ 0.176 ostatka = 15 % svih
    train, val = train_test_split(ostatak, test_size=val_udio,
                                  stratify=ostatak["melanom"], random_state=SEED)     # ostatak -> trening i validacija

    # 4) svaka slika dobiva skup svoje lezije
    ham["skup"] = ""                                                           # prazan stupac
    ham.loc[ham["lesion_id"].isin(train["lesion_id"]), "skup"] = "train"       # slike trening lezija
    ham.loc[ham["lesion_id"].isin(val["lesion_id"]), "skup"] = "val"           # slike validacijskih lezija
    ham.loc[ham["lesion_id"].isin(test["lesion_id"]), "skup"] = "test"         # slike testnih lezija

    # 5) provjera: nijedna lezija ne smije biti u dva skupa
    skupova_po_leziji = ham.groupby("lesion_id")["skup"].nunique()             # u koliko skupova je svaka lezija
    if (skupova_po_leziji > 1).sum() > 0 or (ham["skup"] == "").sum() > 0:     # lezija u 2 skupa ili slika bez skupa
        raise ValueError("Greška u podjeli: lezija u više skupova ili slika bez skupa!")   # zaustavi program
    print("Provjera: svaka lezija je u točno jednom skupu. OK\n")

    # 6) pregled: slike i lezije po skupu i klasi
    print("SLIKE po skupu (0 = ostalo, 1 = melanom):")
    print(pd.crosstab(ham["skup"], ham["melanom"], margins=True, margins_name="ukupno"))   # skup × klasa – slike
    lez = ham.drop_duplicates(subset="lesion_id")                                         # jedan redak po leziji
    print("\nLEZIJE po skupu (0 = ostalo, 1 = melanom):")
    print(pd.crosstab(lez["skup"], lez["melanom"], margins=True, margins_name="ukupno"))   # skup × klasa – lezije

    # 7) spremi popis podjele (ponovljivost)
    PODJELA_CSV.parent.mkdir(parents=True, exist_ok=True)                                          # napravi mapu rezultati
    ham[["image_id", "lesion_id", "dx", "dx_type", "melanom", "skup"]].to_csv(PODJELA_CSV, index=False)   # spremi popis
    print(f"\nPopis podjele spremljen: {PODJELA_CSV}")

    # 8) kopiraj slike u data_split/<skup>/<klasa>/ (izvornici ostaju)
    putanje = {p.stem: p for p in HAM_DIR.rglob("*.jpg")}                      # kazalo: image_id -> putanja
    for _, red in ham.iterrows():                                              # za svaku sliku
        odrediste = SPLIT_DIR / red["skup"] / MAPE_KLASA[red["melanom"]]       # npr. data_split/train/1_melanom
        odrediste.mkdir(parents=True, exist_ok=True)                           # napravi mapu ako ne postoji
        shutil.copy(putanje[red["image_id"]], odrediste / f'{red["image_id"]}.jpg')   # kopiraj sliku

    print("\nBroj slika po mapi:")
    for mapa in sorted(SPLIT_DIR.glob("*/*")):                                 # svaka mapa skup/klasa
        print(f"   {mapa}: {len(list(mapa.glob('*.jpg')))}")                   # broj slika u njoj
    print("Gotovo.")


if __name__ == "__main__":     # pokreće se samo kad skriptu pokrenemo izravno (python podijeli_ham.py)
    main()
