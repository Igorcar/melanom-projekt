# Prepoznavanje melanoma na dermatoskopskim slikama primjenom prijenosnog učenja

Završni projekt programa **„Stručnjak/inja za umjetnu inteligenciju”**, AI centar Lipik (2025./2026.).
Autor: Igor (liječnik, ambulanta opće medicine / HMP).

> ⚠️ **Edukacijski projekt.** Model nije namijenjen kliničkoj uporabi i ne zamjenjuje dermatologa ni patohistološku dijagnozu.

---

## 1. Problem i zašto

Binarna klasifikacija dermatoskopske slike: **je li lezija melanom (1) ili nije (0)**.

Klinički cilj je **ne propustiti melanom**. Zato je glavna metrika **osjetljivost (recall) za melanom**. Lažno pozitivan nalaz (nepotrebno upućivanje dermatologu) prihvatljiviji je od lažno negativnog (propušten melanom). Točnost se navodi samo informativno: melanoma je oko 11 %, pa bi model koji uvijek kaže „nije melanom” imao oko 89 % točnosti i ne bi pronašao nijedan melanom.

## 2. Podaci

Podaci **nisu u repozitoriju** (veličina i licence). Preuzimaju se s izvora:

| Skup | Uloga | Izvor | Licenca |
|---|---|---|---|
| **HAM10000** | trening / validacija / test | Tschandl i sur., *Scientific Data* 2018; [Harvard Dataverse, doi:10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T); [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | CC BY-NC 4.0 |
| **HIBA Skin Lesions** | vanjska provjera 1 (Buenos Aires) | Hospital Italiano de Buenos Aires, ISIC Archive, [doi:10.34970/559884](https://doi.org/10.34970/559884) | CC BY |
| **BCN20000** | vanjska provjera 2 (Barcelona) – završni neovisni test | Hospital Clínic de Barcelona, [ISIC Archive](https://www.isic-archive.com) | CC BY-NC |

**Očekivana struktura mapa nakon preuzimanja:**
```
data/
  ham10000/  HAM10000_images_part_1/  HAM10000_images_part_2/  HAM10000_metadata.csv
  hiba/      images/  metadata.csv
  bcn20000/  ISIC-images/  bcn20000_metadata_<datum>.csv
```

**Glavne brojke (iz EDA-e):**
- **HAM10000:** 10 015 slika, 7 470 lezija; melanom 1 113 slika / 614 lezija (11,1 % slika), svi histopatološki potvrđeni; sve slike 600 × 450.
  Nevusi su samo u 37 % slučajeva histološki potvrđeni (ostalo praćenje / konsenzus).
- **HIBA** (samo dermatoskopske): 1 280 slika, 1 201 lezija, 589 pacijenata; melanom 196 slika (15,3 %); 181 različita veličina slike; tip kože po Fitzpatricku I–IV (tipova V–VI nema).
- **BCN20000:** isključeno 1 156 slika bez dijagnoze i 633 metastaze melanoma; jedna slika po leziji → 4 848 lezija, 695 melanoma (14,3 %); sve slike 1024 × 1024.

## 3. Pristup / model

1. **Podjela HAM10000 po leziji** (`podijeli_ham.py`): 70 / 15 / 15 % lezija (trening / validacija / test), stratificirano po melanomu, seed 42.
   Sve slike iste lezije u istom skupu, da model na testu ne „prepoznaje” lezije iz treninga (curenje podataka).
2. **Prijenosno učenje** (`train_melanom.py`): ResNet-18 s ImageNet težinama, novi zadnji sloj 512 → 2, uči se cijela mreža; Adam (lr = 1e-4), batch 32, 10 epoha, CPU.
   - `CrossEntropyLoss` s **težinama klasa** izračunatima samo iz trening skupa (ostalo 0,56; melanom 4,46).
   - Augmentacija samo u treningu: `RandomResizedCrop`, horizontalno i vertikalno zrcaljenje, rotacija ±30°, blagi `ColorJitter`. **Bez `RandomGrayscale`** – boja je dijagnostički važna (ABCD pravilo).
   - Validacija / test: `Resize(256)` + `CenterCrop(224)` (čuva omjer stranica, tj. oblik lezije).
   - Sprema se epoha s **najboljim validacijskim AUC-om** (ne točnošću).
3. **Prag odluke** (`evaluate_melanom.py`): biran **samo na validaciji** – najviši prag uz osjetljivost ≥ 95 % (rezerva, jer osjetljivost na novim podacima obično padne). Prag se zaključa (`prag.json`) i jednom primijeni na test, HIBA-u i BCN20000.
4. **Vanjske provjere** (`evaluate_hiba.py`, `evaluate_bcn.py`): isti model i isti prag, ništa se ne podešava.
5. **Analiza:** lokalna kalibracija praga na HIBA-i, analiza pogrešaka, Grad-CAM, demo (u bilježnici).

Sve odluke zapisane su **prije** evaluacije. Razvojna verzija 1 otkrila je da prag biran točno na 90 % osjetljivosti na testu daje 84 %, pa je u konačnoj verziji uvedena rezerva (95 %).

## 4. Rezultati

Model iz 6. epohe (val AUC 0,902). Prag = **0,289**. Postupak je **reproducibilan**: dva neovisna treniranja dala su identične rezultate.

### HAM10000 test (1 504 slike, 166 melanoma)

| Prag | Osjetljivost | Specifičnost | Preciznost | F1 | Balanced acc. | AUC |
|---|---|---|---|---|---|---|
| 0,5 | 0,837 (139/166) | 0,762 | 0,304 | 0,446 | 0,800 | 0,894 |
| **0,289** | **0,886 (147/166)** | 0,673 | 0,252 | 0,392 | 0,780 | 0,894 |

### Jedan model, tri bolnice (prag 0,289)

| Skup | AUC | Osjetljivost | Specifičnost |
|---|---|---|---|
| HAM10000 test (isti izvor kao trening) | 0,894 | 0,886 | 0,673 |
| HIBA – Buenos Aires | 0,823 | 0,719 | 0,769 |
| BCN20000 – Barcelona | 0,754 | 0,439 | 0,879 |

**Lokalna kalibracija praga na HIBA-i** (pacijenti podijeljeni na pola): lokalni prag 0,067 (umjesto 0,289) na drugoj polovici vraća osjetljivost s 0,737 na **0,926**, ali specifičnost pada s 0,760 na 0,501.

**Tip kože (HIBA):** osjetljivost I–II 0,740 (125/169), III–IV 0,647 (11/17 – premalo melanoma za zaključak); tipova V–VI nema u podacima.

**Analiza pogrešaka i Grad-CAM:**
- Model griješi sa sigurnošću: dio propuštenih melanoma ima vjerojatnost ≈ 0.
- Velik dio lažnih uzbuna na HIBA-i su drugi karcinomi kože (BCC i SCC: 124 od 250).
- Grad-CAM: kod pogođenih melanoma model gleda leziju; kod propuštenih često nigdje ne nalazi dokaz za melanom; u pregledanom uzorku ne koristi mjerilo, rub dermatoskopa ni dlake.

### Zaključak

Model dobro radi na podacima iz istog izvora, ali u drugim bolnicama znatno slabije prepoznaje melanom. **Raznolikost podataka važnija je od same mreže**; model bi se prije svake primjene morao lokalno provjeriti i kalibrirati, a za pouzdan rad trenirati na podacima iz više ustanova.

Grafovi i tablice: mapa `rezultati/` (EDA, evaluacija, HIBA, BCN, analiza pogrešaka, Grad-CAM, demo).

## 5. Ograničenja

- Retrospektivni javni podaci; HAM10000 pretežno svijetla koža (Austrija, Australija); tamna koža (V–VI) nije zastupljena ni u vanjskim skupovima.
- Samo dermatoskopske slike; oznake nevusa u HAM10000 dijelom bez histološke potvrde.
- HAM10000 nema `patient_id`: podjela po leziji, ne po pacijentu.
- Prag biran na malom broju melanoma (164 na validaciji); rezultati se izvješćuju po slici (na BCN-u jedna slika po leziji).
- Model ne zamjenjuje dermatologa ni patohistologiju.

**Što bi se moglo poboljšati:** trening na podacima iz više bolnica (npr. HAM + BCN + ISIC), lokalna kalibracija u svakoj ustanovi, skupovi s tamnom kožom, veći model ili dulje fino podešavanje uz GPU.

## 6. Upute za pokretanje

**Okruženje:** Python 3.12, CPU je dovoljan (Windows 11 testirano).
```bash
conda create -n melanom python=3.12
conda activate melanom
pip install -r requirements.txt
```

**Redoslijed** (bilježnica `melanom_v2.ipynb` vodi kroz sve korake):
1. Preuzmi podatke u mapu `data/` (vidi poglavlje 2).
2. `python podijeli_ham.py` – podjela po leziji, kopira slike u `data_split/`.
3. `python train_melanom.py` – treniranje (najprije s `PROBNI_RAD = True`, zatim `False`; ~50 min na CPU). Model: `runs/melanom_resnet18/best_model.pt`.
4. `python evaluate_melanom.py` – odabir praga na validaciji i jedna provjera na testu.
5. `python evaluate_hiba.py` i `python evaluate_bcn.py` – vanjske provjere (BCN zahtijeva popis iz bilježnice, ćelija 10b).
6. Bilježnica: lokalna kalibracija, analiza pogrešaka, Grad-CAM i demo.

Spremljeni model (`runs/melanom_resnet18/best_model.pt`, ~45 MB) uključen je u repozitorij, pa se evaluacija i demo mogu pokrenuti bez treniranja.

## 7. Struktura repozitorija

```
melanom_v2.ipynb        bilježnica: EDA, pokretanje skripti, kalibracija, analiza pogrešaka, Grad-CAM, demo
podijeli_ham.py         podjela HAM10000 po leziji
train_melanom.py        treniranje ResNet-18
evaluate_melanom.py     prag na validaciji + test
evaluate_hiba.py        vanjska provjera HIBA
evaluate_bcn.py         završni test BCN20000
requirements.txt
runs/melanom_resnet18/  best_model.pt, TensorBoard zapisi
rezultati/              grafovi, tablice, predikcije, prag.json, podjela_ham.csv
```
