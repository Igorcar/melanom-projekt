# Plan verzije 2 – odluke zaključane UNAPRIJED (8. 11. 2026., revidirano isti dan prije ijedne evaluacije)

> **Napomena (10. 11. 2026.):** tekst ispod je izvorni plan, nepromijenjen. Naknadno, nakon vanjskih recenzija, dodane su opisne analize koje plan nije predviđao (razina lezije, intervali pouzdanosti, kalibracija i dr.); one ne mijenjaju model ni prag. „Lokalna kalibracija praga” danas se zove „lokalna prilagodba praga odluke”. Vidi `DNEVNIK_EKSPERIMENATA.md`.

Cilj: završni rad za ispit – kod što sličniji onome što smo učili (DL, ML, CV, prošli projekt), što jednostavniji, da ga Igor može obraniti red po red.
Nije znanstveni rad: bez naprednih statističkih metoda koje nismo učili.

Verzija 1 (rezultati u `Rezultati.md`) bila je razvojna: prag biran točno na 90 % osjetljivosti na validaciji na testu je dao 84 %; pomak domene na HIBA-i je velik.

## Isto kao v1 (namjerno se ne mijenja)
- Podjela HAM10000: po lesion_id, 70/15/15, stratificirano po melanomu, seed 42.
- Treniranje: ResNet-18 (ImageNet), fc 512→2, cijela mreža, Adam lr 1e-4, batch 32, 10 epoha, težine klasa iz treninga,
  ista augmentacija (RandomResizedCrop 0,7–1,0, H/V flip, rotacija ±30°, blagi ColorJitter, bez grayscale),
  val/test obrada Resize(256)+CenterCrop(224); spremanje epohe s najboljim val AUC-om.
- Izvješćivanje po slici (kao na predavanjima); razina lezije i intervali pouzdanosti se ne računaju – navode se kao ograničenje.

## Novo u v2
1. **Prag s rezervom:** na validaciji biramo najviši prag uz osjetljivost ≥ **95 %** (umjesto 90 %), da na novim podacima ostane oko 90 %.
   Za usporedbu se izvješćuje i prag 0,5. Prag se sprema u prag.json i ne mijenja nakon pogleda na test.
2. Uredna bilježnica i skripte, grafovi na hrvatskom, pokretanje `!python -u`.

## Glavni ishod
Osjetljivost, specifičnost, preciznost, F1, balanced accuracy i AUC na HAM10000 testu uz zaključani prag.

## Sekundarni ishodi
- HIBA (dermatoskopske): isti model i prag; po tipu kože (I–II, III–IV, nema podatka).
- **Lokalna kalibracija praga na HIBA-i:** pacijenti (patient_id) podijeljeni 50/50 (train_test_split, stratificirano po tome ima li pacijent melanom, seed 42);
  na prvoj polovici prag istim pravilom (osjetljivost ≥ 95 %); na drugoj polovici usporedba HAM praga i lokalnog praga.
- **Grad-CAM:** kvalitativno, po 4 slike iz HAM testa i HIBA-e (pogođeni melanomi, propušteni melanomi, lažne uzbune), odabrane sa seed 42 – gleda li model leziju ili rub/dlake/mjerilo.
- **BCN20000 – završni neovisni test** (nikad prije gledan): isključiti „Melanoma metastasis” i slike bez dijagnoze; EDA samo na metapodacima;
  evaluacija se pokreće **jednom**, na kraju, s istim modelom i pragom.

## Organizacija
- v1 arhivirana u `arhiva_v1/`; v2 u `melanom_v2.ipynb`; skripte s `%%writefile`; lekcije dostupne na Google disku (DL, ML, CV, NLP knjige).
