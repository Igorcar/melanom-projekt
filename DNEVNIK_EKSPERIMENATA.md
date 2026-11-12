# Dnevnik eksperimenata

Kronološki zapis svega što je napravljeno, što se znalo u kojem trenutku i što je promijenjeno.
Svrha: iskreno pokazati koji su skupovi podataka bili „viđeni” prije konačnih odluka.

---

## Verzija 1 – razvojna (8. 11. 2026.)

**Postupak:**
- Podjela HAM10000 po leziji (70 / 15 / 15, stratificirano, seed 42).
- ResNet-18 (ImageNet), cijela mreža, Adam lr 1e-4, batch 32, 10 epoha, težine klasa iz treninga.
- Najbolja epoha 6 (val AUC 0,902).
- Prag biran na validaciji: najviši prag uz osjetljivost **≥ 90 %** → prag **0,4646**.

**Rezultati verzije 1 (prag 0,4646):**
| skup | osjetljivost | specifičnost | AUC |
|---|---|---|---|
| HAM validacija | 0,902 (148/164) | 0,738 | 0,902 |
| HAM test | **0,843** (140/166) | 0,748 | 0,894 |
| HIBA | 0,622 (122/196) | 0,844 | 0,823 |

**Što smo naučili:**
- Prag biran *točno* na 90 % na validaciji na testu daje samo 84 %. Pravilu je nedostajala rezerva, jer osjetljivost na novim podacima uvijek malo padne.
- Na HIBA-i osjetljivost jako pada (pomak domene).

**Što je viđeno:** HAM test i HIBA. BCN20000 **nije** bio otvoren (ni metapodaci ni slike).

**Odluka:** prag se nakon pogleda na test NE mijenja (to bi bilo podešavanje prema testu). Umjesto toga cijeli postupak se ponavlja kao verzija 2, s pravilom za prag koje ima rezervu i zapisano je unaprijed. Verzija 1 je arhivirana.

---

## Verzija 2 – konačna (8. 11. 2026.)

**Unaprijed zapisane odluke** (`Plan_v2.md`, napisan prije ijedne evaluacije verzije 2):
- Sve isto kao v1 (podjela, model, augmentacija, treniranje).
- **Jedina promjena postupka:** ciljna osjetljivost na validaciji **95 %** umjesto 90 %, kao rezerva za pad na novim podacima.
- Novi sekundarni ishodi: lokalna prilagodba praga odluke na HIBA-i (pacijenti 50/50), Grad-CAM, **BCN20000 kao završni neovisni test, pokrenut jednom**.

**Ponovljivost:** podjela i treniranje dali su rezultate identične verziji 1 do 4. decimale (svaki gubitak, svaki val AUC). Isti model (epoha 6), samo drugi prag: **0,2894**.

**Rezultati verzije 2 (prag 0,2894):**
| skup | osjetljivost | specifičnost | AUC | status skupa |
|---|---|---|---|---|
| HAM validacija | 0,951 | 0,652 | 0,902 | korišten za odabir praga |
| HAM test | 0,886 | 0,673 | 0,894 | viđen u v1 |
| HIBA | 0,719 | 0,769 | 0,823 | viđen u v1 |
| **BCN20000** | **0,439** | 0,879 | 0,754 | **prvi put otvoren nakon zaključavanja v2** |

**Iskrena ograda:**
- HAM test i HIBA su viđeni u verziji 1. Pravilo za prag (95 %) nastalo je *nakon* što smo vidjeli da 90 % na testu daje 84 %. Rezultati na HAM testu i HIBA-i u verziji 2 zato **nisu potpuno netaknuti**.
- **BCN20000 jest potpuno netaknut.** Metapodaci, isključenja i slike prvi put su otvoreni tek nakon što su model i prag zaključani. Evaluacija je pokrenuta jednom.
- Zato je BCN20000 najpošteniji pokazatelj kako se model ponaša u novoj ustanovi.

---

## Dodatne analize nakon vanjskih recenzija (9.–10. 11. 2026.)

Nakon osam vanjskih recenzija dodane su analize koje **ne mijenjaju model ni prag** i ne traže ponovno treniranje. Sve koriste već spremljene predikcije:
- objedinjena tablica za sva tri skupa s 95 % intervalima pouzdanosti (Wilson) i brojem upućivanja po pronađenom melanomu;
- ROC krivulje s označenom radnom točkom (zaključani prag);
- rezultati po leziji (prosjek skora slika iste lezije);
- propušteni melanomi po dobi, spolu i lokalizaciji;
- raspodjela skora pravih melanoma u tri skupa;
- provjera neovisnosti skupova (zajednički ISIC brojevi / lezije: sve 0);
- dijagram kalibracije;
- demo s opreznim porukama („melanom NIJE isključen”).

**Promjena naziva:** ono što je ranije nazvano „lokalna kalibracija praga” sada se zove **„lokalna prilagodba praga odluke”**, jer se mijenja samo prag, a ne kalibrira skor. Datoteka `rezultati/hiba/lokalna_kalibracija.csv` zadržala je staro ime.

Ove analize su **naknadne i opisne**. Nisu bile u `Plan_v2.md` i ne utječu ni na jednu odluku o modelu ili pragu.

---

## Provjera ponovljivosti evaluacije (10. 11. 2026.)

Nakon uređivanja bilježnice (dodani su samo komentari, logika je ista) ponovno su pokrenute sve ćelije **osim treniranja**: evaluacija, HIBA, BCN20000, analize, Grad-CAM, demo i dodatne analize.

- Isti model (epoha 6, val AUC 0,9017) i ponovno izabran **isti prag 0,2894**.
- Svi rezultati **identični** prethodnima: HAM test 147 / 19 / 901 / 437, HIBA 141 / 55 / 834 / 250, BCN20000 305 / 390 / 3 652 / 501; isti AUC-ovi i intervali pouzdanosti.
- Ništa nije podešavano. Ponovljen je isti postupak, pa ponovno pokretanje ne narušava neovisnost BCN20000 testa.

---

## Unaprijed zapisane usporedbe (11. 11. 2026., PRIJE pokretanja)

Nakon recenzija R9–R14 dogovorene su tri usporedbe. Zapisane su ovdje prije nego što je ijedna pokrenuta. **Glavni model (epoha 6) i prag 0,2894 ostaju glavni rezultat bez obzira na ishod usporedbi.**

| usporedba | što se mijenja | sve ostalo |
|---|---|---|
| **Osnovna linija** | logistička regresija na histogramu boja (24 značajke: 8 razreda × R, G, B), `class_weight="balanced"` | isti HAM trening / validacija / test |
| **Zamrznuta mreža** | ResNet-18, uči se samo zadnji sloj (512 → 2), stopa učenja 1e-3 | ista augmentacija, težine klasa, 10 epoha, najbolja epoha po val AUC-u |
| **Dva dodatna seeda** | seed treninga 123 i 2024 (podjela podataka ostaje seed 42) | sve isto kao glavni model |

**Pravila:**
- Za svaki model prag se bira istim pravilom na HAM validaciji (najviši prag uz osjetljivost ≥ 95 %).
- Modeli se **uspoređuju na validaciji**. Rezultati na HAM testu, HIBA-i i BCN-u samo se prikazuju i **ne služe za odabir modela**, jer su ti skupovi već viđeni.
- Seedovi se izvješćuju kao srednja vrijednost ± SD (glavni model + 2 seeda).
- Usporedni modeli spremaju se u zasebne mape (`runs/usporedba_*`, `rezultati/usporedba/`); glavni model i `prag.json` se ne diraju.

**Dodatne analize bez treniranja (isti dan):** tablica osjetljivosti i specifičnosti za više pragova (eksploratorno, ne bira se novi prag), Brier score, Grad-CAM na BCN-u, i usporedba jedinica evaluacije (prva slika / nasumična slika / prosjek po leziji) na sva tri skupa. Za posljednju se računa skor za sve BCN slike (17 157). Glavni BCN rezultat ostaje prva slika po leziji.

---

## Rezultati usporedbi i dodatnih analiza (11.–12. 11. 2026.)

Sve je pokrenuto prema gornjem zapisu. Glavni model i prag nisu mijenjani; kontrolna evaluacija glavnog modela ponovno je dala prag 0,2894 i iste brojeve.

| model | najbolja epoha | prag (val) | AUC val | AUC test | AUC HIBA | AUC BCN | osj. test | osj. BCN |
|---|---|---|---|---|---|---|---|---|
| logistička regresija (boje) | – | 0,3505 | 0,777 | 0,722 | 0,660 | 0,602 | 0,843 | 0,986* |
| zamrznuti ResNet-18 | 8 | 0,3455 | 0,852 | 0,826 | 0,733 | 0,639 | 0,898 | 0,632 |
| glavni (seed 42) | 6 | 0,2894 | 0,902 | 0,894 | 0,823 | 0,754 | 0,886 | 0,439 |
| seed 123 | 5 | 0,2803 | 0,916 | 0,895 | 0,818 | 0,728 | 0,855 | 0,479 |
| seed 2024 | 7 | 0,1393 | 0,912 | 0,920 | 0,820 | 0,685 | 0,946 | 0,432 |

\* uz specifičnost 0,022 – model na bojama na BCN-u označava gotovo sve.

**Zaključci (na validaciji):** fino podešavanje cijele mreže bolje je od zamrznute mreže i od osnovne linije. Glavni model nije najbolji od tri seeda na validaciji, pa nije biran prema rezultatu. Prag je osjetljiv na slučajnost treninga.

**Dodatne analize:** BCN nasumična slika po leziji ≈ prva slika (osj. 0,453 / 0,439); prosjek po leziji AUC 0,782, osj. 0,472. Kontrola: skor prve slike u ponovnom izračunu identičan završnom testu (razlika 0,0). Brier: HAM 0,152 (ref. 0,098), HIBA 0,136 (0,130), BCN 0,116 (0,123).
