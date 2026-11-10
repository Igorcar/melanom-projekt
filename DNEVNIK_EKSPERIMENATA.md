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
