Styl pro LaTeX umožňující sazbu dokumentů podle vizuálního stylu Technické
univerzity v Liberci.

Návod k instalaci a použití najdete v souboru manual-tul.pdf a
manual-tulthesis.pdf.

Pozor na verzi titlesec!
========================

Pro úpravu nadpisů částí textu se používá balík titlesec. Jeho verze 2.10.1
z března 2016 obsahuje chybu, která způsobuje, že se nezobrazují čísla
v nadpisech \section. Ujistěte se, že máte instalovánu verzi alespoň 2.10.2.
Viz též:

* aktuální verze balíku titlesec
  https://www.ctan.org/pkg/titlesec

* diskuse k problému s číslováním
  http://tex.stackexchange.com/questions/299969/titlesec-loss-of-section-numbering-with-the-new-update-2016-03-15

Licence
=======

Dílo je vydáno pod licencí Creative Commons CC BY. Můžete je
rozmnožovat a distribuovat, remixovat, změnit a vyjít z původního díla
pro jakýkoliv účel, a to i komerční.

Historie
========

verze 2.3 - únor 2025
---------------------
- odstraněno nekorektní vkládání seznamu zkratek do seznamu tabulek

verze 2.2 - prosinec 2024
-------------------------
- přidána volba localfonts

verze 2.1 - únor 2024
---------------------
- přidána volba sfheadings
- balík tabularx nahrazen tabularray
- přepracován seznam zkratek
- opraven anglický název CXI

verze 2.0 - září 2022
---------------------
- implementována nová vizuální identita TUL
- přidány stránkové styly pro záhlaví/zápatí
- v tulthesis sloučeny různé varianty projektů do obecné kategorie "Projekt",
  přidána semestrální práce

verze 1.10 - leden 2021
-----------------------
- v tulthesis opravena titulní strana a dokumentace pro habilitační práce
- do tulthesis přidána volba article pro změnu výchozí třídy dokumentu

verze 1.9 - leden 2021
-----------------------
- do tulthesis přidána volba Hab pro habilitační práce
- upraven kód i návod tulthesis pro vkládání úvodních stránek
  generovaných IS STAG
- v tul odstraněna volba cmyk při volání balíku xcolor, která
  zkreslovala černou barvu

verze 1.8 - březen 2020
-----------------------
- do tulthesis přidána volba Autoref
- nová verze studentského prohlášení v tulthesis

verze 1.7 - březen 2019
-----------------------
- upraven text prohlášení v tulthesis
- \ThesisStart generuje jen jednu jazykovou verzi titulní strany
- prostředí abstractCZ a anbstractEN generují i název práce

verze 1.6 - červen 2018
-----------------------
- tul méně agresivně mění rozměry zrcadla a okrajů
- nový příkaz \TULwide
- do tulthesis přidáno prostředí pro klíčová slova
- do dokumentace doplněn přehled příkazů

verze 1.5 - prosinec 2017
-------------------------
- tulthesis umožňuje uvést na titulní straně konzultanta
- nová volba tulthesis pro teze disertační práce

verze 1.4 - červen 2017
-----------------------
- \ThesisStart umožňuje vložit oskenované zadání a prohlášení
- dílčí úpravy v dokumentaci třídy tulthesis

verze 1.3 - říjen 2016
---------------------
- Ústav zdravotních studií změněn na Fakultu zdravotních studií
- změněna vnitřní logika přepínání částí TUL a příslušných voleb

verze 1.2 - listopad 2015
-------------------------
- opraven anglický název DP na master thesis
- doplněna anglická verze úvodního prohlášení
- doplněna licence

verze 1.1 - duben 2015
----------------------
- přidána možnost opakovaného použití \TULbranch pro studenty kombinace
  několika oborů
- přidána volba wide pro abstrakty a poděkování
- z anglického abstraktu vypuštěn příkaz \clearpage
- volba EN třídy tulthesis sází nejprve anglickou titulní stranu a generuje
  více textů v angličtině
- odstraněna tečka za číslem v nadpisu kapitoly
- aktualizována adresa FP
- odstraněn problém s druhým řádkem adresy

verze 1.0 - duben 2014
----------------------
- upraveno podle nové vyhlášky rektora - do čestného prohlášení přidána věta o
  shodě elektronické a tištěné verze práce
- na titulní straně tulthesis zrušen převod názvu práce na verzálky
- tulthesis umožňuje několik oborů
- grafické úpravy \TULfancytitlepage a titulní strany tulthesis
- odstraněna tabulka v zápatí, způsobovala problémy s balíkem colortbl
- opravy drobných chyb

verze 0.91 - leden 2014
-----------------------
- nahrazeno opomenuté "á" na titulní stránce

verze 0.9 - červenec 2013
-------------------------
- kód generující zápatí byl přepracován
- opravena chyba v zápatí, která znemožňovala překlad pro většinu fakult
- odstraněno varování fancyhdr při jednostranné sazbě
- loga převedena z PDF verze 1.5 na 1.4 (verzi 1.5 nepodporují starší
  implementace, zároveň došlo k poklesu velikosti souborů)

verze 0.8 - červenec 2013
-------------------------
- doplněna anglická verze - příkazy pro anglická loga, volba EN
- nové příkazy \CZlogo, \ENlogo, \TULsoucast, \CZTULsoucast, \ENTULsoucast
- nová podoba titulní stránky absolventských prací ve třídě tulthesis

verze 0.7 - březen 2013
-----------------------
- třídě tulthesis doplněna volba MP pro sazbu zprávy magisterského projektu
- prohlášení o autorství ve třídě tulthesis nyní podporuje všechny typy prací

verze 0.6 - březen 2013
-----------------------
- balík tul převeden na univerzální kódování akcentovaných znaků
- odstraněn problém s dvojím logem na titulní stránce
- do třídy tulthesis přidán příkaz \TULthesisType
- drobné doplňky manuálu tulthesis

verze 0.5 - únor 2013
---------------------
- přidána třída tulthesis pro sazbu absolventských prací

verze 0.2 - srpen 2012
----------------------
- přidána možnost změnit kontaktní mail a telefon v zápatí (příkazy \TULmail a
  \TULphone)
- odstraněno prostředí picture ze zápatí

verze 0.1 - červen 2012
-----------------------
- první veřejná verze
