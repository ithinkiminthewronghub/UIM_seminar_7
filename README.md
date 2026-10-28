# Cvičení 7: Rozhodovací stromy a náhodný les

Sedmé praktické cvičení předmětu **Umělá inteligence v medicíně** pokračuje
v bloku **učení s učitelem**, který otevřelo Cvičení 06. Vzorovou datovou sadou
zůstává **Breast Cancer Wisconsin** — 569 nádorů popsaných 30 číselnými příznaky
a binární diagnózou (**1 = maligní, 0 = benigní**).

Zatímco kNN z Cvičení 06 se prakticky *neučil* (jen si zapamatoval trénovací
množinu), **rozhodovací strom** je první model kurzu, který si z dat skutečně
odvodí **explicitní, čitelné pravidlo**: posloupnost otázek tvaru „je příznak *j*
menší nebo roven prahu *t*?". Cvičení staví strom od nuly, pak z něj kompozicí
udělá **náhodný les**, a jako vedlejší produkt téhož výpočtu získá **důležitost
příznaků** — třetí, *vestavěný* přístup k výběru příznaků vedle filtračního
(Cvičení 05) a obalového (Cvičení 06).

Cvičení má tři těžiště: **dělení prostoru** osově zarovnanými řezy (a proč to
vede k pravoúhlým oblastem, na rozdíl od hladké hranice kNN); **hladové
rekurzivní dělení** jako zcela jiný druh „učení", než jaký znáte z optimalizace;
a **ansámblové metody** — sdružování mnoha slabších modelů do jednoho silnějšího.
Rozdíl mezi **baggingem** a **boostingem** je pojmovým vrcholem cvičení; bagging
si naprogramujete, boosting se pouze demonstruje.

---

## Obsah

1. [Cíle cvičení](#cíle-cvičení)
2. [Struktura repozitáře](#struktura-repozitáře)
3. [Instalace a spuštění](#instalace-a-spuštění)
4. [Teoretický základ](#teoretický-základ)
5. [Konfigurace projektu](#konfigurace-projektu)
6. [Pokyny k vypracování](#pokyny-k-vypracování)
7. [Lokální testování](#lokální-testování)
8. [Doplňkové (papírové) příklady](#doplňkové-papírové-příklady)
9. [Odevzdání](#odevzdání)

---

## Cíle cvičení

Po dokončení tohoto cvičení student:

1. **Implementuje rozhodovací strom od základů** — spočítá nečistotu skupiny
   (Gini), informační zisk řezu, najde nejlepší řez přes všechny uvažované
   příznaky a prahy a rekurzivně z nich sestaví strom. Predikce je průchod
   stromem od kořene k listu.
2. **Chápe, jak strom dělí příznakový prostor** — každý řez je **osově zarovnaná
   nadrovina**, takže výsledné oblasti jsou vždy pravoúhlé „krabice". Umí to
   postavit do kontrastu s hladkou hranicí kNN (Cvičení 06) a s jedinou přímkou
   lineárního modelu. Na hradlech AND / OR / XOR vidí, že *lineární
   separovatelnost* a *jeden osově zarovnaný řez* nejsou totéž.
3. **Rozumí tomu, co tu hraje roli ztrátové funkce** — učení stromu **není**
   gradientní sestup. Je to **hladové rekurzivní dělení** řízené poklesem
   nečistoty; konceptuálně je blíž rekurzivnímu dělení z Cvičení 02 než
   optimalizaci vah.
4. **Odvodí důležitost příznaků jako vedlejší produkt učení** — tentýž zisk,
   který řídí volbu řezů, se akumuluje po příznacích a po normalizaci dává
   `feature_importances_`. Zná i **meze** této míry: zkreslení ve prospěch
   spojitých příznaků s mnoha úrovněmi a rozpad důležitosti mezi korelované
   příznaky, tedy proč **nízká důležitost neznamená zbytečný příznak**.
5. **Zařadí důležitost mezi metody výběru příznaků** — jako **vestavěnou**
   (embedded) metodu vedle **filtrační** z Cvičení 05 a **obalové** z Cvičení 06,
   a ví, že každá z těch tří má jiné slepé místo.
6. **Implementuje bagging a náhodný les** — losování bootstrapového výběru
   s opakováním, natrénování mnoha stromů a většinové hlasování. Chápe, že
   **náhodný les = bagging + losování podmnožiny příznaků v každém uzlu**, a že
   ten druhý krok slouží k **dekorelaci** stromů.
7. **Rozliší bagging a boosting** — paralelní a nezávislé sdružování snižující
   **rozptyl** proti sekvenčnímu a závislému sdružování snižujícímu **zkreslení**.
   Boosting je v pipeline pouze demonstrován (`sklearn`), neprogramuje se.
8. **Volí formát perzistence podle tvaru stavu** — natrénovaný strom se ukládá
   do **JSON**, ne do `.npz` jako PCA (Cvičení 05) a kNN (Cvičení 06), protože
   jeho stav je zanořená stromová struktura, nikoli obdélníková pole.
9. **Rozliší kompozici a dědičnost** — `RandomForest` **má** seznam stromů
   (`self.trees_`), ale **není** stromem; obě třídy jsou sourozenecké potomky
   společného rozhraní `Classifier`.
10. **Pracuje s typovanou konfigurací** — čte hyperparametry z `config.yaml`
    přes dataclassy (`cfg.forest.n_estimators` místo `cfg["forest"]["n_estimators"]`),
    stejný vzor jako v Cvičení 03–06.

---

## Struktura repozitáře

```
cviceni-07-template/
├── cviceni_07.py            # Hlavní pipeline — spusťte pro průběžné ověření (PŘEDVYPLNĚNO)
├── config.yaml              # Konfigurace experimentu (YAML)
├── priklady_07.md           # Papírové (teoretické) příklady — BEZ řešení v repozitáři
├── requirements.txt         # Python závislosti (zamčené verze)
├── .gitignore
├── src/
│   ├── __init__.py          # Re-exporty balíčku (neupravujte)
│   ├── node.py              # Node — dataclass uzlu stromu (PŘEDVYPLNĚNO)
│   ├── base.py              # Classifier (ABC) + konkrétní save/load do JSON (PŘEDVYPLNĚNO)
│   ├── decision_tree.py     # DecisionTree — ÚKOL: 8 metod; __init__ a fit() předvyplněny
│   └── random_forest.py     # RandomForest — ÚKOL: 6 metod + bonusové oob_score_; __init__ předvyplněn
├── dataio/
│   ├── __init__.py          # Re-exporty balíčku (neupravujte)
│   ├── gates.py             # make_gate() — logická hradla AND/OR/XOR/XNOR/IMPLY (předvyplněno)
│   ├── loader.py            # load_breast_cancer_data() — načtení dat (předvyplněno)
│   ├── config_manager.py    # Dataclassy + load_config + validate_config (předvyplněno)
│   └── plotting.py          # Rozhodovací plochy, důležitost příznaků, strom vs. les (předvyplněno)
├── graphs/                  # Výstupní složka pro grafy (generuje se automaticky)
│   └── .gitkeep
├── models/                  # Výstupní složka pro modely (generuje se automaticky)
│   └── .gitkeep
└── test_cviceni_07.py       # Automatické testy (pytest)
```

> **Poznámka k souborům `__init__.py`:** Každá složka s Python kódem (`src/`,
> `dataio/`) obsahuje `__init__.py`, který ji označuje jako balíček a definuje
> veřejné API. Díky tomu lze psát `from src import DecisionTree` místo
> `from src.decision_tree import DecisionTree`. **Tyto soubory neupravujte.**

> **Balíček `dataio/` je v tomto cvičení předvyplněn celý** — generování hradel,
> načítání dat, konfigurace i vykreslování. Na rozdíl od Cvičení 06 (kde byly
> úkoly v `preprocessing.py`) se v `dataio/` nevyskytuje žádný
> `NotImplementedError`. Veškerá vaše práce je ve dvou souborech: `src/decision_tree.py`
> a `src/random_forest.py`.

> **Předvyplněné jádro: `src/node.py` a `src/base.py`.** `Node` je záměrně
> **čistý nositel dat** — pět polí a žádná metoda; veškerá logika žije
> v `DecisionTree`. `Classifier` je **tenká** abstraktní báze: sdílí rozhraní
> a mechanické ukládání, nikoli algoritmus. Ani jeden soubor neupravujte.

> **Žádná injektovaná závislost.** Na rozdíl od Cvičení 03/04/06 se sem
> **nekopíruje `src/distance.py`** a v testech není `DummyDistance`. Rozhodovací
> strom nikde nepočítá vzdálenost mezi dvojicemi bodů — porovnává jednu hodnotu
> příznaku s prahem. Zavádět sem nepoužitou třídu `Distance` by bylo umělé.

> **Jen jeden druh `NotImplementedError`.** Každá zpráva začíná `Úkol:`
> (u bonusu `Úkol: (BONUS)`) a každou je potřeba doplnit. Strom i les jsou
> **induktivní** — naučené pravidlo se dá aplikovat na libovolná nová data —
> takže tu neexistuje důvod pro trvalé omezení, jaké mělo `predict()`
> u transduktivních metod z Cvičení 04.

> **Ukládání do JSON, ne do `.npz`.** Formát se řídí tvarem stavu. PCA (Cv. 05)
> a kNN (Cv. 06) ukládaly obdélníková pole, pro něž je `.npz` nativní. Natrénovaný
> strom je ale **zanořená struktura s nestejně dlouhými větvemi**, která se
> přirozeně mapuje na zanořené slovníky — a ty jsou přesně to, co JSON umí.
> `pickle` se nepoužívá nikdy: je to černá skříňka a bezpečnostní riziko.

---

## Instalace a spuštění

### 1. Vytvoření virtuálního prostředí

```bash
python -m venv .venv
```

Aktivace (Windows):
```bash
.venv\Scripts\activate
```

Aktivace (Linux / macOS):
```bash
source .venv/bin/activate
```

### 2. Instalace závislostí

```bash
pip install -r requirements.txt
```

### 3. Spuštění

```bash
python cviceni_07.py
```

Pipeline je rozdělena do fází a **každá fáze má vlastní ošetření chyb**. Dokud
nejsou úkoly hotové, fáze, která na nedokončenou metodu narazí, se ukončí
hláškou `[NENI HOTOVO] Úkol: …` a pipeline **pokračuje další fází**. Z jednoho
spuštění tak vidíte, co všechno ještě chybí; nikdy nedostanete holý traceback.

> **Jediná výjimka — načtení konfigurace.** `load_config()` volá
> `validate_config()`; kdyby v `config.yaml` byla nesmyslná hodnota, pipeline se
> korektně ukončí hláškou `[CHYBA KONFIGURACE]` hned na začátku. Obě funkce jsou
> předvyplněné, takže při nezměněné konfiguraci tento stav nenastane.

Jednotlivé metody lze mezitím ověřovat přes `pytest`, viz
[Lokální testování](#lokální-testování).

---

## Teoretický základ

Cvičení má dvě vrstvy: **geometrickou** (jak model dělí prostor a proč to vypadá
jinak než u kNN) a **ansámblovou** (proč je průměr mnoha slabých modelů lepší než
jeden). Obě vedou ke stejnému vedlejšímu produktu — důležitosti příznaků.

### 1. Rozhodovací strom jako dělení prostoru

#### 1.1 Osově zarovnané řezy

Rozhodovací strom klade posloupnost otázek tvaru

$$x_j \le t \quad ?$$

kde $j$ je index příznaku a $t$ práh. Každá taková otázka je **osově zarovnaná
nadrovina** — nadrovina kolmá na osu $j$, nezávislá na všech ostatních
příznacích. Vnitřní uzel obsahuje dvojici $(j, t)$, list obsahuje predikovanou
třídu.

Protože se řezy skládají, je oblast příslušná každému listu **průnikem
polorovin kolmých na osy**, tedy vždy **pravoúhlá krabice** (obecně
$k$-rozměrný kvádr, případně neomezený). Rozhodovací hranice stromu je proto
vždy „schodovitá" — nikdy šikmá, nikdy zaoblená.

> **Srovnání tvaru rozhodovací hranice napříč kurzem**
>
> | Model | Tvar hranice | Čím je dán |
> |:---|:---|:---|
> | Lineární model (např. logistická regrese) | **jedna přímka / nadrovina**, obecně šikmá | jeden vektor vah přes všechny příznaky současně |
> | kNN (Cvičení 06) | **hladká, členitá**, kopíruje rozložení dat | vzdálenost k trénovacím bodům, mění se spojitě |
> | **Rozhodovací strom (Cvičení 07)** | **schodovitá, pravoúhlá** | posloupnost nezávislých řezů kolmých na osy |
>
> Šikmou hranici strom nedokáže vyjádřit přesně — aproximuje ji schody. Zaplatí
> za to hloubkou: čím jemnější schody, tím hlubší strom, tím větší riziko přeučení.

#### 1.2 Hradla AND, OR a XOR

Pipeline vykreslí rozhodovací plochu stromu na třech miniaturních úlohách se
čtyřmi body $\{0,1\}^2$. Nejde o hračku — ukazují rozdíl mezi dvěma pojmy,
které se snadno zaměňují.

| Hradlo | Popisky bodů $(0,0), (0,1), (1,0), (1,1)$ | Lineárně separovatelné? | Kolik osově zarovnaných řezů strom potřebuje |
|:---|:---|:---|:---|
| AND | $0, 0, 0, 1$ | **ano** (např. $x_1 + x_2 > 1{,}5$) | **dva** (hloubka 2, tři listy) |
| OR | $0, 1, 1, 1$ | **ano** (např. $x_1 + x_2 > 0{,}5$) | **dva** (hloubka 2, tři listy) |
| XOR | $0, 1, 1, 0$ | **ne** — žádná přímka nefunguje | **dva** (hloubka 2, čtyři listy) |

Z tabulky plynou dvě různá poučení:

- **AND a OR jsou lineárně separovatelné, a přesto na ně jeden řez stromu
  nestačí.** Oddělující přímka je totiž **šikmá** a strom umí řezat jen kolmo na
  osy. Musí ji tedy poskládat ze dvou schodů. *Lineární separovatelnost
  neznamená „jeden řez stromu".*
- **XOR není lineárně separovatelný vůbec** — je to klasický protipříklad, na
  kterém selže každý čistě lineární model. Strom ho přesto vyřeší přesně, protože
  jeho řezy jsou **podmíněné**: druhá otázka se ptá až *uvnitř* oblasti vymezené
  první, a může proto v levé a v pravé části odpovědět opačně. Právě tahle
  podmíněnost dává stromu schopnost popsat interakci mezi příznaky.

### 2. Co u stromu hraje roli ztrátové funkce: nečistota

Strom se **neučí gradientním sestupem**. Nemá váhy, které by se posouvaly proti
gradientu; nemá ani diferencovatelnou ztrátovou funkci. Místo toho má
**kritérium nečistoty**, které měří, jak promíchané jsou třídy v jedné skupině
vzorků, a **hladově** volí ten řez, který nečistotu sníží nejvíc.

Nechť $S$ je skupina vzorků a $p_c$ podíl třídy $c$ v ní.

**Giniho index** (výchozí v tomto cvičení):

$$G(S) = 1 - \sum_{c} p_c^2$$

**Entropie** (bonusová varianta, `criterion: entropy`):

$$H(S) = - \sum_{c} p_c \log_2 p_c$$

| | Gini | Entropie |
|:---|:---|:---|
| Čistá skupina ($p = 1$) | $0$ | $0$ |
| Maximum (binárně, $p = 0{,}5$) | $0{,}5$ | $1$ bit |
| Interpretace | pravděpodobnost chyby při náhodném přiřazení třídy podle rozložení ve skupině | průměrný počet bitů potřebných k zakódování třídy |
| Výpočet | rychlejší (bez logaritmu) | pomalejší |

V praxi vedou obě kritéria k velmi podobným stromům; volba mezi nimi je zřídka
tím, co rozhoduje o kvalitě modelu.

**Učení = hladové rekurzivní dělení.** Algoritmus v každém uzlu vyzkouší
kandidátní řezy, vezme ten nejlepší **právě teď**, a pak tentýž postup zopakuje
zvlášť na levé a zvlášť na pravé části. Nikdy se nevrací a nikdy neuvažuje, zda
by horší řez teď nevedl k lepšímu stromu později.

> **Proč hladově?** Nalezení *optimálního* rozhodovacího stromu je NP-těžký
> problém — počet možných stromů roste kombinatoricky. Hladová heuristika je
> proto standard: je rychlá, dává srozumitelné výsledky, a její slabinu
> (nestabilitu) řeší až ansámbl v oddíle 5.

> **Kam to patří v kurzu.** Konceptuálně je tento druh učení nejblíž
> **rekurzivnímu dělení** z Cvičení 02 (hierarchické shlukování), ne optimalizaci.
> Rozdíl je v tom, co dělení řídí: v Cvičení 02 to byla vzdálenost mezi shluky
> **bez znalosti popisků**, tady je to pokles nečistoty **popisků** — proto strom
> patří k učení s učitelem.

**Kandidátní prahy.** Pro spojitý příznak se jako kandidáti berou **středy mezi
sousedními různými hodnotami**, které se v uzlu vyskytují. Pro binární hradla
s hodnotami $\{0, 1\}$ z toho vyjde jediný kandidát $t = 0{,}5$.

### 3. Informační zisk

**Informační zisk** řezu je pokles nečistoty, kterého se řezem dosáhne — rodič
minus **vzorky vážený** průměr obou potomků:

$$\mathrm{IG}(S, j, t) = I(S) - \frac{|S_L|}{|S|} I(S_L) - \frac{|S_R|}{|S|} I(S_R)$$

kde $I$ je zvolené kritérium nečistoty ($G$ nebo $H$), $S_L = \{ x \in S : x_j \le t \}$
a $S_R = \{ x \in S : x_j > t \}$.

**Vážení počtem vzorků je podstatné.** Bez něj by algoritmus preferoval řezy,
které odštípnou jediný vzorek do dokonale čistého listu — takový list má
nečistotu 0, ale o datech nevypovídá skoro nic. Váha $|S_L|/|S|$ takovému řezu
přisoudí odpovídajícím způsobem malý přínos.

**Číselná ilustrace (hradlo AND).** Skupina má popisky $\{0,0,0,1\}$, tedy
$p_0 = 0{,}75$, $p_1 = 0{,}25$ a

$$G = 1 - (0{,}75^2 + 0{,}25^2) = 0{,}375.$$

Řez $x_2 \le 0{,}5$ dá $S_L = \{0, 0\}$ (čisté, $G = 0$) a $S_R = \{0, 1\}$
($G = 0{,}5$), takže

$$\mathrm{IG} = 0{,}375 - \tfrac{2}{4}\cdot 0 - \tfrac{2}{4}\cdot 0{,}5 = 0{,}125.$$

> **Pozor — nulový zisk není důvod k zastavení.** U hradla XOR má rodič
> $G = 0{,}5$ a **oba** možné řezy mají zisk **přesně 0**: ať už řežeme podle
> $x_1$, nebo podle $x_2$, v obou polovinách zůstane jedna nula a jedna jednička.
> Kdyby algoritmus zastavoval na „zisk je nulový", XOR by se nikdy nenaučil.
> Pokračuje se ale dál a **v další úrovni** má každý řez zisk $0{,}5$ — úloha se
> vyřeší přesně. Proto se strom zastavuje **jen** na čistém uzlu, dosažené
> maximální hloubce, příliš malém počtu vzorků, nebo když žádný kandidátní práh
> neexistuje. **Nikdy ne na nulovém zisku.**

### 4. Důležitost příznaků

Zisk už máme spočítaný — řídí volbu každého řezu. Stačí ho **cestou akumulovat**
a získáme míru, jak moc který příznak k modelu přispěl. To je pointa celého
oddílu: **důležitost příznaku není nový výpočet, je to vedlejší produkt učení.**

Při každém provedeném řezu podle příznaku $j$ se přičte

$$\mathrm{imp}[j] \mathrel{+}= \frac{|S|}{n} \cdot \mathrm{IG}(S, j, t),$$

kde $|S|$ je počet vzorků v dělěném uzlu a $n$ počet **všech** trénovacích
vzorků. Zisky z hlubších uzlů tak váží méně — týkají se menší části dat.
Nakonec se vektor normalizuje na součet 1:

$$\mathrm{importance}[j] = \frac{\mathrm{imp}[j]}{\sum_k \mathrm{imp}[k]}.$$

Pro **les** je důležitost **průměrem** přes všechny stromy. Průměrování je právě
to, co dělá tuto míru použitelnou: jednotlivý strom je nestabilní, ale průměr
přes stovku stromů natrénovaných na různých bootstrapových výběrech už je
rozumně stabilní.

> **Meze této míry — nízká důležitost neznamená zbytečný příznak.**
>
> 1. **Zkreslení ve prospěch spojitých příznaků a příznaků s mnoha úrovněmi.**
>    Čím víc různých hodnot příznak má, tím víc kandidátních prahů nabízí, a tím
>    větší je šance, že se některý z nich náhodou trefí. Binární příznak nabízí
>    jediný práh. Srovnávat důležitost spojitého a binárního příznaku je proto
>    srovnávání nesrovnatelného.
> 2. **Korelované příznaky si důležitost rozdělí — nebo ji jeden vezme celou.**
>    Nesou-li dva příznaky skoro stejnou informaci, první řez použije jeden
>    z nich; druhý už pak nemá co přidat a jeho zisk klesne. Který z dvojice
>    vyhraje, může rozhodnout náhoda. Nízká hodnota tedy znamená „tento příznak
>    nepřinesl **navíc** oproti tomu, co model už použil", ne „tento příznak
>    nenese informaci".
> 3. **Konkrétní důkaz — XOR.** U hradla XOR vyjde vektor důležitostí
>    $[1{,}0,\ 0{,}0]$. Kořenový řez má zisk 0 (viz oddíl 3), takže příznaku
>    v kořeni nepřičte nic, zatímco oba řezy o úroveň níž jdou celé druhému
>    příznaku. Přitom **bez obou příznaků úlohu vyřešit nelze** — XOR na jediném
>    příznaku nezávisí. Příznak s důležitostí 0 je zde nepostradatelný.

> **Tři rodiny výběru příznaků — každá s jiným slepým místem**
>
> | Rodina | Cvičení | Jak funguje | Slepé místo |
> |:---|:---|:---|:---|
> | **Filtrační** (filter) | 05 | ohodnotí každý příznak zvlášť statistickým testem, model se vůbec neptá | nevidí **interakce** — příznak užitečný jen v kombinaci s jiným propadne |
> | **Obalová** (wrapper) | 06 | opakovaně trénuje model na podmnožinách a vybírá podle jeho skóre | **výpočetně drahá**; výsledek platí pro ten jeden model a hrozí přeučení výběru |
> | **Vestavěná** (embedded) | **07** | výběr je součástí učení samotného — strom si příznaky vybírá při každém řezu | zkreslení a rozpad důležitosti u korelovaných příznaků (viz výše) |
>
> Pipeline vykreslí všechny tři pohledy vedle sebe. Cílem není určit vítěze, ale
> vidět, že se **neshodnou** — a rozumět proč.

### 5. Od jednoho stromu k lesu: bagging a boosting

Jediný hluboký strom má typickou vadu: **vysoký rozptyl**. Malá změna
trénovacích dat (pár vzorků navíc) může změnit kořenový řez a tím i celý strom
pod ním. Zkreslení má naopak nízké — dost hluboký strom se natrénovaná data
naučí přesně. To je učebnicový kandidát na **sdružování (ansámbl)**.

#### 5.1 Bootstrap a bagging

**Bootstrap** znáte z Cvičení 06 jako validační strategii. Tady se používá
jinak: k výrobě **různých trénovacích množin** ze stejných dat. Z $n$ vzorků se
losuje $n$ indexů **s opakováním**, takže některé vzorky se objeví vícekrát
a jiné vůbec.

Pravděpodobnost, že se konkrétní vzorek do jednoho výběru **nedostane**, je

$$\left(1 - \frac{1}{n}\right)^{n} \xrightarrow[n \to \infty]{} e^{-1} \approx 0{,}368,$$

takže každý strom vidí zhruba **63,2 %** různých vzorků a zbylých asi **36,8 %**
tvoří jeho **out-of-bag (OOB)** množinu — vzorky, které tento konkrétní strom
při učení neviděl a na kterých ho lze poctivě otestovat bez odděleného
testovacího souboru. (Odhad OOB je v tomto cvičení bonusový úkol.)

**Bagging** (bootstrap aggregating) je pak přímočarý: natrénuj $B$ stromů, každý
na vlastním bootstrapovém výběru, a predikuj **většinovým hlasem**.

$$\hat{y}(\mathbf{x}) = \operatorname*{arg\,max}_{c} \sum_{b=1}^{B} \mathbb{1}\big[\,\hat{y}_b(\mathbf{x}) = c\,\big]$$

#### 5.2 Proč to funguje — a proč nestačí

Mají-li jednotlivé modely rozptyl $\sigma^2$ a **párovou korelaci** $\rho$, má
jejich průměr rozptyl

$$\rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2.$$

Druhý člen jde s rostoucím počtem stromů $B$ k nule — to je ten očekávaný přínos
průměrování. **První člen ale na $B$ vůbec nezávisí.** Sebevětší počet stromů
nepomůže, pokud jsou stromy navzájem silně korelované. A přesně to se v baggingu
děje: všechny stromy vidí (skoro) tatáž data, takže silný příznak si vybere do
kořene skoro každý z nich a stromy si jsou navzájem podobné.

#### 5.3 Náhodný les = bagging + losování příznaků

Náhodný les přidává k baggingu druhý zdroj náhody, mířený přímo na $\rho$:
**v každém uzlu se uvažuje jen náhodná podmnožina `max_features` příznaků.**
Strom tak občas nemůže sáhnout po nejsilnějším příznaku a je nucen využít jiný.
Jednotlivé stromy tím trochu ztratí na přesnosti (zkreslení mírně vzroste), ale
**přestanou si být podobné** — $\rho$ klesne, a s ním i ten neredukovatelný člen.

Obvyklá volba je $\texttt{max\_features} \approx \sqrt{p}$; pro 30 příznaků
datasetu breast cancer tedy zhruba 5, což je hodnota v `config.yaml`.

> **Klíčové rozlišení.** Bagging = **bootstrap + hlasování**. Náhodný les =
> **bagging + náhodná podmnožina příznaků v každém uzlu**. Druhý krok je to, co
> dělá les lesem, a v kódu odpovídá parametru `max_features`, který proto
> **bydlí už ve `DecisionTree`** — les mu jen předá menší hodnotu.

#### 5.4 Bagging proti boostingu

| | **Bagging / náhodný les** | **Boosting** |
|:---|:---|:---|
| Jak vznikají modely | **paralelně, nezávisle** — pořadí nehraje roli | **sekvenčně, závisle** — každý další opravuje chyby předchozích |
| Trénovací data pro model $b$ | bootstrapový výběr z původních dat | převáženy vzorky (nebo rezidua) podle chyb modelu $b-1$ |
| Co primárně snižuje | **rozptyl** (variance) | **zkreslení** (bias) |
| Typický základní model | **hluboký** strom (nízké zkreslení, vysoký rozptyl) | **mělký** strom / pařez (vysoké zkreslení, nízký rozptyl) |
| Paralelizace | triviální | z podstaty ne |
| Citlivost na šum a odlehlé body | nízká | **vyšší** — chybné vzorky dostávají stále větší váhu |
| V tomto cvičení | **implementujete** | pouze **demonstrace** (`sklearn.ensemble.GradientBoostingClassifier`) |

Obě rodiny sdružují stromy, ale **řeší opačný problém**. Bagging bere modely,
které jsou samy o sobě dost silné, ale nestabilní, a stabilizuje je průměrem.
Boosting bere modely, které jsou samy o sobě slabé (často jen jeden řez), a
skládá je do silného tak, že se každý další soustředí na to, co předchozí
nezvládly. Proto se do baggingu dávají **hluboké** stromy a do boostingu
**mělké** — což je rozdíl, který si stojí za to zapamatovat jako jednu větu.

---

## Konfigurace projektu

### Soubor `config.yaml`

```yaml
tree:
  max_depth: 5             # maximalni hloubka stromu (null = bez omezeni)
  criterion: gini          # gini | entropy (entropie je bonusova varianta)

forest:
  n_estimators: 100        # pocet stromu v lese
  max_features: 5          # pocet priznaku losovanych v kazdem uzlu (~sqrt(30))
  max_depth: 5             # maximalni hloubka jednotlivych stromu v lese

data:
  random_state: 42         # seed pro bootstrap, losovani priznaku i rozdeleni dat
```

### Typovaná konfigurace (dataclassy)

```
ExperimentConfig
├── tree:   TreeConfig(max_depth, criterion)
├── forest: ForestConfig(n_estimators, max_features, max_depth)
└── data:   DataConfig(random_state)
```

K hodnotám se přistupuje **přes atributy, nikdy přes klíče slovníku**:

```python
# Místo:   cfg["forest"]["n_estimators"]   ← runtime chyba při překlepu
# Správně: cfg.forest.n_estimators          ← editor odhalí překlep okamžitě
```

Překlep v názvu atributu odhalí editor (nebo statická kontrola) hned při psaní,
zatímco `KeyError` u slovníku přijde až za běhu — často po několika minutách
trénování. `validate_config()` navíc ověří rozsahy hodnot (`max_depth >= 1` nebo
`null`, `criterion` z množiny `{gini, entropy}`, `n_estimators >= 1`,
`1 <= max_features <= 30`) a při porušení vyhodí `ValueError` se srozumitelnou
hláškou.

---

## Pokyny k vypracování

> **Předpoklad z předchozích cvičení: žádný.** Stejně jako v Cvičení 05 se sem
> nekopíruje `src/distance.py` ani nic jiného — strom nepočítá vzdálenosti. Vše
> potřebné je v repozitáři.

Vaše práce je ve **dvou souborech**: `src/decision_tree.py` (Blok I) a
`src/random_forest.py` (Blok II). Bloky mají být vypracovány **v tomto pořadí** —
les se skládá ze stromů, takže bez funkčního stromu nelze les otestovat.

### Blok I: Rozhodovací strom — `src/decision_tree.py`

Předvyplněné jsou `__init__` a `fit`. Metoda `fit` nastaví `self.n_features_in_`,
`self.n_samples_in_`, `self.classes_`, vynuluje `self._importances_` a zavolá
`self._build(x, y, 0)` — vaším úkolem je zbytek.

#### `_impurity(y)`

```
# Ověřte (assert), že y je jednorozměrné pole.
# Spočítejte podíly jednotlivých tříd v y (np.unique s return_counts=True).
# Pro criterion == "gini" vraťte  1 - součet čtverců podílů.
# (BONUS) Pro criterion == "entropy" vraťte  -součet p*log2(p); členy s p == 0
#         přeskočte, log2(0) není definován.
# Čistá skupina (jediná třída) musí dát přesně 0.0. Prázdná skupina také 0.0.
```

#### `_information_gain(parent_y, left_y, right_y)`

```
# Ověřte (assert), že len(left_y) + len(right_y) == len(parent_y).
# Vraťte  _impurity(parent_y) − (n_L/n)·_impurity(left_y) − (n_R/n)·_impurity(right_y).
# Vážení podílem vzorků je podstatné — viz teorie, odd. 3.
```

#### `_best_split(x, y)`

```
# Vyberte, přes které příznaky se bude hledat:
#   max_features is None → všechny příznaky
#   jinak                → náhodná podmnožina velikosti max_features BEZ opakování
#                          (losujte přes self._rng, aby byl běh reprodukovatelný)
# Pro každý uvažovaný příznak j:
#   1. vezměte setříděné RŮZNÉ hodnoty x[:, j] v tomto uzlu
#   2. kandidátní prahy = středy mezi sousedními různými hodnotami
#   3. pro každý práh rozdělte y na levou (x[:, j] <= t) a pravou (> t) část
#      a spočítejte _information_gain
# Vraťte trojici (feature, threshold, gain) s NEJVĚTŠÍM ziskem.
# Neexistuje-li ANI JEDEN kandidátní práh (všechny uvažované příznaky mají
# v tomto uzlu konstantní hodnotu), vraťte (None, None, 0.0).
# POZOR: nulový nejlepší zisk NENÍ důvod vrátit (None, None, 0.0) — u XOR je
#        zisk v kořeni nulový a přesto se musí řezat dál (viz teorie, odd. 3).
```

#### `_build(x, y, depth)`

```
# Zastavte a vraťte LIST  Node(value=<většinová třída v y>), pokud platí:
#   - uzel je čistý (v y je jediná třída), NEBO
#   - self.max_depth is not None a depth >= self.max_depth, NEBO
#   - len(y) < 2, NEBO
#   - _best_split nevrátil použitelný řez (feature is None).
# NIKDY nezastavujte kvůli nulovému zisku.
# Jinak:
#   1. (feature, threshold, gain) = self._best_split(x, y)
#   2. AKUMULUJTE DŮLEŽITOST:
#          self._importances_[feature] += (len(y) / self.n_samples_in_) * gain
#   3. rozdělte data maskou x[:, feature] <= threshold na levou a pravou část
#   4. rekurzivně sestavte oba potomky s depth + 1
#   5. vraťte Node(feature=…, threshold=…, left=…, right=…)  — bez value!
# Invariant: list má value != None a feature == None; vnitřní uzel naopak.
```

#### `predict(x)`

```
# Ověřte (assert), že model je nafitován (self.root_ není None)
# a že x.shape[1] == self.n_features_in_.
# Pro každý řádek x zvlášť: začněte v self.root_ a dokud jste ve vnitřním uzlu
# (uzel.value is None), jděte doleva při x[uzel.feature] <= uzel.threshold,
# jinak doprava. V listu vraťte uzel.value.
# Vraťte np.ndarray tvaru (n_vzorků,) typu int.
```

#### `feature_importances_` (property)

```
# Ověřte (assert), že model je nafitován.
# Vraťte self._importances_ vydělené jejich součtem, aby dávaly dohromady 1.
# Je-li součet nulový (strom je jediný list), vraťte pole nul beze změny —
# NEDĚLTE nulou.
```

#### `_to_dict()` a `_from_dict(data)`

```
# _to_dict: vraťte slovník s hyperparametry (max_depth, max_features, criterion,
#   random_state), naučenými atributy (n_features_in_, n_samples_in_, classes_,
#   _importances_) a REKURZIVNĚ převedeným stromem pod klíčem "root".
#   Uzel převeďte na {"feature":…, "threshold":…, "left":…, "right":…, "value":…},
#   kde left/right jsou opět takové slovníky, nebo None u listu.
# POZOR: JSON neumí typy numpy. int(...) / float(...) / .tolist() použijte
#   všude, kde hodnota pochází z numpy pole — jinak json.dump vyhodí TypeError.
# _from_dict: přesná inverze. Vytvořte instanci s uloženými hyperparametry,
#   obnovte naučené atributy a rekurzivně složte strom zpět z vnořených slovníků.
# Musí platit: model.save(p); DecisionTree.load(p).predict(x) dá tytéž predikce.
```

### Blok II: Náhodný les — `src/random_forest.py`

Předvyplněný je pouze `__init__`. Les **komponuje** stromy — drží si je
v `self.trees_` a volá jejich metody. **Nedědí** z `DecisionTree`; obě třídy jsou
sourozenecké potomky `Classifier`.

#### `_bootstrap_sample(x, y)`

```
# Ověřte (assert), že x a y mají stejný počet řádků.
# Vylosujte n = len(y) indexů S OPAKOVÁNÍM z rozsahu 0..n-1 (self._rng.integers).
# Vraťte (x[indices], y[indices]).
# Je to tentýž bootstrap jako v Cvičení 06, jen použitý k jinému účelu:
# tam k validaci, tady k výrobě různých trénovacích množin.
```

#### `fit(x, y)`

```
# Ověřte (assert), že x má dva rozměry a y odpovídající délku.
# Nastavte self.n_features_in_ a self.classes_, vyprázdněte self.trees_.
# n_estimators-krát:
#   1. (x_b, y_b) = self._bootstrap_sample(x, y)
#   2. strom = DecisionTree(max_depth=self.max_depth,
#                           max_features=self.max_features,
#                           random_state=<odvozený od self.random_state>)
#   3. strom.fit(x_b, y_b)  a  self.trees_.append(strom)
# Každý strom musí dostat JINÝ seed, jinak by všechny stromy byly totožné
# a losování příznaků by ztratilo smysl.
# Vraťte self.
```

#### `predict(x)`

```
# Ověřte (assert), že les je nafitován (self.trees_ není prázdný).
# Nechte každý strom predikovat celé x, výsledky složte do matice
# (n_stromů, n_vzorků) a pro každý sloupec vraťte NEJČASTĚJŠÍ třídu.
# Remízu rozhodněte deterministicky ve prospěch nižší třídy.
```

#### `feature_importances_` (property)

```
# Ověřte (assert), že les je nafitován.
# Vraťte PRŮMĚR vektorů feature_importances_ přes všechny stromy.
# Průměr je už normalizovaný (každý sčítanec dává 1), ale pro jistotu
# normalizujte znovu.
```

#### `_to_dict()` a `_from_dict(data)`

```
# _to_dict: hyperparametry lesa + seznam slovníků jednotlivých stromů
#   pod klíčem "trees" — DELEGUJTE na strom.__to_dict__, nepřepisujte
#   serializaci uzlů znovu.
# _from_dict: přesná inverze, stromy obnovte přes DecisionTree._from_dict.
```

#### `oob_score_` (property) — **BONUS**

```
# Úkol: (BONUS)
# Pro každý vzorek seberte predikce jen těch stromů, které ho ve svém
# bootstrapovém výběru NEMĚLY (out-of-bag), a udělejte většinový hlas.
# Vraťte podíl správně klasifikovaných vzorků, které měly aspoň jeden OOB strom.
# Vyžaduje, aby si fit() zapamatoval použité indexy jednotlivých stromů.
```

---

## Lokální testování

Spusťte automatické testy příkazem:

```bash
python -m pytest test_cviceni_07.py -v
```

| Třída testů | Co ověřuje |
|:---|:---|
| `TestImpurity` | `_impurity` na ručně spočítaných skupinách (čistá skupina → 0, `{0,0,0,1}` → 0,375, vyvážená → 0,5); `_information_gain` proti ručnímu výpočtu a vážení podílem vzorků. |
| `TestBestSplit` | `_best_split` najde na dobře oddělených datech správný příznak i práh; při konstantním vstupu vrací `(None, None, 0.0)`; `max_features` skutečně omezuje počet uvažovaných příznaků. |
| `TestDecisionTree` | strom se shoduje se `sklearn.tree.DecisionTreeClassifier` v přesnosti na oddělitelných datech; **hradla AND / OR / XOR se naučí přesně** (XOR je kontrola, že se nezastavuje na nulovém zisku); `predict` má správný tvar a validační `assert`y. |
| `TestFeatureImportance` | `feature_importances_` má správnou délku, je nezáporná a sčítá se na 1; na hradle AND se shoduje se `sklearn` (`[0,667, 0,333]` až na záměnu rovnocenných řezů). |
| `TestRandomForest` | les se shoduje se `sklearn.ensemble.RandomForestClassifier` v řádu přesnosti; les složený ze 100 stromů není horší než jeden strom; **`RandomForest` není podtřídou `DecisionTree`** (kontrola kompozice); průměrovaná důležitost se sčítá na 1. |
| `TestPersistence` | round-trip `save` → `load` přes JSON: znovunačtený strom i les dávají **shodné predikce** a nesou tytéž naučené atributy. |

Dokud nejsou příslušné metody hotové, testy, které je volají, se hlásí jako
**`xfail`** (očekávané selhání na `NotImplementedError`) a celá sada skončí
s návratovým kódem 0 — jde o záměrné chování, nikoli o chybu. Jakmile metodu
doplníte, stejný test začne procházet (`xpass` → `pass`).

Průběžně ověřujte i celou pipeline:

```bash
python cviceni_07.py
```

Kroky s neimplementovanými metodami se přeskočí s hláškou `[NENI HOTOVO] Úkol: …`;
ostatní proběhnou normálně a uloží grafy do `graphs/`.

---

## Doplňkové (papírové) příklady

Soubor `priklady_07.md` obsahuje číselné příklady k ručnímu výpočtu, navázané
přímo na metody, které programujete: Giniho index malé skupiny, informační zisk
konkrétního řezu, akumulace důležitosti z malého stromu, průchod zadaným stromem
při predikci a bootstrap s odhadem out-of-bag podílu.

Čísla jsou volena tak, aby se dala spočítat na papíře. **Řešení nejsou součástí
repozitáře** — výsledky si ověřte u vyučujícího nebo výpočtem v `numpy` /
`scikit-learn`.

---

## Odevzdání

Úloha se odevzdává prostřednictvím systému **GitHub Classroom**. Po dokončení
implementace proveďte:

```bash
git add src/decision_tree.py src/random_forest.py
git commit -m "Implementace cvičení 7"
git push
```

Po přijetí příkazu `push` se automaticky spustí testovací skripty, které ověří
správnost výpočtů. Výsledek bude zobrazen přímo v rozhraní GitHub u vašeho
repozitáře formou zelené fajfky (úspěch) nebo červeného křížku (neúspěch).

> **Soubory, které se neodevzdávají:** `src/__init__.py`, `src/node.py`,
> `src/base.py`, `dataio/__init__.py`, `dataio/gates.py`, `dataio/loader.py`,
> `dataio/config_manager.py`, `dataio/plotting.py`, `cviceni_07.py`
> a `test_cviceni_07.py` jsou předvyplněny nebo se nemají měnit. Systém hodnotí
> soubory `src/decision_tree.py` a `src/random_forest.py`.
