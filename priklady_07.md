# Papírové příklady — Cvičení 7

Rozhodovací stromy, informační zisk, důležitost příznaků a bootstrap. Příklady
procvičují **přesně tu notaci a ty výpočty**, které implementujete ve
`src/decision_tree.py` a `src/random_forest.py`. Čísla jsou volena tak, aby se
dala spočítat na papíře a hezky se sčítala.

**Řešení nejsou součástí repozitáře.** Výsledky si ověřte u vyučujícího nebo
výpočtem v `numpy` / `scikit-learn`.

Značení je stejné jako ve stub docstringech:

- $I(S)$ — nečistota skupiny $S$; pro Giniho index $G(S) = 1 - \sum_c p_c^2$,
  pro entropii $H(S) = -\sum_c p_c \log_2 p_c$, kde $p_c$ je podíl třídy $c$.
- $\mathrm{IG}(S, j, t) = I(S) - \frac{|S_L|}{|S|} I(S_L) - \frac{|S_R|}{|S|} I(S_R)$
  — informační zisk řezu podle příznaku $j$ prahem $t$, kde
  $S_L = \{x \in S : x_j \le t\}$ a $S_R = \{x \in S : x_j > t\}$.
- $\mathrm{imp}[j] \mathrel{+}= \frac{|S|}{n}\,\mathrm{IG}(S, j, t)$ — příspěvek
  jednoho řezu k důležitosti příznaku $j$; $n$ je počet **všech** trénovacích
  vzorků. Výsledek se nakonec normalizuje na součet 1.

---

## Příklad 1 — Giniho index a entropie malé skupiny

Spočítejte nečistotu následujících skupin popisků. U bodů a)–c) uveďte
**Giniho index i entropii**, u bodu d) stačí Giniho index.

- **a)** skupina s 8 vzorky třídy A a 2 vzorky třídy B
- **b)** skupina s 6 vzorky třídy A a 6 vzorky třídy B
- **c)** skupina s 10 vzorky třídy A a 0 vzorky třídy B
- **d)** skupina se třemi třídami, po 2 vzorcích od každé (A, A, B, B, C, C)

**Otázky k zamyšlení:**

- **e)** Pro kterou z uvedených skupin je nečistota největší a proč?
- **f)** Skupina d) má Giniho index vyšší než 0,5, což je maximum pro dvě třídy.
  Vysvětlete, proč to není v rozporu s tvrzením z přednášky, že „Gini je vždy
  menší než 1".

---

## Příklad 2 — Informační zisk řezu

Uzel obsahuje **10 vzorků: 5 třídy A a 5 třídy B**. Uvažujeme dva kandidátní
řezy:

- **řez I** rozdělí uzel na levou část se 4 vzorky A a 1 vzorkem B a pravou
  část s 1 vzorkem A a 4 vzorky B,
- **řez II** rozdělí uzel na levou část s 5 vzorky A a 3 vzorky B a pravou část
  s 0 vzorky A a 2 vzorky B.

**Úkoly:**

- **a)** Spočítejte Giniho index rodičovského uzlu.
- **b)** Spočítejte Giniho index obou částí pro **řez I** a jeho informační zisk
  $\mathrm{IG}$.
- **c)** Totéž pro **řez II**.
- **d)** Který řez by hladový algoritmus (`_best_split`) zvolil?
- **e)** Řez II vytvořil dokonale čistý list (0 A, 2 B). Přesto nemá největší
  zisk. Vysvětlete pomocí **vážení počtem vzorků**, proč tomu tak je — a co by
  se stalo, kdyby se nečistoty potomků nevážily.

---

## Příklad 3 — Důležitost příznaků z malého stromu

Rozhodovací strom byl natrénován na **12 vzorcích** (6 třídy A, 6 třídy B) a
vypadá takto:

```
kořen:  řez podle příznaku f0 prahem t0
        ├── levá větev (8 vzorků: 6 A, 2 B)
        │       řez podle příznaku f1 prahem t1
        │       ├── list  →  třída A   (6 vzorků: 6 A, 0 B)
        │       └── list  →  třída B   (2 vzorky: 0 A, 2 B)
        └── pravá větev (4 vzorky: 0 A, 4 B)
                list  →  třída B
```

**Úkoly:**

- **a)** Spočítejte informační zisk **kořenového** řezu (podle f0).
- **b)** Spočítejte informační zisk řezu v **levé větvi** (podle f1).
- **c)** Pomocí pravidla
  $\mathrm{imp}[j] \mathrel{+}= \frac{|S|}{n}\,\mathrm{IG}(S, j, t)$
  (kde $n = 12$) spočítejte nenormalizované příspěvky příznaků f0 a f1.
- **d)** Normalizujte vektor důležitostí na součet 1.
- **e)** Zisk řezu podle f1 je **větší** než zisk kořenového řezu podle f0, a
  přesto oba příznaky vyjdou stejně důležité. Vysvětlete, jak to způsobuje
  **váha $|S|/n$** (kořen vidí všech 12 vzorků, hlubší uzel jen 8).

---

## Příklad 4 — Průchod stromem při predikci

Je dán natrénovaný strom se dvěma příznaky $x = (x_0, x_1)$:

```
kořen:  x0 <= 2.5 ?
        ├── ano →  x1 <= 1.0 ?
        │          ├── ano →  list: třída 0
        │          └── ne  →  list: třída 1
        └── ne  →  list: třída 1
```

**Úkoly:**

- **a)** Určete predikci pro každý z bodů:
  $(1{,}0;\ 0{,}5)$, $(2{,}0;\ 3{,}0)$, $(5{,}0;\ 0{,}0)$,
  $(2{,}5;\ 1{,}0)$, $(2{,}6;\ 9{,}9)$.
- **b)** Body s $x_0 = 2{,}5$ a $x_0 = 2{,}6$ leží těsně vedle sebe, přesto
  mohou skončit v různých listech. Ukažte to na konkrétní dvojici a vysvětlete,
  proč je rozhodovací hranice stromu **schodovitá** (osově zarovnaná).
- **c)** Nakreslete rozhodovací oblasti tohoto stromu v rovině $(x_0, x_1)$ pro
  $x_0, x_1 \in [0, 4]$. Kolik obdélníkových oblastí vznikne a které třídě
  každá patří?

---

## Příklad 5 — Bootstrap a out-of-bag

**Úkoly:**

- **a)** Z trénovací množiny o $n$ vzorcích losujeme bootstrapový výběr — $n$
  indexů **s opakováním**. Odvoďte pravděpodobnost, že **konkrétní vzorek se do
  výběru nedostane**, a spočítejte ji pro $n = 5$, $n = 10$, $n = 20$ a
  $n = 100$.
- **b)** K jaké hodnotě tato pravděpodobnost konverguje pro $n \to \infty$?
  Vyjádřete ji jedním výrazem a číselně (na tři desetinná místa).
- **c)** Konkrétní malý bootstrap: $n = 5$, indexy $\{0, 1, 2, 3, 4\}$, výběr
  vylosoval postupně $[2, 2, 0, 4, 2]$. Vypište množinu indexů **ve výběru**,
  množinu **out-of-bag** indexů a jejich podíly.
- **d)** Les má 100 stromů, každý natrénovaný na vlastním bootstrapovém výběru.
  Zhruba kolik stromů (v průměru) **nevidělo** daný konkrétní trénovací vzorek?
  Jak se toho využívá při odhadu `oob_score_` (bonusová část úlohy)?

---

## Příklad 6 — Bagging, náhodný les a dekorelace *(pojmový)*

**Úkoly:**

- **a)** Vysvětlete jednou větou rozdíl mezi **baggingem** a **náhodným lesem**.
  Který krok navíc dělá z baggingu náhodný les a v kterém parametru kódu se
  projeví?
- **b)** Rozptyl průměru $B$ modelů s rozptylem $\sigma^2$ a párovou korelací
  $\rho$ je $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$. Co se stane s tímto
  výrazem, když $B \to \infty$? Který člen zůstane a co z toho plyne pro
  smysluplnost přidávání dalších stromů?
- **c)** Dataset breast cancer má 30 příznaků. Jakou hodnotu `max_features`
  odpovídá obvyklému doporučení $\sqrt{p}$? Porovnejte s hodnotou v
  `config.yaml`.
- **d)** Rozhodněte pro každou dvojici, co patří k baggingu a co k boostingu:
  *(hluboké stromy / mělké stromy)*, *(paralelní trénink / sekvenční trénink)*,
  *(snižuje rozptyl / snižuje zkreslení)*, *(odolnější vůči šumu / citlivější
  na šum)*.

---

## Příklad 7 — Hradlo XOR a nulový zisk *(propojení s kódem)*

Hradlo XOR má body $(0,0)\to 0$, $(0,1)\to 1$, $(1,0)\to 1$, $(1,1)\to 0$.

**Úkoly:**

- **a)** Spočítejte Giniho index kořenového uzlu (všechny čtyři body).
- **b)** Spočítejte informační zisk řezu $x_0 \le 0{,}5$ **a** řezu
  $x_1 \le 0{,}5$. Jaká hodnota vyjde v obou případech?
- **c)** Kdyby `_build` obsahoval pravidlo „když je nejlepší zisk 0, udělej
  list", jak by vypadal výsledný strom pro XOR a jaká by byla jeho přesnost?
- **d)** Spočítejte zisk řezu $x_1 \le 0{,}5$ **uvnitř** levé větve (tj. mezi
  body s $x_0 = 0$: $(0,0)\to 0$ a $(0,1)\to 1$). Vysvětlete, proč se XOR
  naučí přesně, když se algoritmus na nulovém zisku **nezastaví**.
- **e)** Jaký vektor důležitostí $[\,\mathrm{imp}(x_0),\ \mathrm{imp}(x_1)\,]$
  pro XOR vyjde (po normalizaci)? Který příznak dostane nulu — a je opravdu
  zbytečný?
