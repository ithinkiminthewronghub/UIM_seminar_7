"""Rozhodovaci strom staveny od nuly — induktivni model s citelnym pravidlem.

Na rozdil od kNN z cviceni 06, ktere si jen zapamatovalo trenovaci mnozinu,
si ``DecisionTree`` z dat odvodi **explicitni pravidlo**: posloupnost otazek
tvaru ``x_j <= t ?``. Naucene pravidlo je **induktivni** — da se aplikovat na
libovolna nova data, ne jen na trenovaci vzorky.

Uceni NENI gradientni sestup
----------------------------
Strom nema vahy, ktere by se posouvaly proti gradientu, ani diferencovatelnou
ztratovou funkci. Misto toho ma **kriterium necistoty** (Gini, pripadne
entropie), ktere meri promichanost trid ve skupine, a uci se **hladovym
rekurzivnim delenim**: v kazdem uzlu vyzkousi kandidatni rezy, vezme ten
s nejvetsim poklesem necistoty (informacnim ziskem) prave ted, a tyz postup
zopakuje zvlast na leve a zvlast na prave casti. Nikdy se nevraci. Konceptualne
je to blize rekurzivnimu deleni z cviceni 02 nez optimalizaci vah.

Rezy jsou osove zarovnane
-------------------------
Kazdy rez ``x_j <= t`` je nadrovina kolma na osu ``j``. Oblast prislusna
kazdemu listu je proto prunik polorovin kolmych na osy, tedy vzdy **pravouhla
krabice**; rozhodovaci hranice stromu je schodovita, nikdy sikma.

Dulezitost priznaku jako vedlejsi produkt
-----------------------------------------
Tyz zisk, ktery ridi volbu kazdeho rezu, se cestou akumuluje po priznacich
(viz ``_build`` a ``feature_importances_``). Dulezitost priznaku tedy neni
novy vypocet, je to vedlejsi produkt uceni.

Proc je ``max_features`` uz tady
--------------------------------
Parametr ``max_features`` (pocet priznaku losovanych v kazdem uzlu) bydli uz
ve ``DecisionTree``, prestoze samotny strom ho bezne necha na ``None`` (vsechny
priznaky). Nahodny les z cviceni Blok II ho jen vyuzije mensi hodnotou:
nahodny les = bagging + losovani podmnoziny priznaku v kazdem uzlu, a prave
ten druhy krok dekoreluje stromy.

Rozdeleni prace
---------------
``__init__`` a ``fit`` jsou PREDVYPLNENE. Osm dalsich metod je ukol: ``_impurity``,
``_information_gain``, ``_best_split``, ``_build``, ``predict``,
``feature_importances_``, ``_to_dict`` a ``_from_dict``.
"""

from __future__ import annotations

import numpy as np

from src.base import Classifier
from src.node import Node


class DecisionTree(Classifier):
    """Rozhodovaci strom (potomek ``Classifier``, pracuje s uzly ``Node``).

    Hyperparametry (ulozene v ``__init__``):

    * ``max_depth`` : int | None — maximalni hloubka; ``None`` = bez omezeni.
    * ``max_features`` : int | None — kolik priznaku se losuje v kazdem uzlu;
      ``None`` = uvazuji se vsechny.
    * ``criterion`` : str — ``"gini"`` (vychozi) nebo ``"entropy"`` (bonus).
    * ``random_state`` : int | None — seed pro ``self._rng`` (losovani priznaku).

    Atributy naucene ve ``fit`` (konvence sklearn — podtrzitko na konci):

    * ``root_`` : Node | None — koren postaveneho stromu.
    * ``n_features_in_`` : int — pocet priznaku trenovaci matice.
    * ``n_samples_in_`` : int — pocet VSECH trenovacich vzorku; slouzi jako
      vaha ``len(y) / n_samples_in_`` pri akumulaci dulezitosti.
    * ``classes_`` : np.ndarray — setridene unikatni tridy.
    * ``_importances_`` : np.ndarray — akumulovany vazeny zisk po priznacich
      (jeste nenormalizovany; normalizuje az ``feature_importances_``).
    """

    def __init__(
        self,
        max_depth: int | None = None,
        max_features: int | None = None,
        criterion: str = "gini",
        random_state: int | None = None,
    ) -> None:
        """Ulozi hyperparametry a pripravi generator nahody; PREDVYPLNENO.

        Zadny vypocet se tu nedeje — jen se zapamatuji hyperparametry, vytvori
        se ``self._rng`` a naucene atributy se nastavi na ``None``.

        Parametry
        ---------
        max_depth : int | None
            Maximalni hloubka stromu. ``None`` = bez omezeni.
        max_features : int | None
            Pocet priznaku losovanych v kazdem uzlu. ``None`` = vsechny.
        criterion : str
            Kriterium necistoty: ``"gini"`` nebo ``"entropy"`` (bonus).
        random_state : int | None
            Seed pro ``numpy.random.default_rng``. ``None`` = nedeterministicke.
        """
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state

        # Generator nahody pro losovani podmnoziny priznaku v _best_split.
        self._rng = np.random.default_rng(random_state)

        # Atributy naucene ve fit(); do zavolani fit() jsou None / prazdne.
        self.root_: Node | None = None
        self.n_features_in_: int | None = None
        self.n_samples_in_: int | None = None
        self.classes_: np.ndarray | None = None
        self._importances_: np.ndarray | None = None

    def _impurity(self, y: np.ndarray) -> float:
        """Necistota skupiny popisku ``y`` podle zvoleneho kriteria.

        Pro ``criterion == "gini"``::

            G(y) = 1 - suma_c p_c**2

        kde ``p_c`` je podil tridy ``c`` v ``y``. Pro ``criterion == "entropy"``
        (bonus)::

            H(y) = - suma_c p_c * log2(p_c)

        pricemz cleny s ``p_c == 0`` se preskoci (``log2(0)`` neni definovan).

        Cista skupina (jedina trida) i prazdna skupina daji presne ``0.0``.

        Priklady (gini)
        ---------------
        * ``y = {0, 0, 0, 1}`` -> ``p_0 = 0.75``, ``p_1 = 0.25`` ->
          ``G = 1 - (0.75**2 + 0.25**2) = 0.375``.
        * vyvazena binarni ``y = {0, 0, 1, 1}`` -> ``p = 0.5`` ->
          ``G = 1 - (0.25 + 0.25) = 0.5`` (maximum pro dve tridy).

        Parametry
        ---------
        y : np.ndarray
            Jednorozmerne pole celociselnych popisku.

        Navratova hodnota
        -----------------
        float
            Necistota v intervalu ``[0, 1)`` pro Gini (resp. ``[0, log2 k]``
            pro entropii nad ``k`` tridami).
        """
        # assert  Ověřte, že y je jednorozmerne pole (y.ndim == 1)
        raise NotImplementedError(
            "Úkol: spoctete podily trid v y (np.unique s return_counts=True) a vratte "
            "pro criterion 'gini' hodnotu 1 - suma p**2, pro 'entropy' -suma p*log2(p) "
            "(cleny s p == 0 preskocte); cista i prazdna skupina daji 0.0."
        )

    def _information_gain(
        self,
        parent_y: np.ndarray,
        left_y: np.ndarray,
        right_y: np.ndarray,
    ) -> float:
        """Pokles necistoty dosazeny rezem — necistota rodice minus vazeny prumer potomku.

        ::

            IG = _impurity(parent_y)
                 - (n_L / n) * _impurity(left_y)
                 - (n_R / n) * _impurity(right_y)

        kde ``n_L = len(left_y)``, ``n_R = len(right_y)`` a ``n = len(parent_y)``.

        Proc se potomci vazi poctem vzorku
        ----------------------------------
        Bez vazeni by vyhraval rez, ktery odstipne jediny vzorek do dokonale
        cisteho listu: takovy list ma necistotu 0, ale o datech nevypovida
        skoro nic. Vaha ``n_L / n`` takovemu rezu prisoudi odpovidajicim
        zpusobem maly prinos.

        Cislena ilustrace (hradlo AND)
        ------------------------------
        Rodic ``{0, 0, 0, 1}`` ma ``G = 0.375``. Rez ``x2 <= 0.5`` da
        ``left = {0, 0}`` (``G = 0``) a ``right = {0, 1}`` (``G = 0.5``), takze
        ``IG = 0.375 - (2/4)*0 - (2/4)*0.5 = 0.125``.

        Parametry
        ---------
        parent_y : np.ndarray
            Popisky v delenem uzlu.
        left_y : np.ndarray
            Popisky vzorku s ``x_j <= t``.
        right_y : np.ndarray
            Popisky vzorku s ``x_j > t``.

        Navratova hodnota
        -----------------
        float
            Informacni zisk rezu (nezaporny, muze byt i presne ``0.0``).
        """
        # assert  Ověřte, že len(left_y) + len(right_y) == len(parent_y)
        raise NotImplementedError(
            "Úkol: vratte _impurity(parent_y) minus vzorky vazeny prumer necistot "
            "potomku: (n_L/n)*_impurity(left_y) + (n_R/n)*_impurity(right_y)."
        )

    def _best_split(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[int | None, float | None, float]:
        """Najde rez ``(feature, threshold)`` s nejvetsim informacnim ziskem.

        Vyber uvazovanych priznaku
        --------------------------
        * ``self.max_features is None`` -> uvazuji se VSECHNY priznaky
          ``0 .. n_features - 1``.
        * jinak -> **nahodna podmnozina bez opakovani** velikosti
          ``self.max_features`` vylosovana pres ``self._rng`` (napr.
          ``self._rng.choice(n_features, size=self.max_features, replace=False)``).
          Losovani jde pres ``self._rng`` kvuli reprodukovatelnosti behu.

        Kandidatni prahy pro priznak ``j``
        ----------------------------------
        Vezmou se **setridene RUZNE** hodnoty ``x[:, j]`` v tomto uzlu; kandidati
        jsou **stredy mezi sousednimi ruznymi hodnotami**, tj. ``(v[:-1] + v[1:]) / 2``
        nad setridenymi unikaty ``v``. Ma-li priznak v uzlu jedinou hodnotu,
        nenabizi zadny prah. (Pro binarni hradla ``{0, 1}`` vyjde jediny prah ``0.5``.)

        Pro kazdy prah ``t`` se ``y`` rozdeli na ``left = y[x[:, j] <= t]`` a
        ``right = y[x[:, j] > t]`` a spocte se ``_information_gain``.

        Navratova hodnota
        -----------------
        tuple[int | None, float | None, float]
            Trojice ``(feature, threshold, gain)`` s NEJVETSIM ziskem.
            ``(None, None, 0.0)`` se vraci **jen tehdy**, kdyz neexistuje ANI
            JEDEN kandidatni prah — tedy vsechny uvazovane priznaky maji v tomto
            uzlu konstantni hodnotu.

        POZOR
        -----
        **Nulovy nejlepsi zisk NENI duvod vratit ``(None, None, 0.0)``.** U hradla
        XOR je zisk obou moznych rezu v koreni presne ``0.0`` a ``_best_split``
        presto musi vratit platny rez ``(feature, threshold, 0.0)``, aby se v nem
        ``_build`` mohl delit dal. Prazdny vysledek signalizuje pouze
        "neni podle ceho delit", ne "nejlepsi rez nic nezlepsi".
        """
        # assert  Ověřte, že x.ndim == 2 a x.shape[0] == len(y)
        # assert  Ověřte, že self.max_features je None nebo 1 <= max_features <= x.shape[1]
        raise NotImplementedError(
            "Úkol: pres uvazovane priznaky (vsechny, nebo nahodnou podmnozinu velikosti "
            "max_features losovanou self._rng) zkuste vsechny kandidatni prahy (stredy mezi "
            "sousednimi ruznymi hodnotami) a vratte (feature, threshold, gain) s nejvetsim "
            "ziskem; (None, None, 0.0) jen kdyz neexistuje zadny kandidatni prah."
        )

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Rekurzivne postavi (pod)strom pro data ``(x, y)`` v hloubce ``depth``.

        Zastaveni — vrat LIST ``Node(value=<vetsinova trida v y>)``, prave kdyz
        plati aspon jedna z podminek:

        * uzel je cisty: ``len(np.unique(y)) == 1``,
        * je zadana ``max_depth`` a ``depth >= self.max_depth``,
        * ``len(y) < 2`` (prilis malo vzorku na deleni),
        * ``_best_split`` nevratil pouzitelny rez (``feature is None``).

        NIKDY se nezastavuje kvuli nulovemu zisku. U hradla XOR ma koren
        ``gini = 0.5`` a oba mozne rezy zisk ``0.0``; kdyby ``_build`` zastavil
        na ``gain <= 0``, z XOR by vznikl jediny list a rozhodovaci plocha by
        zkolabovala na jednobarevnou. Pokracuje se dal a o uroven niz ma kazdy
        rez zisk ``0.5`` — uloha se vyresi presne.

        Jinak (vnitrni uzel):

        1. ``feature, threshold, gain = self._best_split(x, y)``.
        2. **Akumulace dulezitosti** — tentyz zisk, ktery ridi deleni, meri
           dulezitost priznaku::

               self._importances_[feature] += (len(y) / self.n_samples_in_) * gain

           Zisky z hlubsich uzlu tak vazi min, tykaji se mensi casti dat.
        3. Rozdeleni maskou ``mask = x[:, feature] <= threshold`` na levou
           (``mask``) a pravou (``~mask``) cast.
        4. Rekurze: ``left = self._build(x[mask], y[mask], depth + 1)`` a
           obdobne ``right`` pro ``~mask``.
        5. Vrat vnitrni ``Node(feature=..., threshold=..., left=..., right=...)``
           **bez ``value``**.

        Reseni remizy
        -------------
        Ma-li nejvyssi cetnost vic trid zaroven, vetsinovou tridou listu je ta
        s NEJNIZSIM ciselnym oznacenim (``np.unique`` vraci tridy setridene
        vzestupne, ``np.argmax`` nad jejich cetnostmi vezme prvni maximum).

        Invariant
        ---------
        List: ``value is not None`` a ``feature is None``.
        Vnitrni uzel: ``value is None`` a ``feature`` i oba potomci vyplneni.

        Parametry
        ---------
        x : np.ndarray
            Priznakova matice vzorku v tomto uzlu.
        y : np.ndarray
            Popisky vzorku v tomto uzlu.
        depth : int
            Hloubka aktualniho uzlu (koren ma ``0``).

        Navratova hodnota
        -----------------
        Node
            Koren postaveneho (pod)stromu — list nebo vnitrni uzel.
        """
        # assert  Ověřte, že self.n_samples_in_ neni None (volano az z fit)
        # assert  Ověřte, že len(x) == len(y)
        raise NotImplementedError(
            "Úkol: vratte list Node(value=vetsinova trida) pri cistem uzlu / dosazene "
            "max_depth / len(y) < 2 / feature is None z _best_split (NIKDY kvuli nulovemu "
            "zisku); jinak akumulujte self._importances_[feature] += "
            "(len(y)/self.n_samples_in_)*gain, "
            "rozdelte maskou x[:, feature] <= threshold, rekurzivne sestavte potomky s depth+1 "
            "a vratte vnitrni Node(feature, threshold, left, right) bez value."
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """Pripravi stav a preda rizeni ``_build``; PREDVYPLNENO.

        Metoda ``fit`` sama nic nepocita. Jen prevede vstupy na pole, ulozi
        ``n_features_in_``, ``n_samples_in_`` a ``classes_``, vynuluje vektor
        ``_importances_`` a zavola ``self._build(x, y, 0)``, ktery strom
        skutecne postavi. Vektor ``self._importances_`` se plni **az uvnitr
        ``_build``** jako vedlejsi produkt hledani rezu — pri kazdem provedenem
        rezu se do nej pricte vazeny zisk.

        ``self.n_samples_in_`` je nutne ulozit prave zde: ``_build`` ho potrebuje
        jako jmenovatel vahy ``len(y) / n_samples_in_`` pri akumulaci dulezitosti.

        Parametry
        ---------
        x : np.ndarray
            Trenovaci priznakova matice tvaru ``(n_samples, n_features)``.
        y : np.ndarray
            Cilovy vektor delky ``n_samples`` s celociselnymi tridami.

        Navratova hodnota
        -----------------
        DecisionTree
            Tato instance (``self``), aby slo retezit ``.fit(...).predict(...)``.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y).ravel().astype(int)
        self.n_features_in_ = x.shape[1]
        self.n_samples_in_ = x.shape[0]          # POZOR: nutne pro vahu pri akumulaci dulezitosti
        self.classes_ = np.unique(y)
        self._importances_ = np.zeros(self.n_features_in_, dtype=np.float64)
        self.root_ = self._build(x, y, 0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Zaradi kazdy radek ``x`` pruchodem stromu od korene k listu.

        Pro kazdy radek ``x`` zvlast: zacni v ``self.root_`` a dokud jsi ve
        vnitrnim uzlu (``node.value is None``), jdi doleva pri
        ``x[node.feature] <= node.threshold``, jinak doprava. V listu vrat
        ``node.value``.

        Parametry
        ---------
        x : np.ndarray
            Priznakova matice tvaru ``(n_samples, n_features)`` se stejnym
            poctem priznaku jako trenovaci data.

        Navratova hodnota
        -----------------
        np.ndarray
            Predikovane tridy tvaru ``(n_samples,)`` typu ``int``.
        """
        # assert  Ověřte, že model je nafitovan (self.root_ není None)
        # assert  Ověřte, že x.shape[1] == self.n_features_in_
        raise NotImplementedError(
            "Úkol: pro kazdy radek projdete strom od self.root_; ve vnitrnim uzlu jdete "
            "doleva pri x[node.feature] <= node.threshold, jinak doprava; v listu vratte "
            "node.value. Vratte np.ndarray tvaru (n_samples,) typu int."
        )

    @property
    def feature_importances_(self) -> np.ndarray:
        """Dulezitost priznaku — akumulovany vazeny zisk normalizovany na soucet 1.

        Vrati ``self._importances_`` vydelene jejich souctem, takze slozky
        daji dohromady ``1``. Je-li soucet nulovy (strom je jediny list, zadny
        rez se neprovedl), vrati pole nul **beze zmeny** — bez deleni nulou.

        Navratova hodnota
        -----------------
        np.ndarray
            Nezaporny vektor delky ``n_features_in_``, souctem ``1`` (nebo cely
            nulovy).
        """
        # assert  Ověřte, že model je nafitovan (self._importances_ není None)
        raise NotImplementedError(
            "Úkol: vratte self._importances_ vydelene souctem (soucet 1); pri nulovem "
            "souctu vratte pole nul beze zmeny, nedelte nulou."
        )

    def _to_dict(self) -> dict:
        """Prevede nauceny strom na slovnik serializovatelny do JSON.

        Slovnik obsahuje:

        * hyperparametry: ``max_depth``, ``max_features``, ``criterion``,
          ``random_state``,
        * naucene atributy: ``n_features_in_``, ``n_samples_in_``, ``classes_``,
          ``_importances_``,
        * strom pod klicem ``"root"`` — rekurzivne prevedeny uzel jako
          ``{"feature": ..., "threshold": ..., "left": ..., "right": ..., "value": ...}``,
          kde ``left`` / ``right`` jsou opet takove slovniky, nebo ``None`` u listu.

        POZOR — JSON neumi typy numpy
        -----------------------------
        Vsude, kde hodnota pochazi z numpy pole nebo numpy skalaru, ji preved
        na cisty Python: ``int(...)`` pro tridy a indexy, ``float(...)`` pro
        prahy a dulezitosti, ``.tolist()`` pro cela pole (``classes_``,
        ``_importances_``). Jinak ``json.dump`` skonci ``TypeError:
        Object of type int64 is not JSON serializable``.

        Navratova hodnota
        -----------------
        dict
            Zanoreny slovnik obsahujici jen typy, ktere umi ``json``.
        """
        # assert  Ověřte, že model je nafitovan (self.root_ není None)
        raise NotImplementedError(
            "Úkol: sestavte slovnik s hyperparametry, naucenymi atributy a rekurzivne "
            "prevedenym stromem pod klicem 'root'; vsude pouzijte int()/float()/.tolist(), "
            "protoze json neumi numpy typy."
        )

    @classmethod
    def _from_dict(cls, data: dict) -> "DecisionTree":
        """Sestavi ``DecisionTree`` ze slovniku — presna inverze ``_to_dict``.

        Postup: vytvor instanci s ulozenymi hyperparametry
        (``max_depth``, ``max_features``, ``criterion``, ``random_state``),
        obnov naucene atributy (``n_features_in_``, ``n_samples_in_``,
        ``classes_`` jako ``np.ndarray``, ``_importances_`` jako ``np.ndarray``)
        a rekurzivne slozi strom z vnorenych slovniku zpet na uzly ``Node``
        (list, kdyz je ``value`` vyplnene, jinak vnitrni uzel s obnovenym
        ``left`` a ``right``).

        Kontrakt
        --------
        ``t.save(p)`` -> ``DecisionTree.load(p)`` -> ``.predict(x)`` musi dat
        **tytez predikce** jako puvodni ``t.predict(x)``.

        Parametry
        ---------
        data : dict
            Slovnik ve tvaru, jaky vraci ``_to_dict``.

        Navratova hodnota
        -----------------
        DecisionTree
            Nova instance s obnovenym naucenym stavem, pripravena k ``predict``.
        """
        # assert  Ověřte, že data obsahuje klic "root"
        raise NotImplementedError(
            "Úkol: vytvorte instanci s ulozenymi hyperparametry, obnovte naucene atributy "
            "(classes_ a _importances_ zpet na np.ndarray) a rekurzivne slozte strom "
            "z vnorenych slovniku na uzly Node; save -> load -> predict musi dat tytez predikce."
        )
