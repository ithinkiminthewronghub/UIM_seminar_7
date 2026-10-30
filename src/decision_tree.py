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
        assert y.ndim == 1
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)

        if self.criterion == "gini":
            return float(1.0 - np.sum(p ** 2))
        elif self.criterion == "entropy":
            p_nonzero = p[p > 0]
            return float(-np.sum(p_nonzero * np.log2(p_nonzero)))
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")


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
        assert len(left_y) + len(right_y) == len(parent_y)

        n = len(parent_y)
        if n == 0:
            return 0.0

        impurity_parent = self._impurity(parent_y)
        impurity_left = self._impurity(left_y)
        impurity_right = self._impurity(right_y)

        gain = impurity_parent - (len(left_y) / n) * impurity_left - (len(right_y) / n) * impurity_right
        return float(max(0.0, gain))

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
        assert x.ndim == 2 and x.shape[0] == len(y)
        n_samples, n_features = x.shape
        assert self.max_features is None or (1 <= self.max_features <= n_features)

        best_feature = None
        best_threshold = None
        best_gain = 0.0
        found_split = False

        if self.max_features is None:
            features = np.arange(n_features)
        else:
            features = self._rng.choice(n_features, size=self.max_features, replace=False)

        for j in features:
            column = x[:, j]
            unique_vals = np.unique(column)
            if len(unique_vals) <= 1:
                continue

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for t in thresholds:
                mask = column <= t
                left_y = y[mask]
                right_y = y[~mask]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gain = self._information_gain(y, left_y, right_y)

                if not found_split or gain > best_gain:
                    best_gain = gain
                    best_feature = int(j)
                    best_threshold = float(t)
                    found_split = True
                elif not found_split and gain == best_gain:
                    pass

        if not found_split:
            return None, None, 0.0

        return best_feature, best_threshold, best_gain

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
        assert self.n_samples_in_ is not None
        assert len(x) == len(y)

        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        majority_class = int(classes[np.argmax(counts)])

        if (
                len(classes) == 1
                or (self.max_depth is not None and depth >= self.max_depth)
                or n_samples < 2
        ):
            return Node(value=majority_class)

        feature, threshold, gain = self._best_split(x, y)

        if feature is None:
            return Node(value=majority_class)

        self._importances_[feature] += (n_samples / self.n_samples_in_) * gain

        mask = x[:, feature] <= threshold
        left_node = self._build(x[mask], y[mask], depth + 1)
        right_node = self._build(x[~mask], y[~mask], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

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
        assert self.root_ is not None
        x = np.asarray(x, dtype=np.float64)
        assert x.shape[1] == self.n_features_in_

        predictions = []
        for row in x:
            node = self.root_
            while node.value is None:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)

        return np.array(predictions, dtype=int)

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
        assert self._importances_ is not None
        total = np.sum(self._importances_)
        if total == 0.0:
            return self._importances_.copy()
        return self._importances_ / total

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
        assert self.root_ is not None

        def node_to_dict(node: Node | None) -> dict | None:
            if node is None:
                return None
            return {
                "feature": int(node.feature) if node.feature is not None else None,
                "threshold": float(node.threshold) if node.threshold is not None else None,
                "left": node_to_dict(node.left),
                "right": node_to_dict(node.right),
                "value": int(node.value) if node.value is not None else None,
            }

        return {
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "criterion": self.criterion,
            "random_state": self.random_state,
            "n_features_in_": int(self.n_features_in_),
            "n_samples_in_": int(self.n_samples_in_),
            "classes_": self.classes_.tolist(),
            "_importances_": self._importances_.tolist(),
            "root": node_to_dict(self.root_),
        }

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
        assert "root" in data

        tree = cls(
            max_depth=data.get("max_depth"),
            max_features=data.get("max_features"),
            criterion=data.get("criterion", "gini"),
            random_state=data.get("random_state"),
        )
        tree.n_features_in_ = data["n_features_in_"]
        tree.n_samples_in_ = data["n_samples_in_"]
        tree.classes_ = np.array(data["classes_"])
        tree._importances_ = np.array(data["_importances_"], dtype=np.float64)

        def dict_to_node(d: dict | None) -> Node | None:
            if d is None:
                return None
            return Node(
                feature=d["feature"],
                threshold=d["threshold"],
                left=dict_to_node(d["left"]),
                right=dict_to_node(d["right"]),
                value=d["value"],
            )

        tree.root_ = dict_to_node(data["root"])
        return tree
