"""Nahodny les -- ansambl rozhodovacich stromu slozeny KOMPOZICI.

Jeden hluboky rozhodovaci strom ma nizke zkresleni, ale vysoky rozptyl: mala
zmena trenovacich dat muze zmenit korenovy rez a tim cely strom pod nim.
Nahodny les tuto vadu tlumi prumerovanim mnoha stromu.

Bagging a nahodny les
---------------------
* **Bagging** (bootstrap aggregating): natrenuj ``n_estimators`` stromu, kazdy
  na vlastnim bootstrapovem vyberu (``n`` vzorku losovanych s opakovanim), a
  predikuj vetsinovym hlasem.
* **Nahodny les = bagging + losovani podmnoziny priznaku v kazdem uzlu.** Ten
  druhy krok (parametr ``max_features``) stromy **dekoreluje** -- brani tomu,
  aby si vsechny vybraly do korene tyz silny priznak a byly si navzajem
  podobne. Prumer korelovanych stromu se totiz da vylepsit jen omezene.

Bagging vs. boosting
--------------------
Bagging stavi stromy **paralelne a nezavisle** a snizuje **rozptyl**; boosting
je stavi **sekvencne**, kazdy dalsi opravuje chyby predchozich, a snizuje
**zkresleni**. Do baggingu patri hluboke stromy, do boostingu melke. Boosting
se v tomto cviceni jen demonstruje (viz ``cviceni_07.py``), neimplementuje.

Kompozice, ne dedicnost
-----------------------
``RandomForest`` **neni** potomkem ``DecisionTree``. Obe tridy jsou sourozenci
-- potomci spolecneho rozhrani ``Classifier``. Les stromy pouze **drzi**
(``self.trees_``) a vola jejich metody; sam zadnou stromovou rekurzi nedela.

Rozdeleni prace
---------------
``__init__`` je PREDVYPLNENY. Sest metod je ukol: ``_bootstrap_sample``,
``fit``, ``predict``, ``feature_importances_``, ``_to_dict``, ``_from_dict``.
Property ``oob_score_`` je bonus.
"""

from __future__ import annotations

import numpy as np

from src.base import Classifier
from src.decision_tree import DecisionTree


class RandomForest(Classifier):
    """Nahodny les (potomek ``Classifier``, KOMPONUJE ``DecisionTree``).

    Hyperparametry (ulozene v ``__init__``):

    * ``n_estimators`` : int -- pocet stromu v lese.
    * ``max_features`` : int | None -- kolik priznaku se losuje v kazdem uzlu
      KAZDEHO stromu; ``None`` = vsechny (pak jde jen o bagging, ne o pravy
      nahodny les).
    * ``max_depth`` : int | None -- maximalni hloubka kazdeho stromu.
    * ``random_state`` : int | None -- seed pro ``self._rng`` (bootstrap i
      odvozene seedy jednotlivych stromu).

    Atributy naucene ve ``fit`` (konvence sklearn -- podtrzitko na konci):

    * ``trees_`` : list[DecisionTree] -- natrenovany ansambl.
    * ``n_features_in_`` : int -- pocet priznaku trenovaci matice.
    * ``classes_`` : np.ndarray -- setridene unikatni tridy.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: int | None = None,
        max_depth: int | None = None,
        random_state: int | None = None,
    ) -> None:
        """Ulozi hyperparametry a pripravi generator nahody; PREDVYPLNENO.

        Zadny vypocet se tu nedeje -- jen se zapamatuji hyperparametry, vytvori
        se ``self._rng`` a naucene atributy se nastavi na ``None`` / prazdne.

        Parametry
        ---------
        n_estimators : int
            Pocet stromu v lese.
        max_features : int | None
            Pocet priznaku losovanych v kazdem uzlu kazdeho stromu.
            ``None`` = vsechny (cisty bagging).
        max_depth : int | None
            Maximalni hloubka jednotlivych stromu. ``None`` = bez omezeni.
        random_state : int | None
            Seed pro ``numpy.random.default_rng``. ``None`` = nedeterministicke.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state

        # Generator nahody pro bootstrap a pro odvozeni seedu jednotlivych stromu.
        self._rng = np.random.default_rng(random_state)

        # Atributy naucene ve fit(); do zavolani fit() jsou None / prazdne.
        self.trees_: list[DecisionTree] = []
        self.n_features_in_: int | None = None
        self.classes_: np.ndarray | None = None

    def _bootstrap_sample(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vylosuje jeden bootstrapovy vyber ``(x_b, y_b)`` -- ``n`` vzorku s opakovanim.

        Z rozsahu ``0 .. n-1`` (kde ``n = len(y)``) se vylosuje ``n`` indexu
        **s opakovanim** (napr. ``self._rng.integers(0, n, size=n)``), takze
        nektere vzorky se ve vyberu objevi vicekrat a jine vubec. Vrati se
        ``(x[indices], y[indices])``.

        Je to tyz bootstrap jako v cviceni 06, jen pouzity k jinemu ucelu: tam
        slouzil k validaci, tady k vyrobe RUZNYCH trenovacich mnozin ze stejnych
        dat -- kazdy strom lesa dostane trochu jina data, a proto se stromy
        navzajem lisi.

        Parametry
        ---------
        x : np.ndarray
            Priznakova matice tvaru ``(n_samples, n_features)``.
        y : np.ndarray
            Cilovy vektor delky ``n_samples``.

        Navratova hodnota
        -----------------
        tuple[np.ndarray, np.ndarray]
            Dvojice ``(x_b, y_b)`` stejnych tvaru jako vstup.
        """
        assert x.shape[0] == len(y)
        n = len(y)
        indices = self._rng.integers(0, n, size=n)
        return x[indices], y[indices]

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForest":
        """Natrenuje ``n_estimators`` stromu, kazdy na vlastnim bootstrapovem vyberu.

        Postup:

        1. Preved ``x`` na ``float64`` a ``y`` na cela cisla; uloz
           ``self.n_features_in_`` a ``self.classes_``; vyprazdni
           ``self.trees_``.
        2. ``n_estimators``-krat:

           a. ``x_b, y_b = self._bootstrap_sample(x, y)``,
           b. vytvor ``strom = DecisionTree(max_depth=self.max_depth,
              max_features=self.max_features, random_state=<odvozeny seed>)``,
           c. ``strom.fit(x_b, y_b)`` a ``self.trees_.append(strom)``.

        Kazdy strom musi dostat **jiny** seed (napr. vylosovany pres
        ``self._rng.integers``), jinak by vsechny stromy losovaly tytez
        priznaky ve tychz uzlech, byly by temer totozne a losovani priznaku by
        ztratilo smysl.

        Parametry
        ---------
        x : np.ndarray
            Trenovaci priznakova matice tvaru ``(n_samples, n_features)``.
        y : np.ndarray
            Cilovy vektor delky ``n_samples`` s celociselnymi tridami.

        Navratova hodnota
        -----------------
        RandomForest
            Tato instance (``self``).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y).ravel().astype(int)
        assert x.ndim == 2 and x.shape[0] == len(y)

        self.n_features_in_ = x.shape[1]
        self.classes_ = np.unique(y)
        self.trees_ = []

        tree_seeds = self._rng.integers(0, 2 ** 31 - 1, size=self.n_estimators)

        for i in range(self.n_estimators):
            x_b, y_b = self._bootstrap_sample(x, y)
            tree = DecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=int(tree_seeds[i]),
            )
            tree.fit(x_b, y_b)
            self.trees_.append(tree)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Zaradi kazdy radek ``x`` vetsinovym hlasem napric stromy lesa.

        Kazdy strom v ``self.trees_`` predikuje cele ``x``; predikce se slozi do
        matice tvaru ``(n_stromu, n_samples)`` a pro kazdy sloupec (vzorek) se
        vrati **nejcastejsi trida**.

        Reseni remizy
        -------------
        Ma-li nejvyssi pocet hlasu vic trid zaroven, vrat tridu s NEJNIZSIM
        ciselnym oznacenim (deterministicky, napr. pres ``np.unique`` +
        ``np.argmax`` nad cetnostmi).

        Parametry
        ---------
        x : np.ndarray
            Priznakova matice tvaru ``(n_samples, n_features)``.

        Navratova hodnota
        -----------------
        np.ndarray
            Predikovane tridy tvaru ``(n_samples,)`` typu ``int``.
        """
        assert len(self.trees_) > 0
        x = np.asarray(x, dtype=np.float64)
        assert x.shape[1] == self.n_features_in_

        n_samples = x.shape[0]
        predictions_matrix = np.zeros((len(self.trees_), n_samples), dtype=int)

        for i, tree in enumerate(self.trees_):
            predictions_matrix[i] = tree.predict(x)

        final_predictions = np.zeros(n_samples, dtype=int)
        for j in range(n_samples):
            col = predictions_matrix[:, j]
            classes, counts = np.unique(col, return_counts=True)
            max_count = np.max(counts)
            best_classes = classes[counts == max_count]
            final_predictions[j] = int(np.min(best_classes))

        return final_predictions

    @property
    def feature_importances_(self) -> np.ndarray:
        """Dulezitost priznaku lesa -- prumer ``feature_importances_`` pres vsechny stromy.

        Kazdy strom vraci vlastni vektor dulezitosti (souctem 1). Les je
        zprumeruje pres cely ansambl. Prave prumerovani dela tuto miru
        pouzitelnou: jednotlivy strom je nestabilni, ale prumer pres stovku
        stromu na ruznych bootstrapovych vyberech uz je rozumne stabilni.

        Prumer je opet souctem 1 (kazdy scitanec dava 1), pro jistotu ho
        normalizujte znovu.

        Navratova hodnota
        -----------------
        np.ndarray
            Nezaporny vektor delky ``n_features_in_``, souctem ``1``.
        """
        assert len(self.trees_) > 0

        importances = np.zeros(self.n_features_in_, dtype=np.float64)
        for tree in self.trees_:
            importances += tree.feature_importances_

        importances /= len(self.trees_)
        total = np.sum(importances)
        if total > 0.0:
            importances /= total

        return importances

    def _to_dict(self) -> dict:
        """Prevede nauceny les na slovnik serializovatelny do JSON.

        Slovnik obsahuje hyperparametry lesa (``n_estimators``, ``max_features``,
        ``max_depth``, ``random_state``), naucene atributy (``n_features_in_``,
        ``classes_``) a seznam slovniku jednotlivych stromu pod klicem
        ``"trees"``.

        DELEGUJTE na strom
        ------------------
        Serializaci uzlu NEPISTE znovu -- kazdy strom uz to umi. Slovnik stromu
        ziskate jako ``strom._to_dict()`` a jen je posbirejte do seznamu.

        Navratova hodnota
        -----------------
        dict
            Zanoreny slovnik obsahujici jen typy, ktere umi ``json`` (numpy
            hodnoty preved pres ``int()`` / ``.tolist()``).
        """
        assert len(self.trees_) > 0

        return {
            "n_estimators": int(self.n_estimators),
            "max_features": self.max_features,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "n_features_in_": int(self.n_features_in_),
            "classes_": self.classes_.tolist(),
            "trees": [tree._to_dict() for tree in self.trees_],
        }

    @classmethod
    def _from_dict(cls, data: dict) -> "RandomForest":
        """Sestavi ``RandomForest`` ze slovniku -- presna inverze ``_to_dict``.

        Postup: vytvor instanci s ulozenymi hyperparametry, obnov naucene
        atributy (``n_features_in_``, ``classes_`` jako ``np.ndarray``) a
        naplni ``self.trees_`` tak, ze KAZDY slovnik ze seznamu ``"trees"``
        predas do ``DecisionTree._from_dict`` (opet delegace na strom).

        Kontrakt
        --------
        ``rf.save(p)`` -> ``RandomForest.load(p)`` -> ``.predict(x)`` musi dat
        **tytez predikce** jako puvodni ``rf.predict(x)``.

        Parametry
        ---------
        data : dict
            Slovnik ve tvaru, jaky vraci ``_to_dict``.

        Navratova hodnota
        -----------------
        RandomForest
            Nova instance s obnovenym ansamblem, pripravena k ``predict``.
        """
        assert "trees" in data

        forest = cls(
            n_estimators=data.get("n_estimators", 100),
            max_features=data.get("max_features"),
            max_depth=data.get("max_depth"),
            random_state=data.get("random_state"),
        )
        forest.n_features_in_ = data["n_features_in_"]
        forest.classes_ = np.array(data["classes_"])
        forest.trees_ = [DecisionTree._from_dict(tree_dict) for tree_dict in data["trees"]]

        return forest


@property
def oob_score_(self) -> float:
    """Presnost lesa na out-of-bag vzorcich; BONUS.

    Out-of-bag (OOB) vzorky konkretniho stromu jsou ty, ktere se do jeho
    bootstrapoveho vyberu NEDOSTALY (v prumeru asi 36,8 %). Pro kazdy vzorek
    se poridi predikce jen tech stromu, ktere ho ve svem vyberu nemely, a
    udela se z nich vetsinovy hlas. ``oob_score_`` je podil spravne
    klasifikovanych vzorku (z tech, ktere maji aspon jeden OOB strom).

    Vyzaduje, aby si ``fit`` u kazdeho stromu zapamatoval pouzite
    bootstrapove indexy (jinak nelze OOB mnozinu rekonstruovat).

    Navratova hodnota
    -----------------
    float
        Presnost na OOB vzorcich v intervalu ``[0, 1]``.
    """
    # assert  Ověřte, že les je nafitovan a fit si zapamatoval bootstrapove indexy
    raise NotImplementedError(
        "Úkol: (BONUS) pro kazdy vzorek seberte predikce stromu, ktere ho nemely "
        "ve svem bootstrapovem vyberu, udelejte vetsinovy hlas a vratte podil "
        "spravne klasifikovanych OOB vzorku."
    )
