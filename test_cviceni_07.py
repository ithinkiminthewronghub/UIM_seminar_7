# -*- coding: utf-8 -*-

"""
Created on 10. 09. 2026 at 12:10:00

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:

    Testy pro cviceni 07 -- rozhodovaci stromy a nahodny les.

    Spousteni:  pytest -v

    Ve stavu stubu se sada NACTE a jednotlive testy, ktere volaji nedokoncene
    ukoly, se oznaci jako xfail (ocekavane selhani s NotImplementedError) --
    sada nikdy neskonci holym tracebackem. Po dokonceni ukolu se z nich stanou
    xpass a nasledne plne prochazejici testy.

    Toto cviceni NEMA branu Distance -- strom nepocita parove vzdalenosti, takze
    v testech neni zadna DummyDistance (neni co odpojovat).

    Strom se porovnava se sklearn.tree.DecisionTreeClassifier (presnost na
    oddelitelnych datech, tolerance na remizy pri volbe rovnocennych rezu) a na
    hradlech AND / OR / XOR, kde se ma naucit presne -- XOR je kontrola, ze se
    _build nezastavuje na nulovem zisku. Nahodny les se porovnava se
    sklearn.ensemble.RandomForestClassifier v radu presnosti a testuje se, ze
    KOMPONUJE strom (neni jeho podtridou). Perzistence overuje round-trip
    save -> load pres JSON pro strom i les.
================================================================================
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.tree import DecisionTreeClassifier as SKDecisionTree

from src.base import Classifier
from src.decision_tree import DecisionTree
from src.node import Node
from src.random_forest import RandomForest

STUB = pytest.mark.xfail(raises=NotImplementedError, strict=False,
                         reason="studentsky ukol jeste neni dokoncen")


# --------------------------------------------------------------------------- #
#  Fixtury                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def gate_xor() -> tuple[np.ndarray, np.ndarray]:
    """Hradlo XOR -- ctyri body, neni linearne separovatelne."""
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    return x, y


@pytest.fixture
def gate_and() -> tuple[np.ndarray, np.ndarray]:
    """Hradlo AND -- ctyri body."""
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 0, 1])
    return x, y


@pytest.fixture
def separable_2class() -> tuple[np.ndarray, np.ndarray]:
    """Dobre oddelene dvojrozmerne shluky dvou trid."""
    rng = np.random.default_rng(42)
    n = 60
    x0 = rng.normal(-2.0, 0.6, size=(n, 2))
    x1 = rng.normal(2.0, 0.6, size=(n, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * n + [1] * n)
    return x, y


@pytest.fixture
def blobs_3feat() -> tuple[np.ndarray, np.ndarray]:
    """Tri priznaky, dve tridy, jen prvni dva priznaky nesou signal."""
    rng = np.random.default_rng(7)
    n = 80
    x = rng.normal(0.0, 1.0, size=(2 * n, 3))
    x[:n, 0] -= 3.0
    x[n:, 0] += 3.0
    x[:n, 1] -= 2.0
    x[n:, 1] += 2.0
    y = np.array([0] * n + [1] * n)
    return x, y


# --------------------------------------------------------------------------- #
#  Necistota a informacni zisk                                               #
# --------------------------------------------------------------------------- #
class TestImpurity:
    """Studentske metody ``_impurity`` a ``_information_gain``."""

    @STUB
    def test_gini_ciste_skupiny_je_nula(self) -> None:
        """Gini ciste skupiny (jedna trida) je presne 0."""
        t = DecisionTree(criterion="gini")
        assert t._impurity(np.array([1, 1, 1, 1])) == pytest.approx(0.0)

    @STUB
    def test_gini_znameho_rozlozeni(self) -> None:
        """Gini pro ``{0,0,0,1}`` je 0.375 a pro vyvazenou dvojici 0.5."""
        t = DecisionTree(criterion="gini")
        assert t._impurity(np.array([0, 0, 0, 1])) == pytest.approx(0.375)
        assert t._impurity(np.array([0, 0, 1, 1])) == pytest.approx(0.5)

    @STUB
    def test_gini_prazdne_skupiny_je_nula(self) -> None:
        """Gini prazdne skupiny je 0 (bez deleni nulou)."""
        t = DecisionTree(criterion="gini")
        assert t._impurity(np.array([], dtype=int)) == pytest.approx(0.0)

    @STUB
    def test_information_gain_znameho_rezu(self, gate_and: tuple[np.ndarray, np.ndarray]) -> None:
        """Zisk rezu ``x2 <= 0.5`` na hradle AND je 0.125 (viz README, odd. 3)."""
        t = DecisionTree(criterion="gini")
        _, y = gate_and
        parent = y
        left = np.array([0, 0])     # x2 = 0
        right = np.array([0, 1])    # x2 = 1
        assert t._information_gain(parent, left, right) == pytest.approx(0.125)

    @STUB
    def test_information_gain_je_vazen_poctem_vzorku(self) -> None:
        """Odstepnuti jednoho cisteho vzorku ma nizsi zisk nez vyvazeny rez."""
        t = DecisionTree(criterion="gini")
        parent = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # vyvazeny rez: cista polovina vlevo, cista vpravo -> zisk = 0.5
        vyvazeny = t._information_gain(parent, np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1]))
        # odsteplo se jen jedno "1" doprava
        odsteple = t._information_gain(
            parent, np.array([0, 0, 0, 0, 1, 1, 1]), np.array([1])
        )
        assert vyvazeny == pytest.approx(0.5)
        assert odsteple < vyvazeny


# --------------------------------------------------------------------------- #
#  Nejlepsi rez                                                              #
# --------------------------------------------------------------------------- #
class TestBestSplit:
    """Studentska metoda ``_best_split``."""

    @STUB
    def test_najde_spravny_priznak_a_prah(
        self, blobs_3feat: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Na dobre oddelenych datech je nejlepsi rez na priznaku 0 poblíz nuly."""
        x, y = blobs_3feat
        t = DecisionTree()
        t.n_features_in_ = x.shape[1]
        t.n_samples_in_ = x.shape[0]
        feature, threshold, gain = t._best_split(x, y)
        assert feature == 0
        assert abs(threshold) < 1.5
        assert gain > 0.0

    @STUB
    def test_konstantni_vstup_vraci_prazdny_rez(self) -> None:
        """Ma-li kazdy priznak v uzlu jedinou hodnotu, vrati se ``(None, None, 0.0)``."""
        x = np.ones((6, 3))
        y = np.array([0, 0, 0, 1, 1, 1])
        t = DecisionTree()
        t.n_features_in_ = 3
        t.n_samples_in_ = 6
        feature, threshold, gain = t._best_split(x, y)
        assert feature is None
        assert threshold is None
        assert gain == pytest.approx(0.0)

    @STUB
    def test_max_features_omezuje_pocet_uvazovanych_priznaku(
        self, blobs_3feat: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Pri ``max_features=1`` musi rez padnout na jeden z priznaku 0..2 a byt platny."""
        x, y = blobs_3feat
        t = DecisionTree(max_features=1, random_state=0)
        t.n_features_in_ = x.shape[1]
        t.n_samples_in_ = x.shape[0]
        feature, threshold, _ = t._best_split(x, y)
        assert feature in (0, 1, 2)
        assert threshold is not None


# --------------------------------------------------------------------------- #
#  Cely rozhodovaci strom                                                    #
# --------------------------------------------------------------------------- #
class TestDecisionTree:
    """Stavba stromu, predikce a shoda se sklearn."""

    @STUB
    def test_shoda_se_sklearn_na_oddelitelnych_datech(
        self, separable_2class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Vlastni strom i sklearn strom klasifikuji oddelitelna data bezchybne."""
        x, y = separable_2class
        mine = DecisionTree(max_depth=5, random_state=0).fit(x, y).predict(x)
        ref = SKDecisionTree(max_depth=5, random_state=0).fit(x, y).predict(x)
        assert np.mean(np.asarray(mine).ravel() == y) >= 0.98
        assert np.mean(ref == y) >= 0.98

    @STUB
    @pytest.mark.parametrize(
        "y_gate",
        [
            pytest.param(np.array([0, 0, 0, 1]), id="AND"),
            pytest.param(np.array([0, 1, 1, 1]), id="OR"),
            pytest.param(np.array([0, 1, 1, 0]), id="XOR"),
        ],
    )
    def test_hradla_se_nauci_presne(self, y_gate: np.ndarray) -> None:
        """AND, OR i XOR se strom nauci se 100% presnosti.

        XOR je klicovy: zisk obou rezu v koreni je 0, takze pokud by se ``_build``
        zastavil na nulovem zisku, XOR by nedal 100 %.
        """
        x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        pred = DecisionTree(random_state=0).fit(x, y_gate).predict(x)
        assert np.array_equal(np.asarray(pred).ravel(), y_gate)

    @STUB
    def test_predict_ma_spravny_tvar(
        self, separable_2class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """predict vraci 1D pole delky poctu radku vstupu."""
        x, y = separable_2class
        pred = DecisionTree(max_depth=3, random_state=0).fit(x, y).predict(x[:10])
        assert np.asarray(pred).shape == (10,)

    @STUB
    def test_predict_bez_fit_vyvola_chybu(self) -> None:
        """predict pred fit musi selhat (assert v kostre), ne vratit nesmysl."""
        t = DecisionTree()
        with pytest.raises((AssertionError, AttributeError, ValueError, TypeError)):
            t.predict(np.zeros((3, 2)))

    @STUB
    def test_fit_vraci_self(self, gate_and: tuple[np.ndarray, np.ndarray]) -> None:
        """fit vraci tutez instanci (aby slo retezit .fit(...).predict(...))."""
        x, y = gate_and
        t = DecisionTree(random_state=0)
        assert t.fit(x, y) is t

    @STUB
    def test_koren_je_node(self, gate_and: tuple[np.ndarray, np.ndarray]) -> None:
        """Po fit je ``root_`` instance ``Node``."""
        x, y = gate_and
        t = DecisionTree(random_state=0).fit(x, y)
        assert isinstance(t.root_, Node)


# --------------------------------------------------------------------------- #
#  Dulezitost priznaku                                                       #
# --------------------------------------------------------------------------- #
class TestFeatureImportance:
    """Studentska property ``feature_importances_`` (strom)."""

    @STUB
    def test_ma_spravnou_delku_a_scita_se_na_jedna(
        self, blobs_3feat: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Vektor dulezitosti ma delku n_features, je nezaporny a scita se na 1."""
        x, y = blobs_3feat
        imp = DecisionTree(max_depth=5, random_state=0).fit(x, y).feature_importances_
        imp = np.asarray(imp)
        assert imp.shape == (3,)
        assert np.all(imp >= -1e-12)
        assert imp.sum() == pytest.approx(1.0)

    @STUB
    def test_nedulezity_priznak_ma_malou_dulezitost(
        self, blobs_3feat: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Treti priznak (samy sum) ma vyrazne mensi dulezitost nez prvni dva."""
        x, y = blobs_3feat
        imp = DecisionTree(max_depth=5, random_state=0).fit(x, y).feature_importances_
        imp = np.asarray(imp)
        assert imp[2] < max(imp[0], imp[1])

    @STUB
    def test_shoda_se_sklearn_na_hradle_and(
        self, gate_and: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Na hradle AND se serazene dulezitosti shoduji se sklearn (az na volbu rezu)."""
        x, y = gate_and
        mine = np.sort(np.asarray(
            DecisionTree(random_state=0).fit(x, y).feature_importances_
        ))
        ref = np.sort(SKDecisionTree(random_state=0).fit(x, y).feature_importances_)
        assert mine == pytest.approx(ref, abs=1e-9)


# --------------------------------------------------------------------------- #
#  Nahodny les                                                               #
# --------------------------------------------------------------------------- #
class TestRandomForest:
    """Stavba lesa, hlasovani, kompozice a prumerovana dulezitost."""

    def test_les_neni_podtridou_stromu(self) -> None:
        """RandomForest KOMPONUJE DecisionTree -- nesmi z nej dedit (neni ukol)."""
        assert issubclass(RandomForest, Classifier)
        assert not issubclass(RandomForest, DecisionTree)

    def test_les_drzi_stromy_v_seznamu(self) -> None:
        """Atribut ``trees_`` je seznam (pred fit prazdny) -- struktura kompozice."""
        rf = RandomForest(n_estimators=5)
        assert isinstance(rf.trees_, list)
        assert rf.trees_ == []

    @STUB
    def test_shoda_se_sklearn_v_radu_presnosti(
        self, separable_2class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Vlastni les i sklearn les klasifikuji oddelitelna data temer bezchybne."""
        x, y = separable_2class
        mine = RandomForest(n_estimators=25, random_state=0).fit(x, y).predict(x)
        ref = SKRandomForest(n_estimators=25, random_state=0).fit(x, y).predict(x)
        assert np.mean(np.asarray(mine).ravel() == y) >= 0.95
        assert np.mean(ref == y) >= 0.95

    @STUB
    def test_les_neni_horsi_nez_jeden_strom(
        self, blobs_3feat: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Les ze 30 stromu neni na trenovacich datech vyrazne horsi nez jeden strom."""
        x, y = blobs_3feat
        strom_acc = np.mean(
            DecisionTree(max_depth=4, random_state=0).fit(x, y).predict(x) == y
        )
        les_acc = np.mean(
            RandomForest(n_estimators=30, max_depth=4, random_state=0).fit(x, y).predict(x) == y
        )
        assert les_acc >= strom_acc - 0.05

    @STUB
    def test_prumerovana_dulezitost_se_scita_na_jedna(
        self, blobs_3feat: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """feature_importances_ lesa (prumer stromu) je nezaporny vektor souctem 1."""
        x, y = blobs_3feat
        imp = np.asarray(
            RandomForest(n_estimators=15, random_state=0).fit(x, y).feature_importances_
        )
        assert imp.shape == (3,)
        assert np.all(imp >= -1e-12)
        assert imp.sum() == pytest.approx(1.0)

    @STUB
    def test_predict_ma_spravny_tvar(
        self, separable_2class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """predict lesa vraci 1D pole delky poctu radku vstupu."""
        x, y = separable_2class
        pred = RandomForest(n_estimators=10, random_state=0).fit(x, y).predict(x[:12])
        assert np.asarray(pred).shape == (12,)


# --------------------------------------------------------------------------- #
#  Perzistence (JSON round-trip)                                             #
# --------------------------------------------------------------------------- #
class TestPersistence:
    """save -> load pres JSON pro strom i les (kontrakt: shodne predikce)."""

    @STUB
    def test_strom_round_trip(
        self, tmp_path, separable_2class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Znovunacteny strom dava stejne predikce jako puvodni."""
        x, y = separable_2class
        strom = DecisionTree(max_depth=4, random_state=0).fit(x, y)
        p = tmp_path / "strom.json"
        strom.save(str(p))
        obnoveny = DecisionTree.load(str(p))
        assert np.array_equal(
            np.asarray(strom.predict(x)).ravel(),
            np.asarray(obnoveny.predict(x)).ravel(),
        )

    @STUB
    def test_strom_ulozeny_soubor_je_validni_json(
        self, tmp_path, gate_and: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """save vytvori soubor, ktery jde nacist jako JSON (ne pickle)."""
        import json

        x, y = gate_and
        strom = DecisionTree(random_state=0).fit(x, y)
        p = tmp_path / "strom.json"
        strom.save(str(p))
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "root" in data

    @STUB
    def test_les_round_trip(
        self, tmp_path, separable_2class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Znovunacteny les dava stejne predikce jako puvodni."""
        x, y = separable_2class
        les = RandomForest(n_estimators=8, random_state=0).fit(x, y)
        p = tmp_path / "les.json"
        les.save(str(p))
        obnoveny = RandomForest.load(str(p))
        assert np.array_equal(
            np.asarray(les.predict(x)).ravel(),
            np.asarray(obnoveny.predict(x)).ravel(),
        )
