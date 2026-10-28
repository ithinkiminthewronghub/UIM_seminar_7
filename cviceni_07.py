# -*- coding: utf-8 -*-

"""
Created on 10. 09. 2026 at 11:55:00

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
    Vstupni bod cviceni 07 (rozhodovaci stromy a nahodny les). Pipeline projde
    sest fazi, ktere dohromady tvori oblouk od jednoho stromu k ansamblu:

      1. deleni prostoru: rozhodovaci strom na hradlech AND / OR / XOR
         + rozhodovaci povrch kazdeho hradla,
      2. dulezitost priznaku: strom na datasetu Breast Cancer Wisconsin,
         sloupcovy graf akumulovaneho zisku po priznacich,
      3. perzistence: save -> load stromu pres JSON a overeni shody predikci,
      4. nahodny les: bagging + losovani priznaku, strom vs. les vedle sebe,
         zprumerovana dulezitost priznaku,
      5. bagging vs. boosting: demonstrace sklearn GradientBoostingClassifier
         a kontrast zkresleni / rozptylu (v komentari),
      6. tri rodiny vyberu priznaku: filtracni (cv5) vs. obalova (cv6) vs.
         vestavena (cv7) vedle sebe.

    Repozitar bezi v kazde fazi. Dokud nejsou ukoly hotove, faze se zastavi jen
    hlaskou "[NENI HOTOVO] Úkol: ..." a pipeline pokracuje dal -- nikdy
    nezpracovanym tracebackem.
================================================================================
"""

from __future__ import annotations

import sys

import numpy as np

# --- Import guard: srozumitelna hlaska misto holeho ImportError ----------------
try:
    from dataio.config_manager import load_config
    from dataio.gates import make_gate
    from dataio.loader import load_breast_cancer_data
    from dataio.plotting import (
        plot_decision_surface,
        plot_feature_importance,
        plot_importance_comparison,
        plot_tree_vs_forest,
    )
    from src.decision_tree import DecisionTree
    from src.random_forest import RandomForest
except ImportError as exc:  # pragma: no cover - jen ochranna hlaska
    print(f"[CHYBA IMPORTU] Nepodarilo se nacist moduly projektu: {exc}")
    print("Zkontrolujte, ze spoustite skript z korene repozitare a mate "
          "nainstalovane zavislosti (pip install -r requirements.txt).")
    sys.exit(1)

GRAPHS_DIR = "graphs"        # vystupni grafy (.png)
MODELS_DIR = "models"            # sem se uklada natrenovany strom (.json)
MODEL_PATH = f"{MODELS_DIR}/strom_hradlo_xor.json"


def _banner(text: str) -> None:
    """Vypise oddelovaci nadpis faze pipeline."""
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def _faze_neni_hotova(exc: NotImplementedError) -> None:
    """Vypise pratelskou hlasku, kdyz faze narazi na nedokonceny ukol."""
    print(f"  [NENI HOTOVO] {exc}")
    print("  -> Tuto cast dokoncite v ramci ukolu; pipeline pokracuje dal.")


def _top2_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vrati indexy dvou priznaku s nejvetsim |Cohenovym d| mezi tridami.

    Pouziva se jen k tomu, aby sla vykreslit 2D rozhodovaci plocha na realnych
    datech (strom se pak uci jen z tech dvou priznaku). Zadny import z cv5 --
    velikost ucinku se pocita inline.
    """
    a, b = x[y == 0], x[y == 1]
    pooled_sd = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2.0)
    pooled_sd = np.where(pooled_sd == 0.0, 1.0, pooled_sd)
    cohen_d = np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled_sd
    return np.argsort(cohen_d)[::-1][:2]


def _cohen_d_importance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Filtracni (cv5) mira -- |Cohenovo d| po priznacich, normalizovano na soucet 1."""
    a, b = x[y == 0], x[y == 1]
    pooled_sd = np.sqrt((a.var(axis=0, ddof=1) + b.var(axis=0, ddof=1)) / 2.0)
    pooled_sd = np.where(pooled_sd == 0.0, 1.0, pooled_sd)
    d = np.abs(a.mean(axis=0) - b.mean(axis=0)) / pooled_sd
    total = d.sum()
    return d / total if total > 0 else d


def _single_feature_importance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Obalova (cv6) mira -- presnost jednopriznakoveho referencniho stromu.

    Pro kazdy priznak zvlast se natrenuje maly sklearn strom jen z toho jednoho
    priznaku a zmeri se presnost na trenovacich datech. Neni to plny wrapper
    (ten by prohledaval podmnoziny), ale zachycuje jeho myslenku: hodnotime
    priznak podle toho, jak dobre s nim model klasifikuje. Normalizovano.
    """
    from sklearn.tree import DecisionTreeClassifier

    accs = np.zeros(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        clf = DecisionTreeClassifier(max_depth=3, random_state=0)
        clf.fit(x[:, [j]], y)
        accs[j] = clf.score(x[:, [j]], y)
    accs = np.clip(accs - 0.5, 0.0, None)  # odecti nahodne hadani
    total = accs.sum()
    return accs / total if total > 0 else accs


def faze_hradla(cfg) -> None:
    """Faze 1 -- rozhodovaci strom na hradlech AND / OR / XOR."""
    _banner("Faze 1: Deleni prostoru -- hradla AND / OR / XOR")

    for gate in ("and", "or", "xor"):
        x, y = make_gate(gate)
        try:
            strom = DecisionTree(
                max_depth=cfg.tree.max_depth,
                criterion=cfg.tree.criterion,
                random_state=cfg.data.random_state,
            ).fit(x, y)
            y_pred = strom.predict(x)
            acc = float(np.mean(y_pred == y))
            cesta = f"{GRAPHS_DIR}/hradlo_{gate}.png"
            plot_decision_surface(strom, x, y, f"Hradlo {gate.upper()}", save_path=cesta)
            print(f"  {gate.upper():>3}: presnost na 4 bodech = {acc:.2f}  "
                  f"(XOR je test, ze se nezastavujeme na nulovem zisku)")
            print(f"       rozhodovaci povrch ulozen: {cesta}")
        except NotImplementedError as exc:
            _faze_neni_hotova(exc)
            return


def faze_dulezitost(cfg, x: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    """Faze 2 -- strom na breast cancer + dulezitost priznaku."""
    _banner("Faze 2: Dulezitost priznaku jako vedlejsi produkt uceni")

    try:
        strom = DecisionTree(
            max_depth=cfg.tree.max_depth,
            criterion=cfg.tree.criterion,
            random_state=cfg.data.random_state,
        ).fit(x, y)
        acc = float(np.mean(strom.predict(x) == y))
        imp = strom.feature_importances_
        poradi = np.argsort(imp)[::-1]
        print(
            f"  Strom (max_depth={cfg.tree.max_depth}) presnost na trenovacich datech = {acc:.3f}")
        print("  Peti nejdulezitejsich priznaku:")
        for i in poradi[:5]:
            print(f"    {names[i]:<28} {imp[i]:.3f}")
        print(f"  Soucet dulezitosti = {imp.sum():.3f}  (ma byt 1.0)")
        cesta = f"{GRAPHS_DIR}/dulezitost_priznaku_strom.png"
        plot_feature_importance(imp, names, save_path=cesta)
        print(f"  Graf ulozen: {cesta}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def faze_perzistence(cfg) -> None:
    """Faze 3 -- model = ulozene naucene parametry: save -> load pres JSON."""
    _banner("Model jako ulozene parametry: save / load (JSON, ne .npz)")

    x, y = make_gate("xor")
    try:
        strom = DecisionTree(
            max_depth=cfg.tree.max_depth,
            criterion=cfg.tree.criterion,
            random_state=cfg.data.random_state,
        ).fit(x, y)
        y_pred_puvod = strom.predict(x)

        strom.save(MODEL_PATH)
        obnoveny = DecisionTree.load(MODEL_PATH)
        y_pred_obnov = obnoveny.predict(x)

        shoda = bool(np.array_equal(np.asarray(y_pred_puvod).ravel(),
                                    np.asarray(y_pred_obnov).ravel()))
        print(f"  Strom ulozen do {MODEL_PATH}, znovu nacten.")
        print(f"  load(...).predict(x) == puvodni predikce:  {shoda}")
        print("  -> stav stromu je zanoreny slovnik (uzly), proto JSON, ne pole v .npz.")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def faze_les(cfg, x: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    """Faze 4 -- nahodny les: strom vs. les + zprumerovana dulezitost."""
    _banner("Faze 4: Nahodny les -- bagging + losovani priznaku")

    sel = _top2_features(x, y)
    x2 = x[:, sel]

    try:
        strom = DecisionTree(
            max_depth=cfg.tree.max_depth,
            random_state=cfg.data.random_state,
        ).fit(x2, y)
        les = RandomForest(
            n_estimators=cfg.forest.n_estimators,
            max_features=min(cfg.forest.max_features, 2),
            max_depth=cfg.forest.max_depth,
            random_state=cfg.data.random_state,
        ).fit(x2, y)

        acc_strom = float(np.mean(strom.predict(x2) == y))
        acc_les = float(np.mean(les.predict(x2) == y))
        print(f"  2D data ({names[sel[0]]} vs. {names[sel[1]]}):")
        print(f"    jeden strom -- presnost {acc_strom:.3f}")
        print(f"    les ({cfg.forest.n_estimators} stromu) -- presnost {acc_les:.3f}")
        cesta = f"{GRAPHS_DIR}/strom_vs_les.png"
        plot_tree_vs_forest(strom, les, x2, y, save_path=cesta)
        print(f"    porovnani hranic ulozeno: {cesta}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)

    # --- Zprumerovana dulezitost priznaku pres cely les (vsech 30 priznaku) ---
    try:
        les_plny = RandomForest(
            n_estimators=cfg.forest.n_estimators,
            max_features=cfg.forest.max_features,
            max_depth=cfg.forest.max_depth,
            random_state=cfg.data.random_state,
        ).fit(x, y)
        imp = les_plny.feature_importances_
        poradi = np.argsort(imp)[::-1]
        print("  Zprumerovana dulezitost pres les (top 5):")
        for i in poradi[:5]:
            print(f"    {names[i]:<28} {imp[i]:.3f}")
        cesta = f"{GRAPHS_DIR}/dulezitost_priznaku_les.png"
        plot_feature_importance(imp, names, save_path=cesta)
        print(f"  Graf ulozen: {cesta}")
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)


def faze_boosting(cfg, x: np.ndarray, y: np.ndarray) -> None:
    """Faze 5 -- bagging vs. boosting (boosting jen demonstrovan pres sklearn).

    Kontrast, ktery si stoji za zapamatovani:

      * Bagging / nahodny les: stromy vznikaji PARALELNE a NEZAVISLE, kazdy na
        vlastnim bootstrapovem vyberu; prumer snizuje ROZPTYL. Zakladni model je
        HLUBOKY strom (nizke zkresleni, vysoky rozptyl).
      * Boosting: stromy vznikaji SEKVENCNE, kazdy dalsi se soustredi na vzorky,
        ktere predchozi spletl; skladani snizuje ZKRESLENI. Zakladni model je
        MELKY strom / paren (vysoke zkresleni, nizky rozptyl).

    Boosting se v tomto cviceni NEIMPLEMENTUJE -- jen se spusti referencni
    GradientBoostingClassifier ze sklearn, aby bylo videt, ze i mnoho MELKYCH
    stromu (max_depth=1) da silny model.
    """
    _banner("Faze 5: Bagging vs. boosting (boosting = demonstrace sklearn)")

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.model_selection import train_test_split

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.3, random_state=cfg.data.random_state, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=cfg.forest.n_estimators,
        max_features=cfg.forest.max_features,
        max_depth=cfg.forest.max_depth,
        random_state=cfg.data.random_state,
    ).fit(x_tr, y_tr)
    gb = GradientBoostingClassifier(
        n_estimators=cfg.forest.n_estimators,
        max_depth=1,  # melke stromy -- typicke pro boosting
        random_state=cfg.data.random_state,
    ).fit(x_tr, y_tr)

    print(f"  Bagging (RandomForest, hluboke stromy) -- presnost na testu = "
          f"{rf.score(x_te, y_te):.3f}")
    print(f"  Boosting (GradientBoosting, max_depth=1) -- presnost na testu = "
          f"{gb.score(x_te, y_te):.3f}")
    print("  Obe rodiny sdruzuji stromy, ale resi opacny problem: bagging tlumi "
          "rozptyl hlubokych stromu, boosting tlumi zkresleni melkych.")


def faze_vyber_priznaku(cfg, x: np.ndarray, y: np.ndarray, names: list[str]) -> None:
    """Faze 6 -- tri rodiny vyberu priznaku vedle sebe (cv5 / cv6 / cv7)."""
    _banner("Faze 6: Filtracni (cv5) vs. obalova (cv6) vs. vestavena (cv7)")

    sets: dict[str, np.ndarray] = {
        "cv5 filtr (|Cohen d|)": _cohen_d_importance(x, y),
        "cv6 wrapper (1 priznak)": _single_feature_importance(x, y),
    }
    try:
        strom = DecisionTree(
            max_depth=cfg.tree.max_depth,
            criterion=cfg.tree.criterion,
            random_state=cfg.data.random_state,
        ).fit(x, y)
        sets["cv7 strom (zisk)"] = strom.feature_importances_
    except NotImplementedError as exc:
        _faze_neni_hotova(exc)
        print("  -> porovnani se vykresli az s hotovou dulezitosti stromu; "
              "kreslim aspon filtracni a obalovou.")

    cesta = f"{GRAPHS_DIR}/porovnani_vyberu_priznaku.png"
    plot_importance_comparison(sets, names, save_path=cesta)
    print(f"  Graf ulozen: {cesta}")
    print("  Metody se casto NESHODNOU -- kazda ma jine slepe misto (viz README).")


def main() -> None:
    """Spusti celou pipeline cviceni 07 s ochrannymi bloky u kazde faze."""
    _banner("CVICENI 07 -- Rozhodovaci stromy a nahodny les -- start")

    # --- Config guard -----------------------------------------------------------
    try:
        cfg = load_config()
    except (ValueError, AssertionError, FileNotFoundError) as exc:
        print(f"[CHYBA KONFIGURACE] {exc}")
        sys.exit(1)

    # --- Data loading ---------------------------------------------------------
    x, y, names = load_breast_cancer_data(cfg.data.random_state)
    print(f"  Data: x {x.shape}, malignich vzorku {int(y.sum())} / {len(y)} "
          f"(pozitivni trida = maligni).")

    faze_hradla(cfg)
    faze_dulezitost(cfg, x, y, names)
    faze_perzistence(cfg)
    faze_les(cfg, x, y, names)
    faze_boosting(cfg, x, y)
    faze_vyber_priznaku(cfg, x, y, names)

    _banner("CVICENI 07 -- konec")


if __name__ == "__main__":
    main()
