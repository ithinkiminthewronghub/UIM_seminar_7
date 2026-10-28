"""Vykreslovani vysledku cviceni 07 -- vse PREDVYPLNENE, bohate.

Grafy jsou v tomto cviceni explicitni priorita: rozhodovaci povrch modelu ve
2D (vybarvene oblasti rozhodnuti pres mrizku), sloupcovy graf dulezitosti
priznaku, porovnani hranice stromu vs. lesa vedle sebe a skupinove porovnani
dulezitosti priznaku napric metodami z cviceni 05/06/07.

Vsechny funkce pouzivaji neinteraktivni backend ``Agg``: figuru sestavi,
volitelne ulozi do ``save_path`` (vcetne vytvoreni nadrazeneho adresare) a
vzdy figuru zavrou. Funkce ``plt.show`` se nikdy nevola (bezi to bezhlave).
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # neinteraktivni backend, vykreslujeme jen do souboru

import matplotlib.pyplot as plt  # noqa: E402  (musi az po matplotlib.use)
import numpy as np  # noqa: E402

# Sdilene barvy trid (0 = benigni / nepravda, 1 = maligni / pravda).
_CLASS_COLORS: dict[int, str] = {0: "#2ca02c", 1: "#d62728"}
_CLASS_LABELS: dict[int, str] = {0: "trida 0", 1: "trida 1"}


def _save_and_close(fig: plt.Figure, save_path: str | None) -> None:
    """Pomocna funkce: ulozi figuru do ``save_path`` a zavre ji.

    Pokud je ``save_path`` ``None``, figura se pouze zavre. Nadrazeny
    adresar se v pripade potreby vytvori.
    """
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _draw_surface_on_ax(
    ax: plt.Axes,
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
) -> None:
    """Vykresli rozhodovaci povrch ``model`` do zadaneho ``ax``.

    Pres rovinu dvou priznaku se polozi jemna mrizka ``(300, 300)``, model
    kazdy jeji bod klasifikuje a vysledek se vykresli jako barevne pozadi
    (``contourf``); pres nej jsou trenovaci body. Meze se rozsiruji o okraj,
    aby byly videt i body na kraji dat (u hradel vsechny ctyri rohy ctverce).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)

    x_min, x_max = float(x[:, 0].min()), float(x[:, 0].max())
    y_min, y_max = float(x[:, 1].min()), float(x[:, 1].max())
    x_pad = max(0.5, 0.1 * (x_max - x_min))
    y_pad = max(0.5, 0.1 * (y_max - y_min))

    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, 300),
        np.linspace(y_min - y_pad, y_max + y_pad, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    # ``predict`` vraci pole tvaru (n_bodu,) -- pretvarujeme zpet na tvar mrizky.
    zz = np.asarray(model.predict(grid)).reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, zz, colors="#555555", linewidths=0.8, levels=[0.5])

    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            s=45,
            alpha=0.9,
            color=_CLASS_COLORS.get(int(cls), "#1f77b4"),
            edgecolor="white",
            linewidth=0.5,
            label=_CLASS_LABELS.get(int(cls), f"trida {cls}"),
        )

    ax.set_xlabel("priznak 1")
    ax.set_ylabel("priznak 2")
    ax.set_title(title)
    ax.legend(loc="best")


def plot_decision_surface(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    save_path: str | None = None,
) -> None:
    """Vykresli rozhodovaci povrch modelu ve 2D -- klicovy vizual cviceni.

    Parametry
    ---------
    model:
        Nafitovany klasifikator s metodou ``predict`` (vlastni
        ``DecisionTree`` / ``RandomForest`` i ``sklearn`` model). Musi byt
        trenovany na dvourozmernych datech odpovidajicich ``x``.
    x:
        Priznakova matice tvaru ``(n_samples, 2)`` -- dva priznaky nebo 2D
        projekce. Funguje i pro ctyri body logickeho hradla.
    y:
        Stitky delky ``n_samples`` (cela cisla ``{0, 1}``).
    title:
        Titulek grafu.
    save_path:
        Cesta k vystupnimu PNG, nebo ``None`` (pak se figura jen zavre).

    U hradel AND/OR je hranice jedina primka; u XOR strom vykrouzi
    schodovitou hranici ze dvou osove zarovnanych rezu kolem dvou
    protilehlych rohu.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _draw_surface_on_ax(ax, model, x, y, title)
    _save_and_close(fig, save_path)


def plot_feature_importance(
    importances: Sequence[float],
    feature_names: Sequence[str],
    save_path: str | None = None,
) -> None:
    """Vykresli dulezitosti priznaku jako vodorovny sloupcovy graf.

    Parametry
    ---------
    importances:
        Dulezitosti priznaku (delka ``n_features``), napr.
        ``model.feature_importances_``.
    feature_names:
        Nazvy priznaku (delka ``n_features``).
    save_path:
        Cesta k vystupnimu PNG, nebo ``None``.

    Sloupce jsou serazene sestupne podle dulezitosti. Ma-li dataset hodne
    priznaku (napr. 30 u breast cancer), zobrazi se jen rozumny pocet
    nejdulezitejsich (horni 15) a je to uvedeno v titulku.

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``importances`` a ``feature_names`` nemaji stejnou delku.
    """
    importances = np.asarray(importances, dtype=np.float64)
    names = [str(n) for n in feature_names]
    if importances.shape[0] != len(names):
        raise ValueError(
            "importances a feature_names musi mit stejnou delku, zadano: "
            f"{importances.shape[0]} vs. {len(names)}"
        )

    order = np.argsort(importances)[::-1]
    top_n = min(15, len(names))
    top_idx = order[:top_n]

    if top_n < len(names):
        title = f"Dulezitost priznaku (horni {top_n} z {len(names)})"
    else:
        title = "Dulezitost priznaku"

    # Vodorovne sloupce: nejdulezitejsi nahore.
    pos = np.arange(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 0.4 * top_n + 1.5))
    ax.barh(pos, importances[top_idx], color="#1f77b4", edgecolor="white")
    ax.set_yticks(pos)
    ax.set_yticklabels([names[i] for i in top_idx])
    ax.set_xlabel("dulezitost (podil na celkovem zisku)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    _save_and_close(fig, save_path)


def plot_tree_vs_forest(
    tree: Any,
    forest: Any,
    x: np.ndarray,
    y: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Vykresli rozhodovaci povrch stromu a lesa vedle sebe.

    Parametry
    ---------
    tree:
        Nafitovany rozhodovaci strom s metodou ``predict``.
    forest:
        Nafitovany nahodny les s metodou ``predict``.
    x:
        Priznakova matice tvaru ``(n_samples, 2)``.
    y:
        Stitky delky ``n_samples`` (cela cisla ``{0, 1}``).
    save_path:
        Cesta k vystupnimu PNG, nebo ``None``.

    Oba subploty pouzivaji stejny rozhodovaci povrch jako
    ``plot_decision_surface``. Vlevo je strom (ostra, schodovita hranice),
    vpravo les (prumer mnoha stromu hranici vyhladi).
    """
    fig, (ax_tree, ax_forest) = plt.subplots(1, 2, figsize=(13, 5.5))
    _draw_surface_on_ax(ax_tree, tree, x, y, "Rozhodovaci strom")
    _draw_surface_on_ax(ax_forest, forest, x, y, "Nahodny les")
    fig.suptitle("Strom vs. les: prumer stromu vyhladi rozhodovaci hranici")
    _save_and_close(fig, save_path)


def plot_importance_comparison(
    importance_sets: dict[str, np.ndarray],
    feature_names: Sequence[str],
    save_path: str | None = None,
) -> None:
    """Vykresli skupinove porovnani dulezitosti priznaku napric metodami.

    Parametry
    ---------
    importance_sets:
        Slovnik ``{nazev_metody: dulezitosti}``, kde kazde pole ma delku
        ``len(feature_names)`` (napr. klice ``"cv5 filtr"``, ``"cv6 wrapper"``,
        ``"cv7 strom"``).
    feature_names:
        Nazvy priznaku (delka ``n_features``).
    save_path:
        Cesta k vystupnimu PNG, nebo ``None``.

    Jedna skupina sloupcu na priznak, jedna barva na metodu. Zobrazi se jen
    hornich ``N`` priznaku podle prumeru dulezitosti pres metody.

    Vyjimky
    -------
    ``ValueError``:
        Pokud je ``importance_sets`` prazdny nebo pokud nektera sada nema
        delku ``len(feature_names)``.
    """
    names = [str(n) for n in feature_names]
    if not importance_sets:
        raise ValueError("importance_sets nesmi byt prazdny slovnik")

    methods = list(importance_sets.keys())
    arrays: dict[str, np.ndarray] = {}
    for method in methods:
        arr = np.asarray(importance_sets[method], dtype=np.float64)
        if arr.shape != (len(names),):
            raise ValueError(
                f"sada {method!r} musi mit delku {len(names)} (podle feature_names), "
                f"zadano: {arr.shape[0] if arr.ndim == 1 else arr.shape}"
            )
        arrays[method] = arr

    stacked = np.vstack([arrays[m] for m in methods])
    mean_importance = stacked.mean(axis=0)
    order = np.argsort(mean_importance)[::-1]
    top_n = min(12, len(names))
    top_idx = order[:top_n]

    group_pos = np.arange(top_n)
    bar_width = 0.8 / len(methods)
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(0.9 * top_n + 3.0, 5.0))
    for i, method in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2.0) * bar_width
        ax.bar(
            group_pos + offset,
            arrays[method][top_idx],
            width=bar_width,
            color=cmap(i % 10),
            edgecolor="white",
            label=method,
        )

    ax.set_xticks(group_pos)
    ax.set_xticklabels([names[i] for i in top_idx], rotation=45, ha="right")
    ax.set_ylabel("dulezitost priznaku")
    ax.set_title(f"Porovnani dulezitosti priznaku (horni {top_n}) napric metodami")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    _save_and_close(fig, save_path)
