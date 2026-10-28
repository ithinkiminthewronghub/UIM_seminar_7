"""Typovana sprava konfigurace nad ``config.yaml`` pro cviceni 07.

Modul definuje vnorene dataclassy odpovidajici sekcim ``config.yaml`` a
dve funkce: ``load_config`` (naparsuje YAML, sestavi dataclassy, zvaliduje
a vrati) a ``validate_config`` (rozsahove kontroly s ceskymi chybovymi
hlaskami).

K hodnotam se pristupuje pres atributy (napr. ``cfg.tree.max_depth``), nikdy
ne pres klice slovniku. Atributovy pristup je typovany (IDE i typovy kontroler
znaji jmena a typy poli, preklep se odhali staticky), zatimco ``cfg["tree"]``
je jen dynamicke vyhledani v ``dict`` -- preklep spadne az za behu a navic
nenese informaci, ze ``max_depth`` smi byt ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class TreeConfig:
    """Nastaveni jednoho rozhodovaciho stromu (sekce ``tree``)."""

    max_depth: int | None
    criterion: str


@dataclass
class ForestConfig:
    """Nastaveni nahodneho lesa (sekce ``forest``)."""

    n_estimators: int
    max_features: int | None
    max_depth: int | None


@dataclass
class DataConfig:
    """Nastaveni dat (sekce ``data``)."""

    random_state: int


@dataclass
class ExperimentConfig:
    """Korenova konfigurace experimentu slozena ze vsech dilcich sekci."""

    tree: TreeConfig
    forest: ForestConfig
    data: DataConfig


def _opt_int(value: Any) -> int | None:
    """Prevede hodnotu z YAML na ``int``, nebo na ``None`` je-li ``null``.

    YAML ``null`` se nacte jako ``None``; ``int(None)`` by spadlo, proto se
    ``None`` propousti beze zmeny.
    """
    if value is None:
        return None
    return int(value)


def load_config(filepath: str = "config.yaml") -> ExperimentConfig:
    """Nacte a zvaliduje konfiguraci z YAML souboru.

    Parametry
    ---------
    filepath:
        Cesta k YAML souboru s konfiguraci.

    Navratova hodnota
    -----------------
    ``ExperimentConfig`` s vnorenymi dataclassami ``TreeConfig``,
    ``ForestConfig`` a ``DataConfig``.

    Vyjimky
    -------
    ``FileNotFoundError``:
        Pokud soubor neexistuje.
    ``ValueError``:
        Pokud nektera hodnota nesplnuje rozsahove kontroly ve
        ``validate_config``.
    """
    with open(filepath, "r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    cfg = ExperimentConfig(
        tree=TreeConfig(
            max_depth=_opt_int(raw["tree"]["max_depth"]),
            criterion=str(raw["tree"]["criterion"]),
        ),
        forest=ForestConfig(
            n_estimators=int(raw["forest"]["n_estimators"]),
            max_features=_opt_int(raw["forest"]["max_features"]),
            max_depth=_opt_int(raw["forest"]["max_depth"]),
        ),
        data=DataConfig(
            random_state=int(raw["data"]["random_state"]),
        ),
    )

    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    """Zkontroluje rozsahy hodnot v konfiguraci.

    Pri poruseni nektere podminky vyhodi ``ValueError`` se srozumitelnou
    ceskou hlaskou obsahujici zadanou hodnotu. Kontroluji se:

    - ``tree.max_depth`` je ``None`` nebo ``>= 1``
    - ``tree.criterion`` je ``"gini"`` nebo ``"entropy"``
    - ``forest.n_estimators >= 1``
    - ``forest.max_depth`` je ``None`` nebo ``>= 1``
    - ``forest.max_features`` je ``None`` nebo ``1 <= max_features <= 30``
      (30 = pocet priznaku datasetu breast cancer)

    Navratova hodnota je ``None`` -- funkce pouze validuje.
    """
    tree = cfg.tree
    forest = cfg.forest

    if tree.max_depth is not None and tree.max_depth < 1:
        raise ValueError(
            f"tree.max_depth musi byt None nebo >= 1, zadano: {tree.max_depth}"
        )
    if tree.criterion not in {"gini", "entropy"}:
        raise ValueError(
            f"tree.criterion musi byt 'gini' nebo 'entropy', zadano: {tree.criterion!r}"
        )
    if forest.n_estimators < 1:
        raise ValueError(
            f"forest.n_estimators musi byt >= 1, zadano: {forest.n_estimators}"
        )
    if forest.max_depth is not None and forest.max_depth < 1:
        raise ValueError(
            f"forest.max_depth musi byt None nebo >= 1, zadano: {forest.max_depth}"
        )
    if forest.max_features is not None and not 1 <= forest.max_features <= 30:
        raise ValueError(
            "forest.max_features musi byt None nebo v rozsahu 1..30, "
            f"zadano: {forest.max_features}"
        )
