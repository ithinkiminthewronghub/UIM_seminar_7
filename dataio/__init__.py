"""Verejne API balicku ``dataio`` pro cviceni 07.

Nacitani dat (Breast Cancer Wisconsin), pravdivostni tabulky logickych hradel
(AND, OR, XOR) jako male 2D datasety, typovana sprava konfigurace a bohate
vykreslovani (rozhodovaci povrch, dulezitost priznaku, strom vs. les,
porovnani dulezitosti napric cvicenimi).

Verejne API
-----------
- ``make_gate`` -- ``(x, y)`` pro logicke hradlo z
  ``{"and", "or", "xor", "xnor", "imply"}`` (pipeline pouziva zakladni trojici)
- ``load_breast_cancer_data`` -- ``(x, y, feature_names)`` datasetu breast cancer
- ``load_config`` / ``validate_config`` -- typovana konfigurace nad ``config.yaml``
- ``TreeConfig`` / ``ForestConfig`` / ``DataConfig`` / ``ExperimentConfig`` -- dataclassy
- ``plot_decision_surface`` / ``plot_feature_importance`` /
  ``plot_tree_vs_forest`` / ``plot_importance_comparison`` -- vykreslovani

Na rozdil od cviceni 06 (kde byly studentske ukoly v ``dataio/preprocessing.py``)
je v cviceni 07 cely balicek ``dataio/`` **predvyplneny** -- zadny studentsky
ukol, zadny ``NotImplementedError``. Studentske ukoly cviceni 07 zijou pouze
v ``src/decision_tree.py`` a ``src/random_forest.py``.

**Tento soubor (__init__.py) neupravujte** -- re-exporty verejneho API zustavaji
beze zmeny.
"""

from __future__ import annotations

from dataio.config_manager import (
    DataConfig,
    ExperimentConfig,
    ForestConfig,
    TreeConfig,
    load_config,
    validate_config,
)
from dataio.gates import make_gate
from dataio.loader import load_breast_cancer_data
from dataio.plotting import (
    plot_decision_surface,
    plot_feature_importance,
    plot_importance_comparison,
    plot_tree_vs_forest,
)

__all__ = [
    "make_gate",
    "load_breast_cancer_data",
    "load_config",
    "validate_config",
    "TreeConfig",
    "ForestConfig",
    "DataConfig",
    "ExperimentConfig",
    "plot_decision_surface",
    "plot_feature_importance",
    "plot_tree_vs_forest",
    "plot_importance_comparison",
]
