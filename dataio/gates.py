"""Pravdivostni tabulky logickych hradel jako male 2D datasety (4 body).

Didakticka pointa cviceni: rozhodovaci strom deli prostor **osove zarovnanymi**
rezy. Nektera hradla jsou linearne separovatelna (jedina primka oddeli tridu 0
od tridy 1), ale i tak na ne strom potrebuje dva rezy, protoze ta oddelujici
primka je sikma a strom umi rezat jen kolmo na osy. Hradlo XOR (a jeho negace
XNOR) linearne separovatelne **neni** vubec -- zadna jedina primka body
nerozdeli. Strom ho presto zvladne: sklada za sebe **dva** osove zarovnane rezy
(nejdriv podle jednoho priznaku, pak v kazde vetvi podle druheho), takze druha
otazka smi v leve a v prave casti odpovedet opacne.

Nabizena hradla (poradi radku ``[0,0], [0,1], [1,0], [1,1]``):

======  ================  =========================  ==========================
hradlo  vystup ``y``       linearne separovatelne?    kolik rezu strom potrebuje
======  ================  =========================  ==========================
AND     ``[0, 0, 0, 1]``  ano (sikma primka)         dva
OR      ``[0, 1, 1, 1]``  ano (sikma primka)         dva
XOR     ``[0, 1, 1, 0]``  **ne**                     dva (ctyri listy)
XNOR    ``[1, 0, 0, 1]``  **ne** (negace XOR)        dva (ctyri listy)
IMPLY   ``[1, 1, 0, 1]``  ano (izoluje jediny roh)   dva
======  ================  =========================  ==========================

AND/OR/XOR jsou zakladni trojice, kterou prochazi pipeline; XNOR a IMPLY jsou
navic k volnemu experimentovani (napr. ``make_gate("xnor")`` a vlastni volani
``plot_decision_surface``).
"""

from __future__ import annotations

import numpy as np

# Ctyri kombinace vstupu v pevnem poradi; sdilene vsemi hradly.
_INPUTS: np.ndarray = np.array(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64
)

# Pravdivostni tabulky (vystup pro poradi radku v ``_INPUTS``).
_TRUTH_TABLES: dict[str, list[int]] = {
    "and": [0, 0, 0, 1],
    "or": [0, 1, 1, 1],
    "xor": [0, 1, 1, 0],
    "xnor": [1, 0, 0, 1],   # ekvivalence (A = B); negace XOR, rovnez nelinearni
    "imply": [1, 1, 0, 1],  # implikace (A => B); nepravda jen pro (1, 0)
}


def make_gate(gate: str) -> tuple[np.ndarray, np.ndarray]:
    """Vrati ``(x, y)`` pro zadane logicke hradlo.

    Parametry
    ---------
    gate:
        Jmeno hradla z mnoziny ``{"and", "or", "xor", "xnor", "imply"}``.
        Velikost pismen ani okrajove mezery nevadi -- hodnota se normalizuje
        pres ``gate.lower().strip()``. Trojice ``and`` / ``or`` / ``xor`` je
        zakladni (prochazi ji pipeline); ``xnor`` a ``imply`` jsou navic
        k volnemu experimentovani.

    Navratova hodnota
    -----------------
    x:
        ``np.ndarray`` tvaru ``(4, 2)`` typu ``float64`` se ctyrmi radky
        ``[0, 0]``, ``[0, 1]``, ``[1, 0]``, ``[1, 1]`` v tomto poradi.
    y:
        ``np.ndarray`` tvaru ``(4,)`` typu ``int64`` s vystupem hradla pro
        odpovidajici radky ``x``:

        - AND   -> ``[0, 0, 0, 1]``
        - OR    -> ``[0, 1, 1, 1]``
        - XOR   -> ``[0, 1, 1, 0]``
        - XNOR  -> ``[1, 0, 0, 1]``   (ekvivalence, negace XOR)
        - IMPLY -> ``[1, 1, 0, 1]``   (implikace A => B)

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``gate`` po normalizaci neni ``"and"``, ``"or"``, ``"xor"``,
        ``"xnor"`` nebo ``"imply"``.
    """
    key = gate.lower().strip()
    if key not in _TRUTH_TABLES:
        povolene = ", ".join(sorted(_TRUTH_TABLES))
        raise ValueError(
            f"Nezname hradlo: {gate!r}. Povolene hodnoty jsou: {povolene}."
        )

    x = _INPUTS.copy()
    y = np.array(_TRUTH_TABLES[key], dtype=np.int64)
    return x, y
