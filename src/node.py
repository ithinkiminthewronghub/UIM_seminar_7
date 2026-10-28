"""Uzel rozhodovaciho stromu — cisty nositel stavu, bez jakekoli logiky.

``Node`` je zamerne pouha datova struktura (``@dataclass`` s peti poli). Veskera
logika — hledani nejlepsiho rezu, rekurzivni stavba stromu, prochazeni pri
predikci i (de)serializace do JSON — zije ve tride ``DecisionTree`` v modulu
``src.decision_tree``. Strom je tedy dvojice "hloupa data + chytra trida":
uzly drzi stav, ``DecisionTree`` s nimi pracuje.

Invariant listu
---------------
``value is not None``  <=>  uzel je list.

* U **listu** je vyplnene pouze ``value`` (predikovana trida); ``feature``,
  ``threshold``, ``left`` i ``right`` zustavaji ``None``.
* U **vnitrniho uzlu** je ``value`` rovno ``None`` a vyplnene jsou ``feature``,
  ``threshold`` a oba potomci ``left`` a ``right``.

Konvence rezu
-------------
Vnitrni uzel deli vzorky podle jedineho priznaku ``feature`` prahem
``threshold``: vzorky s hodnotou ``<= threshold`` pokracuji do ``left``,
vzorky s hodnotou ``> threshold`` do ``right``.

Poznamka k rozsireni
--------------------
Uzel by slo rozsirit o vlastni metody — napriklad ``predict``, ktera by se
rekurzivne ptala svych potomku a sama dosla az do listu. Zamerne to
NEDELAME: v tomto cviceni ma uzel zustat cistym nositelem dat a rozhodovaci
logika ma byt na jednom miste ve ``DecisionTree``. Zajemci si ``Node`` o
takove metody rozsirit mohou, neni to ale soucast zadani.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Node:
    """Jeden uzel rozhodovaciho stromu (list nebo vnitrni uzel).

    Vsech pet poli ma vychozi hodnotu ``None``; ktera jsou vyplnena, rozhoduje
    o tom, zda je uzel list (viz invariant v modulovem docstringu).
    """

    feature: int | None = None        # index priznaku, podle ktereho se deli (None u listu)
    threshold: float | None = None    # prahova hodnota rezu (<= vlevo, > vpravo)
    left: "Node | None" = None        # potomek pro hodnoty <= threshold
    right: "Node | None" = None       # potomek pro hodnoty > threshold
    value: int | None = None          # predikovana trida (jen list; list <=> value is not None)
