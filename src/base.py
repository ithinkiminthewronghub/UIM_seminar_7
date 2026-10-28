"""Tenka spolecna baze klasifikatoru cviceni 07 — rozhrani a mechanicke I/O.

Tenka baze bez dependency injection
-----------------------------------
``Classifier`` sdili jen to, co maji ``DecisionTree`` a ``RandomForest``
opravdu spolecne: ctyri metody rozhrani (``fit``, ``predict``, ``_to_dict``,
``_from_dict``) a mechanicke ukladani do souboru (``save`` / ``load``).
ZADNA skutecna logika algoritmu tu neni — Gini, informacni zisk, rekurzivni
stavba stromu i hlasovani lesa zijou az v podtridach.

Na rozdil od cviceni 03/04/06, kde se do klasifikatoru injektovala vymenitelna
hierarchie ``Distance``, tady se nic neinjektuje. Strom ani les nemaji zadnou
vymenitelnou zavislost: rozhoduji se podle prahu na priznacich, nepocitaji
parove vzdalenosti. Jedina "volba chovani" je kriterium necistoty, a to je
pouhy retezec (``"gini"`` / ``"entropy"``), ktery si podtrida zpracuje sama —
neni to spolupracujici objekt, takze neni co injektovat ani co gateovat.

Proc JSON, a ne ``.npz``
------------------------
Cviceni 05 (PCA) i cviceni 06 (kNN) ukladaji stav do ``.npz``, protoze jejich
naucenym stavem jsou obdelnikova numpy pole (prumer, vlastni vektory, cela
trenovaci matice) — a ``.npz`` je pro pole nativni format.

Natrenovany rozhodovaci strom je ale neco jineho: zanorena stromova struktura
s nestejne hlubokymi a nestejne velkymi vetvemi. Takovy stav se prirozene
mapuje na zanorene slovniky (uzel -> ``{"feature": ..., "left": {...},
"right": {...}}``), ne na pole pevneho tvaru. Format perzistence se ridi
tvarem stavu: zanoreny strom -> zanorene slovniky -> JSON. JSON je navic
citelny (da se otevrit a zkontrolovat) a bezpecny (na rozdil od ``pickle``
nespousti pri nacteni zadny kod).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

import numpy as np


class Classifier(ABC):
    """Spolecne rozhrani klasifikatoru cviceni 07 (strom i les jsou potomci).

    Delba prace mezi metodami:

    * ``save`` / ``load`` jsou MECHANIKA a jsou hotove. Vedi jen "vezmi
      slovnik, zapis ho jako JSON" a "nacti JSON, predej ho zpet". O obsahu
      slovniku nevedi nic.
    * ``_to_dict`` / ``_from_dict`` odpovidaji na otazku "co ten model
      vlastne JE" — ktera naucena pole tvori jeho stav a jak se serializuji
      do zanorenych slovniku a zpet. Prave tohle doplnuje student v
      podtridach ``DecisionTree`` a ``RandomForest``.
    * ``fit`` / ``predict`` jsou vlastni algoritmus, rovnez v podtridach.

    ``save`` a ``load`` jsou jedine KONKRETNI metody teto tridy; vse ostatni
    je abstraktni.
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Classifier":
        """Nauci klasifikator z trenovacich dat a vrati ``self``.

        Parametry
        ---------
        x : np.ndarray
            Matice priznaku tvaru ``(n_samples, n_features)``.
        y : np.ndarray
            Cilovy vektor delky ``n_samples`` s celociselnymi tridami.

        Navratova hodnota
        -----------------
        Classifier
            Tato instance (``self``), aby slo retezit ``.fit(...).predict(...)``.
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Zaradi kazdy radek ``x`` do jedne z naucenych trid.

        Parametry
        ---------
        x : np.ndarray
            Matice priznaku tvaru ``(n_samples, n_features)`` se stejnym
            poctem priznaku jako trenovaci data.

        Navratova hodnota
        -----------------
        np.ndarray
            Predikovane tridy tvaru ``(n_samples,)``.
        """

    @abstractmethod
    def _to_dict(self) -> dict:
        """Prevede nauceny stav modelu na slovnik vhodny k serializaci do JSON.

        Navratova hodnota
        -----------------
        dict
            Zanoreny slovnik obsahujici vse potrebne k pozdejsi rekonstrukci
            modelu (hyperparametry i naucena pole). Smi obsahovat jen typy,
            ktere umi ``json`` (``dict``, ``list``, ``str``, ``int``,
            ``float``, ``bool``, ``None``).
        """

    @classmethod
    @abstractmethod
    def _from_dict(cls, data: dict) -> "Classifier":
        """Sestavi novou instanci z jejiho slovnikoveho popisu (inverze k ``_to_dict``).

        Parametry
        ---------
        data : dict
            Slovnik ve tvaru, jaky vraci ``_to_dict``.

        Navratova hodnota
        -----------------
        Classifier
            Nova instance s obnovenym naucenym stavem, pripravena k ``predict``.
        """

    def save(self, path: str) -> None:
        """Ulozi nauceny model do souboru ve formatu JSON.

        KONKRETNI, predvyplnene. Metoda je jen mechanika: zavola
        ``self._to_dict()`` a vysledny slovnik zapise jako JSON. O tom, co
        slovnik obsahuje, rozhoduje ``_to_dict`` v podtride.

        Nadrazeny adresar se v pripade potreby vytvori. JSON se zapisuje s
        ``encoding="utf-8"``, ``ensure_ascii=False`` (diakritika se ulozi
        primo, ne jako escapovane kody) a ``indent=2`` (citelne odsazeni).

        Parametry
        ---------
        path : str
            Cesta k vystupnimu souboru ``.json``.
            Ukládejte do složky ``./models/``, aby se nepletly s ostatními soubory.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Classifier":
        """Nacte model ulozeny metodou ``save`` a vrati hotovou instanci.

        KONKRETNI, predvyplnene. Metoda je inverzi k ``save`` a je opet jen
        mechanika: nacte JSON ze souboru a preda vysledny slovnik do
        ``cls._from_dict(data)``, ktery uz vi, jak z nej model sestavit.

        Parametry
        ---------
        path : str
            Cesta k souboru ``.json`` vytvorenemu metodou ``save``.

        Navratova hodnota
        -----------------
        Classifier
            Nova instance s obnovenym naucenym stavem.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)
