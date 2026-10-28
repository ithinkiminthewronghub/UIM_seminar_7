"""Nacitani dat Breast Cancer Wisconsin pro cviceni 07.

Modul poskytuje jedinou funkci ``load_breast_cancer_data``, ktera vrati
priznakovou matici, binarni cilovou promennou a nazvy priznaku.

Data se zde **nestandardizuji** a u rozhodovaciho stromu to nevadi. Strom deli
data prahem na jednotlivych priznacich zvlast (``priznak_j <= prah``); vysledek
takoveho porovnani se nezmeni pri libovolne rostouci (monotonni) transformaci
skaly daneho priznaku. Strom je tedy vuci prenasobeni ci posunu skaly
invariantni. To je zasadni rozdil oproti cviceni 06 (kNN pocita euklidovske
vzdalenosti pres vsechny priznaky najednou, takze priznak s velkym rozsahem
dominuje) a cviceni 05 (PCA hleda smery maximalniho rozptylu, ktere se
preskalovanim priznaku posunou). Tam na meritku zaleželo zasadne, tady ne.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer


def load_breast_cancer_data(
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Nacte dataset Breast Cancer Wisconsin a vrati ``(x, y, feature_names)``.

    Parametry
    ---------
    random_state:
        Je v podpisu pouze kvuli konzistenci s ostatnimi nacitaci funkcemi
        v kurzu. Dataset je pevny a deterministicky, takze tento parametr
        nic neovlivnuje (zadne michani ani vzorkovani se zde nedeje).

    Navratova hodnota
    -----------------
    x:
        ``np.ndarray`` tvaru ``(569, 30)`` typu ``float64``. Priznaky se
        zamerne **nestandardizuji** -- viz modul-docstring (strom je vuci
        meritku priznaku invariantni).
    y:
        ``np.ndarray`` tvaru ``(569,)`` typu ``int64`` s hodnotami
        ``{0, 1}``, kde **1 = maligni (zhoubny)** a **0 = benigni
        (nezhoubny)**. Plati ``y.sum() == 212`` (pocet malignich vzorku).
    feature_names:
        Seznam 30 nazvu priznaku (``list[str]``).

    Poznamka ke kodovani cilove promenne
    ------------------------------------
    ``sklearn.datasets.load_breast_cancer`` koduje ``target`` **opacne**,
    nez potrebujeme: 0 = malignant (zhoubny), 1 = benign (nezhoubny).
    Kurz vsak pracuje s konvenci "vystup 1 -> maligni", proto se stitky
    prohazuji vztahem ``y = 1 - dataset.target``. Kodovani je konzistentni
    s cvicenim 05 i 06. Po prohozeni plati ``y.sum() == 212``.

    Pozitivni trida pro Se/Sp
    -------------------------
    Ve vyhodnoceni klasifikace je **pozitivni tridou maligni nador
    (y = 1)**. Z toho plyne cteni metrik:

    - ``Se = recall = sensitivita = TPR`` -- podil spravne odhalenych
      malignich nadoru; "nepropasnout zhoubny nador".
    - ``Sp = specificita = TNR`` -- podil spravne oznacenych benignich
      pripadu.

    V medicinskem kontextu obvykle sensitivita (Se) prevazuje nad
    precizi -- cena za propasnuty maligni nador je vyssi nez za falesny
    poplach.
    """
    dataset = load_breast_cancer()
    x = np.asarray(dataset.data, dtype=np.float64)
    # Prohozeni stitku: sklearn ma 0 = malignant, 1 = benign; kurz chce 1 = maligni.
    y = 1 - np.asarray(dataset.target, dtype=np.int64)
    feature_names = [str(name) for name in dataset.feature_names]

    return x, y, feature_names
