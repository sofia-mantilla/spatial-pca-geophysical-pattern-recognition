"""Names of the twelve known deposits, as the paper's tables give them.

The shapefile shipped with the data carries placeholder names for deposits 8
to 12 ("Deposit 8" and so on), which is why earlier versions of the maps
labelled five of the twelve by number while the other seven were named. The
identifications were confirmed by P. Mejia-Herrera; Table 1 of the paper is
the authority for them. Figures import this module so that a figure and a
table never disagree.
"""
from __future__ import annotations

NAMES = {
    1: "Tucumã",
    2: "Pedra Branca",
    3: "Alemão",
    4: "Furnas",
    5: "Salobo",
    6: "Paulo Afonso",
    7: "Cabano",
    8: "AQW1 and AQW2",
    9: "Pantera",
    10: "Guepardo",
    11: "Sossego",
    12: "Estrela",
}


def label(dep_1based: int) -> str:
    """"6. Paulo Afonso", the form the maps and the tables share."""
    return "%d. %s" % (dep_1based, NAMES[dep_1based])
