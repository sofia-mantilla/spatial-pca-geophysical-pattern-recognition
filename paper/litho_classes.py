"""Lithological grouping of the Para 1:100,000 units (R2.6).

The 1:100,000 sheets list the lithotypes of each mapped unit in LITOTIPOS, in
alphabetical order and with no indication of which dominates. A figure cannot
carry 103 distinct lithotype strings, so the units are grouped into nine
lithological classes. Each listed lithotype is matched against the keyword sets
below, the class with the most matches wins, and ties go to the class listed
first. Generic terms that say little about the rock, such as a bare "xisto",
count half. A mafic granulite counts as a mafic rock and not as a gneiss. Iron formation is kept separate from the metasedimentary rocks it sits
in, because whether the ranking favours iron formation is the question
Reviewer 2 raises.

Colours follow the usual conventions for the rock types rather than the
geological timescale, so this map reads as lithology and the era map
(Fig26, Fig27) reads as age.
"""
from __future__ import annotations

# (key, label, colour, keywords). Order is the tie-break: earlier wins.
CLASSES = [
    ("bif", "Iron formation, jaspilite, magnetitite", "#7E1F3D",
     ("formação ferrífera", "formacao ferrifera", "jaspelito", "magnetitito",
      "quartzito ferruginoso")),
    ("gneiss", "Gneiss, granulite, migmatite", "#B07C57",
     ("gnaisse", "ortognaisse", "paragnaisse", "migmatito", "granulito",
      "charnockito")),
    ("maficint", "Mafic to ultramafic intrusive", "#2F5E3F",
     ("gabro", "gabronorito", "norito", "diabásio", "diabasio", "dunito",
      "peridotito", "piroxenito", "serpentinito", "hornblendito",
      "metagabro", "metadiabásio", "metadiabasio")),
    ("maficvolc", "Mafic metavolcanic rock and amphibolite", "#6FA84D",
     ("metabasalto", "basalto", "anfibolito", "ortoanfibolito",
      "metakomatiíto", "metakomatiito", "metamáfica", "metamafica",
      "metavulcânica máfica", "metavulcanica mafica", "metaultramafito",
      "rocha metaultramáfica", "rocha metaultramafica", "xisto ultramáfico",
      "actinolita xisto", "metandesito", "andesito")),
    ("felsicvolc", "Felsic metavolcanic rock", "#D9834F",
     ("metarriolito", "metadacito", "metavulcânica félsica",
      "metavulcanica felsica", "metapiroclástica", "metapiroclastica",
      "metatufo", "riolito", "dacito")),
    ("granitoid", "Granitoid", "#E8707C",
     ("granito", "granitoide", "monzogranito", "sienogranito", "granodiorito",
      "tonalito", "trondhjemito", "diorito", "metagranito",
      "metamonzogranito", "metassienogranito", "metagranodiorito",
      "metatonalito", "metatrondhjemito", "metadiorito", "microgranito",
      "albitito", "quartzo diorito", "quartzo gabro")),
    ("metased", "Schist, quartzite, metasedimentary rock", "#8EA9C1",
     ("xisto", "quartzito", "metarenito", "metapelito", "metassiltito",
      "metagrauvaca", "metaconglomerado", "metassedimentar", "mármore",
      "marmore", "ardósia", "ardosia", "chert", "metachert",
      "metassilexito")),
    ("laterite", "Laterite and ferruginous duricrust", "#EFB264",
     ("laterita", "canga", "crosta laterítica", "crosta lateritica")),
    ("clastic", "Sand, clay, gravel, sandstone, conglomerate", "#F2E394",
     ("areia", "argila", "cascalho", "silte", "siltito", "arenito",
      "conglomerado", "lateritas imaturas")),
]
UNMAPPED = ("Not mapped at 1:100,000", "#E4E4E4")
# Terms that name a rock only loosely, so they should not outvote a specific one.
GENERIC = ("xisto", "rocha metassedimentar", "metamáfica", "metamafica",
           "granitoide", "metassedimentar clástica")
ORDER = [c[0] for c in CLASSES]
LABEL = {c[0]: c[1] for c in CLASSES}
COLOUR = {c[0]: c[2] for c in CLASSES}


def classify(litotipos: str) -> str:
    """Lithological class of one mapped unit, from its LITOTIPOS string."""
    parts = [p.strip().lower() for p in str(litotipos or "").split(",") if p.strip()]
    best, best_n = None, 0.0
    for key, _label, _colour, words in CLASSES:
        n = 0.0
        for part in parts:
            if key == "gneiss" and ("máfic" in part or "mafic" in part):
                continue            # a mafic granulite is a mafic rock
            if any(w in part for w in words):
                n += 0.5 if part in GENERIC else 1.0
        if n > best_n:
            best, best_n = key, n
    return best or "unmapped"
