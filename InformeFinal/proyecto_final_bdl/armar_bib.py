"""Arma referencias.bib con las entradas que cita la sección: las toma de la bibliografía de la versión anterior de la
sección y de la del libro del proyecto (sin modificarlas) y añade las tres que ninguna de las dos tenía."""
import re
from pathlib import Path

AQUI = Path(__file__).resolve().parent
FUENTES = [c for c in (AQUI.parent / "Resultados Entregable 3" / "referencias.bib",                  # carpeta de trabajo
                       AQUI.parent / "resultados" / "referencias.bib",                               # InformeFinal/ del repo
                       AQUI.parent / "entrega 3" / "repo_entregable3" / "ProyectoFinal" / "references.bib",
                       AQUI.parent.parent / "ProyectoFinal" / "references.bib") if c.exists()]
CLAVES = ["munoz2025", "nadeau2003", "diebold1995", "harvey1997", "corani2015", "benavoli2017", "hansen2011mcs",
          "holm1979", "demsar2006", "iman1980", "efron1987", "wilcoxon1945", "breiman2001", "chen2016xgboost", "ke2017",
          "nori2019interpretml", "hajjem2014", "sigrist2022", "kaufman2012", "kapoor2023", "akiba2019", "dempster1977",
          "friedman2001", "schwarz1978", "vovk2005", "cawley2010"]
NUEVAS = r"""
@article{vargha2000,
  author  = {Vargha, Andr{\'a}s and Delaney, Harold D.},
  title   = {A Critique and Improvement of the {CL} Common Language Effect Size Statistics of {McGraw} and {Wong}},
  journal = {Journal of Educational and Behavioral Statistics},
  year    = {2000}, volume = {25}, number = {2}, pages = {101--132}, doi = {10.3102/10769986025002101}
}

@article{mackay1992,
  author  = {MacKay, David J. C.},
  title   = {Bayesian Interpolation},
  journal = {Neural Computation},
  year    = {1992}, volume = {4}, number = {3}, pages = {415--447}, doi = {10.1162/neco.1992.4.3.415}
}

@book{cohen1988,
  author    = {Cohen, Jacob},
  title     = {Statistical Power Analysis for the Behavioral Sciences},
  edition   = {2},
  publisher = {Lawrence Erlbaum Associates},
  address   = {Hillsdale, NJ},
  year      = {1988}
}
"""


def entradas(texto: str) -> dict[str, str]:
    salida, pos = {}, 0
    for m in re.finditer(r"@\w+\{([^,]+),", texto):
        inicio, nivel = m.start(), 0
        i = texto.index("{", inicio)                      # la llave que abre la entrada
        while i < len(texto):
            if texto[i] == "{":
                nivel += 1
            elif texto[i] == "}":
                nivel -= 1
                if nivel == 0:
                    break
            i += 1
        salida[m.group(1).strip()] = texto[inicio:i + 1]
    return salida


todas = {}
for f in FUENTES:
    for k, v in entradas(f.read_text(encoding="utf-8")).items():
        todas.setdefault(k, v)
faltan = [k for k in CLAVES if k not in todas]
if faltan:
    raise SystemExit(f"Faltan entradas: {faltan}")
(AQUI / "referencias.bib").write_text("\n\n".join(todas[k] for k in CLAVES) + "\n" + NUEVAS, encoding="utf-8")
print(f"referencias.bib: {len(CLAVES) + 3} entradas")
