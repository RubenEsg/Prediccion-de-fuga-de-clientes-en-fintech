# -*- coding: utf-8 -*-
"""Arma el informe final a partir de las dos secciones escritas con las guías del curso.

    python armar.py      -> copia metodologia.tex, resultados.tex, tablas/ y figures/, y une las referencias

Toma la sección de metodología de ../Metodologia Entregable 3 (o ../metodologia, en el repositorio) y la de
resultados de ../Resultados Entregable 3 (o ../resultados), y hace los únicos ajustes que necesita la unión: la
primera frase de la metodología remite a la sección de resultados, y la tabla de optimizadores de los resultados
cambia de etiqueta (tab:optimizadores_res) porque la metodología ya usa tab:optimizadores. Las copias no se editan
aquí: cualquier cambio se hace en la carpeta de origen y se vuelve a armar. Después: python compilar.py.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def origen(*nombres: str) -> Path:
    for nombre in nombres:
        carpeta = AQUI.parent / nombre
        if carpeta.is_dir():
            return carpeta
    sys.exit(f"No se encontró ninguna de las carpetas {nombres} en {AQUI.parent}")


MET = origen("Metodologia Entregable 3", "metodologia")
RES = origen("Resultados Entregable 3", "resultados")
AVISO = "% Copia que genera armar.py desde {origen} de la carpeta de su sección; los cambios se hacen allí y se vuelve a armar.\n"


def reemplazar(texto: str, viejo: str, nuevo: str) -> str:
    n = texto.count(viejo)
    if n != 1:
        sys.exit(f"Se esperaba una aparición de {viejo!r} y hay {n}")
    return texto.replace(viejo, nuevo)


def copiar_tex(origen_tex: Path, destino: Path, cambios: list[tuple[str, str]]) -> None:
    texto = origen_tex.read_text(encoding="utf-8")
    for viejo, nuevo in cambios:
        texto = reemplazar(texto, viejo, nuevo)
    destino.write_text(AVISO.format(origen=origen_tex.name) + texto, encoding="utf-8")


def copiar_carpeta(origen_dir: Path, destino: Path, patron: str, transformar=None) -> list[str]:
    destino.mkdir(exist_ok=True)
    nombres = []
    for archivo in sorted(origen_dir.glob(patron)):
        if transformar is not None:
            texto = transformar(archivo.name, archivo.read_text(encoding="utf-8"))
            (destino / archivo.name).write_text(texto, encoding="utf-8")
        else:
            shutil.copy2(archivo, destino / archivo.name)
        nombres.append(archivo.name)
    return nombres


def entradas_bib(ruta: Path) -> dict[str, str]:
    partes = re.split(r"(?m)^(?=@)", ruta.read_text(encoding="utf-8"))
    entradas = {}
    for parte in partes:
        m = re.match(r"@\w+\{([^,\s]+),", parte)
        if m:
            entradas[m.group(1)] = parte.strip()
    return entradas


def unir_bib() -> int:
    met, res = entradas_bib(MET / "referencias.bib"), entradas_bib(RES / "referencias.bib")
    for clave in sorted(set(met) & set(res)):
        if re.sub(r"\s+", " ", met[clave]) != re.sub(r"\s+", " ", res[clave]):
            sys.exit(f"La entrada {clave} difiere entre las dos secciones; unifíquela antes de armar")
    todas = dict(met)
    todas.update({k: v for k, v in res.items() if k not in met})
    cabecera = ("% Referencias del informe final: la unión de las de la sección de metodología y las de la sección de\n"
                "% resultados; armar.py las une y comprueba que las entradas comunes sean idénticas. No editar aquí.\n\n")
    cuerpo = "\n\n".join(todas[k] for k in sorted(todas))
    (AQUI / "referencias.bib").write_text(cabecera + cuerpo + "\n", encoding="utf-8")
    return len(todas)


def main() -> None:
    copiar_tex(MET / "metodologia.tex", AQUI / "metodologia.tex", [
        ("se reservan para la sección de resultados.", "se reservan para la sección~\\ref{sec:resultados}."),
    ])
    copiar_tex(RES / "resultados.tex", AQUI / "resultados.tex", [
        ("\\ref{tab:optimizadores}", "\\ref{tab:optimizadores_res}"),
    ])

    def etiqueta(nombre: str, texto: str) -> str:
        if nombre == "optimizadores.tex":
            texto = reemplazar(texto, "\\label{tab:optimizadores}", "\\label{tab:optimizadores_res}")
        return texto

    tablas = copiar_carpeta(RES / "tablas", AQUI / "tablas", "*.tex", etiqueta)
    figuras = copiar_carpeta(MET / "figures", AQUI / "figures", "*.pdf") + copiar_carpeta(RES / "figures", AQUI / "figures", "*.pdf")
    if len(set(figuras)) != len(figuras):
        sys.exit("Dos figuras de las secciones tienen el mismo nombre; renombre una antes de armar")
    referencias = unir_bib()
    print(f"metodologia.tex y resultados.tex copiados; {len(tablas)} tablas, {len(figuras)} figuras y "
          f"{referencias} referencias ({MET.name} + {RES.name})")


if __name__ == "__main__":
    main()
