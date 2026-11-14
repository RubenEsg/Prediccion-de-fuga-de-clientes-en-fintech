# -*- coding: utf-8 -*-
"""Compila main.tex con Tectonic (LaTeX + BibTeX en una sola pasada), cuenta las palabras de cada sección y revisa
el registro de la compilación.

    python compilar.py      -> main.pdf (si cambió alguna de las dos secciones, antes: python armar.py)

Tectonic se busca en el PATH, en la variable de entorno TECTONIC, en ../entrega 3/articulo/herramientas/tectonic.exe
(carpeta de trabajo) o en ../../Entrega3/articulo/herramientas/tectonic.exe (repositorio). El conteo de palabras
ignora comandos, tablas, figuras, algoritmos y comentarios; las cifras, también las escritas en modo matemático,
cuentan como palabras, igual que en un contador sobre el PDF.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
CANDIDATOS_TECTONIC = (
    AQUI.parent / "entrega 3" / "articulo" / "herramientas" / "tectonic.exe",
    AQUI.parent.parent / "Entrega3" / "articulo" / "herramientas" / "tectonic.exe",
)
# Sección -> (tope de palabras de la guía del curso o None, si las cifras en modo matemático cuentan). La guía de
# metodología excluye las ecuaciones y la de resultados solo tablas, figuras y código, así que cada sección se cuenta
# con su propia regla, la misma de su carpeta de origen.
SECCIONES = {"introduccion.tex": (None, True), "metodologia.tex": (3500, False), "resultados.tex": (3000, True),
             "conclusiones.tex": (None, True)}
ENTORNOS = ("equation", r"equation\*", "align", r"align\*", "table", r"table\*", "figure", r"figure\*", "tabular",
            "algorithm", "algorithmic", "thebibliography")


def tectonic() -> str:
    candidatos = [os.environ.get("TECTONIC"), shutil.which("tectonic")] + [str(c) for c in CANDIDATOS_TECTONIC]
    for candidato in candidatos:
        if candidato and Path(candidato).exists():
            return candidato
    sys.exit("No se encontró Tectonic: https://tectonic-typesetting.github.io/ (o fije la variable TECTONIC)")


def limpiar_latex(texto: str, contar_matematicas: bool = True) -> str:
    texto = re.sub(r"(?<!\\)%.*", "", texto)
    for env in ENTORNOS:
        texto = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", " ", texto, flags=re.S)
    texto = re.sub(r"\$\$.*?\$\$", " ", texto, flags=re.S)
    texto = re.sub(r"\\\[.*?\\\]", " ", texto, flags=re.S)
    if contar_matematicas:
        texto = re.sub(r"\$([^$]*)\$", lambda m: " " + m.group(1).replace("{,}", ",") + " ", texto)
    else:
        texto = re.sub(r"\$[^$]*\$", " ", texto)
    texto = re.sub(r"\\(cite[pt]?|ref|eqref|label|autoref|url|href|input|includegraphics|caption)\*?(\[[^\]]*\])?\{[^}]*\}", " ", texto)
    texto = re.sub(r"\\[A-Za-z@]+\*?(\[[^\]]*\])?", " ", texto)
    texto = re.sub(r"[{}~]", " ", texto)
    return texto


def palabras(texto: str, contar_matematicas: bool = True) -> int:
    return len(limpiar_latex(texto, contar_matematicas).split())


def resumen(main_tex: str) -> int:
    m = re.search(r"\\begin\{abstract\}(.*?)\\medskip", main_tex, flags=re.S)
    return palabras(m.group(1)) if m else 0


def revisar_log(ruta: Path) -> None:
    if not ruta.exists():
        print("(sin main.log)")
        return
    log = ruta.read_text(encoding="utf-8", errors="replace")
    overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt", log)
    print(f"Overfull hbox: {len(overfull)}" + (f" (máximo {max(map(float, overfull)):.1f} pt)" if overfull else ""))
    for patron, nombre in [(r"Reference `[^']*' on page \d+ undefined", "referencias cruzadas sin definir"),
                           (r"Citation `[^']*' on page \d+ undefined", "citas sin definir"),
                           (r"Label `[^']*' multiply defined", "etiquetas repetidas"),
                           (r"Missing character: [^\n]*", "caracteres sin glifo")]:
        avisos = sorted(set(re.findall(patron, log)))
        if avisos:
            print(f"{nombre}: {len(avisos)}")
            for a in avisos[:10]:
                print("   ", a)


def main() -> None:
    comando = [tectonic(), "--keep-logs", "--chatter", "minimal", "main.tex"]
    print("$", " ".join(comando))
    r = subprocess.run(comando, cwd=AQUI, capture_output=True, text=True, encoding="utf-8", errors="replace")
    salida = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        print(salida[-4000:])
        sys.exit(f"Tectonic terminó con código {r.returncode}")
    avisos = [l for l in salida.splitlines() if "warning" in l.lower() and "microtype" not in l.lower()]
    if avisos:
        print("\n".join(avisos[:30]))
    pdf = AQUI / "main.pdf"
    try:
        from pypdf import PdfReader
        print(f"{pdf.name}: {len(PdfReader(str(pdf)).pages)} páginas, {pdf.stat().st_size / 1e6:.2f} MB")
    except ImportError:
        print(f"{pdf.name}: {pdf.stat().st_size / 1e6:.2f} MB")
    revisar_log(AQUI / "main.log")

    print("\nPalabras (sin tablas, figuras ni algoritmos; en la metodología tampoco cuentan las matemáticas, como en su guía)")
    print(f"  {resumen((AQUI / 'main.tex').read_text(encoding='utf-8')):5d}  Resumen")
    total = 0
    for nombre, (tope, matematicas) in SECCIONES.items():
        n = palabras((AQUI / nombre).read_text(encoding="utf-8"), matematicas)
        total += n
        print(f"  {n:5d}  {nombre}" + (f"  (guía: {tope})" if tope else ""))
    print(f"  {total:5d}  total de las cuatro secciones")
    citas = set()
    for nombre in SECCIONES:
        for grupo in re.findall(r"\\cite[pt]?\{([^}]*)\}", (AQUI / nombre).read_text(encoding="utf-8")):
            citas.update(c.strip() for c in grupo.split(","))
    print(f"Referencias citadas: {len(citas)}")


if __name__ == "__main__":
    main()
