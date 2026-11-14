"""Compila main.tex con Tectonic (LaTeX + BibTeX en una sola pasada) y cuenta las palabras de la sección.

    python compilar.py          -> main.pdf (desde esta carpeta o desde cualquier otra)

Tectonic se busca en el PATH, en la variable TECTONIC, en ../entrega 3/articulo/herramientas/tectonic.exe (carpeta
de trabajo) o en ../../Entrega3/articulo/herramientas/tectonic.exe (repositorio publicado).
El conteo de palabras ignora comandos, matemáticas, tablas, figuras, algoritmos y comentarios, como pide la guía
(«sin contar tablas, figuras, ecuaciones ni pseudocódigo»).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
TECTONIC_LOCAL = (AQUI.parent / "entrega 3" / "articulo" / "herramientas" / "tectonic.exe",
                  AQUI.parent.parent / "Entrega3" / "articulo" / "herramientas" / "tectonic.exe")


def tectonic() -> str:
    for candidato in [os.environ.get("TECTONIC"), shutil.which("tectonic"), *map(str, TECTONIC_LOCAL)]:
        if candidato and Path(candidato).exists():
            return candidato
    sys.exit("No se encontró Tectonic: https://tectonic-typesetting.github.io/ (o fija la variable TECTONIC)")


def limpiar_latex(texto: str) -> str:
    texto = re.sub(r"(?<!\\)%.*", "", texto)                                   # comentarios
    for env in ("equation", "equation\\*", "align", "align\\*", "table", "figure", "algorithm", "algorithmic",
                "tabular", "thebibliography"):
        texto = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", " ", texto, flags=re.S)
    texto = re.sub(r"\$\$.*?\$\$", " ", texto, flags=re.S)
    texto = re.sub(r"\\\[.*?\\\]", " ", texto, flags=re.S)
    texto = re.sub(r"\$[^$]*\$", " ", texto)
    texto = re.sub(r"\\(cite[pt]?|ref|eqref|label|autoref|url|href|input|includegraphics|caption)\*?(\[[^\]]*\])?\{[^}]*\}", " ", texto)
    texto = re.sub(r"\\[A-Za-z@]+\*?(\[[^\]]*\])?", " ", texto)                 # otros comandos (se conserva su argumento)
    texto = re.sub(r"[{}~]", " ", texto)
    return texto


def contar_palabras(ruta: Path) -> tuple[int, dict[str, int]]:
    texto = ruta.read_text(encoding="utf-8")
    partes = re.split(r"\\subsection\{([^}]*)\}", texto)
    por_subseccion: dict[str, int] = {}
    for titulo, cuerpo in zip(partes[1::2], partes[2::2]):
        por_subseccion[titulo] = len(limpiar_latex(cuerpo).split())
    return len(limpiar_latex(texto).split()), por_subseccion


def main() -> None:
    exe = tectonic()
    comando = [exe, "--keep-logs", "--chatter", "minimal", "main.tex"]
    print("$", " ".join(comando))
    r = subprocess.run(comando, cwd=AQUI, capture_output=True, text=True, encoding="utf-8", errors="replace")
    salida = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        print(salida[-4000:])
        sys.exit(f"Tectonic terminó con código {r.returncode}")
    avisos = [l for l in salida.splitlines() if "warning" in l.lower() and "microtype" not in l.lower()
              and "algorithm.sty" not in l]
    if avisos:
        print("\n".join(avisos[:30]))
    pdf = AQUI / "main.pdf"
    try:
        from pypdf import PdfReader
        paginas = len(PdfReader(str(pdf)).pages)
        print(f"{pdf.name}: {paginas} páginas, {pdf.stat().st_size / 1e6:.2f} MB")
    except ImportError:
        print(f"{pdf.name}: {pdf.stat().st_size / 1e6:.2f} MB")

    total, por_sub = contar_palabras(AQUI / "metodologia.tex")
    presupuesto = {"Formulación del problema y notación": 300, "Datos, análisis exploratorio y preprocesamiento": 650,
                   "Modelo propuesto": 1100, "Entrenamiento y optimización": 450, "Modelos de referencia": 250,
                   "Protocolo de evaluación": 450, "Diseño del estudio de ablación": 200,
                   "Disponibilidad de código y datos": 100}
    print(f"\nPalabras de la sección (sin tablas, figuras, ecuaciones ni pseudocódigo): {total}")
    for titulo, n in por_sub.items():
        ref = next((v for k, v in presupuesto.items() if titulo.startswith(k.split(' ')[0]) and k.split(' ')[1] in titulo), None)
        print(f"  {n:5d}  {titulo}" + (f"  (guía: {ref})" if ref else ""))


if __name__ == "__main__":
    main()
