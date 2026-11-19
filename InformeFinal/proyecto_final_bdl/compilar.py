"""Compila main.tex con Tectonic (LaTeX + BibTeX en una sola pasada) y cuenta las palabras de la sección.

    python compilar.py          -> main.pdf (desde esta carpeta o desde cualquier otra)

Tectonic se busca en el PATH, en la variable TECTONIC, en ../entrega 3/articulo/herramientas/tectonic.exe (carpeta
de trabajo) o en ../../Entrega3/articulo/herramientas/tectonic.exe (repositorio publicado).
El conteo de palabras ignora comandos, tablas, figuras y comentarios; las cifras, también las escritas en modo
matemático, cuentan como palabras, igual que en un contador sobre el PDF. La guía de resultados pide 3.000 palabras
como máximo, «sin contar tablas, figuras ni código».
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
PRESUPUESTO = {"Resumen del montaje experimental": 150, "Desempeño predictivo global": 500,
               "Resultados de las pruebas de significancia": 800, "Resultados del estudio de ablación": 400,
               "Sensibilidad y robustez": 300, "Costo computacional": 150,
               "Interpretabilidad y análisis de errores": 400, "Discusión, limitaciones y amenazas a la validez": 300}


def tectonic() -> str:
    for candidato in [os.environ.get("TECTONIC"), shutil.which("tectonic"), *map(str, TECTONIC_LOCAL)]:
        if candidato and Path(candidato).exists():
            return candidato
    sys.exit("No se encontró Tectonic: https://tectonic-typesetting.github.io/ (o fija la variable TECTONIC)")


def limpiar_latex(texto: str) -> str:
    texto = re.sub(r"(?<!\\)%.*", "", texto)                                   # comentarios
    for env in ("equation", "equation\\*", "align", "align\\*", "table", "figure", "tabular", "thebibliography"):
        texto = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", " ", texto, flags=re.S)
    texto = re.sub(r"\$\$.*?\$\$", " ", texto, flags=re.S)
    texto = re.sub(r"\\\[.*?\\\]", " ", texto, flags=re.S)
    texto = re.sub(r"\$([^$]*)\$", lambda m: " " + m.group(1).replace("{,}", ",") + " ", texto)   # las cifras cuentan
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
    avisos = [l for l in salida.splitlines() if "warning" in l.lower() and "microtype" not in l.lower()]
    if avisos:
        print("\n".join(avisos[:30]))
    log = (AQUI / "main.log").read_text(encoding="utf-8", errors="replace") if (AQUI / "main.log").exists() else ""
    print(f"overfull: {log.count('Overfull')}  |  referencias sin definir: {log.count('undefined')}")
    pdf = AQUI / "main.pdf"
    try:
        from pypdf import PdfReader
        paginas = len(PdfReader(str(pdf)).pages)
        print(f"{pdf.name}: {paginas} páginas, {pdf.stat().st_size / 1e6:.2f} MB")
    except ImportError:
        print(f"{pdf.name}: {pdf.stat().st_size / 1e6:.2f} MB")

    total, por_sub = contar_palabras(AQUI / "resultados.tex")
    print(f"\nPalabras de la sección de resultados (sin tablas ni figuras; las cifras cuentan): {total}  (guía: 3000)")
    for titulo, n in por_sub.items():
        ref = PRESUPUESTO.get(titulo)
        print(f"  {n:5d}  {titulo}" + (f"  (guía: {ref})" if ref else ""))
    for parte in ("introduccion", "metodologia", "conclusiones"):
        ruta = AQUI / f"{parte}.tex"
        if ruta.exists():
            print(f"Palabras de {parte}.tex (mismo criterio): {contar_palabras(ruta)[0]}")


if __name__ == "__main__":
    main()
