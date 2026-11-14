"""Figuras de la sección de metodología, en PDF vectorial y con texto en inglés (formato de la guía del curso).

Se generan todas desde este único guion, con los datos del proyecto (entrega 3/datos/customer_data.csv) y la
misma partición 80/20 con semilla 42 del Entregable 3:

    python figuras.py            (desde esta carpeta o desde cualquier otra)

Escribe en figures/:
    fig_target.pdf               distribución del objetivo en el entrenamiento y la escalera de active_products
    fig_ebm_architecture.pdf     arquitectura del modelo propuesto (EBM: suma de funciones de forma)
    fig_pipeline_nested_cv.pdf   pipeline combinatorio y validación cruzada anidada 5 x 3

Cada PDF se comprueba al guardarlo: ancho de 6,3 in como máximo (el ancho del texto, para que LaTeX no lo
reduzca al incluirlo con width=\\linewidth y el texto conserve sus 12 pt) y ningún texto fuera del lienzo.
Con la variable de entorno FIGURAS_PNG=<carpeta> guarda además una vista previa PNG de cada figura.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from pypdf import PdfReader
from sklearn.model_selection import train_test_split

AQUI = Path(__file__).resolve().parent
FIGURAS = AQUI / "figures"


def datos_clientes() -> Path:
    """customer_data.csv: en ../entrega 3/datos (carpeta de trabajo), en ../../datos (repositorio publicado, donde va
    comprimido y se extrae la primera vez) o en la carpeta de la variable de entorno COFINFAD."""
    candidatas = [Path(c) for c in (os.environ.get("COFINFAD"), AQUI.parent / "entrega 3" / "datos",
                                    AQUI.parent.parent / "datos") if c]
    for carpeta in candidatas:
        if (carpeta / "customer_data.csv").exists():
            return carpeta / "customer_data.csv"
    for carpeta in candidatas:
        if (carpeta / "customer_data.zip").exists():
            import zipfile
            with zipfile.ZipFile(carpeta / "customer_data.zip") as comprimido:
                comprimido.extractall(carpeta)
            return carpeta / "customer_data.csv"
    sys.exit("No se encontró customer_data.csv; fije la variable COFINFAD con la carpeta de los datos")


DATOS = datos_clientes()
SEMILLA, PRUEBA = 42, 0.20                      # partición del Entregable 3 (src/config.py)
ANCHO_MAX = 6.3                                 # pulgadas: ancho del texto en main.tex

# Estilo pedido por la guía: sans-serif (también en las fórmulas), 12 pt base, 13 pt en los ejes, líneas de
# 2 pt, cuadrícula tenue y fuentes incrustadas (TrueType).
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Latin Modern Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
NAVY, AMBER, GREEN, GREY = "#1F3A5F", "#E0A526", "#2E8B57", "#6E6E6E"
CLARO, MEDIO, FUERTE = "#EAF0F7", "#D5E0EE", "#EDC24A"   # FUERTE se distingue de MEDIO también en gris


def comprobar_textos(fig: plt.Figure) -> None:
    """Falla si algún texto queda fuera del lienzo (en los diagramas el lienzo es el PDF entero)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w, h = fig.bbox.width, fig.bbox.height
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > w + 0.5 or bb.y1 > h + 0.5:
            raise SystemExit(f"texto fuera del lienzo: {t.get_text()[:50]!r} -> {bb}")


def guardar(fig: plt.Figure, nombre: str, recortar: bool) -> None:
    """Guarda el PDF (recortando los márgenes solo en las gráficas con ejes) y comprueba su ancho."""
    FIGURAS.mkdir(exist_ok=True)
    ruta = FIGURAS / nombre
    if not recortar:
        comprobar_textos(fig)
    opciones = {"bbox_inches": "tight", "pad_inches": 0.05} if recortar else {}
    fig.savefig(ruta, **opciones)
    if os.environ.get("FIGURAS_PNG"):
        vista = Path(os.environ["FIGURAS_PNG"])
        vista.mkdir(parents=True, exist_ok=True)
        fig.savefig(vista / nombre.replace(".pdf", ".png"), dpi=150, **opciones)
    plt.close(fig)
    caja = PdfReader(str(ruta)).pages[0].mediabox
    ancho, alto = float(caja.width) / 72, float(caja.height) / 72
    if ancho > ANCHO_MAX + 1e-3:
        raise SystemExit(f"{nombre}: {ancho:.2f} in de ancho (> {ANCHO_MAX} in): LaTeX la reduciría al incluirla")
    print(f"{ruta.relative_to(AQUI)}  ({ancho:.2f} x {alto:.2f} in, {ruta.stat().st_size / 1024:.0f} KB)")


def caja(ax, x, y, w, h, texto, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=color, ec=NAVY, lw=2.0))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", linespacing=1.3)


def flecha(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16, lw=2.0, color=NAVY,
                                 shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ Figura 1: el objetivo
def fig_target() -> None:
    df = pd.read_csv(DATOS, usecols=["churn_probability", "active_products"])
    idx_tr, _ = train_test_split(df.index, test_size=PRUEBA, random_state=SEMILLA)
    tr = df.loc[idx_tr]
    y = tr["churn_probability"]
    mediana, p75 = y.median(), y.quantile(0.75)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})

    ax1.hist(y, bins=60, color=NAVY, alpha=0.85, edgecolor="none")
    ax1.axvline(mediana, color=AMBER, ls="--", label=f"median = {mediana:.3f}")
    ax1.axvline(p75, color=GREEN, ls=":", label=f"75th percentile = {p75:.3f}")
    ax1.set_xlabel("Churn probability score")
    ax1.set_ylabel("Training clients")
    ax1.set_ylim(0, 2900)
    ax1.legend(frameon=False, loc="upper left", handlelength=1.5)

    g = tr.groupby("active_products")["churn_probability"].agg(["mean", "std"])
    ax2.errorbar(g.index, g["mean"], yerr=g["std"], color=NAVY, marker="o", ms=7, capsize=4, lw=2.0,
                 label="mean ± 1 sd")
    ax2.set_xlabel("Active products")
    ax2.set_ylabel("Churn probability score")
    ax2.set_xticks(g.index)
    ax2.set_ylim(0.08, None)
    pasos = -np.diff(g["mean"].values)
    ax2.text(0.97, 0.95, f"constant step\n≈ {pasos.mean():.3f}", transform=ax2.transAxes, ha="right", va="top",
             color=GREY)
    ax2.legend(frameon=False, loc="lower left")

    fig.tight_layout(w_pad=1.6)
    guardar(fig, "fig_target.pdf", recortar=True)


# ------------------------------------------------------------------ Figura 2: la EBM
def fig_ebm_architecture() -> None:
    fig = plt.figure(figsize=(6.3, 4.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.985, "shape functions: step curves on bins quantile intervals,\n"
                         "learned by cyclic boosting of one-variable trees\n"
                         "(L leaves, min m per leaf, rate η), averaged over B bags",
            ha="center", va="top")

    filas = [0.655, 0.48, 0.255]
    etiquetas = ["x₁", "x₂", "xₚ"]
    for y0, et in zip(filas, etiquetas):
        caja(ax, 0.02, y0 - 0.055, 0.10, 0.11, et, CLARO)
    ax.text(0.07, 0.375, "⋮", ha="center", va="center", fontsize=16)

    rng = np.random.default_rng(3)
    formas = [np.cumsum(rng.normal(0, 1, 12)), -np.abs(np.linspace(-1.5, 1.5, 12)) ** 2 + 1.5,
              np.tanh(np.linspace(-2, 2, 12)) * 1.2]
    for y0, curva, et in zip(filas, formas, ["f₁(x₁)", "f₂(x₂)", "fₚ(xₚ)"]):
        caja(ax, 0.20, y0 - 0.07, 0.23, 0.14, "", MEDIO)
        ins = ax.inset_axes((0.215, y0 - 0.055, 0.12, 0.11))
        ins.step(np.arange(12), curva, where="mid", color=NAVY, lw=2.0)
        ins.set_xticks([]); ins.set_yticks([]); ins.grid(False)
        for s in ins.spines.values():
            s.set_visible(False)
        ax.text(0.385, y0, et, ha="center", va="center")
        flecha(ax, (0.12, y0), (0.20, y0))
        flecha(ax, (0.43, y0), (0.56, 0.48 + (y0 - 0.48) * 0.2))
    ax.text(0.315, 0.375, "⋮", ha="center", va="center", fontsize=16)
    caja(ax, 0.20, 0.03, 0.23, 0.12, "fⱼₖ(xⱼ, xₖ)\nK pairs (FAST)", MEDIO)
    flecha(ax, (0.43, 0.09), (0.56, 0.445))

    caja(ax, 0.56, 0.43, 0.11, 0.10, "Σ + β₀", FUERTE)
    flecha(ax, (0.67, 0.48), (0.73, 0.48))
    caja(ax, 0.73, 0.43, 0.11, 0.10, "g⁻¹", CLARO)
    flecha(ax, (0.84, 0.48), (0.91, 0.48))
    ax.text(0.955, 0.48, "ŷ", ha="center", va="center", fontsize=14)

    ax.text(0.775, 0.33, "g = logit (classification)\nor identity (regression)", ha="center", va="top")
    ax.text(0.775, 0.19, "sample weights $w_i = n / (2\\,n_{z_i})$\nfor class balancing;\n"
                         "prediction = p + K table look-ups", ha="center", va="top")
    guardar(fig, "fig_ebm_architecture.pdf", recortar=False)


# ------------------------------------------------------------------ Figura 3: pipeline y validación anidada
def fig_pipeline_nested_cv() -> None:
    fig = plt.figure(figsize=(6.3, 7.8))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pasos = [
        ("COFINFAD: 48,723 clients × 54 columns;\n3,159,157 transactions (2023)", CLARO),
        ("ETL: 12 transaction features per client; 17 columns removed\n"
         "(active_products excluded: target leakage);\n48 predictors → 78 columns after preprocessing", CLARO),
        ("80/20 split by index, seed 42: 38,978 train / 9,745 test;\n"
         "high-risk label threshold τ estimated on the training set only", CLARO),
        ("160 nested runs:\n{7 classifiers + EBM} × 4 balancing × 4 optimizers (128)\n"
         "{7 regressors + EBM} × 4 optimizers (32)", MEDIO),
        ("each run: nested cross-validation 5 × 3 → final refit\n→ test set scored once", FUERTE),
        ("statistical protocol: Friedman + Nemenyi, DeLong + Holm,\nModel Confidence Set, Diebold–Mariano", CLARO),
    ]
    y0 = 0.98
    alto, hueco = {2: 0.074, 3: 0.103}, 0.023
    bordes_inferiores = []
    for texto, color in pasos:
        h = alto[texto.count("\n") + 1]
        caja(ax, 0.02, y0 - h, 0.96, h, texto, color)
        bordes_inferiores.append(y0 - h)
        y0 = y0 - h - hueco
    for i in range(len(pasos) - 1):      # del borde inferior de cada caja al superior de la siguiente
        flecha(ax, (0.50, bordes_inferiores[i]), (0.50, bordes_inferiores[i] - hueco))

    # esquema de la validación anidada
    base = y0 - 0.012
    ax.text(0.02, base, "Nested cross-validation inside the training set (one run):", ha="left", va="top")
    xi, ancho = 0.02, 0.96
    fila1 = base - 0.068
    for k in range(5):
        color = AMBER if k == 1 else NAVY
        ax.add_patch(FancyBboxPatch((xi + k * ancho / 5 + 0.004, fila1), ancho / 5 - 0.008, 0.036,
                                    boxstyle="square,pad=0", fc=color, ec="none"))
    ax.text(0.02, fila1 - 0.012, "outer loop: 5 folds (stratified in classification), the same for\n"
                                 "every model; fold 2 shown as validation", ha="left", va="top")
    fila2 = fila1 - 0.115
    ancho_int = ancho * 0.8
    for j in range(3):
        color = GREEN if j == 1 else MEDIO
        ax.add_patch(FancyBboxPatch((xi + j * ancho_int / 3 + 0.004, fila2), ancho_int / 3 - 0.008, 0.036,
                                    boxstyle="square,pad=0", fc=color, ec=NAVY, lw=2.0))
    ax.text(0.02, fila2 - 0.012,
            "inner loop: the four outer training folds, pooled, are split into\n"
            "3 folds that choose the hyperparameters with one optimizer (Grid,\n"
            "Random, TPE or genetic; budget = grid size, 30–48 evaluations);\n"
            "low fidelity for the EBM (B = 2) and Random Forest (100 trees)", ha="left", va="top")
    guardar(fig, "fig_pipeline_nested_cv.pdf", recortar=False)


if __name__ == "__main__":
    fig_target()
    fig_ebm_architecture()
    fig_pipeline_nested_cv()
