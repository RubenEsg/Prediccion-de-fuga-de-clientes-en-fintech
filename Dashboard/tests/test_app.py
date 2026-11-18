"""Pruebas de la app: cada sección y vista responde, sus callbacks corren con los valores iniciales y las cifras son las
de los datos. Corren probar_app.py, verificar_cifras.py y la parte estática de verificar_estilo.py (el sistema de diseño:
colores, tipografía y componentes), que no necesitan navegador."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from conftest import RAIZ


def _correr(guion: str) -> subprocess.CompletedProcess:
    entorno = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    entorno.pop("FUGA_SOLO", None)
    return subprocess.run([sys.executable, "-B", str(RAIZ / guion)], cwd=RAIZ, env=entorno, capture_output=True, text=True,
                          encoding="utf-8", timeout=900)


def test_rutas_y_redirecciones():
    import app
    c = app.server.test_client()
    for ruta in ["/", "/eda"] + [app.ruta_vista(v) for v, *_ in app.VISTAS]:
        assert c.get(ruta).status_code == 200, ruta
    r = c.get("/cliente?fila=4")
    assert r.status_code == 302 and r.headers["Location"].endswith("/modelos/cliente?fila=4")
    r = c.get("/original")                       # el atajo a la subsección del modelo original
    assert r.status_code == 302 and r.headers["Location"].endswith(app.ruta_vista("original"))
    assert [g for g, _n in app.GRUPOS] == ["todos", "original", "ebm"]
    assert {g for _v, _n, _p, g, *_ in app.VISTAS} == {"todos", "original", "ebm"}
    assert c.get("/documento/no-existe").status_code == 404


@pytest.mark.parametrize("guion", ["probar_app.py", "verificar_cifras.py", "verificar_estilo.py"])
def test_guion_pasa(guion):
    r = _correr(guion)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
