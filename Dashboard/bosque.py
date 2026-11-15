"""Random Forest de clasificación en forma compacta: los mismos árboles en arreglos de numpy, para caber en 512 MB.

El artefacto artifacts/clasificacion/random_forest/modelo.joblib (300 árboles, 574.556 nodos) ocupa unos 85 MB en memoria
como objeto de scikit-learn; en el servidor gratuito de Render (512 MB en total) eso no cabe junto con los demás modelos.
exportar() guarda sus árboles en bosque.npz (rasgo, umbral, hijos y fracción de la clase positiva de cada nodo; unos 16 MB
en memoria) y el preprocesado aparte, y comprueba que BosqueCompacto da EXACTAMENTE las mismas probabilidades que el
artefacto en los 9.745 clientes de prueba. Es lo mismo que se hizo con la EBM (motor.py): el modelo no cambia, solo su
representación.

Cómo reproduce a scikit-learn 1.9 bit a bit: (1) la entrada se convierte a float32, como hace el bosque al predecir; (2) en
cada nodo se va a la izquierda si x[rasgo] <= umbral, comparando el float32 con el umbral float64; (3) la probabilidad de un
árbol es la fracción guardada en la hoja (scikit-learn ya no la normaliza al predecir); (4) el bosque suma las de sus
árboles en orden, en float64, y divide entre su número, como ForestClassifier.predict_proba con n_jobs = 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class BosqueCompacto:
    """Los árboles de un RandomForestClassifier binario en arreglos planos con índices globales. predict_proba(z) recibe
    la matriz ya preprocesada, como el estimador dentro de la tubería."""

    def __init__(self, d: dict) -> None:
        self.rasgo = np.asarray(d["rasgo"], dtype=np.int32)          # -2 en las hojas
        self.umbral = np.asarray(d["umbral"], dtype=np.float64)
        self.izq = np.asarray(d["izq"], dtype=np.int32)              # índices globales; -1 en las hojas
        self.der = np.asarray(d["der"], dtype=np.int32)
        self.p1 = np.asarray(d["p1"], dtype=np.float64)              # fracción de la clase positiva en el nodo
        self.p0 = np.asarray(d["p0"], dtype=np.float64)
        self.raices = np.asarray(d["raices"], dtype=np.int64)
        self.profundidad = int(d["profundidad"])
        self.classes_ = np.asarray(d["clases"])
        self.n_features_in_ = int(d["n_rasgos"])

    @property
    def n_estimators(self) -> int:
        return len(self.raices)

    def apply(self, z) -> np.ndarray:
        """La hoja (índice global) a la que llega cada fila en cada árbol: (n, árboles). Solo avanzan los pares
        (fila, árbol) que aún no llegaron a una hoja."""
        x = np.asarray(z, dtype=np.float32)
        n, arboles = x.shape[0], len(self.raices)
        nodo = np.tile(self.raices, n)
        fila = np.repeat(np.arange(n), arboles)
        activos = np.arange(n * arboles)
        while activos.size:
            nd = nodo[activos]
            r = self.rasgo[nd]
            sigue = r >= 0
            activos, nd, r = activos[sigue], nd[sigue], r[sigue]
            if not activos.size:
                break
            valor = x[fila[activos], r].astype(np.float64)
            nodo[activos] = np.where(valor <= self.umbral[nd], self.izq[nd], self.der[nd])
        return nodo.reshape(n, arboles)

    def predict_proba(self, z) -> np.ndarray:
        hojas = self.apply(z)
        a1 = np.zeros(hojas.shape[0])
        a0 = np.zeros(hojas.shape[0])
        for t in range(hojas.shape[1]):              # en orden, como el bosque con un solo hilo
            a0 += self.p0[hojas[:, t]]
            a1 += self.p1[hojas[:, t]]
        a0 /= hojas.shape[1]
        a1 /= hojas.shape[1]
        return np.column_stack([a0, a1])

    def predict(self, z) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(z), axis=1)]


def arreglos(rf) -> dict:
    """Los árboles de un RandomForestClassifier binario como arreglos planos (ver BosqueCompacto)."""
    rasgo, umbral, izq, der, p0, p1, raices = [], [], [], [], [], [], []
    base, profundidad = 0, 0
    for e in rf.estimators_:
        t = e.tree_
        raices.append(base)
        rasgo.append(t.feature.astype(np.int32))
        umbral.append(t.threshold.astype(np.float64))
        izq.append(np.where(t.children_left < 0, -1, t.children_left + base).astype(np.int32))
        der.append(np.where(t.children_right < 0, -1, t.children_right + base).astype(np.int32))
        p0.append(t.value[:, 0, 0].astype(np.float64))
        p1.append(t.value[:, 0, 1].astype(np.float64))
        profundidad = max(profundidad, int(t.max_depth))
        base += t.node_count
    return {"rasgo": np.concatenate(rasgo), "umbral": np.concatenate(umbral), "izq": np.concatenate(izq),
            "der": np.concatenate(der), "p0": np.concatenate(p0), "p1": np.concatenate(p1),
            "raices": np.asarray(raices, dtype=np.int64), "profundidad": profundidad, "clases": rf.classes_,
            "n_rasgos": rf.n_features_in_}


class TuberiaCompacta:
    """Preprocesado + BosqueCompacto con la interfaz de la tubería que usa la app (named_steps, steps, predict_proba,
    predict). No es una Pipeline de scikit-learn porque esta exige que el último paso sea un estimador ajustable."""

    def __init__(self, preprocesado, modelo: BosqueCompacto) -> None:
        self.steps = [("preprocesado", preprocesado), ("modelo", modelo)]
        self.named_steps = dict(self.steps)

    def predict_proba(self, X) -> np.ndarray:
        return self.named_steps["modelo"].predict_proba(self.named_steps["preprocesado"].transform(X))

    def predict(self, X) -> np.ndarray:
        return self.named_steps["modelo"].predict(self.named_steps["preprocesado"].transform(X))


def cargar(carpeta: Path) -> TuberiaCompacta:
    """La tubería (preprocesado + BosqueCompacto) guardada por exportar() en la carpeta del artefacto."""
    import joblib
    d = dict(np.load(carpeta / "bosque.npz"))
    return TuberiaCompacta(joblib.load(carpeta / "preprocesado.joblib"), BosqueCompacto(d))


def exportar(carpeta: Path, X_prueba) -> dict:
    """Escribe bosque.npz y preprocesado.joblib junto a modelo.joblib, comprueba la igualdad exacta con el artefacto en
    X_prueba y la anota en metadata.json. Devuelve la verificación."""
    import joblib
    pipe = joblib.load(carpeta / "modelo.joblib")
    rf = pipe.named_steps["modelo"]
    if rf.n_classes_ != 2:
        raise ValueError("solo bosques binarios")
    np.savez_compressed(carpeta / "bosque.npz", **arreglos(rf))
    joblib.dump(pipe.named_steps["preprocesado"], carpeta / "preprocesado.joblib", compress=3)
    compacto = cargar(carpeta)
    a = pipe.predict_proba(X_prueba)
    b = compacto.predict_proba(X_prueba)
    verif = {"filas": int(len(X_prueba)), "identicas": bool(np.array_equal(a, b)),
             "max_dif": float(np.max(np.abs(a - b))), "nodos": int(len(compacto.named_steps["modelo"].rasgo)),
             "arboles": int(compacto.named_steps["modelo"].n_estimators),
             "archivos": ["bosque.npz", "preprocesado.joblib"],
             "motivo": "la app usa esta representación (unos 16 MB en memoria frente a unos 85 MB del objeto de scikit-learn) "
                       "para caber en los 512 MB del plan gratuito de Render; el modelo es el mismo"}
    if not verif["identicas"]:
        raise AssertionError(f"el bosque compacto no reproduce el artefacto: {verif}")
    ruta = carpeta / "metadata.json"
    meta = json.loads(ruta.read_text(encoding="utf-8"))
    meta["representacion_compacta"] = verif
    ruta.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return verif
