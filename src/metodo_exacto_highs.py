"""
metodo_exacto_highs.py
----------------------
Resuelve el Set Cover con el solver HiGHS (via scipy.optimize.milp) para obtener
el optimo certificado de la instancia y comparar contra el Algoritmo Genetico.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from utils import cargar_instancia, costo_solucion, es_solucion_factible


def resolver_highs(A, c, limite_s=1800):
    n = A.shape[1]
    # Cada cliente debe quedar cubierto por al menos una antena
    cobertura = LinearConstraint(A, lb=np.ones(A.shape[0]), ub=np.inf)
    inicio = time.time()
    res = milp(c=c, constraints=[cobertura], integrality=np.ones(n),
               bounds=Bounds(0, 1), options={"time_limit": limite_s, "disp": False})
    return res, time.time() - inicio


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    A, c = cargar_instancia(base / "data" / "Proyecto_clase_set_cover.csv",
                            base / "data" / "Proyecto_clase_Costo__1_.xlsx")
    res, segundos = resolver_highs(A, c)
    x = np.round(res.x).astype(int)
    resumen = {
        "estado": res.message,
        "optimo_certificado": bool(res.status == 0),
        "costo": costo_solucion(x, c),
        "cota_inferior": float(getattr(res, "mip_dual_bound", np.nan)),
        "gap": float(getattr(res, "mip_gap", np.nan)),
        "n_antenas": int(x.sum()),
        "factible": es_solucion_factible(x, A),
        "tiempo_s": round(segundos, 2),
        "antenas": np.flatnonzero(x).tolist(),
    }
    salida = base / "results_highs"
    salida.mkdir(exist_ok=True)
    np.save(salida / "x_highs.npy", x)
    (salida / "resumen_highs.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
