"""
exact_solver.py
---------------
Solucion exacta del problema SET COVER mediante:

    1. Relajacion lineal (LP)        -> resuelta por SIMPLEX (cota inferior)
    2. Programacion lineal entera    -> resuelta por BRANCH & CUT (CBC)

El SET COVER es NP-Hard. Para instancias reales el solver puede no probar
optimalidad en tiempo razonable; en ese caso reportamos:
    - z_best        : mejor solucion entera encontrada (cota superior)
    - z_lower       : cota inferior actual (mejor cota dual)
    - gap_pct       : (z_best - z_lower) / z_best * 100
    - optimo_probado: True solo si CBC certifico optimalidad

Esto justifica el uso de metaheuristicas como el Algoritmo Genetico.
"""

import re
import time
from pathlib import Path

import numpy as np
import pulp

from utils import cargar_instancia, resumen_instancia, es_solucion_factible


# ---------------------------------------------------------------------------
#  1. Relajacion LP  (Simplex via CBC/CLP)
# ---------------------------------------------------------------------------
def resolver_lp_relajado(A: np.ndarray, c: np.ndarray, verbose: bool = False):
    """
    Relajacion lineal: variables continuas en [0,1]. Resuelta por Simplex.
    El optimo z_LP es una COTA INFERIOR del optimo entero z*.
    """
    m, n = A.shape

    prob = pulp.LpProblem("SetCover_LP_Relajado", pulp.LpMinimize)
    x = [
        pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Continuous")
        for j in range(n)
    ]

    prob += pulp.lpSum(c[j] * x[j] for j in range(n))
    for i in range(m):
        cubren = np.where(A[i] == 1)[0]
        prob += pulp.lpSum(x[j] for j in cubren) >= 1, f"cliente_{i}"

    solver = pulp.PULP_CBC_CMD(msg=verbose)
    t0 = time.perf_counter()
    prob.solve(solver)
    tiempo = time.perf_counter() - t0

    x_val = np.array([v.value() for v in x])
    n_frac = int(np.sum((x_val > 1e-6) & (x_val < 1 - 1e-6)))

    return {
        "status": pulp.LpStatus[prob.status],
        "z_LP": float(pulp.value(prob.objective)),
        "x_LP": x_val,
        "tiempo": tiempo,
        "n_fraccionarias": n_frac,
    }


# ---------------------------------------------------------------------------
#  2. PLE   (Branch & Cut con CBC)
# ---------------------------------------------------------------------------
def resolver_ple(
    A: np.ndarray,
    c: np.ndarray,
    time_limit: int = 300,
    verbose: bool = True,
    log_file=None,
):
    """
    PLE: variables binarias, B&C. Si time_limit se agota antes de probar
    optimalidad, devuelve la mejor solucion encontrada y la cota inferior.
    """
    m, n = A.shape

    prob = pulp.LpProblem("SetCover_PLE", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(n)]

    prob += pulp.lpSum(c[j] * x[j] for j in range(n))
    for i in range(m):
        cubren = np.where(A[i] == 1)[0]
        prob += pulp.lpSum(x[j] for j in cubren) >= 1, f"cliente_{i}"

    log_path = str(log_file) if log_file else None
    solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit, logPath=log_path)
    t0 = time.perf_counter()
    prob.solve(solver)
    tiempo = time.perf_counter() - t0

    x_val = np.array([int(round(v.value() or 0)) for v in x])
    z_best = float(pulp.value(prob.objective))
    status_pulp = pulp.LpStatus[prob.status]
    factible = es_solucion_factible(x_val, A)

    # PuLP reporta "Optimal" aunque CBC se haya detenido por limite de tiempo con una
    # solucion factible, asi que la optimalidad solo se acepta si no se agoto el tiempo
    detenido_por_tiempo = time_limit is not None and tiempo >= 0.99 * time_limit

    # Extraer cota inferior y estado del log de CBC
    z_lower = float("nan")
    optimo_probado = False
    if log_path and Path(log_path).exists():
        log_text = Path(log_path).read_text()
        detenido_por_tiempo = detenido_por_tiempo or "Stopped on time" in log_text
        m1 = re.search(r"Lower bound:\s+([-\d.]+)", log_text)
        if m1:
            z_lower = float(m1.group(1))
        m2 = re.search(r"best possible\s+([-\d.]+)", log_text)
        if m2:
            z_lower = max(z_lower, float(m2.group(1))) if m1 else float(m2.group(1))
    if status_pulp == "Optimal" and not detenido_por_tiempo:
        optimo_probado = True
        z_lower = z_best

    gap_pct = ((z_best - z_lower) / z_best * 100) if z_best > 0 else 0.0

    return {
        "status": status_pulp,
        "optimo_probado": optimo_probado,
        "z_best": z_best,
        "z_lower": z_lower,
        "gap_pct": gap_pct,
        "x_best": x_val,
        "antenas": np.where(x_val == 1)[0],
        "n_antenas": int(x_val.sum()),
        "tiempo": tiempo,
        "factible": factible,
    }


# ---------------------------------------------------------------------------
#  3. Reporte comparativo
# ---------------------------------------------------------------------------
def reporte_comparativo(A, c, time_limit_ple: int = 300, verbose: bool = False):
    res_dir = Path(__file__).resolve().parent.parent / "results"
    res_dir.mkdir(exist_ok=True)
    log_ple = res_dir / "cbc_log.txt"

    print("\n>>> Resolviendo RELAJACION LP (Simplex)...")
    lp = resolver_lp_relajado(A, c, verbose=False)
    print(
        f"    status={lp['status']}, z_LP={lp['z_LP']:.2f}, "
        f"tiempo={lp['tiempo']:.3f}s, "
        f"variables fraccionarias={lp['n_fraccionarias']}"
    )

    print(f"\n>>> Resolviendo PLE (Branch & Cut, limite {time_limit_ple}s)...")
    ple = resolver_ple(
        A, c, time_limit=time_limit_ple, verbose=verbose, log_file=log_ple
    )
    print(
        f"    z_best={ple['z_best']:.2f}, z_lower={ple['z_lower']:.2f}, "
        f"antenas={ple['n_antenas']}, "
        f"gap={ple['gap_pct']:.4f}%, "
        f"tiempo={ple['tiempo']:.2f}s"
    )
    print(
        f"    optimalidad probada: {ple['optimo_probado']}, "
        f"factible: {ple['factible']}"
    )

    gap_lp_pct = (ple["z_best"] - lp["z_LP"]) / ple["z_best"] * 100

    print("\n" + "=" * 64)
    print("COMPARATIVA")
    print("=" * 64)
    print(f"  z_LP  (Simplex, cota inferior raiz)    = {lp['z_LP']:>12,.2f}")
    print(f"  z_lower (cota inferior tras B&C)       = {ple['z_lower']:>12,.2f}")
    print(f"  z_best (mejor solucion entera)         = {ple['z_best']:>12,.2f}")
    print(f"  Gap de integralidad (raiz)             = {gap_lp_pct:>12.4f} %")
    print(f"  Gap final B&C (z_best vs z_lower)      = {ple['gap_pct']:>12.4f} %")
    print(f"  # antenas seleccionadas                = {ple['n_antenas']}")
    print(f"  Tiempo Simplex                         = {lp['tiempo']:>12.3f} s")
    print(f"  Tiempo B&C                             = {ple['tiempo']:>12.2f} s")
    print(f"  Optimalidad probada                    = {ple['optimo_probado']}")
    print("=" * 64)

    return {"lp": lp, "ple": ple, "gap_lp_pct": gap_lp_pct}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    data = base / "data"
    res_dir = base / "results"
    res_dir.mkdir(exist_ok=True)

    A, c = cargar_instancia(
        data / "Proyecto_clase_set_cover.csv", data / "Proyecto_clase_Costo__1_.xlsx"
    )
    resumen_instancia(A, c)

    resultados = reporte_comparativo(A, c, time_limit_ple=600, verbose=False)

    np.save(res_dir / "x_exacto.npy", resultados["ple"]["x_best"])
    with open(res_dir / "exacto_resumen.txt", "w") as f:
        f.write(f"z_LP        = {resultados['lp']['z_LP']:.4f}\n")
        f.write(f"z_lower_BC  = {resultados['ple']['z_lower']:.4f}\n")
        f.write(f"z_best      = {resultados['ple']['z_best']:.4f}\n")
        f.write(f"gap_LP_pct  = {resultados['gap_lp_pct']:.6f}\n")
        f.write(f"gap_BC_pct  = {resultados['ple']['gap_pct']:.6f}\n")
        f.write(f"n_antenas   = {resultados['ple']['n_antenas']}\n")
        f.write(f"optimo_probado = {resultados['ple']['optimo_probado']}\n")
        f.write(f"tiempo_LP   = {resultados['lp']['tiempo']:.4f}\n")
        f.write(f"tiempo_BC   = {resultados['ple']['tiempo']:.4f}\n")
        f.write(f"antenas     = {resultados['ple']['antenas'].tolist()}\n")

    print(f"\nResultados guardados en: {res_dir}")
