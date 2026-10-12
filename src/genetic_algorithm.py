"""
genetic_algorithm.py
--------------------
Algoritmo Genetico (GA) para el problema SET COVER.

Diseno general
--------------
Codificacion         : vector binario de longitud n  (1 = antena instalada)
Poblacion inicial    : heuristica greedy aleatorizada (asegura factibilidad)
Funcion fitness      : costo total + penalizacion por clientes no cubiertos
Seleccion            : torneo de tamano k=3
Cruce                : uniforme con probabilidad p_cross
Mutacion             : bit-flip con probabilidad p_mut por gen
Reparacion           : si la solucion no es factible, agregar antenas greedy
                       luego eliminar antenas redundantes
Elitismo             : se conservan los E mejores individuos
Parada               : numero maximo de generaciones o estancamiento

Referencias clave (para tu marco teorico):
    - Beasley & Chu (1996) "A genetic algorithm for the set covering problem"
    - Holland (1992)  "Adaptation in Natural and Artificial Systems"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils import (cargar_instancia, resumen_instancia,
                   es_solucion_factible, costo_solucion)


# ===========================================================================
#  PARAMETROS DEL GA
# ===========================================================================
@dataclass
class GAParams:
    """Hiperparametros del Algoritmo Genetico."""
    pop_size:       int   = 120         # tamano de la poblacion
    n_generaciones: int   = 300         # maximo de generaciones
    p_cross:        float = 0.85        # probabilidad de cruce
    p_mut:          float = 0.01        # prob. de mutacion por gen
    torneo_k:       int   = 3           # tamano del torneo
    elite:          int   = 2           # cuantos mejores se conservan
    paciencia:      int   = 80          # paro temprano si no mejora en X gens
    penal_lambda:   float = 1e6         # penalizacion por cliente no cubierto
    seed:           int   = 42          # semilla aleatoria

    def __post_init__(self):
        if self.p_mut is None:
            self.p_mut = -1.0  # se ajusta luego conociendo n


# ===========================================================================
#  OPERADORES
# ===========================================================================
def construir_individuo_greedy(A: np.ndarray, c: np.ndarray,
                                rng: np.random.Generator,
                                aleatoriedad: float = 0.3) -> np.ndarray:
    """
    Construye una solucion factible mediante una heuristica greedy
    aleatorizada. La regla greedy clasica para Set Cover es:

        en cada paso, agregar la antena que minimice
                costo / clientes_nuevos_que_cubre

    Para introducir diversidad en la poblacion inicial, en vez de elegir
    siempre la mejor antena, elegimos una al azar entre las top-k.

    Parameters
    ----------
    aleatoriedad : float in [0,1]
        0 -> greedy puro, 1 -> totalmente aleatorio.
    """
    m, n = A.shape
    x = np.zeros(n, dtype=np.int8)
    cubiertos = np.zeros(m, dtype=bool)

    while not cubiertos.all():
        # Calcular ratio costo / nueva_cobertura para cada antena
        nuevos_por_antena = A[~cubiertos].sum(axis=0)  # cuanto cubriria
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(nuevos_por_antena > 0,
                             c / nuevos_por_antena,
                             np.inf)
        # Antenas ya seleccionadas no compiten
        ratio[x == 1] = np.inf

        # Top-k candidatas
        k = max(1, int(aleatoriedad * 5) + 1)
        idx_orden = np.argsort(ratio)
        candidatas = idx_orden[:k]
        elegida = rng.choice(candidatas)

        x[elegida] = 1
        cubiertos |= (A[:, elegida] == 1)

    return x


def reparar(x: np.ndarray, A: np.ndarray, c: np.ndarray,
            rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Operador de reparacion para SET COVER:

      1. AGREGAR : si x no cubre a todos los clientes, agrega antenas con
                   regla greedy (costo / nueva_cobertura) hasta cubrir todo.

      2. PODAR   : intenta eliminar antenas redundantes (que su retiro no
                   rompe la cobertura). Para EVITAR CONVERGENCIA PREMATURA,
                   el orden de eliminacion es ALEATORIO (cuando se da rng).
                   Asi distintos individuos encuentran distintos optimos
                   locales y se preserva diversidad en la poblacion.
    """
    x = x.copy()

    # ---- 1. Reparacion: agregar antenas hasta cubrir todo ----
    cubiertos = (A @ x) >= 1
    while not cubiertos.all():
        no_cubiertos = ~cubiertos
        nuevos = A[no_cubiertos].sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(nuevos > 0, c / nuevos, np.inf)
        ratio[x == 1] = np.inf
        elegida = int(np.argmin(ratio))
        x[elegida] = 1
        cubiertos = (A @ x) >= 1

    # ---- 2. Eliminar redundantes (orden aleatorio para diversidad) ----
    seleccionadas = np.where(x == 1)[0]
    if rng is not None:
        rng.shuffle(seleccionadas)
    else:
        # fallback determinista: por costo descendente
        seleccionadas = seleccionadas[np.argsort(-c[seleccionadas])]
    for j in seleccionadas:
        x[j] = 0
        if not ((A @ x) >= 1).all():
            x[j] = 1
    return x


def fitness(x: np.ndarray, A: np.ndarray, c: np.ndarray,
            penal_lambda: float) -> float:
    """
    Fitness = costo + penalizacion * (# clientes no cubiertos).
    Como minimizamos, fitness pequeno = mejor.
    """
    no_cubiertos = int(((A @ x) < 1).sum())
    return float(c @ x) + penal_lambda * no_cubiertos


def seleccion_torneo(poblacion: np.ndarray, fits: np.ndarray,
                     k: int, rng: np.random.Generator) -> np.ndarray:
    """Selecciona un individuo por torneo de tamano k."""
    idx = rng.choice(len(poblacion), size=k, replace=False)
    ganador = idx[np.argmin(fits[idx])]   # menor fitness gana
    return poblacion[ganador].copy()


def cruce_uniforme(p1: np.ndarray, p2: np.ndarray,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Cruce uniforme: cada gen viene del padre 1 o del padre 2 con p=0.5."""
    mascara = rng.random(len(p1)) < 0.5
    h1 = np.where(mascara, p1, p2)
    h2 = np.where(mascara, p2, p1)
    return h1.astype(np.int8), h2.astype(np.int8)


def mutar(x: np.ndarray, p_mut: float, rng: np.random.Generator) -> np.ndarray:
    """Bit-flip: cada gen se invierte con probabilidad p_mut."""
    flip = rng.random(len(x)) < p_mut
    x_mut = x.copy()
    x_mut[flip] = 1 - x_mut[flip]
    return x_mut


# ===========================================================================
#  BUCLE PRINCIPAL DEL GA
# ===========================================================================
@dataclass
class GAResultado:
    """Resultado de una corrida del GA."""
    mejor_x: np.ndarray
    mejor_costo: float
    mejor_n_antenas: int
    factible: bool
    historial_mejor: list = field(default_factory=list)
    historial_promedio: list = field(default_factory=list)
    generaciones_corridas: int = 0
    tiempo: float = 0.0
    seed: int = 0


def algoritmo_genetico(A: np.ndarray, c: np.ndarray,
                       params: GAParams = None,
                       verbose: bool = True) -> GAResultado:
    """
    Ejecuta el GA completo. Retorna la mejor solucion encontrada y el
    historial para graficar la convergencia.
    """
    if params is None:
        params = GAParams()
    m, n = A.shape

    # Si p_mut quedo en -1, fijar a 1/n (regla clasica de De Jong)
    p_mut = params.p_mut if params.p_mut > 0 else 1.0 / n

    rng = np.random.default_rng(params.seed)
    t0 = time.perf_counter()

    # ------ Poblacion inicial: greedy con MUCHA diversidad ------
    # Variamos el grado de aleatoriedad de cada individuo para no todos
    # caigan en la misma solucion. Esto es clave para evitar convergencia
    # prematura.
    poblacion = np.zeros((params.pop_size, n), dtype=np.int8)
    for i in range(params.pop_size):
        # aleatoriedad creciente: 0 (greedy puro) a 0.95 (casi aleatorio)
        aleat = 0.05 + 0.9 * (i / max(1, params.pop_size - 1))
        ind = construir_individuo_greedy(A, c, rng, aleatoriedad=aleat)
        poblacion[i] = reparar(ind, A, c, rng)  # podar redundancias

    fits = np.array([fitness(ind, A, c, params.penal_lambda)
                     for ind in poblacion])

    historial_mejor, historial_prom = [], []
    mejor_global_x = poblacion[np.argmin(fits)].copy()
    mejor_global_fit = fits.min()
    sin_mejora = 0

    if verbose:
        print(f"  Gen   0: mejor={mejor_global_fit:>10.0f}  "
              f"prom={fits.mean():>10.0f}")

    # ------------------ Evolucion ------------------
    for gen in range(1, params.n_generaciones + 1):
        # Elitismo: pasar los E mejores a la siguiente generacion
        elite_idx = np.argsort(fits)[:params.elite]
        nueva_pob = [poblacion[i].copy() for i in elite_idx]

        # Generar el resto por seleccion + cruce + mutacion + reparacion
        while len(nueva_pob) < params.pop_size:
            p1 = seleccion_torneo(poblacion, fits, params.torneo_k, rng)
            p2 = seleccion_torneo(poblacion, fits, params.torneo_k, rng)

            if rng.random() < params.p_cross:
                h1, h2 = cruce_uniforme(p1, p2, rng)
            else:
                h1, h2 = p1.copy(), p2.copy()

            h1 = mutar(h1, p_mut, rng)
            h2 = mutar(h2, p_mut, rng)

            # Reparacion -> garantiza factibilidad y depura redundancias
            # (orden aleatorio para mantener diversidad)
            h1 = reparar(h1, A, c, rng)
            h2 = reparar(h2, A, c, rng)

            nueva_pob.append(h1)
            if len(nueva_pob) < params.pop_size:
                nueva_pob.append(h2)

        poblacion = np.array(nueva_pob, dtype=np.int8)
        fits = np.array([fitness(ind, A, c, params.penal_lambda)
                         for ind in poblacion])

        # Tracking
        gen_mejor = fits.min()
        gen_prom  = fits.mean()
        historial_mejor.append(gen_mejor)
        historial_prom.append(gen_prom)

        if gen_mejor < mejor_global_fit - 1e-9:
            mejor_global_fit = gen_mejor
            mejor_global_x   = poblacion[np.argmin(fits)].copy()
            sin_mejora = 0
        else:
            sin_mejora += 1

        if verbose and (gen % 10 == 0 or gen == 1):
            print(f"  Gen {gen:>3}: mejor={gen_mejor:>10.0f}  "
                  f"prom={gen_prom:>10.0f}  "
                  f"sin_mejora={sin_mejora}")

        # Parada por estancamiento
        if sin_mejora >= params.paciencia:
            if verbose:
                print(f"  Parada temprana en generacion {gen} "
                      f"({params.paciencia} gens sin mejora).")
            break

    tiempo_total = time.perf_counter() - t0
    factible = es_solucion_factible(mejor_global_x, A)

    return GAResultado(
        mejor_x=mejor_global_x,
        mejor_costo=costo_solucion(mejor_global_x, c),
        mejor_n_antenas=int(mejor_global_x.sum()),
        factible=factible,
        historial_mejor=historial_mejor,
        historial_promedio=historial_prom,
        generaciones_corridas=gen,
        tiempo=tiempo_total,
        seed=params.seed,
    )


# ===========================================================================
#  Main: corrida unica de demostracion
# ===========================================================================
if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    A, c = cargar_instancia(base / "data" / "Proyecto_clase_set_cover.csv",
                            base / "data" / "Proyecto_clase_Costo__1_.xlsx")
    resumen_instancia(A, c)

    params = GAParams(seed=42)   # usa los defaults optimizados

    print("\n>>> Ejecutando Algoritmo Genetico...")
    res = algoritmo_genetico(A, c, params, verbose=True)

    print("\n" + "=" * 60)
    print("RESULTADO DEL GA (semilla 42)")
    print("=" * 60)
    print(f"  Costo total          = {res.mejor_costo:>12,.2f}")
    print(f"  Numero de antenas    = {res.mejor_n_antenas}")
    print(f"  Factible             = {res.factible}")
    print(f"  Generaciones         = {res.generaciones_corridas}")
    print(f"  Tiempo               = {res.tiempo:.2f} s")
    print("=" * 60)

    # Guardar resultado
    res_dir = base / "results"
    res_dir.mkdir(exist_ok=True)
    np.save(res_dir / "x_ga.npy", res.mejor_x)
    with open(res_dir / "ga_resumen.txt", "w") as f:
        f.write(f"costo        = {res.mejor_costo:.2f}\n")
        f.write(f"n_antenas    = {res.mejor_n_antenas}\n")
        f.write(f"factible     = {res.factible}\n")
        f.write(f"generaciones = {res.generaciones_corridas}\n")
        f.write(f"tiempo       = {res.tiempo:.4f}\n")
        f.write(f"seed         = {res.seed}\n")
        f.write(f"antenas      = {np.where(res.mejor_x==1)[0].tolist()}\n")
