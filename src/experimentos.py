"""
experimentos.py
---------------
Bateria de experimentos para el proyecto SET COVER.

Ejecuta el Algoritmo Genetico N_RUNS veces con semillas distintas y produce:

    1. Tabla CSV con resultados de cada corrida
    2. Estadisticas resumen (mejor, peor, promedio, desviacion, gaps)
    3. Grafica de convergencia (mejor por generacion, promediada)
    4. Grafica de variabilidad (boxplot/distribucion de costos)
    5. Archivo de texto con la comparativa final exacto vs GA

Todos los productos se guardan en /results y se pueden incluir directamente
en el documento de Overleaf.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import cargar_instancia, resumen_instancia, costo_solucion
from genetic_algorithm import algoritmo_genetico, GAParams


# ===========================================================================
#  CONFIGURACION DEL EXPERIMENTO
# ===========================================================================
N_RUNS    = 30        # numero de corridas del GA con semillas distintas
SEED_BASE = 42        # semillas seran 42, 43, ..., 42+N_RUNS-1


# ===========================================================================
#  CARGA DE RESULTADOS DEL EXACTO (para calcular gaps)
# ===========================================================================
def cargar_exacto(res_dir: Path) -> dict:
    """Lee el resumen del solver exacto. Devuelve dict o None si no existe."""
    archivo = res_dir / "exacto_resumen.txt"
    if not archivo.exists():
        print("  [aviso] No se encuentra exacto_resumen.txt; "
              "el GA se reportara sin gap vs exacto.")
        return None

    datos = {}
    for linea in archivo.read_text().splitlines():
        if "=" in linea and "antenas" not in linea.split("=")[0].lower()[:8]:
            k, v = linea.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                datos[k] = float(v)
            except ValueError:
                datos[k] = v
    return datos


# ===========================================================================
#  EJECUTAR LAS N_RUNS CORRIDAS
# ===========================================================================
def ejecutar_bateria(A, c, n_runs: int, seed_base: int):
    """Corre el GA n_runs veces y devuelve un DataFrame con los resultados."""
    registros = []
    historiales = []   # para grafica de convergencia

    print(f"\n>>> Ejecutando {n_runs} corridas del GA...\n")
    print(f"  {'Run':>4} {'Seed':>5} {'Costo':>10} {'Antenas':>8} "
          f"{'Gens':>5} {'Tiempo':>8}")
    print("  " + "-" * 50)

    t_total = time.perf_counter()
    for i in range(n_runs):
        seed = seed_base + i
        params = GAParams(seed=seed)
        res = algoritmo_genetico(A, c, params, verbose=False)

        registros.append({
            "run":         i + 1,
            "seed":        seed,
            "costo":       res.mejor_costo,
            "n_antenas":   res.mejor_n_antenas,
            "factible":    res.factible,
            "generaciones": res.generaciones_corridas,
            "tiempo_s":    res.tiempo,
        })
        historiales.append(res.historial_mejor)

        print(f"  {i+1:>4} {seed:>5} {res.mejor_costo:>10,.0f} "
              f"{res.mejor_n_antenas:>8} {res.generaciones_corridas:>5} "
              f"{res.tiempo:>7.1f}s")

    t_total = time.perf_counter() - t_total
    print(f"\n  Tiempo total de la bateria: {t_total:.1f} s "
          f"({t_total/60:.1f} min)")

    return pd.DataFrame(registros), historiales


# ===========================================================================
#  GRAFICAS
# ===========================================================================
def graficar_convergencia(historiales, exacto_z, ruta_png: Path):
    """
    Grafica la convergencia: mejor costo por generacion.
    Muestra todas las corridas (lineas finas) + promedio (linea gruesa)
    + linea horizontal con el resultado del exacto.
    """
    plt.figure(figsize=(10, 6))

    # Igualar longitudes (rellenar con el ultimo valor las corridas mas cortas)
    max_len = max(len(h) for h in historiales)
    matriz = np.full((len(historiales), max_len), np.nan)
    for i, h in enumerate(historiales):
        matriz[i, :len(h)] = h
        if len(h) < max_len:
            matriz[i, len(h):] = h[-1]   # mantener el ultimo valor

    promedio = np.nanmean(matriz, axis=0)
    minimos  = np.nanmin(matriz, axis=0)
    maximos  = np.nanmax(matriz, axis=0)
    gens = np.arange(1, max_len + 1)

    # Banda min-max
    plt.fill_between(gens, minimos, maximos, alpha=0.15, color="steelblue",
                     label="Rango (min-max entre corridas)")
    # Cada corrida como linea fina
    for h in historiales:
        plt.plot(range(1, len(h) + 1), h, color="steelblue", alpha=0.25, lw=0.6)
    # Promedio
    plt.plot(gens, promedio, color="navy", lw=2.5,
             label=f"Promedio ({len(historiales)} corridas)")

    # Linea del exacto
    if exacto_z is not None:
        plt.axhline(exacto_z, color="crimson", lw=2, ls="--",
                    label=f"Mejor del exacto = {exacto_z:,.0f}")

    plt.xlabel("Generacion")
    plt.ylabel("Mejor costo encontrado")
    plt.title("Convergencia del Algoritmo Genetico")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=150)
    plt.close()
    print(f"  [graf] {ruta_png.name} guardada.")


def graficar_distribucion(df: pd.DataFrame, exacto_z, ruta_png: Path):
    """Boxplot + histograma de los costos finales del GA."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    axes[0].boxplot(df["costo"], vert=True, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor="lightsteelblue"))
    axes[0].set_ylabel("Costo final")
    axes[0].set_title(f"Boxplot ({len(df)} corridas del GA)")
    axes[0].set_xticklabels(["GA"])
    if exacto_z is not None:
        axes[0].axhline(exacto_z, color="crimson", ls="--", lw=2,
                        label=f"Exacto = {exacto_z:,.0f}")
        axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Histograma
    axes[1].hist(df["costo"], bins=10, color="steelblue", edgecolor="black",
                 alpha=0.8)
    axes[1].set_xlabel("Costo final")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribucion de costos finales")
    if exacto_z is not None:
        axes[1].axvline(exacto_z, color="crimson", ls="--", lw=2,
                        label=f"Exacto = {exacto_z:,.0f}")
        axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta_png, dpi=150)
    plt.close()
    print(f"  [graf] {ruta_png.name} guardada.")


# ===========================================================================
#  REPORTE FINAL
# ===========================================================================
def reporte_final(df: pd.DataFrame, exacto: dict, ruta_txt: Path):
    """Genera un texto resumen comparando exacto vs GA."""
    mejor = df["costo"].min()
    peor  = df["costo"].max()
    media = df["costo"].mean()
    std   = df["costo"].std()
    t_med = df["tiempo_s"].mean()

    lineas = []
    lineas.append("=" * 64)
    lineas.append("REPORTE COMPARATIVO: EXACTO vs ALGORITMO GENETICO")
    lineas.append("=" * 64)

    if exacto is not None:
        z_exacto = exacto.get("z_best")
        t_exacto = exacto.get("tiempo_BC", exacto.get("tiempo_BC ", None))
        lineas.append("")
        lineas.append("METODO EXACTO (Branch & Cut, CBC)")
        lineas.append("-" * 64)
        lineas.append(f"  z_LP (relajacion lineal, Simplex) : "
                      f"{exacto.get('z_LP', 0):>12,.2f}")
        lineas.append(f"  z_lower (cota inferior B&C)       : "
                      f"{exacto.get('z_lower_BC', 0):>12,.2f}")
        lineas.append(f"  z_best (mejor solucion entera)    : "
                      f"{z_exacto:>12,.2f}")
        lineas.append(f"  Gap final                         : "
                      f"{exacto.get('gap_BC_pct', 0):>12.4f} %")
        lineas.append(f"  Numero de antenas                 : "
                      f"{int(exacto.get('n_antenas', 0)):>12d}")
        lineas.append(f"  Tiempo                            : "
                      f"{exacto.get('tiempo_BC', 0):>12.2f} s")
        lineas.append(f"  Optimalidad probada               : "
                      f"{exacto.get('optimo_probado', 'False')}")

    lineas.append("")
    lineas.append(f"ALGORITMO GENETICO ({len(df)} corridas)")
    lineas.append("-" * 64)
    lineas.append(f"  Mejor costo encontrado            : {mejor:>12,.2f}")
    lineas.append(f"  Peor costo encontrado             : {peor:>12,.2f}")
    lineas.append(f"  Costo promedio                    : {media:>12,.2f}")
    lineas.append(f"  Desviacion estandar               : {std:>12,.2f}")
    lineas.append(f"  Coeficiente de variacion          : "
                  f"{(std/media*100):>12.4f} %")
    lineas.append(f"  # antenas (min/max/moda)          : "
                  f"{df['n_antenas'].min()} / {df['n_antenas'].max()} / "
                  f"{int(df['n_antenas'].mode().iloc[0])}")
    lineas.append(f"  Tiempo medio por corrida          : {t_med:>12.2f} s")
    lineas.append(f"  Generaciones medias               : "
                  f"{df['generaciones'].mean():>12.1f}")
    lineas.append(f"  Todas factibles                   : "
                  f"{bool(df['factible'].all())}")

    if exacto is not None and z_exacto is not None:
        gap_mejor = (mejor - z_exacto) / z_exacto * 100
        gap_medio = (media - z_exacto) / z_exacto * 100
        lineas.append("")
        lineas.append("COMPARACION GA vs EXACTO")
        lineas.append("-" * 64)
        lineas.append(f"  Gap del MEJOR GA vs exacto        : "
                      f"{gap_mejor:>+12.4f} %  "
                      f"({'MEJOR' if gap_mejor < 0 else 'peor'} que exacto)")
        lineas.append(f"  Gap del PROMEDIO GA vs exacto     : "
                      f"{gap_medio:>+12.4f} %")
        lineas.append(f"  Speedup en tiempo (exacto/GA)     : "
                      f"{exacto.get('tiempo_BC', 0)/t_med:>12.2f} x")

    lineas.append("=" * 64)

    texto = "\n".join(lineas)
    print("\n" + texto)
    ruta_txt.write_text(texto)
    print(f"\n  [out] Reporte guardado en {ruta_txt.name}")


# ===========================================================================
#  MAIN
# ===========================================================================
if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"
    res_dir  = base / "results"
    res_dir.mkdir(exist_ok=True)

    A, c = cargar_instancia(data_dir / "Proyecto_clase_set_cover.csv",
                            data_dir / "Proyecto_clase_Costo__1_.xlsx")
    resumen_instancia(A, c)

    exacto = cargar_exacto(res_dir)

    df, historiales = ejecutar_bateria(A, c, N_RUNS, SEED_BASE)

    # Guardar tabla de corridas
    df.to_csv(res_dir / "ga_corridas.csv", index=False)
    print(f"\n  [out] Tabla guardada: ga_corridas.csv")

    # Guardar mejor solucion encontrada por el GA en toda la bateria
    idx_mejor = df["costo"].idxmin()
    seed_mejor = int(df.loc[idx_mejor, "seed"])
    print(f"  [out] Mejor corrida: seed={seed_mejor}, "
          f"costo={df.loc[idx_mejor, 'costo']:,.0f}")

    # Graficas
    z_exacto = exacto["z_best"] if exacto else None
    graficar_convergencia(historiales, z_exacto, res_dir / "convergencia.png")
    graficar_distribucion(df, z_exacto, res_dir / "distribucion.png")

    # Reporte final
    reporte_final(df, exacto, res_dir / "reporte_final.txt")

    print(f"\n  Todo guardado en: {res_dir}")
    print("  Archivos generados:")
    print("    - ga_corridas.csv      (tabla de las 30 corridas)")
    print("    - convergencia.png     (grafica de convergencia)")
    print("    - distribucion.png     (boxplot + histograma)")
    print("    - reporte_final.txt    (comparativa exacto vs GA)")
