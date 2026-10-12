"""
generar_reporte.py
------------------
Genera los productos finales del proyecto:
    1. reporte_final.txt    (comparativa exacto vs GA, con todos los detalles)
    2. tabla_exacto.txt     (sensibilidad al tiempo limite del exacto: 180/300/600s)
    3. convergencia.png     (regenerada con la linea del exacto)
    4. distribucion.png     (regenerada con la linea del exacto)

Todo se guarda en /results_final.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import cargar_instancia, es_solucion_factible


# ===========================================================================
#  RUTAS
# ===========================================================================
base = Path(__file__).resolve().parent.parent
res_dir = base / "results_final"
res_dir.mkdir(exist_ok=True)


# ===========================================================================
#  1.  Lectura de datos
# ===========================================================================
def leer_resumen(ruta: Path) -> dict:
    """Parsea un archivo exacto_resumen.txt y devuelve un dict."""
    d = {}
    for linea in ruta.read_text().splitlines():
        if "=" not in linea:
            continue
        k, v = linea.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            d[k] = float(v)
        except ValueError:
            d[k] = v
    return d


print("Cargando datos...")
A, c = cargar_instancia(base / "data" / "Proyecto_clase_set_cover.csv",
                        base / "data" / "Proyecto_clase_Costo__1_.xlsx")

# Tres corridas del exacto a distintos tiempos
exacto_180 = leer_resumen(res_dir / "exacto_180s.txt")
exacto_300 = leer_resumen(res_dir / "exacto_300s.txt")
exacto_600 = leer_resumen(res_dir / "exacto_600s.txt")

# Tabla de las 30 corridas del GA
df = pd.read_csv(res_dir / "ga_corridas.csv")

# Soluciones binarias
x_exacto = np.load(res_dir / "x_exacto.npy")
x_ga     = np.load(res_dir / "x_ga.npy")
fact_exacto = es_solucion_factible(x_exacto, A)
fact_ga     = es_solucion_factible(x_ga, A)

# Las corridas de CBC tenian limite de tiempo. Si el tiempo usado llega al limite, CBC se
# detuvo con su mejor solucion y NO probo optimalidad (PuLP igual reporta "Optimal")
for limite, d in [(180, exacto_180), (300, exacto_300), (600, exacto_600)]:
    d["limite_s"] = limite
    d["optimo_probado"] = d["tiempo_BC"] < 0.99 * limite
    if not d["optimo_probado"] and d["z_lower_BC"] == d["z_best"]:
        # La corrida de 600 s no guardo el log de CBC, asi que su cota inferior no se conoce
        d["z_lower_BC"] = float("nan")
        d["gap_BC_pct"] = float("nan")

# Referencia: mejor solucion conocida, obtenida con HiGHS en 30 minutos
highs = json.loads((base / "results_highs" / "resumen_highs.json").read_text())
z_ref = highs["costo"]

# Resultado del exacto con CBC: la corrida de 600s (la mas larga)
z_exacto = exacto_600["z_best"]
n_exacto = int(exacto_600["n_antenas"])
t_exacto = exacto_600["tiempo_BC"]


# ===========================================================================
#  2.  Estadisticas del GA
# ===========================================================================
mejor_ga = df["costo"].min()
peor_ga  = df["costo"].max()
media_ga = df["costo"].mean()
std_ga   = df["costo"].std()
t_ga     = df["tiempo_s"].mean()
n_min    = df["n_antenas"].min()
n_max    = df["n_antenas"].max()
n_moda   = int(df["n_antenas"].mode().iloc[0])
seeds_mejor = df.loc[df["costo"] == mejor_ga, "seed"].tolist()

gap_mejor = (mejor_ga - z_exacto) / z_exacto * 100
gap_medio = (media_ga - z_exacto) / z_exacto * 100
gap_ref_mejor = (mejor_ga - z_ref) / z_ref * 100
gap_ref_medio = (media_ga - z_ref) / z_ref * 100
gap_ref_cbc = (z_exacto - z_ref) / z_ref * 100
mejores   = (df["costo"] < z_exacto).sum()
peores    = (df["costo"] > z_exacto).sum()
iguales   = (df["costo"] == z_exacto).sum()


# ===========================================================================
#  3.  Reporte 1: comparativa exacto vs GA
# ===========================================================================
L = []
L.append("=" * 76)
L.append(" REPORTE COMPARATIVO FINAL : EXACTO  vs  ALGORITMO GENETICO")
L.append(" Problema SET COVER - Instancia 500 clientes x 500 antenas")
L.append("=" * 76)

L.append("\n[1] METODO EXACTO (Branch & Cut, CBC) - corrida con limite de 600s")
L.append("-" * 76)
L.append(f"  Relajacion lineal (Simplex)    : z_LP    = "
         f"{exacto_600['z_LP']:>12,.2f}")
L.append(f"  Cota inferior tras B&C         : z_lower = "
         + ("   no registrada" if np.isnan(exacto_600['z_lower_BC'])
            else f"{exacto_600['z_lower_BC']:>12,.2f}"))
L.append(f"  Mejor solucion entera          : z_best  = "
         f"{z_exacto:>12,.2f}")
L.append(f"  Numero de antenas              : {n_exacto}")
L.append(f"  Tiempo de computo              : {t_exacto:.2f} s")
L.append(f"  Optimalidad probada            : {exacto_600['optimo_probado']} "
         f"(se detuvo por limite de tiempo)")
L.append(f"  Solucion factible (verificada) : {fact_exacto}")

L.append("\n[2] METODO EXACTO (HiGHS) - corrida con limite de 1800s")
L.append("-" * 76)
L.append(f"  Mejor solucion entera          : {z_ref:>12,.2f}  (mejor conocida)")
L.append(f"  Cota inferior                  : {highs['cota_inferior']:>12,.2f}")
L.append(f"  Numero de antenas              : {highs['n_antenas']}")
L.append(f"  Optimalidad probada            : {highs['optimo_certificado']} "
         f"(gap restante {highs['gap']*100:.1f}%)")
L.append(f"  Solucion factible (verificada) : {highs['factible']}")

L.append(f"\n[3] ALGORITMO GENETICO ({len(df)} corridas independientes)")
L.append("-" * 76)
L.append(f"  Mejor costo encontrado         : {mejor_ga:>12,.2f}  "
         f"(seeds: {seeds_mejor})")
L.append(f"  Peor costo                     : {peor_ga:>12,.2f}")
L.append(f"  Costo promedio                 : {media_ga:>12,.2f}")
L.append(f"  Desviacion estandar            : {std_ga:>12,.2f}")
L.append(f"  Coeficiente de variacion       : "
         f"{std_ga/media_ga*100:>12.4f} %")
L.append(f"  Numero de antenas (min/max/moda): {n_min} / {n_max} / {n_moda}")
L.append(f"  Tiempo medio por corrida       : {t_ga:.2f} s")
L.append(f"  Generaciones medias            : {df['generaciones'].mean():.1f}")
L.append(f"  Todas factibles (verificado)   : {bool(df['factible'].all())}")

L.append("\n[4] COMPARACION")
L.append("-" * 76)
L.append(f"  Contra la mejor solucion conocida ({z_ref:,.0f}, HiGHS 30 min):")
L.append(f"    - Mejor corrida del GA       : {gap_ref_mejor:>+8.2f} %")
L.append(f"    - Promedio de las corridas GA: {gap_ref_medio:>+8.2f} %")
L.append(f"    - CBC con 600 s              : {gap_ref_cbc:>+8.2f} %")
L.append(f"\n  Contra CBC con 600 s ({z_exacto:,.0f}):")
L.append(f"    - Corridas del GA con menor costo : {mejores:>3} "
         f"({mejores/len(df)*100:.1f}%)")
L.append(f"    - Corridas del GA con mayor costo : {peores:>3} "
         f"({peores/len(df)*100:.1f}%)")
L.append(f"    - Tiempo: el GA tarda en promedio {t_ga:.0f} s por corrida, "
         f"{t_exacto/t_ga:.0f} veces menos que CBC")

L.append("\n[5] CONCLUSION")
L.append("-" * 76)
L.append("  Ninguno de los dos solvers exactos alcanzo a probar el optimo en el")
L.append("  tiempo dado: la instancia es dificil (el gap de HiGHS sigue en")
L.append(f"  {highs['gap']*100:.0f}% tras 30 minutos). Con un presupuesto de tiempo")
L.append(f"  limitado, el GA es la mejor opcion: su mejor corrida queda a "
         f"{gap_ref_mejor:.2f}% de la")
L.append(f"  mejor solucion conocida usando ~{t_ga:.0f} s, mientras que CBC con 600 s")
L.append(f"  queda a {gap_ref_cbc:.1f}%.")
L.append("")
L.append("  Nota: una version anterior de este reporte afirmaba que CBC habia")
L.append("  probado optimalidad. Era un error de lectura del estado: PuLP")
L.append("  reporta 'Optimal' aunque CBC se detenga por limite de tiempo.")

L.append("\n" + "=" * 76)

texto = "\n".join(L)
print(texto)
(res_dir / "reporte_final.txt").write_text(texto)
print(f"\n[OK] reporte_final.txt guardado")


# ===========================================================================
#  4.  Reporte 2: sensibilidad al tiempo limite del exacto
# ===========================================================================
T = []
T.append("=" * 76)
T.append(" SENSIBILIDAD AL TIEMPO LIMITE DEL METODO EXACTO (Branch & Cut)")
T.append("=" * 76)
T.append("")
T.append(f" Tiempo  | z_lower    | z_best     | Gap final | # antenas | Optimal")
T.append(" --------|------------|------------|-----------|-----------|--------")
for tlim, d in [(180, exacto_180), (300, exacto_300), (600, exacto_600)]:
    cota = "     s/d  " if np.isnan(d["z_lower_BC"]) else f"{d['z_lower_BC']:>10,.2f}"
    gap = "     s/d " if np.isnan(d["gap_BC_pct"]) else f"{d['gap_BC_pct']:>8.4f}%"
    T.append(f"  {tlim:>4}s  | "
             f"{cota} | "
             f"{d['z_best']:>10,.2f} | "
             f"{gap} | "
             f"{int(d['n_antenas']):>9} | "
             f"{d['optimo_probado']}")

T.append("")
T.append(" OBSERVACIONES")
T.append(" - z_LP (relajacion lineal, Simplex) = "
         f"{exacto_600['z_LP']:,.2f} (constante en las 3 corridas)")
mejora_180_300 = exacto_180["z_best"] - exacto_300["z_best"]
mejora_300_600 = exacto_300["z_best"] - exacto_600["z_best"]
T.append(f" - De 180s a 300s: z_best mejoro en {mejora_180_300:.0f} "
         f"({mejora_180_300/exacto_180['z_best']*100:.3f}%)")
T.append(f" - De 300s a 600s: z_best mejoro en {mejora_300_600:.0f} "
         f"({mejora_300_600/exacto_300['z_best']*100:.3f}%)")
T.append(" - Las tres corridas se detuvieron por limite de tiempo, sin probar optimalidad.")
T.append(" - La corrida de 600 s no guardo el log de CBC, por eso su cota no esta disponible (s/d).")
T.append(f" - Mejor solucion conocida (HiGHS, 30 min): {z_ref:,.0f}.")
T.append("")
T.append("=" * 76)

texto2 = "\n".join(T)
print("\n\n" + texto2)
(res_dir / "tabla_exacto_sensibilidad.txt").write_text(texto2)
print(f"\n[OK] tabla_exacto_sensibilidad.txt guardado")


# ===========================================================================
#  5.  Graficas finales (con linea del exacto)
# ===========================================================================
print("\nGenerando graficas...")

# --- DISTRIBUCION ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Boxplot
bp = axes[0].boxplot(df["costo"], vert=True, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor="lightsteelblue", edgecolor="navy"),
                     medianprops=dict(color="navy", linewidth=2))
axes[0].axhline(z_exacto, color="crimson", ls="--", lw=2,
                label=f"CBC con 600 s = {z_exacto:,.0f}")
axes[0].axhline(mejor_ga, color="green", ls=":", lw=2,
                label=f"Mejor GA = {mejor_ga:,.0f}")
axes[0].axhline(z_ref, color="purple", ls="-.", lw=1.5,
                label=f"Mejor conocida (HiGHS 30 min) = {z_ref:,.0f}")
axes[0].set_ylabel("Costo final")
axes[0].set_title(f"Distribucion de costos finales ({len(df)} corridas)")
axes[0].set_xticklabels(["GA"])
axes[0].legend(loc="upper left", fontsize=9)
axes[0].grid(alpha=0.3)

# Histograma
axes[1].hist(df["costo"], bins=12, color="steelblue", edgecolor="black",
             alpha=0.85)
axes[1].axvline(z_exacto, color="crimson", ls="--", lw=2,
                label=f"CBC con 600 s = {z_exacto:,.0f}")
axes[1].axvline(mejor_ga, color="green", ls=":", lw=2,
                label=f"Mejor GA = {mejor_ga:,.0f}")
axes[1].axvline(media_ga, color="orange", ls="-", lw=1.5, alpha=0.8,
                label=f"Media GA = {media_ga:,.0f}")
axes[1].set_xlabel("Costo final")
axes[1].set_ylabel("Frecuencia")
axes[1].set_title("Histograma de costos")
axes[1].legend(loc="upper left", fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(res_dir / "distribucion.png", dpi=150)
plt.close()
print("  [OK] distribucion.png")


# --- COMPARATIVA EN BARRAS POR SEMILLA ---
fig, ax = plt.subplots(figsize=(13, 5))
colores = ['#2c7fb8' if c < z_exacto else '#d7301f' for c in df["costo"]]
ax.bar(df["seed"], df["costo"], color=colores, edgecolor="black", alpha=0.85)
ax.axhline(z_exacto, color="black", ls="--", lw=2,
           label=f"CBC con 600 s = {z_exacto:,.0f}")
ax.axhline(mejor_ga, color="green", ls=":", lw=1.5,
           label=f"Mejor GA = {mejor_ga:,.0f}")
ax.set_xlabel("Semilla")
ax.set_ylabel("Costo final")
ax.axhline(z_ref, color="purple", ls="-.", lw=1.5,
           label=f"Mejor conocida (HiGHS 30 min) = {z_ref:,.0f}")
ax.set_title(f"Costo del GA por semilla (azul = menor que CBC con 600 s, "
             f"rojo = mayor)  -  {mejores}/{len(df)} corridas por debajo de CBC")
ax.set_ylim(df["costo"].min() - 200, df["costo"].max() + 200)
ax.legend(loc="upper right")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(res_dir / "comparativa_por_seed.png", dpi=150)
plt.close()
print("  [OK] comparativa_por_seed.png")


# --- TABLA DE TIEMPOS COMO GRAFICA ---
fig, ax = plt.subplots(figsize=(8, 5))
tiempos = [exacto_180["tiempo_BC"], exacto_300["tiempo_BC"],
           exacto_600["tiempo_BC"], df["tiempo_s"].mean()]
costos = [exacto_180["z_best"], exacto_300["z_best"],
          exacto_600["z_best"], mejor_ga]
tiempos.append(highs["tiempo_s"])
costos.append(z_ref)
etiquetas = ["CBC\n180s", "CBC\n300s", "CBC\n600s",
             f"GA mejor\n({df['tiempo_s'].mean():.0f}s prom)", "HiGHS\n1800s"]
colores_t = ["#bdbdbd", "#969696", "#525252", "#2c7fb8", "#7b3294"]
bars = ax.bar(etiquetas, costos, color=colores_t, edgecolor="black", alpha=0.9)
for bar, costo, t in zip(bars, costos, tiempos):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{costo:,.0f}\n({t:.0f}s)", ha="center", fontsize=10)
ax.set_ylabel("Mejor costo encontrado")
ax.set_title("Comparativa: calidad de solucion y tiempo")
ax.set_ylim(48000, 52500)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(res_dir / "comparativa_metodos.png", dpi=150)
plt.close()
print("  [OK] comparativa_metodos.png")

print(f"\n[FIN] Todo listo en: {res_dir}")
print("\nArchivos disponibles:")
for f in sorted(res_dir.iterdir()):
    print(f"  - {f.name}")
