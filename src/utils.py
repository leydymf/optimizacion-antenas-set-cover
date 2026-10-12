"""
utils.py
--------
Funciones de carga y validacion de datos para el problema SET COVER.

La instancia se compone de:
    - Matriz A (m x n) binaria, donde A[i,j] = 1 si la antena j cubre al cliente i
    - Vector c (n,) con el costo de instalar cada antena j
"""

import numpy as np
import pandas as pd
from pathlib import Path


def cargar_instancia(ruta_csv: str, ruta_xlsx: str):
    """
    Carga la matriz de cobertura y el vector de costos desde disco.

    Parameters
    ----------
    ruta_csv : str
        Ruta al CSV con la matriz binaria de cobertura (clientes x antenas).
    ruta_xlsx : str
        Ruta al XLSX con los costos por antena.

    Returns
    -------
    A : np.ndarray (m, n) de int8
        Matriz de cobertura.
    c : np.ndarray (n,) de float
        Vector de costos.
    """
    # Matriz de cobertura: las cabeceras son indices 0..n-1
    A_df = pd.read_csv(ruta_csv)
    A = A_df.to_numpy(dtype=np.int8)

    # Vector de costos: 1 fila, primera columna es la etiqueta "Costo"
    C_df = pd.read_excel(ruta_xlsx)
    c = C_df.iloc[0, 1:].to_numpy(dtype=float)

    # Validaciones basicas
    m, n = A.shape
    assert c.shape[0] == n, f"Costos ({c.shape[0]}) no coinciden con antenas ({n})"
    assert set(np.unique(A)).issubset({0, 1}), "La matriz debe ser binaria"

    return A, c


def resumen_instancia(A: np.ndarray, c: np.ndarray) -> None:
    """Imprime un resumen descriptivo de la instancia."""
    m, n = A.shape
    cobertura_por_cliente = A.sum(axis=1)
    cobertura_por_antena = A.sum(axis=0)

    print("=" * 60)
    print("RESUMEN DE LA INSTANCIA")
    print("=" * 60)
    print(f"  Clientes (m):                {m}")
    print(f"  Antenas  (n):                {n}")
    print(f"  Densidad de la matriz:       {A.mean():.4f}")
    print(f"  Cobertura por cliente:       min={cobertura_por_cliente.min()}, "
          f"max={cobertura_por_cliente.max()}, "
          f"media={cobertura_por_cliente.mean():.1f}")
    print(f"  Clientes cubiertos por cada antena: "
          f"min={cobertura_por_antena.min()}, "
          f"max={cobertura_por_antena.max()}, "
          f"media={cobertura_por_antena.mean():.1f}")
    print(f"  Costos:                      min={c.min():.0f}, "
          f"max={c.max():.0f}, "
          f"media={c.mean():.1f}")
    print("=" * 60)


def es_solucion_factible(x: np.ndarray, A: np.ndarray) -> bool:
    """
    Verifica que la seleccion x cubra todos los clientes.

    Parameters
    ----------
    x : np.ndarray (n,) binario
    A : np.ndarray (m, n) binario

    Returns
    -------
    bool
    """
    cobertura = A @ x
    return bool(np.all(cobertura >= 1))


def costo_solucion(x: np.ndarray, c: np.ndarray) -> float:
    """Costo total de instalar las antenas seleccionadas."""
    return float(c @ x)


if __name__ == "__main__":
    # Prueba rapida del modulo
    base = Path(__file__).resolve().parent.parent / "data"
    A, c = cargar_instancia(base / "Proyecto_clase_set_cover.csv",
                            base / "Proyecto_clase_Costo__1_.xlsx")
    resumen_instancia(A, c)
