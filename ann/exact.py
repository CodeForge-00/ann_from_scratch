"""
0. Busqueda EXACTA (fuerza bruta): el punto de partida.

    python3 -m ann.exact

Antes de aproximar hay que entender que es lo que se aproxima y por que hace falta. Esta pantalla recorre:
  1. el problema (vecino mas cercano de una consulta),
  2. la solucion exacta: medir la distancia a TODOS los puntos,
  3. cuanto cuesta eso al crecer n y d,
  4. la maldicion de la dimensionalidad (por que las "ideas obvias" fallan),
  5. el recall: la vara de medir de todo el curso,
  6. una referencia de lo peor que puede hacerse: mirar puntos al azar.
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from .common import (COLORS, Frame, Param, annotate_box, cli, exact_knn, make_blobs, mark_points, recall_at_k, scatter,
                     sqdist)


# --------------------------------------------------------------------------------------------------------------------
# El algoritmo (tres lineas)
# --------------------------------------------------------------------------------------------------------------------

def brute_force(X: np.ndarray, q: np.ndarray, k: int):
    """Distancia a los n puntos + ordenar. Coste: O(n * d) por consulta. Es exacto y es la referencia."""
    return exact_knn(X, q, k)


@lru_cache(maxsize=1)
def measure_scaling(d: int = 64):
    """Mide de verdad cuanto tarda una consulta exacta al crecer n (con vectores de d dimensiones)."""
    rng = np.random.default_rng(0)
    ns = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
    times = []
    for n in ns:
        X = rng.random((n, d), dtype=np.float32)
        q = rng.random(d, dtype=np.float32)
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            dist = ((X - q) ** 2).sum(1)
            np.argpartition(dist, 10)[:10]
            best = min(best, time.perf_counter() - t0)
        times.append(best * 1000)
    return ns, times


@lru_cache(maxsize=1)
def measure_contrast():
    """(d_max - d_min) / d_min para puntos uniformes: mide cuanto se distingue 'el mas cercano' del 'mas lejano'."""
    rng = np.random.default_rng(0)
    dims = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    out = []
    for d in dims:
        X = rng.random((2000, d))
        q = rng.random(d)
        dist = np.sqrt(((X - q) ** 2).sum(1))
        out.append((dist.max() - dist.min()) / dist.min())
    return dims, out


# --------------------------------------------------------------------------------------------------------------------
# El recorrido
# --------------------------------------------------------------------------------------------------------------------

def build(p, q):
    X = make_blobs()
    k = p["k"]
    idx, dist = brute_force(X, q, k)
    all_d = np.sqrt(sqdist(q[None, :], X)[0])

    def base(ax):
        scatter(ax, X)

    def heat(ax):
        sc = ax.scatter(X[:, 0], X[:, 1], s=22, c=all_d, cmap="viridis_r", linewidths=0, zorder=2)
        return sc

    def neighbors(ax):
        heat(ax)
        mark_points(ax, X, idx, color="#d62728", s=110)
        for i in idx:
            ax.plot([q[0], X[i, 0]], [q[1], X[i, 1]], color="#d62728", lw=0.8, alpha=0.5, zorder=3)
        ax.add_patch(__import__("matplotlib").patches.Circle(q, dist[-1], fill=False, ls="--", ec="#d62728", lw=1.2))

    def hist_aux(ax):
        ax.hist(all_d, bins=30, color="0.7")
        ax.axvline(dist[-1], color="#d62728", ls="--")
        ax.set_title(f"Distancias de q a los {len(X)} puntos", fontsize=10)
        ax.set_xlabel("distancia a q")
        ax.set_ylabel("cuantos puntos")
        ax.text(dist[-1], ax.get_ylim()[1] * 0.9, f" los {k} mas cercanos", color="#d62728", fontsize=9)

    def cost_aux(ax):
        ns, ts = measure_scaling()
        ax.loglog(ns, ts, "o-", color="#1f77b4")
        ax.set_title("Tiempo medido de UNA consulta exacta (d=64)", fontsize=10)
        ax.set_xlabel("n = numero de puntos")
        ax.set_ylabel("milisegundos")
        ax.grid(alpha=0.3, which="both")
        ax.text(ns[0], ts[-1] * 0.6, "el coste crece en linea recta con n:\n10x mas datos = 10x mas tiempo", fontsize=9)

    def curse_aux(ax):
        dims, c = measure_contrast()
        ax.semilogx(dims, c, "o-", color="#ff7f0e")
        ax.set_title("Maldicion de la dimensionalidad", fontsize=10)
        ax.set_xlabel("dimension d (2000 puntos uniformes al azar)")
        ax.set_ylabel("(d_max - d_min) / d_min")
        ax.grid(alpha=0.3, which="both")
        ax.text(1.2, max(c) * 0.55, "cuando el contraste tiende a 0, 'el mas\ncercano' apenas se distingue del\nmas lejano", fontsize=9)

    # ---- recall con un "algoritmo" muy malo: mirar una fraccion al azar --------------------------------------------------
    # Si la muestra contiene a un vecino verdadero, este sale siempre entre los k mejores de la muestra (es de los mas
    # cercanos), asi que recall = (vecinos verdaderos que cayeron en la muestra) / k. Se promedia sobre 300 sorteos.
    fracs = np.linspace(0.02, 1.0, 50)
    ranks = np.array([np.argsort(np.random.default_rng(s).permutation(len(X)))[idx] for s in range(300)])
    recalls = [float(np.mean(ranks < max(k, int(f * len(X))))) for f in fracs]
    frac_seen = 0.2
    order = np.random.default_rng(0).permutation(len(X))      # un sorteo concreto, el que se dibuja
    seen = order[: int(frac_seen * len(X))]
    found = seen[np.argsort(np.sqrt(sqdist(q[None, :], X[seen])[0]))[:k]]

    def sample_draw(ax):
        scatter(ax, X, color="0.85")
        scatter(ax, X[seen], color="#1f77b4", s=18)
        mark_points(ax, X, idx, color="#d62728", s=110)
        mark_points(ax, X, found, color="#2ca02c", s=60)

    def recall_aux(ax):
        ax.plot(fracs, recalls, color="#1f77b4", label="mirar puntos al azar")
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="y = x")
        ax.set_title("Recall segun la fraccion de datos examinada", fontsize=10)
        ax.set_xlabel("fraccion de puntos examinados (coste)")
        ax.set_ylabel("recall@k")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.text(0.45, 0.12, "un buen algoritmo ANN debe quedar\nMUY por encima de esta diagonal", fontsize=9)

    r20 = recall_at_k(found, idx)
    return [
        Frame("el problema",
              "Tenemos n puntos (aqui 400, en 2D para poder verlos) y una consulta q (la estrella roja). Queremos los k puntos mas "
              "cercanos a q: el vecino mas cercano (k=1) o los k vecinos. Es la operacion detras de buscadores de imagenes, "
              "recomendadores y la memoria de los LLM (RAG). Haz clic en el grafico para mover q y usa el deslizador k.",
              base),
        Frame("solucion exacta: medirlo todo",
              f"La forma mas simple: calcular la distancia de q a CADA uno de los n puntos. El color indica esa distancia "
              f"(amarillo = cerca, azul oscuro = lejos). Hicimos n = {len(X)} calculos de distancia, cada uno de coste d.",
              heat, hist_aux),
        Frame("ordenar y quedarse con los k mejores",
              f"Ordenamos las distancias y nos quedamos con las {k} menores (anillos rojos). La circunferencia punteada pasa por el "
              f"k-esimo vecino: todo lo que queda fuera es mas lejano. Este resultado es EXACTO, y sera la 'verdad' contra la que "
              f"mediremos todos los algoritmos aproximados.",
              neighbors, hist_aux),
        Frame("el problema: el coste",
              "Cada consulta cuesta O(n·d). Con 400 puntos es instantaneo, pero con 100 millones de vectores de 768 dimensiones "
              "(un indice de texto realista) son ~10^11 operaciones por consulta. La grafica es una medicion real en tu ordenador: "
              "el tiempo crece en linea recta con n. Idea de los ANN: renunciar a la garantia de acertar SIEMPRE a cambio de mirar "
              "solo una pequena parte de los datos.",
              base, cost_aux),
        Frame("el otro problema: la dimension",
              "Los arboles espaciales funcionan de maravilla en 2D, y en dimension alta se degradan. Motivo: al crecer d, las "
              "distancias entre puntos se parecen cada vez mas entre si (la grafica mide ese contraste con datos reales). Si "
              "'cerca' y 'lejos' casi no se distinguen, es dificil descartar zonas enteras sin mirarlas. Lo veremos medido en kdtree.",
              base, curse_aux),
        Frame("como medir una aproximacion: recall",
              f"Un algoritmo aproximado devuelve k candidatos; el RECALL@k es la fraccion de los k vecinos verdaderos (anillos rojos) "
              f"que ha acertado (anillos verdes). Ejemplo ingenuo: mirar solo el {int(frac_seen * 100)}% de los puntos, elegidos al azar "
              f"(puntos azules), da recall = {r20:.2f} en este sorteo. La curva (media de 300 sorteos) muestra que, al azar, recall ≈ fraccion examinada. Todos los algoritmos "
              f"siguientes buscan lo mismo: MUCHO recall mirando POCOS puntos.",
              sample_draw, recall_aux),
    ]


PARAMS = {"k": Param("k vecinos", 1, 15, 5)}

if __name__ == "__main__":
    cli("0. Busqueda exacta: el punto de partida", build, PARAMS)
