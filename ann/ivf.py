"""
4. IVF (Inverted File Index): "no busques en todo el almacen; busca solo en los pasillos que importan".

    python3 -m ann.ivf

La idea: partir los datos en `nlist` celdas con k-means. Cada celda tiene un centroide y una LISTA de los puntos que le
pertenecen (la "lista invertida": centroide -> ids). Para consultar:

    1. se mide la distancia de q a los nlist centroides (barato: nlist << n),
    2. se eligen las `nprobe` celdas con centroide mas cercano,
    3. se calcula la distancia exacta SOLO a los puntos de esas celdas.

Es el mismo esquema "candidatos + reordenar" de LSH, pero las celdas se aprenden de los datos (k-means) en vez de sortearse.
Su punto debil esta en el borde de las celdas: un vecino verdadero puede vivir en la celda de al lado, y para encontrarlo hay
que mirar mas de una (nprobe > 1).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon

from .common import (COLORS, Frame, Param, annotate_box, cli, exact_knn, highd_benchmark, kmeans, make_blobs, mark_points,
                     recall_at_k, scatter, sqdist, voronoi_cells)

K_NEIGHBORS = 5


# --------------------------------------------------------------------------------------------------------------------
# El algoritmo
# --------------------------------------------------------------------------------------------------------------------

class IVF:
    """k-means + una lista de ids por celda. Buscar = ordenar los centroides por cercania y recorrer las nprobe primeras listas."""

    def __init__(self, X: np.ndarray, nlist: int, seed: int = 0, iters: int = 25):
        self.X = X
        self.km = kmeans(X, nlist, iters, seed)
        self.centers = self.km.centers
        self.labels = sqdist(X, self.centers).argmin(1)                  # celda de cada punto (respecto a los centroides finales)
        self.lists = [np.where(self.labels == j)[0] for j in range(len(self.centers))]

    def rank_cells(self, q: np.ndarray) -> np.ndarray:
        """Las celdas ordenadas de la mas cercana a la mas lejana (por la distancia de q al centroide)."""
        return np.argsort(sqdist(q[None, :], self.centers)[0])

    def candidates(self, q: np.ndarray, nprobe: int):
        """Los ids de las nprobe celdas mas cercanas. Devuelve (candidatos, celdas visitadas)."""
        probed = self.rank_cells(q)[:nprobe]
        return np.concatenate([self.lists[j] for j in probed]), probed

    def search(self, q: np.ndarray, k: int, nprobe: int):
        """Candidatos + ordenacion exacta solo entre ellos. Devuelve (ids, n_candidatos, celdas visitadas)."""
        cand, probed = self.candidates(q, nprobe)
        if len(cand) == 0:
            return cand, 0, probed
        d = sqdist(q[None, :], self.X[cand])[0]
        return cand[np.argsort(d)[:k]], len(cand), probed


# --------------------------------------------------------------------------------------------------------------------
# Medicion en alta dimension (el compromiso real)
# --------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=4)
def ivf_tradeoff(nlist: int, seed: int = 0):
    """
    Para nprobe = 1, 2, 3, ...: (coste, recall@10 medio, nprobe) en el banco de 32 dimensiones. El coste cuenta TODAS las
    distancias: las de los candidatos y tambien las nlist de la consulta a los centroides.
    """
    X, Q, truth = highd_benchmark()
    ivf = IVF(X, nlist, seed)
    probes = [p for p in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256) if p <= nlist]
    cost, rec = [], []
    for npb in probes:
        c, r = [], []
        for q, t in zip(Q, truth):
            ids, n_cand, _ = ivf.search(q, 10, npb)
            c.append((n_cand + nlist) / len(X))
            r.append(recall_at_k(ids, t))
        cost.append(float(np.mean(c)))
        rec.append(float(np.mean(r)))
    return cost, rec, probes


# --------------------------------------------------------------------------------------------------------------------
# El recorrido
# --------------------------------------------------------------------------------------------------------------------

def draw_cells(ax, polys, emphasized=None):
    """Las celdas de Voronoi coloreadas. Si se da `emphasized` (un conjunto de celdas), las demas quedan palidas."""
    for j, poly in enumerate(polys):
        if len(poly) < 3:
            continue
        hi = emphasized is None or j in emphasized
        ax.add_patch(Polygon(poly, closed=True, facecolor=to_rgba(COLORS[j % 10], 0.30 if hi else 0.06),
                             edgecolor="0.35", linewidth=1.0, zorder=0))


def draw_centers(ax, C, size=170):
    for j, c in enumerate(C):
        ax.scatter([c[0]], [c[1]], marker="X", s=size, c=COLORS[j % 10], edgecolors="black", linewidths=1.2, zorder=8)


def build(p, q):
    X = make_blobs()
    n = len(X)
    nlist = p["nlist"]
    nprobe = min(p["nprobe"], nlist)
    ivf = IVF(X, nlist, p["seed"])
    km = ivf.km
    C = ivf.centers
    polys = voronoi_cells(C)
    truth, _ = exact_knn(X, q, K_NEIGHBORS)
    cols = lambda labels: [COLORS[int(c) % 10] for c in labels]

    # inercia (suma de distancias al cuadrado al centroide) tras cada iteracion de k-means
    init_labels = sqdist(X, km.init).argmin(1)
    inertia = [float(((X - km.init[init_labels]) ** 2).sum())]
    for labels, centers in km.steps:
        inertia.append(float(((X - centers[labels]) ** 2).sum()))

    def aux_inertia(upto):
        def aux(ax):
            ax.plot(range(len(inertia)), inertia, "o-", ms=4, color="0.7")
            ax.plot(range(upto + 1), inertia[: upto + 1], "o-", ms=5, color="#d62728")
            ax.set_xlabel("iteracion de k-means")
            ax.set_ylabel("suma de distancias^2 al centroide")
            ax.set_title("Cada iteracion acerca los centroides a sus puntos", fontsize=10)
            ax.grid(alpha=0.3)
        return aux

    # --- paso 1: centroides iniciales ---------------------------------------------------------------------------------
    def draw_init(ax):
        scatter(ax, X, color="0.75")
        draw_centers(ax, km.init)

    def aux_cost(ax):
        annotate_box(ax, [f"sin indice: {n} distancias por consulta",
                          "",
                          f"con IVF ({nlist} celdas, nprobe={nprobe}):",
                          f"  {nlist} distancias a los centroides",
                          f"  + los puntos de {nprobe} celdas",
                          f"  (si las celdas son parejas: ~{nprobe * n // nlist})",
                          "",
                          "Los centroides iniciales se eligen con k-means++:",
                          "cada uno lejos de los anteriores."])

    # --- paso 2: primera iteracion ------------------------------------------------------------------------------------
    lab1, cen1 = km.steps[0]

    def draw_iter1(ax):
        scatter(ax, X, color=cols(lab1), s=20)
        for a, b in zip(km.init, cen1):
            ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="black", lw=1.4), zorder=9)
        ax.scatter(km.init[:, 0], km.init[:, 1], marker="X", s=70, c="white", edgecolors="black", linewidths=1, zorder=8)
        draw_centers(ax, cen1)

    # --- paso 3: k-means convergido -> celdas de Voronoi --------------------------------------------------------------
    def draw_cells_frame(ax):
        draw_cells(ax, polys)
        scatter(ax, X, color=cols(ivf.labels), s=20)
        draw_centers(ax, C)

    # --- paso 4: la lista invertida -----------------------------------------------------------------------------------
    sizes = np.array([len(l) for l in ivf.lists])

    def aux_lists(ax):
        ax.barh(range(nlist), sizes, color=[COLORS[j % 10] for j in range(nlist)])
        for j, s in enumerate(sizes):
            ax.text(s + 0.5, j, f"{s}", va="center", fontsize=8)
        ax.set_yticks(range(nlist))
        ax.set_yticklabels([f"celda {j}" for j in range(nlist)], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("puntos en la lista")
        ax.set_title(f"Lista invertida: celda -> ids   (media {n / nlist:.0f}, max {sizes.max()})", fontsize=10)

    # --- paso 5: elegir las celdas ------------------------------------------------------------------------------------
    order = ivf.rank_cells(q)
    probed = order[:nprobe]
    dq = np.sqrt(sqdist(q[None, :], C)[0])

    def draw_probe(ax):
        draw_cells(ax, polys, emphasized=set(probed.tolist()))
        scatter(ax, X, color="0.78", s=16)
        for rank, j in enumerate(order):
            on = rank < nprobe
            ax.plot([q[0], C[j, 0]], [q[1], C[j, 1]], color=COLORS[j % 10], lw=2.6 if on else 0.8, ls="-" if on else ":",
                    alpha=0.95 if on else 0.55, zorder=4)
            if on:
                ax.text(C[j, 0] + 0.02, C[j, 1] + 0.02, f"#{rank + 1}", fontsize=11, fontweight="bold", zorder=9)
        draw_centers(ax, C, size=110)

    def aux_probe(ax):
        ax.barh(range(nlist), dq[order], color=[COLORS[j % 10] for j in order], alpha=0.9)
        for r in range(nlist):
            ax.text(dq[order][r] + 0.005, r, f"celda {order[r]}", va="center", fontsize=8)
        ax.axhline(nprobe - 0.5, color="black", ls="--", lw=1)
        ax.invert_yaxis()
        ax.set_yticks(range(nlist))
        ax.set_yticklabels([f"#{r + 1}" for r in range(nlist)], fontsize=8)
        ax.set_xlabel("distancia de q al centroide")
        ax.set_title(f"{nlist} distancias (no {n}); se visitan las {nprobe} primeras", fontsize=10)

    # --- paso 6: escanear esas celdas ---------------------------------------------------------------------------------
    ids, n_cand, _ = ivf.search(q, K_NEIGHBORS, nprobe)
    cand = ivf.candidates(q, nprobe)[0]
    rec = recall_at_k(ids, truth)

    def draw_scan(ax):
        draw_cells(ax, polys, emphasized=set(probed.tolist()))
        scatter(ax, X, color="0.85", s=16)
        scatter(ax, X[cand], color="#1f77b4", s=24)
        mark_points(ax, X, truth, color="black", s=130, label="verdad")
        mark_points(ax, X, ids, color="#2ca02c", s=70, label="encontrados")

    def aux_scan(ax):
        annotate_box(ax, [f"celdas visitadas: {nprobe} de {nlist}",
                          f"candidatos: {n_cand} de {n}",
                          f"distancias calculadas: {n_cand + nlist}  ({nlist} centroides + {n_cand} puntos)",
                          f"ahorro: {1 - (n_cand + nlist) / n:.0%}",
                          "",
                          f"recall@{K_NEIGHBORS} de esta consulta: {rec:.2f}",
                          "",
                          "Igual que en LSH: los candidatos se reordenan con la",
                          "distancia exacta, asi que lo devuelto nunca es falso;",
                          "lo unico que puede pasar es PERDER un vecino."])

    # --- paso 7: el borde de las celdas -------------------------------------------------------------------------------
    missed = [t for t in truth if ivf.labels[t] not in set(probed.tolist())]
    cum_cost, cum_found = [], []
    acc, seen = 0, set()
    for r in range(nlist):
        acc += len(ivf.lists[order[r]])
        seen.update(ivf.lists[order[r]].tolist())
        cum_cost.append((acc + nlist) / n)
        cum_found.append(len(seen & set(truth.tolist())) / K_NEIGHBORS)

    def draw_border(ax):
        draw_cells(ax, polys, emphasized=set(probed.tolist()))
        scatter(ax, X, color="0.85", s=16)
        scatter(ax, X[cand], color="#1f77b4", s=22)
        mark_points(ax, X, truth, color="black", s=130)
        if missed:
            ax.scatter(X[missed, 0], X[missed, 1], marker="x", s=110, c="#d62728", linewidths=3, zorder=9)

    def aux_border(ax):
        xs = np.arange(1, nlist + 1)
        ax.plot(xs, cum_cost, "o-", color="#1f77b4", label="coste (fraccion de distancias)")
        ax.plot(xs, cum_found, "s-", color="#2ca02c", label="vecinos verdaderos encontrados")
        ax.axvline(nprobe, color="black", ls="--", lw=1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("nprobe (celdas visitadas)")
        ax.set_title("Visitar mas celdas: mas vecinos, mas coste", fontsize=10)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)

    # --- paso 8: el compromiso, medido --------------------------------------------------------------------------------
    def aux_tradeoff(ax):
        for nl, col in ((64, "#1f77b4"), (256, "#ff7f0e")):
            cost, rc, probes = ivf_tradeoff(nl)
            ax.plot(cost, rc, "o-", ms=3, color=col, label=f"nlist={nl}")
            for c, r, pr in zip(cost, rc, probes):
                if pr in (1, 4, 16, 64):
                    ax.annotate(f"{pr}", (c, r), textcoords="offset points", xytext=(4, -10), fontsize=7, color=col)
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="mirar al azar")
        ax.set_xlabel("coste (fraccion de distancias calculadas)")
        ax.set_ylabel("recall@10")
        ax.set_title("Medido en 4000 puntos de 32 dim.; cada marca = un nprobe (1, 2, 3, 4, 6...)", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    return [
        Frame("elegir los centroides",
              f"IVF empieza dividiendo los datos en nlist = {nlist} celdas. Cada celda estara representada por un centroide (las X "
              f"de colores). Los centroides iniciales se eligen con k-means++: el primero al azar y cada uno de los siguientes "
              f"lejos de los ya elegidos. Todavia no hay celdas; solo {nlist} candidatos a representar a los {n} puntos.",
              draw_init, aux_cost),
        Frame("k-means, iteracion 1: asignar y mover",
              f"Cada iteracion de k-means hace dos cosas: (1) asigna cada punto a su centroide mas cercano (el color del punto) y "
              f"(2) mueve cada centroide a la media de sus puntos (las flechas: de la X blanca, donde estaba, a la X de color, "
              f"donde queda). Las flechas largas indican que los centroides iniciales estaban lejos de donde debian.",
              draw_iter1, aux_inertia(1)),
        Frame("k-means converge: las celdas",
              f"Tras {len(km.steps)} iteraciones los centroides ya no se mueven (si no converge, se corta a 25). Cada punto pertenece "
              f"al centroide mas cercano, asi que el espacio queda dividido en celdas de Voronoi: el color de fondo marca "
              f"la region donde cada centroide gana. La curva muestra que la distancia total a los centroides solo puede bajar, "
              f"y se estabiliza. Cambia nlist con el deslizador para ver celdas mas grandes o mas pequenas.",
              draw_cells_frame, aux_inertia(len(km.steps))),
        Frame("el indice: una lista por celda",
              f"El indice es solo esto: para cada celda, la lista de ids de los puntos que contiene (lista invertida: "
              f"centroide -> puntos). Construirlo cuesta un k-means; despues se guarda y se reutiliza. Observa que las celdas "
              f"NO son del mismo tamano: k-means sigue la densidad de los datos, y una celda grande significa mas trabajo para "
              f"quien consulte ahi.",
              draw_cells_frame, aux_lists),
        Frame(f"consulta, paso 1: las {nprobe} celdas mas cercanas",
              f"Para consultar no tocamos los puntos todavia. Medimos la distancia de q a los {nlist} centroides (las lineas) y "
              f"nos quedamos con los nprobe = {nprobe} mas cercanos (lineas gruesas, numerados). Eso cuesta {nlist} distancias en "
              f"vez de {n}. La idea: si un centroide esta lejos de q, probablemente sus puntos tambien. Mueve la consulta con el "
              f"clic y cambia nprobe con el deslizador.",
              draw_probe, aux_probe),
        Frame("consulta, paso 2: escanear solo esas celdas",
              f"Ahora si miramos puntos, pero solo los de las celdas elegidas (azules): {n_cand} de {n}. Se calcula la distancia "
              f"exacta a esos candidatos y se devuelven los {K_NEIGHBORS} mejores (verde). Aro negro = vecinos verdaderos. "
              f"Recall de esta consulta: {rec:.2f}.",
              draw_scan, aux_scan),
        Frame("el punto debil: el borde de la celda",
              ("Esta consulta no pierde ningun vecino con este nprobe. " if not missed else
               f"Las cruces rojas son vecinos verdaderos que NO se encontraron: viven en una celda que no visitamos. ") +
              "Pasa cuando q esta cerca de la frontera entre celdas: el vecino mas cercano puede estar al otro lado, en una "
              "celda cuyo CENTROIDE esta mas lejos. La solucion es visitar mas celdas (nprobe mayor), a cambio de mas coste. "
              "Prueba a poner q pegada a una frontera y nprobe = 1, y despues subelo.",
              draw_border, aux_border),
        Frame("el compromiso, medido",
              "En 32 dimensiones se mide el compromiso: coste (distancias calculadas, incluidas las nlist a los centroides) "
              "contra recall. Subir nprobe sube las dos cosas. Mas celdas (nlist = 256) significa celdas mas pequenas: cada "
              "visita es mas barata, pero hace falta visitar mas celdas para tener el mismo recall; ademas la fase de elegir "
              "celdas ya cuesta mas. En la practica nlist ~ sqrt(n) es un buen punto de partida.",
              draw_scan, aux_tradeoff),
    ]


PARAMS = {
    "nlist": Param("nlist (celdas)", 2, 20, 8),
    "nprobe": Param("nprobe (celdas visitadas)", 1, 20, 2),
    "seed": Param("semilla", 0, 30, 1),
}

if __name__ == "__main__":
    cli("4. IVF: indice invertido con celdas k-means", build, PARAMS)
