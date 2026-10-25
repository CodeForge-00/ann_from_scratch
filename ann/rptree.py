"""
3. Arboles de proyecciones aleatorias (Random Projection Trees): la idea de Annoy.

    python3 -m ann.rptree

El kd-tree corta solo con hiperplanos paralelos a los ejes, y en dimension alta eso se queda corto. Aqui cada corte es un
hiperplano OBLICUO elegido a partir de los propios datos:

    se toman dos puntos al azar a y b, y se corta por el plano que los separa (la mediatriz de a y b).

Los cortes se adaptan a la forma de los datos y no dependen de los ejes. Pero UN arbol pierde vecinos que quedan al otro lado
de un corte, asi que se construye un BOSQUE de arboles aleatorios y se buscan los candidatos en todos a la vez, con una
cola de prioridad compartida que explora primero las ramas cuyo corte quedaba mas cerca de q.
"""
from __future__ import annotations

import heapq
from functools import lru_cache

import numpy as np
from matplotlib.collections import PolyCollection

from .common import (COLORS, UNIT_SQUARE, Frame, Param, annotate_box, clip_polygon, cli, exact_knn, highd_benchmark,
                     line_through, make_blobs, mark_points, recall_at_k, scatter, sqdist)

K_NEIGHBORS = 5


# --------------------------------------------------------------------------------------------------------------------
# El algoritmo
# --------------------------------------------------------------------------------------------------------------------

class RPNode:
    __slots__ = ("normal", "offset", "left", "right", "idx", "a", "b")

    def __init__(self):
        self.normal = self.offset = self.left = self.right = self.idx = self.a = self.b = None

    @property
    def is_leaf(self):
        return self.idx is not None

    def margin(self, q):
        """Distancia con signo de q al plano de corte: >0 cae a la derecha, <0 a la izquierda."""
        return float(self.normal @ q - self.offset)


def split_node(X, idx, rng):
    """Elige dos puntos al azar y devuelve (normal, offset, a, b, mascara_derecha): el plano equidistante a ambos."""
    for _ in range(8):
        ia, ib = rng.choice(idx, size=2, replace=False)
        a, b = X[ia], X[ib]
        diff = b - a
        norm = np.linalg.norm(diff)
        if norm > 0:
            normal = diff / norm
            offset = float(normal @ (a + b) / 2)
            right = X[idx] @ normal - offset > 0
            if right.any() and not right.all():
                return normal, offset, a, b, right
    right = np.zeros(len(idx), dtype=bool)                  # caso degenerado (puntos repetidos): partir por la mitad
    right[len(idx) // 2:] = True
    return np.zeros(X.shape[1]), 0.0, X[idx[0]], X[idx[-1]], right


def build_tree(X, idx, leaf_size, rng):
    node = RPNode()
    if len(idx) <= leaf_size:
        node.idx = idx
        return node
    node.normal, node.offset, node.a, node.b, right = split_node(X, idx, rng)
    node.left = build_tree(X, idx[~right], leaf_size, rng)
    node.right = build_tree(X, idx[right], leaf_size, rng)
    return node


def leaf_of(node, q):
    """Descenso simple: sigue el lado de cada corte en el que cae q."""
    while not node.is_leaf:
        node = node.right if node.margin(q) > 0 else node.left
    return node


class RPForest:
    def __init__(self, X: np.ndarray, n_trees: int = 8, leaf_size: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.X, self.leaf_size = X, leaf_size
        self.trees = [build_tree(X, np.arange(len(X)), leaf_size, rng) for _ in range(n_trees)]

    def own_leaves(self, q, n_trees=None):
        """Las hojas de q en cada arbol (sin cola de prioridad): la version mas basica."""
        return [leaf_of(t, q) for t in self.trees[: n_trees or len(self.trees)]]

    def collect(self, q: np.ndarray, search_k: int, n_trees: int = None):
        """
        Cola de prioridad compartida por todos los arboles (como Annoy). La prioridad de una rama es el MINIMO, a lo largo
        del camino, de las distancias de q a los cortes que hubo que cruzar: la rama de q tiene prioridad alta, la del otro
        lado de un corte lejano la tiene muy baja. Se reune candidatos hasta tener `search_k`. Devuelve (candidatos, traza).
        """
        trees = self.trees[: n_trees or len(self.trees)]
        heap, counter = [], 0
        for ti, t in enumerate(trees):
            heapq.heappush(heap, (-np.inf, counter, ti, t))
            counter += 1
        found, trace = [], []
        while heap and len(found) < search_k:
            negp, _, ti, node = heapq.heappop(heap)
            pri = -negp
            if node.is_leaf:
                found.extend(node.idx.tolist())
                trace.append((ti, node, pri))
                continue
            m = node.margin(q)
            heapq.heappush(heap, (-min(pri, m), counter, ti, node.right))        # el lado de q tiene margen > 0
            heapq.heappush(heap, (-min(pri, -m), counter + 1, ti, node.left))
            counter += 2
        return np.unique(np.array(found, dtype=int)), trace

    def search(self, q: np.ndarray, k: int, search_k: int, n_trees: int = None):
        cand, trace = self.collect(q, search_k, n_trees)
        d = sqdist(q[None, :], self.X[cand])[0]
        return cand[np.argsort(d)[:k]], len(cand), trace


# --------------------------------------------------------------------------------------------------------------------
# Medicion en alta dimension
# --------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=4)
def forest_tradeoff(n_trees: int):
    X, Q, truth = highd_benchmark()
    forest = RPForest(X, n_trees, leaf_size=16, seed=0)
    frac, rec = [], []
    for sk in (10, 20, 40, 80, 160, 320, 640, 1280, 2500):
        f, r = [], []
        for q, t in zip(Q[:50], truth[:50]):
            ids, nc, _ = forest.search(q, 10, sk)
            f.append(nc / len(X))
            r.append(recall_at_k(ids, t))
        frac.append(float(np.mean(f)))
        rec.append(float(np.mean(r)))
    return frac, rec


# --------------------------------------------------------------------------------------------------------------------
# Dibujo
# --------------------------------------------------------------------------------------------------------------------

def leaf_polygons(node, poly, depth=0, max_depth=None):
    """Poligonos de las celdas del arbol (solo 2D): se recorta el cuadrado con cada corte. Hasta max_depth si se indica."""
    if node.is_leaf or (max_depth is not None and depth == max_depth):
        return [(node, poly)]
    out = []
    for child, positive in ((node.right, True), (node.left, False)):
        sub = clip_polygon(poly, node.normal, node.offset, positive)
        if len(sub) >= 3:
            out += leaf_polygons(child, sub, depth + 1, max_depth)
    return out


def all_indices(node):
    if node.is_leaf:
        return node.idx
    return np.concatenate([all_indices(node.left), all_indices(node.right)])


def find_polygon(polys, leaf):
    for nd, pg in polys:
        if nd is leaf:
            return pg
    return None


def build(p, q):
    X = make_blobs()
    n = len(X)
    T, leaf_size, search_k = p["T"], p["leaf"], p["search_k"]
    forest = RPForest(X, T, leaf_size, seed=p["seed"])
    tree0 = forest.trees[0]
    truth, _ = exact_knn(X, q, K_NEIGHBORS)

    # --- un solo corte ----------------------------------------------------------------------------------------------
    root = tree0
    right_mask = X @ root.normal - root.offset > 0

    def draw_cut(ax):
        scatter(ax, X, color=np.where(right_mask, "#ff7f0e", "#1f77b4"), s=20)
        ax.scatter(*np.array([root.a, root.b]).T, s=170, c="black", marker="D", zorder=8)
        ax.plot(*np.array([root.a, root.b]).T, color="black", ls=":", lw=1.2, zorder=3)
        line_through(ax, root.normal, root.offset, color="black", lw=1.8, zorder=4)

    def aux_cut(ax):
        annotate_box(ax, ["1. elegir dos puntos al azar: a y b (rombos)",
                          "2. el corte es la mediatriz de a y b:",
                          "      n = (b - a) / |b - a|",
                          "      c = n . (a + b) / 2",
                          "3. un punto x va a la derecha si  n . x > c",
                          "",
                          f"izquierda: {int((~right_mask).sum())} puntos",
                          f"derecha:   {int(right_mask.sum())} puntos",
                          "",
                          "Es un solo producto escalar por punto,",
                          "en cualquier dimension d."])

    # --- arbol por niveles -------------------------------------------------------------------------------------------
    def tree_frame(max_depth, title, text):
        polys = leaf_polygons(tree0, UNIT_SQUARE, max_depth=max_depth)
        cols = [COLORS[i % 10] for i in range(len(polys))]
        label = {}
        for ci, (nd, _) in enumerate(polys):
            for i in all_indices(nd):
                label[i] = ci

        def draw(ax):
            ax.add_collection(PolyCollection([pg for _, pg in polys], facecolors=cols, edgecolors="0.25", linewidths=1.2, alpha=0.14, zorder=1))
            scatter(ax, X, color=[COLORS[label[i] % 10] for i in range(n)], s=18)

        def aux(ax):
            sizes = [len(all_indices(nd)) for nd, _ in polys]
            ax.bar(range(len(sizes)), sizes, color=cols)
            ax.set_xlabel("celda")
            ax.set_ylabel("puntos")
            ax.set_title(f"{len(polys)} celdas: puntos por celda", fontsize=10)
        return Frame(title, text, draw, aux)

    polys0 = leaf_polygons(tree0, UNIT_SQUARE)

    # --- un solo arbol -----------------------------------------------------------------------------------------------
    leaf_q = leaf_of(tree0, q)
    cand1 = leaf_q.idx
    miss1 = [i for i in truth if i not in set(cand1)]

    def draw_one_tree(ax):
        ax.add_collection(PolyCollection([pg for _, pg in polys0], facecolors="none", edgecolors="0.5", linewidths=0.7, zorder=1))
        pg = find_polygon(polys0, leaf_q)
        ax.add_collection(PolyCollection([pg], facecolors="#1f77b4", edgecolors="#1f77b4", linewidths=2, alpha=0.2, zorder=1))
        scatter(ax, X, color="0.75", s=14)
        scatter(ax, X[cand1], color="#1f77b4", s=24)
        mark_points(ax, X, truth, color="black", s=130)

    def aux_one_tree(ax):
        annotate_box(ax, [f"candidatos (la hoja de q): {len(cand1)} de {n}",
                          f"vecinos verdaderos capturados: {K_NEIGHBORS - len(miss1)} de {K_NEIGHBORS}",
                          f"vecinos perdidos (aros negros sin punto azul): {len(miss1)}",
                          "",
                          "Si q esta cerca de un corte, sus vecinos",
                          "pueden estar al otro lado: UN solo arbol",
                          "los pierde. Pulsa en el grafico, cerca de",
                          "un borde de celda, para provocarlo."])

    # --- bosque sin prioridad ----------------------------------------------------------------------------------------
    leaves_q = forest.own_leaves(q)
    cand_t = np.unique(np.concatenate([lf.idx for lf in leaves_q]))
    cum_c, cum_f = [], []
    acc = set()
    for lf in leaves_q:
        acc.update(lf.idx.tolist())
        cum_c.append(len(acc))
        cum_f.append(len(acc & set(truth.tolist())))
    all_polys = [leaf_polygons(tr, UNIT_SQUARE) for tr in forest.trees]

    def draw_forest(ax):
        for ti, (tr, polys) in enumerate(zip(forest.trees, all_polys)):
            pg = find_polygon(polys, leaves_q[ti])
            ax.add_collection(PolyCollection([pg], facecolors=COLORS[ti % 10], edgecolors=COLORS[ti % 10], linewidths=1.5, alpha=0.12, zorder=1))
        scatter(ax, X, color="0.78", s=14)
        scatter(ax, X[cand_t], color="#1f77b4", s=24)
        mark_points(ax, X, truth, color="black", s=130)

    def aux_forest(ax):
        xs = np.arange(1, len(leaves_q) + 1)
        ax.plot(xs, np.array(cum_c) / n, "o-", color="#1f77b4", label="fraccion examinada")
        ax.plot(xs, np.array(cum_f) / K_NEIGHBORS, "s-", color="#2ca02c", label="vecinos verdaderos encontrados")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("arboles usados")
        ax.xaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(integer=True))
        ax.set_title("Cada arbol nuevo corta distinto y rescata vecinos", fontsize=10)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)

    # --- cola de prioridad en UN arbol ------------------------------------------------------------------------------
    single = RPForest.__new__(RPForest)
    single.X, single.leaf_size, single.trees = X, leaf_size, [tree0]
    cand_pq1, trace1 = single.collect(q, search_k)

    def draw_pq1(ax):
        ax.add_collection(PolyCollection([pg for _, pg in polys0], facecolors="none", edgecolors="0.5", linewidths=0.7, zorder=1))
        scatter(ax, X, color="0.78", s=14)
        scatter(ax, X[cand_pq1], color="#1f77b4", s=24)
        for order, (_, nd, pri) in enumerate(trace1, 1):
            pg = find_polygon(polys0, nd)
            ax.add_collection(PolyCollection([pg], facecolors="#1f77b4", edgecolors="#1f77b4", linewidths=1.5, alpha=0.18, zorder=1))
            c = pg.mean(0)
            ax.text(c[0], c[1], str(order), ha="center", va="center", fontsize=11, color="#1f4e79", fontweight="bold", zorder=9)
        mark_points(ax, X, truth, color="black", s=130)

    def aux_pq1(ax):
        lines = [f"search_k = {search_k}: se reunen candidatos hasta tener {search_k}",
                 f"hojas visitadas: {len(trace1)}   candidatos: {len(cand_pq1)}", "",
                 "orden  prioridad (margen minimo a un corte)"]
        for order, (_, _, pri) in enumerate(trace1, 1):
            lines.append(f"  {order:>2}      {'inf (hoja de q)' if np.isinf(pri) else f'{pri:.3f}'}")
        lines += ["", "La prioridad de una rama = distancia de q al corte mas", "cercano que hay que cruzar para llegar a ella."]
        annotate_box(ax, lines[:22], fontsize=9)

    # --- bosque con prioridad + search_k -----------------------------------------------------------------------------
    ids_f, ncand_f, trace_f = forest.search(q, K_NEIGHBORS, search_k)
    rec_f = recall_at_k(ids_f, truth)

    def draw_pq_forest(ax):
        scatter(ax, X, color="0.8", s=14)
        cand, _ = forest.collect(q, search_k)
        scatter(ax, X[cand], color="#1f77b4", s=24)
        mark_points(ax, X, truth, color="black", s=130)
        mark_points(ax, X, ids_f, color="#2ca02c", s=70)

    def aux_pq_forest(ax):
        annotate_box(ax, [f"{T} arboles, search_k = {search_k}",
                          f"hojas visitadas: {len(trace_f)}",
                          f"candidatos unicos: {ncand_f} de {n}  ({ncand_f / n:.0%})",
                          f"recall@{K_NEIGHBORS}: {rec_f:.2f}",
                          "",
                          "Dos mandos para ganar precision:",
                          "  * mas arboles (T): mas memoria",
                          "  * mas search_k: mas tiempo de consulta",
                          "Con el mismo bosque se puede elegir, en",
                          "cada consulta, el compromiso que se quiera."])

    def aux_tradeoff(ax):
        for t, col in zip((1, 4, 16), ("#1f77b4", "#ff7f0e", "#2ca02c")):
            fr, rc = forest_tradeoff(t)
            ax.plot(fr, rc, "o-", ms=3, color=col, label=f"{t} arbol(es)")
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="mirar al azar")
        ax.set_xlabel("fraccion de puntos examinados")
        ax.set_ylabel("recall@10")
        ax.set_title("Banco de 32 dimensiones; cada marca = un search_k distinto", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)

    return [
        Frame("un corte oblicuo con dos puntos al azar",
              "Se eligen dos puntos al azar (rombos negros) y se corta por la recta que queda a igual distancia de ambos (su "
              "mediatriz). Los puntos mas cerca de a van a un lado (azul) y los mas cerca de b al otro (naranja). Diferencia clave con "
              "el kd-tree: la recta puede tener cualquier inclinacion y se adapta a donde estan los datos: si hay dos nubes, "
              "tiende a separarlas. Cambia la semilla para ver otros cortes.",
              draw_cut, aux_cut),
        tree_frame(2, "segundo nivel",
                   "Se repite dentro de cada mitad (con otros dos puntos de esa mitad). Las celdas son ahora poligonos convexos "
                   "cualquiera, no rectangulos alineados con los ejes. Como cada corte parte a los puntos en dos grupos por su "
                   "mediatriz, las celdas suelen quedar con tamanos razonables."),
        tree_frame(None, "un arbol completo",
                   f"Se sigue hasta que cada celda tiene <= {leaf_size} puntos. Construir cuesta O(n log n) y el arbol guarda solo "
                   f"un vector normal y un numero por corte. En dimension d, cada corte cuesta un producto escalar, sea d 2 o 2000."),
        Frame("un arbol: la hoja de q",
              f"Buscar con un solo arbol: bajamos hasta la hoja de q y la tomamos como lista de candidatos ({len(cand1)} puntos). "
              f"Es muy barato, pero cuando q esta cerca de un corte, vecinos verdaderos caen al otro lado y se pierden "
              f"(hoy se pierden {len(miss1)} de {K_NEIGHBORS}). Un corte es una decision binaria dura y se equivoca cerca de la frontera.",
              draw_one_tree, aux_one_tree),
        Frame(f"un bosque de {T} arboles",
              f"Solucion: construir T arboles con cortes DISTINTOS (por el azar) y unir las hojas de q en todos. Es la misma idea que "
              f"las tablas de LSH: cada arbol falla en lugares distintos, asi que es raro que TODOS pierdan al mismo vecino. "
              f"Cada color es la celda de q en un arbol. Prueba a cambiar T con el deslizador.",
              draw_forest, aux_forest),
        Frame("cola de prioridad: cruzar el corte mas cercano",
              f"Mejor que 'una hoja por arbol': en vez de cortar la busqueda en la hoja de q, se sigue explorando por orden de "
              f"prioridad. La prioridad de una rama es la distancia de q al corte que habria que cruzar: si el corte esta a "
              f"0.01 de q, el otro lado es MUY prometedor; si esta a 0.3, casi seguro no. Aqui, con un solo arbol, los numeros "
              f"indican el orden de visita hasta reunir search_k = {search_k} candidatos.",
              draw_pq1, aux_pq1),
        Frame("bosque + cola de prioridad + search_k",
              f"Annoy combina las dos cosas: la cola de prioridad es COMPARTIDA por todos los arboles, y se para al reunir search_k "
              f"candidatos. Se ordenan por distancia exacta y se devuelven los {K_NEIGHBORS} mejores: recall {rec_f:.2f} mirando "
              f"el {ncand_f / n:.0%} de los puntos. search_k se elige EN CADA CONSULTA, sin reconstruir el indice.",
              draw_pq_forest, aux_pq_forest),
        Frame("el compromiso, medido",
              "En 32 dimensiones (4000 puntos). Con 1 arbol hay que examinar muchos puntos para lograr buen recall; con 16 la "
              "curva sube mucho antes. El coste de pasar de 1 a 16 arboles es MEMORIA (16 veces mas estructura) y tiempo de "
              "construccion. Es el compromiso tipico de los metodos basados en arboles.",
              lambda ax: scatter(ax, X), aux_tradeoff),
    ]


PARAMS = {
    "T": Param("T (arboles)", 1, 10, 4),
    "leaf": Param("puntos por hoja", 4, 40, 12),
    "search_k": Param("search_k (candidatos)", 5, 200, 40, 5),
    "seed": Param("semilla", 0, 30, 1),
}

if __name__ == "__main__":
    cli("3. Arboles de proyecciones aleatorias (Annoy)", build, PARAMS)
