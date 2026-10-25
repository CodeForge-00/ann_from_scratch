"""
2. KD-tree: dividir el espacio con cortes alineados con los ejes.

    python3 -m ann.kdtree

Un arbol binario: cada nodo corta sus puntos en dos mitades con un hiperplano paralelo a un eje (por la mediana).
Las hojas guardan unos pocos puntos. Para buscar:

  1. se baja hasta la celda que contiene a q y se miran sus puntos -> primer candidato a vecino;
  2. el k-esimo mejor hasta ahora define una bola alrededor de q; solo hay que mirar las celdas que esa bola TOCA;
  3. las celdas que no la tocan se descartan sin mirarlas (poda).

Esto es EXACTO y rapidisimo en pocas dimensiones. Para hacerlo APROXIMADO basta con parar antes: se visitan las celdas en
orden de cercania a q (best-bin-first) y se corta tras un presupuesto de celdas. En dimension alta la poda casi nunca
funciona, y eso tambien se mide aqui.
"""
from __future__ import annotations

import heapq
from functools import lru_cache

import numpy as np
from matplotlib.patches import Circle, Rectangle

from .common import (COLORS, Frame, Param, annotate_box, cli, exact_knn, highd_benchmark, make_blobs, mark_points,
                     recall_at_k, scatter, sqdist)

K_NEIGHBORS = 5


# --------------------------------------------------------------------------------------------------------------------
# El algoritmo
# --------------------------------------------------------------------------------------------------------------------

class Node:
    __slots__ = ("lo", "hi", "depth", "dim", "val", "left", "right", "idx", "all_idx")

    def __init__(self, lo, hi, depth, all_idx):
        self.lo, self.hi, self.depth, self.all_idx = lo, hi, depth, all_idx
        self.dim = self.val = self.left = self.right = self.idx = None

    @property
    def is_leaf(self):
        return self.idx is not None


class KDTree:
    def __init__(self, X: np.ndarray, leaf_size: int = 8, lo=None, hi=None):
        self.X, self.leaf_size = X, leaf_size
        lo = X.min(0) if lo is None else np.asarray(lo, dtype=float)
        hi = X.max(0) if hi is None else np.asarray(hi, dtype=float)
        self.root = self._build(np.arange(len(X)), lo, hi, 0)

    def _build(self, idx, lo, hi, depth):
        node = Node(lo, hi, depth, idx)
        pts = self.X[idx]
        spread = pts.max(0) - pts.min(0)
        if len(idx) <= self.leaf_size or spread.max() == 0:
            node.idx = idx                                           # hoja
            return node
        dim = int(spread.argmax())                                   # cortar por la dimension mas extendida
        order = idx[np.argsort(pts[:, dim], kind="stable")]
        m = len(order) // 2                                          # y por la mediana: mitades equilibradas
        val = float(self.X[order[m], dim])
        node.dim, node.val = dim, val
        left_hi, right_lo = hi.copy(), lo.copy()
        left_hi[dim], right_lo[dim] = val, val
        node.left = self._build(order[:m], lo, left_hi, depth + 1)
        node.right = self._build(order[m:], right_lo, hi, depth + 1)
        return node

    def leaves(self, node=None):
        node = node or self.root
        if node.is_leaf:
            return [node]
        return self.leaves(node.left) + self.leaves(node.right)

    @staticmethod
    def cell_distance2(node: Node, q: np.ndarray) -> float:
        """Cota inferior de la distancia (al cuadrado) de q a CUALQUIER punto de la celda: distancia de q al rectangulo."""
        gap = np.maximum(np.maximum(node.lo - q, q - node.hi), 0.0)
        return float(gap @ gap)

    def search(self, q: np.ndarray, k: int, max_leaves: int = None):
        """
        Best-bin-first: cola de celdas ordenada por su distancia minima a q. Se saca la mas prometedora; si es una hoja se
        miran sus puntos, si no se meten sus dos hijos. Se para cuando la celda mas prometedora ya no puede mejorar el
        k-esimo mejor (exacto) o cuando se agota el presupuesto `max_leaves` (aproximado).

        Devuelve (ids, distancias, n_distancias_calculadas, traza). La traza guarda, tras cada hoja visitada,
        (hoja, n_dist acumuladas, ids de los k mejores actuales, distancia al k-esimo).
        """
        best = []                                                    # max-heap de (-dist2, id) con los k mejores
        heap = [(0.0, 0, self.root)]
        counter, n_dist, trace = 1, 0, []
        while heap:
            if max_leaves is not None and len(trace) >= max_leaves:
                break
            d2, _, node = heapq.heappop(heap)
            if len(best) == k and d2 >= -best[0][0]:
                break                                                # PODA: esta celda y todas las demas quedan fuera del radio
            if not node.is_leaf:
                for child in (node.left, node.right):
                    heapq.heappush(heap, (self.cell_distance2(child, q), counter, child))
                    counter += 1
                continue
            dd = ((self.X[node.idx] - q) ** 2).sum(1)
            n_dist += len(node.idx)
            for i, di in zip(node.idx, dd):
                if len(best) < k:
                    heapq.heappush(best, (-di, i))
                elif di < -best[0][0]:
                    heapq.heapreplace(best, (-di, i))
            ordered = sorted([(-nd, i) for nd, i in best])
            trace.append((node, n_dist, [i for _, i in ordered], float(np.sqrt(ordered[-1][0]))))
        ordered = sorted([(-nd, i) for nd, i in best])
        ids = np.array([i for _, i in ordered], dtype=int)
        return ids, np.sqrt([d for d, _ in ordered]), n_dist, trace


# --------------------------------------------------------------------------------------------------------------------
# Mediciones en alta dimension
# --------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def exact_cost_by_dimension():
    """Fraccion de puntos cuya distancia hay que calcular en una busqueda EXACTA con kd-tree, segun la dimension."""
    rng = np.random.default_rng(0)
    dims = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    out = []
    for d in dims:
        X = rng.normal(size=(4000, d))
        tree = KDTree(X, leaf_size=16)
        fr = [tree.search(rng.normal(size=d), 10)[2] / len(X) for _ in range(20)]
        out.append(float(np.mean(fr)))
    return dims, out


@lru_cache(maxsize=1)
def budget_tradeoff():
    """Recall@10 vs fraccion examinada en el banco de 32 dimensiones, variando el presupuesto de hojas."""
    X, Q, truth = highd_benchmark()
    tree = KDTree(X, leaf_size=16)
    budgets = [1, 2, 4, 8, 16, 32, 64, 128, 250]
    frac, rec = [], []
    for b in budgets:
        f, r = [], []
        for q, t in zip(Q[:50], truth[:50]):
            ids, _, nd, _ = tree.search(q, 10, max_leaves=b)
            f.append(nd / len(X))
            r.append(recall_at_k(ids, t))
        frac.append(float(np.mean(f)))
        rec.append(float(np.mean(r)))
    return budgets, frac, rec


# --------------------------------------------------------------------------------------------------------------------
# Dibujo
# --------------------------------------------------------------------------------------------------------------------

def cells_at_level(node, level):
    if node.depth == level or node.is_leaf:
        return [node]
    return cells_at_level(node.left, level) + cells_at_level(node.right, level)


def draw_rect(ax, node, fc=None, ec="0.25", lw=1.0, alpha=0.25, hatch=None, z=1):
    ax.add_patch(Rectangle(node.lo, *(node.hi - node.lo), facecolor=fc if fc else "none", edgecolor=ec, linewidth=lw,
                           alpha=alpha if fc else 1.0, hatch=hatch, zorder=z))


def draw_tree_diagram(ax, root, max_depth):
    """Dibujo del arbol (hasta max_depth) en el panel auxiliar."""
    ax.set_xlim(0, 1)
    ax.set_ylim(-max_depth - 0.6, 0.6)
    ax.axis("off")
    ax.set_title("El arbol", fontsize=10)

    def rec(node, x, w):
        y = -node.depth
        if node.is_leaf or node.depth == max_depth:
            ax.text(x, y, f"{len(node.all_idx)}", ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round", fc="#cfe8cf" if node.is_leaf else "#eeeeee", ec="0.4"))
            return
        ax.text(x, y, f"x{node.dim} < {node.val:.2f}", ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", fc="#cfe0f5", ec="0.4"))
        for child, cx in ((node.left, x - w / 4), (node.right, x + w / 4)):
            ax.plot([x, cx], [y - 0.12, -child.depth + 0.12], color="0.5", lw=0.8)
            rec(child, cx, w / 2)

    rec(root, 0.5, 1.0)
    ax.text(0.5, -max_depth - 0.45, "azul = corte (dimension, valor); verde = hoja (n de puntos)", ha="center", fontsize=8, color="0.4")


def build(p, q):
    X = make_blobs()
    n = len(X)
    tree = KDTree(X, leaf_size=p["leaf"], lo=[0, 0], hi=[1, 1])
    leaves = tree.leaves()
    truth, _ = exact_knn(X, q, K_NEIGHBORS)
    ids_ex, dist_ex, nd_ex, trace_ex = tree.search(q, K_NEIGHBORS)
    budget = p["hojas"]
    ids_ap, dist_ap, nd_ap, trace_ap = tree.search(q, K_NEIGHBORS, max_leaves=budget)
    max_depth = max(lf.depth for lf in leaves)

    def level_frame(level, title, text):
        cells = cells_at_level(tree.root, level)
        label = {}
        for ci, c in enumerate(cells):
            for i in c.all_idx:
                label[i] = ci

        def draw(ax):
            for ci, c in enumerate(cells):
                draw_rect(ax, c, fc=COLORS[ci % 10], alpha=0.12, ec="0.25", lw=1.4)
            scatter(ax, X, color=[COLORS[label[i] % 10] for i in range(n)], s=18)

        return Frame(title, text, draw, lambda ax: draw_tree_diagram(ax, tree.root, min(level, 4)))

    # --- busqueda: celdas visitadas ----------------------------------------------------------------------------------
    def search_draw(trace, upto, show_ball=True, show_pruned=False, ids_to_mark=None):
        def draw(ax):
            visited = {id(t[0]): ord_ for ord_, t in enumerate(trace[:upto], 1)}
            for lf in leaves:
                if id(lf) in visited:
                    draw_rect(ax, lf, fc="#1f77b4", alpha=0.22, ec="#1f77b4", lw=1.6, z=1)
                    c = (lf.lo + lf.hi) / 2
                    ax.text(c[0], c[1], str(visited[id(lf)]), ha="center", va="center", fontsize=9, color="#1f4e79", zorder=7)
                else:
                    draw_rect(ax, lf, ec="0.6", lw=0.6, hatch="////" if show_pruned else None)
            scatter(ax, X, color="0.6", s=14)
            seen = np.concatenate([lf.idx for lf in [t[0] for t in trace[:upto]]]) if upto else np.array([], dtype=int)
            scatter(ax, X[seen], color="#1f77b4", s=22)
            mark_points(ax, X, truth, color="black", s=130)
            if upto:
                cur = trace[upto - 1]
                mark_points(ax, X, cur[2], color="#2ca02c", s=70)
                if show_ball:
                    ax.add_patch(Circle(q, cur[3], fill=False, ec="#d62728", ls="--", lw=1.3, zorder=5))
        return draw

    def search_aux_plot(trace, budget_mark=None):
        def aux(ax):
            xs = np.arange(1, len(trace) + 1)
            ax.plot(xs, [recall_at_k(t[2], truth) for t in trace], "s-", color="#2ca02c", label="recall")
            ax.plot(xs, [t[1] / n for t in trace], "o-", color="#1f77b4", label="fraccion examinada")
            if budget_mark:
                ax.axvline(budget_mark, color="#d62728", ls="--")
                ax.text(budget_mark, 0.5, " presupuesto", color="#d62728", fontsize=9, rotation=90, va="center")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("celdas (hojas) visitadas, en orden")
            ax.set_title("El recall sube rapido; el resto es 'demostrar' que no hay nada mejor", fontsize=9)
            ax.xaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(integer=True))
            ax.legend(fontsize=9, loc="center right")
            ax.grid(alpha=0.3)
        return aux

    first = trace_ex[0]
    first_text = (f"Bajamos por el arbol hasta la celda que contiene a q y miramos sus {len(first[0].idx)} puntos (azules). Los "
                  f"mejores {K_NEIGHBORS} hasta ahora son los aros verdes; la circunferencia roja pasa por el {K_NEIGHBORS}º. "
                  f"Aros negros = vecinos verdaderos. Con solo {first[1]} distancias ya tenemos una respuesta decente, pero no "
                  f"sabemos si es exacta.")
    cross = [lf for lf in leaves if lf is not first[0] and tree.cell_distance2(lf, q) < first[3] ** 2]

    def draw_ball(ax):
        search_draw(trace_ex, 1)(ax)
        for lf in cross:
            draw_rect(ax, lf, fc="#ff7f0e", alpha=0.3, ec="#ff7f0e", lw=1.4)

    ex_all = len(trace_ex)
    n_leaves = len(leaves)
    rec_ex = recall_at_k(ids_ex, truth)
    rec_ap = recall_at_k(ids_ap, truth)
    d_cols, d_fracs = exact_cost_by_dimension()

    def aux_curse(ax):
        ax.plot(d_cols, d_fracs, "o-", color="#d62728")
        ax.set_xscale("log")
        ax.set_xlabel("dimension d")
        ax.set_ylabel("fraccion de puntos con distancia calculada")
        ax.set_title("Busqueda EXACTA con kd-tree (medido, 4000 puntos)", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, which="both")
        ax.text(1.1, 0.5, "en 1-3D la poda salva casi todo;\npasada la decena de dimensiones\nel arbol visita ~todo: da igual\nque la fuerza bruta", fontsize=9)

    def aux_budget(ax):
        bs, fr, rc = budget_tradeoff()
        ax.plot(fr, rc, "o-", color="#1f77b4", label="kd-tree con presupuesto de hojas")
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="mirar al azar")
        for b, f, r in zip(bs, fr, rc):
            ax.annotate(str(b), (f, r), textcoords="offset points", xytext=(4, -10), fontsize=7)
        ax.set_xlabel("fraccion de puntos examinados")
        ax.set_ylabel("recall@10")
        ax.set_title("Banco de 32 dimensiones (etiquetas = hojas visitadas)", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)

    frames = [
        level_frame(1, "primer corte",
                    "Empezamos con todos los puntos en una caja. Elegimos la dimension en la que los puntos estan mas dispersos "
                    "(x o y) y cortamos por la MEDIANA: la mitad de los puntos a cada lado. Asi el arbol queda equilibrado. "
                    "A la derecha, el nodo del arbol: 'x0 < valor' manda a la izquierda los puntos menores."),
        level_frame(2, "segundo nivel",
                    "Repetimos en cada mitad, con su propia dimension y mediana (el eje puede cambiar de una caja a otra). "
                    "Cada corte reparte los puntos por la mitad, asi que tras h niveles hay 2^h celdas con n/2^h puntos."),
        level_frame(3, "tercer nivel",
                    "Y otra vez. Cada celda es un rectangulo; los rectangulos nunca se solapan y juntos cubren todo el espacio, asi que "
                    "cada punto (y cada consulta) cae en exactamente una celda."),
        level_frame(max_depth, f"arbol completo: {n_leaves} hojas",
                    f"Paramos al llegar a celdas con <= {p['leaf']} puntos (deslizador 'puntos por hoja'): son las HOJAS. "
                    f"Construir cuesta O(n log n); el arbol tiene profundidad ~log2(n/{p['leaf']}) = {max_depth}. "
                    f"Hojas grandes = arbol mas bajo pero mas distancias por hoja."),
        Frame("buscar: bajar hasta la celda de q", first_text, search_draw(trace_ex, 1)),
        Frame("la bola de busqueda cruza fronteras",
              f"El vecino verdadero puede estar en OTRA celda: q esta cerca de un corte y sus vecinos pueden estar al otro lado. "
              f"Pero ahora conocemos un radio: el k-esimo mejor (circunferencia roja). Cualquier celda que la circunferencia "
              f"no toque no puede contener nada mejor, asi que se PODA sin mirarla. Las {len(cross)} celdas naranjas SI la tocan: "
              f"hay que visitarlas.",
              draw_ball),
        Frame("visitar las celdas por orden de cercania",
              f"Se mantiene una cola de celdas pendientes ordenada por su distancia minima a q (best-bin-first). Se visita la mas "
              f"prometedora (los numeros indican el orden). Cada visita puede encoger la circunferencia y podar mas celdas. Aqui "
              f"van las primeras {min(3, ex_all)} celdas.",
              search_draw(trace_ex, min(3, ex_all)), search_aux_plot(trace_ex[:min(3, ex_all)])),
        Frame("busqueda EXACTA terminada",
              f"Seguimos hasta que la celda pendiente mas cercana queda fuera de la circunferencia. Resultado EXACTO (recall "
              f"{rec_ex:.2f}) visitando {ex_all} de {n_leaves} hojas y calculando {nd_ex} de {n} distancias ({nd_ex / n:.0%}). "
              f"Las celdas rayadas ni se tocaron. Fijate en la grafica: el recall suele llegar a 1 en las primeras celdas; el resto de la "
              f"busqueda solo sirve para GARANTIZAR que no hay nada mejor.",
              search_draw(trace_ex, ex_all, show_pruned=True), search_aux_plot(trace_ex)),
        Frame(f"aproximado: parar tras {budget} hojas",
              f"Aqui esta la idea ANN: renunciar a la garantia. Con un presupuesto de {budget} hojas (deslizador 'presupuesto') "
              f"se examinan {nd_ap} puntos ({nd_ap / n:.0%}) y el recall es {rec_ap:.2f}. Prueba presupuestos distintos y haz clic "
              f"en otras zonas: cerca de una frontera entre celdas cuesta mas acertar.",
              search_draw(trace_ap, len(trace_ap), show_pruned=False), search_aux_plot(trace_ex, budget_mark=len(trace_ap))),
        Frame("la maldicion de la dimensionalidad",
              "El kd-tree es una maravilla en 2D, 3D... pero en dimension alta se rompe. En d dimensiones cada corte solo usa UNA "
              "coordenada: con 4000 puntos y 32 dimensiones solo se llega a cortar unas 8 de las 32 dimensiones. Ademas las "
              "distancias se igualan (ver 'exact'), asi que la circunferencia de busqueda toca casi todas las celdas y la poda deja "
              "de funcionar. Por eso en alta dimension se usan otras ideas.",
              lambda ax: scatter(ax, X), aux_curse),
        Frame("kd-tree aproximado en 32 dimensiones",
              "Con presupuesto de hojas el kd-tree sigue siendo aproximado util, pero su curva en alta dimension es mediocre: hay "
              "que examinar una gran fraccion de los datos para lograr buen recall. Los metodos siguientes (arboles de proyecciones "
              "aleatorias, listas invertidas, cuantizacion y grafos) mejoran este compromiso.",
              lambda ax: scatter(ax, X), aux_budget),
    ]
    return frames


PARAMS = {
    "leaf": Param("puntos por hoja", 2, 30, 10),
    "hojas": Param("presupuesto (hojas)", 1, 20, 3),
}

if __name__ == "__main__":
    cli("2. KD-tree: cortes alineados con los ejes", build, PARAMS)
