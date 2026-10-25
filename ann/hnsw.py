"""
6. HNSW (Hierarchical Navigable Small World): "un mapa con autopistas: se baja de lo grosero a lo fino".

    python3 -m ann.hnsw

Hasta ahora los indices PARTIAN el espacio (franjas, celdas, cortes). HNSW no particiona nada: construye un GRAFO donde cada
punto esta unido a unos pocos vecinos cercanos, y busca CAMINANDO por el grafo:

    desde un punto de entrada, ir siempre al vecino que mas se acerque a q; cuando ninguno mejora, se ha llegado.

Dos problemas de ese paseo "voraz" tienen solucion:
  * Un grafo de vecinos mas cercanos queda en fragmentos y el paseo se atasca en un minimo local. Se arregla eligiendo los
    enlaces con una HEURISTICA: no solo los vecinos mas cercanos, sino los que ademas van en direcciones distintas, de modo
    que algunos enlaces cruzan entre grupos.
  * Con muchos puntos el paseo daria miles de pasos. Se arregla con CAPAS, como los mapas con autopistas: la capa 0 tiene
    todos los puntos; cada capa superior tiene una fraccion al azar de la anterior, con enlaces largos. Se empieza arriba,
    se da un paseo grueso, se baja una capa con ese punto y se afina, hasta llegar a la capa 0.
En la capa 0 el paseo no sigue un solo camino sino que mantiene una lista de los `ef` mejores candidatos (busqueda en haz).
"""
from __future__ import annotations

import heapq
from functools import lru_cache

import numpy as np
from matplotlib.collections import LineCollection

from .common import (COLORS, Frame, Param, annotate_box, cli, exact_knn, highd_benchmark, make_blobs, mark_points,
                     recall_at_k, scatter, sqdist)

K_NEIGHBORS = 5
LAYER_COLORS = ["0.55", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#17becf"]


# --------------------------------------------------------------------------------------------------------------------
# El algoritmo
# --------------------------------------------------------------------------------------------------------------------

class HNSW:
    """
    adj[i][l] = lista de vecinos del punto i en la capa l (solo existe para l <= nivel de i).
    Todas las distancias internas son al cuadrado: ordenan igual que las distancias, y ahorran la raiz.
    """

    def __init__(self, X: np.ndarray, M: int = 8, ef_construction: int = 40, seed: int = 0, max_level: int = 5):
        self.X, self.M, self.efC = X, M, ef_construction
        self.Mmax0 = 2 * M                                      # la capa 0 admite el doble de enlaces
        n = len(X)
        rng = np.random.default_rng(seed)
        mL = 1.0 / np.log(max(M, 2))
        # nivel de cada punto: geometrico. P(nivel >= l) = M^-l, asi cada capa tiene ~1/M de los puntos de la anterior
        self.levels = np.minimum((-np.log(1.0 - rng.random(n)) * mL).astype(int), max_level)
        self.adj = [None] * n
        self.entry, self.top = None, -1                         # punto de entrada (el de mas nivel) y su nivel
        for i in rng.permutation(n):
            self.insert(int(i))

    # ---- piezas ---------------------------------------------------------------------------------------------------------
    def _d(self, q: np.ndarray, ids) -> np.ndarray:
        diff = self.X[ids] - q
        return np.einsum("ij,ij->i", diff, diff)

    def search_layer(self, q, eps, ef, layer, trace=None):
        """
        Busqueda en haz en una capa: se mantienen los `ef` mejores puntos vistos (W) y una cola de candidatos por explorar.
        Se saca el candidato mas cercano a q, se miran sus vecinos no vistos y se anaden los que mejoran W. Se para cuando
        el mejor candidato pendiente ya es peor que el peor de W. Con ef=1 es exactamente el paseo voraz.
        Devuelve ([(dist^2, id)] ordenado, numero de puntos vistos = distancias calculadas). `trace` registra cada paso.
        """
        eps = list(eps)
        visited = set(eps)
        d0 = self._d(q, eps)
        cand = list(zip(d0, eps))
        heapq.heapify(cand)
        W = [(-d, e) for d, e in zip(d0, eps)]
        heapq.heapify(W)
        while cand:
            d, c = heapq.heappop(cand)
            if len(W) >= ef and d > -W[0][0]:
                break
            new = [e for e in self.adj[c][layer] if e not in visited]
            visited.update(new)
            if new:
                for e, de in zip(new, self._d(q, new)):
                    if len(W) < ef or de < -W[0][0]:
                        heapq.heappush(cand, (de, e))
                        heapq.heappush(W, (-de, e))
                        if len(W) > ef:
                            heapq.heappop(W)
            if trace is not None:
                trace.append(dict(node=c, new=new, W=[e for _, e in sorted((-nd, e) for nd, e in W)]))
        return sorted((-nd, e) for nd, e in W), len(visited)

    def select(self, W, m: int, heuristic: bool = True):
        """
        Elegir m enlaces entre los candidatos W = [(dist^2 al punto, id)] ordenados. Simple: los m mas cercanos. Heuristica:
        recorrer de cerca a lejos y aceptar un candidato SOLO si esta mas cerca del punto que de cualquiera de los ya
        aceptados; si no, ya hay un enlace que "cubre" esa direccion.
        """
        if not heuristic:
            return [e for _, e in W[:m]]
        chosen = []
        for d, e in W:
            if len(chosen) >= m:
                break
            if not chosen or d < self._d(self.X[e], chosen).min():
                chosen.append(e)
        return chosen

    def insert(self, i: int):
        """Insertar el punto i: buscar su sitio por las capas y enlazarlo bidireccionalmente con M vecinos por capa."""
        x, level = self.X[i], int(self.levels[i])
        self.adj[i] = [[] for _ in range(level + 1)]
        if self.entry is None:
            self.entry, self.top = i, level
            return
        ep = [self.entry]
        for lc in range(self.top, level, -1):                   # capas por encima del nivel de i: solo bajar (paseo voraz)
            ep = [self.search_layer(x, ep, 1, lc)[0][0][1]]
        for lc in range(min(self.top, level), -1, -1):          # de su nivel hacia abajo: buscar y enlazar
            W, _ = self.search_layer(x, ep, self.efC, lc)
            nb = self.select(W, self.M)
            self.adj[i][lc] = nb
            mmax = self.Mmax0 if lc == 0 else self.M
            for e in nb:
                lst = self.adj[e][lc]
                lst.append(i)
                if len(lst) > mmax:                             # el vecino se ha pasado de enlaces: se queda con los mejores
                    self.adj[e][lc] = self.select(sorted(zip(self._d(self.X[e], lst), lst)), mmax)
            ep = [e for _, e in W]
        if level > self.top:
            self.entry, self.top = i, level

    # ---- consulta ---------------------------------------------------------------------------------------------------------
    def search(self, q: np.ndarray, k: int, ef: int, trace: bool = False):
        """
        Bajar por las capas con paseo voraz (ef=1) y buscar en haz con ef en la capa 0. Devuelve un diccionario: ids (los k
        mejores), ndist (distancias calculadas), paths (camino por capa, si trace) y trace0 (pasos de la capa 0).
        """
        ep, ndist, paths = [self.entry], 0, {}
        for lc in range(self.top, 0, -1):
            tr = [] if trace else None
            W, nv = self.search_layer(q, ep, 1, lc, tr)
            ndist += nv
            if trace:
                paths[lc] = [t["node"] for t in tr]
            ep = [W[0][1]]
        tr0 = [] if trace else None
        W, nv = self.search_layer(q, ep, max(ef, k), 0, tr0)
        return dict(ids=np.array([e for _, e in W[:k]], dtype=int), ndist=ndist + nv, paths=paths, trace0=tr0, entry0=ep[0],
                    W=W)

    def plan_insert(self, q: np.ndarray):
        """Como se enlazaria un punto NUEVO q en la capa 0, sin modificar el grafo: candidatos, enlaces simples y heuristicos."""
        ep = [self.entry]
        for lc in range(self.top, 0, -1):
            ep = [self.search_layer(q, ep, 1, lc)[0][0][1]]
        W, _ = self.search_layer(q, ep, self.efC, 0)
        return W, self.select(W, self.M, heuristic=False), self.select(W, self.M, heuristic=True)

    def edges(self, layer: int):
        """Pares (i, j) no repetidos con i-j enlazados en esa capa."""
        out = set()
        for i, a in enumerate(self.adj):
            if len(a) > layer:
                out.update((min(i, j), max(i, j)) for j in a[layer])
        return sorted(out)


# --------------------------------------------------------------------------------------------------------------------
# Medicion
# --------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=8)
def hnsw_index(M: int, ef_construction: int, seed: int) -> HNSW:
    return HNSW(make_blobs(), M, ef_construction, seed)


EFS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)


@lru_cache(maxsize=8)
def ef_curve(M: int, ef_construction: int, seed: int):
    """En 2D: para cada ef, (fraccion de puntos visitados, recall@5) medios sobre 40 consultas al azar."""
    idx = hnsw_index(M, ef_construction, seed)
    X = idx.X
    rng = np.random.default_rng(7)
    Q = X[rng.integers(0, len(X), 40)] + rng.normal(0, 0.03, (40, 2))
    truth = [exact_knn(X, q, K_NEIGHBORS)[0] for q in Q]
    frac, rec = [], []
    for ef in EFS:
        f, r = [], []
        for q, t in zip(Q, truth):
            res = idx.search(q, K_NEIGHBORS, ef)
            f.append(res["ndist"] / len(X))
            r.append(recall_at_k(res["ids"], t))
        frac.append(float(np.mean(f)))
        rec.append(float(np.mean(r)))
    return frac, rec


@lru_cache(maxsize=4)
def hnsw_bench(M: int, ef_construction: int = 40) -> HNSW:
    X, _, _ = highd_benchmark()
    return HNSW(X, M, ef_construction, seed=0)


@lru_cache(maxsize=4)
def hnsw_tradeoff(M: int):
    """Banco de 32 dimensiones: para varios ef, (fraccion de distancias calculadas, recall@10 medio)."""
    X, Q, truth = highd_benchmark()
    idx = hnsw_bench(M)
    frac, rec = [], []
    for ef in (10, 12, 16, 24, 32, 48, 64, 100, 160, 250):
        f, r = [], []
        for q, t in zip(Q, truth):
            res = idx.search(q, 10, ef)
            f.append(res["ndist"] / len(X))
            r.append(recall_at_k(res["ids"], t))
        frac.append(float(np.mean(f)))
        rec.append(float(np.mean(r)))
    return frac, rec


# --------------------------------------------------------------------------------------------------------------------
# El recorrido
# --------------------------------------------------------------------------------------------------------------------

def draw_edges(ax, X, pairs, color="0.7", lw=0.7, alpha=0.8, z=1):
    if pairs:
        ax.add_collection(LineCollection([(X[i], X[j]) for i, j in pairs], colors=color, linewidths=lw, alpha=alpha, zorder=z))


def knn_graph(X: np.ndarray, m: int):
    """El grafo ingenuo: cada punto unido a sus m vecinos mas cercanos (sin direccion)."""
    D = sqdist(X, X)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :m]
    adj = [set() for _ in X]
    for i, row in enumerate(nn):
        for j in row:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    return [sorted(a) for a in adj]


def greedy_walk(X, adj, q, start):
    """Paseo voraz en un grafo plano: ir al vecino mas cercano a q mientras alguno mejore. Devuelve (camino, distancias)."""
    cur = start
    path, dist = [cur], [float(np.linalg.norm(X[cur] - q))]
    while True:
        nb = adj[cur]
        if not nb:
            break
        d = np.linalg.norm(X[nb] - q, axis=1)
        j = int(d.argmin())
        if d[j] >= dist[-1]:
            break
        cur = nb[j]
        path.append(cur)
        dist.append(float(d[j]))
    return path, dist


def build(p, q):
    X = make_blobs()
    n = len(X)
    M, efC, ef, seed = p["M"], p["efC"], p["ef"], p["seed"]
    idx = hnsw_index(M, efC, seed)
    truth, tdist = exact_knn(X, q, K_NEIGHBORS)
    dq = np.sqrt(sqdist(q[None, :], X)[0])
    res = idx.search(q, K_NEIGHBORS, ef, trace=True)
    found = res["ids"]
    rec = recall_at_k(found, truth)
    levels = idx.levels
    layer0 = idx.edges(0)

    # --- paso 1: el grafo ingenuo y su paseo voraz --------------------------------------------------------------------
    A = knn_graph(X, M)
    start = int(np.random.default_rng(seed).integers(n))
    path, pdist = greedy_walk(X, A, q, start)
    knn_pairs = sorted({(min(i, j), max(i, j)) for i, a in enumerate(A) for j in a})
    arrived = path[-1] == int(truth[0])

    def draw_knn(ax):
        draw_edges(ax, X, knn_pairs, color="0.6", lw=0.6, alpha=0.7)
        scatter(ax, X, color="0.55", s=14)
        ax.plot(X[path, 0], X[path, 1], "-o", color="#d62728", lw=2.2, ms=5, zorder=7)
        ax.scatter([X[start, 0]], [X[start, 1]], marker="s", s=120, c="#ff7f0e", edgecolors="black", zorder=8)
        mark_points(ax, X, truth[:1], color="black", s=150)

    def aux_knn(ax):
        ax.plot(range(len(pdist)), pdist, "o-", color="#d62728", label="distancia a q en cada salto")
        ax.axhline(tdist[0], color="black", ls="--", lw=1, label="distancia al vecino verdadero")
        ax.set_xlabel("salto")
        ax.set_ylabel("distancia a q")
        ax.set_title("El paseo voraz baja... hasta que no puede bajar mas", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # --- paso 2: capas -------------------------------------------------------------------------------------------------
    n_layers = int(levels.max()) + 1
    per_layer = [int((levels >= l).sum()) for l in range(n_layers)]

    def draw_levels(ax):
        scatter(ax, X, color="0.85", s=14)
        for l in range(n_layers - 1, 0, -1):
            sel = np.where(levels == l)[0]
            ax.scatter(X[sel, 0], X[sel, 1], s=40 + 55 * l, c=LAYER_COLORS[l], edgecolors="black", linewidths=0.8, zorder=4 + l)

    def aux_levels(ax):
        ax.bar(range(n_layers), per_layer, color=[LAYER_COLORS[l] for l in range(n_layers)])
        for l, c in enumerate(per_layer):
            ax.text(l, c + n * 0.01, str(c), ha="center", fontsize=9)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"capa {l}" for l in range(n_layers)])
        ax.set_ylabel("puntos en la capa")
        ax.set_title(f"Cada punto sube de capa con prob. 1/M = 1/{M}: sin bordes, geometrico", fontsize=10)

    # --- paso 3: insertar = buscar primero -------------------------------------------------------------------------
    W, simple, heur = idx.plan_insert(q)
    W_ids = [e for _, e in W]

    def draw_insert(ax):
        draw_edges(ax, X, layer0, color="0.78", lw=0.5, alpha=0.8)
        scatter(ax, X, color="0.8", s=14)
        ax.scatter(X[W_ids, 0], X[W_ids, 1], s=34, c="#1f77b4", zorder=4)
        mark_points(ax, X, W_ids[: K_NEIGHBORS], color="#1f77b4", s=90)

    def aux_insert(ax):
        annotate_box(ax, ["Insertar el punto nuevo q (la estrella):",
                          "",
                          f"1) bajar las capas {idx.top}..1 con paseo voraz",
                          "   (como en una consulta) hasta la capa 0;",
                          f"2) en la capa 0, busqueda en haz con",
                          f"   ef_construction = {efC}  ->  {len(W)} candidatos",
                          "   (puntos azules: los mas cercanos a q",
                          "    que se encontraron);",
                          f"3) de esos {len(W)}, elegir M = {M} para enlazar.",
                          "",
                          "Insertar es 'consultar y luego enlazar':",
                          "por eso el grafo se construye punto a punto."])

    # --- paso 4: elegir los enlaces ---------------------------------------------------------------------------------
    same = len(set(simple) & set(heur))

    def draw_select(ax):
        draw_edges(ax, X, layer0, color="0.8", lw=0.5, alpha=0.7)
        scatter(ax, X, color="0.85", s=14)
        ax.scatter(X[W_ids, 0], X[W_ids, 1], s=22, c="#aec7e8", zorder=3)
        for e in simple:
            ax.plot([q[0], X[e, 0]], [q[1], X[e, 1]], color="#ff7f0e", lw=2.6, ls="--", zorder=5)
        for e in heur:
            ax.plot([q[0], X[e, 0]], [q[1], X[e, 1]], color="#2ca02c", lw=2.0, zorder=6)
        mark_points(ax, X, simple, color="#ff7f0e", s=130)
        mark_points(ax, X, heur, color="#2ca02c", s=70)

    def aux_select(ax):
        annotate_box(ax, [f"Naranja (rayas): los M = {M} mas cercanos.",
                          f"Verde: la heuristica ({len(heur)} enlaces).",
                          f"Coinciden en {same} de {len(heur)}.",
                          "",
                          "Heuristica: recorrer los candidatos de cerca",
                          "a lejos; aceptar uno solo si esta mas cerca de",
                          "q que de cualquier enlace ya aceptado.",
                          "",
                          "Si los M mas cercanos estan todos en el mismo",
                          "grupo, apuntan en la misma direccion y se",
                          "'tapan' entre si: la heuristica los descarta y",
                          "deja sitio a un enlace hacia otro grupo.",
                          "Sin esos enlaces el grafo se parte en islas."])

    # --- paso 5: las capas del grafo ---------------------------------------------------------------------------------
    def draw_layers(ax):
        draw_edges(ax, X, layer0, color="0.78", lw=0.5, alpha=0.8, z=1)
        for l in range(1, n_layers):
            draw_edges(ax, X, idx.edges(l), color=LAYER_COLORS[l], lw=1.0 + 0.8 * l, alpha=0.9, z=2 + l)
        scatter(ax, X, color="0.85", s=14)
        for l in range(n_layers - 1, 0, -1):
            sel = np.where(levels == l)[0]
            ax.scatter(X[sel, 0], X[sel, 1], s=40 + 55 * l, c=LAYER_COLORS[l], edgecolors="black", linewidths=0.8, zorder=6 + l)

    def aux_layers(ax):
        lines = ["capa   puntos   enlaces/punto (media)", ""]
        for l in range(n_layers - 1, -1, -1):
            deg = [len(a[l]) for a in idx.adj if len(a) > l]
            lines.append(f"  {l}    {len(deg):5d}      {np.mean(deg):4.1f}")
        lines += ["", f"punto de entrada: {idx.entry} (capa {idx.top})", "",
                  "Arriba: pocos puntos y enlaces LARGOS (autopistas).",
                  "Abajo: todos los puntos y enlaces CORTOS (calles).",
                  f"Maximo de enlaces: {2 * M} en la capa 0 y {M} en el resto."]
        annotate_box(ax, lines, fontsize=10)

    # --- paso 6: bajar por las capas --------------------------------------------------------------------------------
    steps = []
    for l in range(idx.top, 0, -1):
        steps += [(l, d) for d in [float(dq[v]) for v in res["paths"][l]]]

    def draw_descent(ax):
        draw_edges(ax, X, layer0, color="0.85", lw=0.4, alpha=0.8)
        scatter(ax, X, color="0.86", s=12)
        for l in range(idx.top, 0, -1):
            pts = res["paths"][l]
            ax.plot(X[pts, 0], X[pts, 1], "-o", color=LAYER_COLORS[l], lw=2.8, ms=7, zorder=7, mec="black", mew=0.6)
        ax.scatter([X[idx.entry, 0]], [X[idx.entry, 1]], marker="s", s=130, c="white", edgecolors="black", linewidths=1.6, zorder=9)
        e0 = res["entry0"]
        ax.scatter([X[e0, 0]], [X[e0, 1]], marker="D", s=110, c="#1f77b4", edgecolors="black", zorder=9)
        mark_points(ax, X, truth, color="black", s=130)

    def aux_descent(ax):
        x0 = 0
        for l in range(idx.top, 0, -1):
            ds = [float(dq[v]) for v in res["paths"][l]]
            ax.plot(range(x0, x0 + len(ds)), ds, "o-", color=LAYER_COLORS[l], label=f"capa {l}")
            x0 += len(ds)
        ax.axhline(tdist[0], color="black", ls="--", lw=1, label="vecino verdadero")
        ax.set_xlabel("salto (concatenando las capas)")
        ax.set_ylabel("distancia a q")
        ax.set_title("Cada capa deja un punto de partida mas cercano para la siguiente", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # --- pasos 7-9: la busqueda en haz en la capa 0 --------------------------------------------------------------------
    tr0 = res["trace0"]
    T = len(tr0)
    e0 = res["entry0"]

    def beam_state(k):
        seen = {e0}
        segs = []
        for t in tr0[:k]:
            seen.update(t["new"])
            segs += [(t["node"], v) for v in t["new"]]
        Wk = tr0[k - 1]["W"] if k else [e0]
        return seen, segs, Wk, [t["node"] for t in tr0[:k]]

    def draw_beam(k):
        def draw(ax):
            seen, segs, Wk, expanded = beam_state(k)
            draw_edges(ax, X, layer0, color="0.87", lw=0.4, alpha=0.8)
            scatter(ax, X, color="0.88", s=12)
            draw_edges(ax, X, segs, color="#1f77b4", lw=1.0, alpha=0.7, z=3)
            sv = sorted(seen)
            ax.scatter(X[sv, 0], X[sv, 1], s=26, c="#9ecae1", zorder=4)
            ax.scatter(X[expanded, 0], X[expanded, 1], s=40, c="#1f77b4", edgecolors="black", linewidths=0.6, zorder=5)
            mark_points(ax, X, Wk, color="#2ca02c", s=100)
            mark_points(ax, X, truth, color="black", s=170)
            ax.scatter([X[e0, 0]], [X[e0, 1]], marker="D", s=110, c="#1f77b4", edgecolors="black", zorder=9)
        return draw

    best = [float(dq[t["W"][0]]) for t in tr0]
    worst = [float(dq[t["W"][-1]]) for t in tr0]

    def aux_beam(k):
        def aux(ax):
            seen, _, Wk, _ = beam_state(k)
            xs = np.arange(1, T + 1)
            ax.plot(xs, best, "-", color="#2ca02c", label="mejor de W")
            ax.plot(xs, worst, "-", color="#d62728", label="peor de W")
            ax.axvline(k, color="black", ls="--", lw=1)
            ax.axhline(tdist[K_NEIGHBORS - 1], color="0.5", ls=":", lw=1, label=f"distancia del vecino {K_NEIGHBORS} verdadero")
            ax.set_xlabel("puntos explorados (expansiones)")
            ax.set_ylabel("distancia a q")
            ax.set_title(f"{len(seen)} puntos vistos de {n}; se para cuando los candidatos ya no mejoran a W", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        return aux

    k1, k2 = max(1, round(T * 0.25)), max(2, round(T * 0.6))

    # --- paso 10: resultado final ---------------------------------------------------------------------------------------
    def draw_final(ax):
        draw_edges(ax, X, layer0, color="0.88", lw=0.4, alpha=0.8)
        seen, _, _, _ = beam_state(T)
        scatter(ax, X, color="0.88", s=12)
        sv = sorted(seen)
        ax.scatter(X[sv, 0], X[sv, 1], s=24, c="#1f77b4", zorder=4)
        mark_points(ax, X, truth, color="black", s=130, label="verdad")
        mark_points(ax, X, found, color="#2ca02c", s=70, label="encontrados")

    def aux_final(ax):
        annotate_box(ax, [f"puntos vistos (distancias calculadas): {res['ndist']} de {n}",
                          f"ahorro: {1 - res['ndist'] / n:.0%}",
                          f"recall@{K_NEIGHBORS} de esta consulta: {rec:.2f}",
                          "",
                          f"capas: {idx.top + 1}   ef = {ef}   M = {M}",
                          "",
                          "Distinto a LSH/IVF: aqui no hay lista de",
                          "candidatos por particion; los candidatos",
                          "salen de CAMINAR. Cuantas mas ramas se",
                          "mantienen abiertas (ef), menos probable es",
                          "quedarse en un minimo local."])

    # --- paso 11: el efecto de ef ---------------------------------------------------------------------------------------
    def aux_ef(ax):
        fr, rc = ef_curve(M, efC, seed)
        ax.plot(EFS, rc, "s-", color="#2ca02c", label="recall@5 (media de 40 consultas)")
        ax.plot(EFS, fr, "o-", color="#1f77b4", label="fraccion de puntos visitados")
        ax.axvline(ef, color="black", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("ef (anchura del haz)")
        ax.set_ylim(0, 1.05)
        ax.set_title("ef es el mando del compromiso: mas ef, mas recall, mas coste", fontsize=10)
        ax.legend(fontsize=9, loc="center right")
        ax.grid(alpha=0.3)

    # --- paso 12: medido ----------------------------------------------------------------------------------------------------
    def aux_tradeoff(ax):
        for m_, col in ((6, "#1f77b4"), (16, "#ff7f0e")):
            fr, rc = hnsw_tradeoff(m_)
            ax.plot(fr, rc, "o-", ms=3, color=col, label=f"M={m_}")
        ax.set_xscale("log")
        ax.set_xlabel("fraccion de puntos visitados (coste, escala logaritmica)")
        ax.set_ylabel("recall@10")
        ax.set_title("4000 puntos, 32 dim.; cada marca = un ef (10 ... 250). Ejes ampliados", fontsize=9)
        ax.set_ylim(0.6, 1.005)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    return [
        Frame("el grafo de vecinos y el paseo voraz",
              f"Primera idea: unir cada punto con sus {M} vecinos mas cercanos (lineas grises). Para buscar, partir de un punto "
              f"cualquiera (cuadrado naranja) e ir saltando siempre al vecino mas cercano a q, hasta que ninguno mejore (camino rojo). "
              + ("En esta ocasion llega al vecino verdadero (aro negro). " if arrived else
                 f"Aqui el paseo se atasca a {pdist[-1]:.2f} de q, y el vecino verdadero esta a {tdist[0]:.2f}: es un minimo local. ")
              + "Un grafo de vecinos mas cercanos suele quedar en islas (una por grupo), y un paseo que empieza en otra isla "
                "no puede salir. Cambia la semilla (otro punto de partida) o mueve q para provocarlo.",
              draw_knn, aux_knn),
        Frame("capas: las autopistas",
              f"Para no depender del punto de partida ni caminar miles de saltos, HNSW usa capas. Cada punto recibe un nivel al "
              f"azar con probabilidad geometrica: nivel >= 1 con prob. 1/M, nivel >= 2 con prob. 1/M^2... (los circulos grandes "
              f"son los de nivel alto). La capa 0 tiene todos los puntos; la capa l tiene los de nivel >= l. Es una skip-list "
              f"en un grafo: arriba hay pocos puntos, asi que sus enlaces son largos.",
              draw_levels, aux_levels),
        Frame("insertar un punto (1): buscar su sitio",
              f"Veamos como se construye el grafo, usando q como punto nuevo. Insertar es consultar: se baja por las capas con "
              f"paseo voraz y en la capa 0 se hace una busqueda en haz con ef_construction = {efC}. Los puntos azules son los "
              f"{len(W)} candidatos que se encontraron; con ellos se decide a quien enlazar. Cambia ef_construction y mira como "
              f"cambia el numero de candidatos: un haz mas ancho encuentra mejores vecinos, pero tarda mas en construir.",
              draw_insert, aux_insert),
        Frame("insertar un punto (2): elegir M enlaces",
              f"De los candidatos hay que elegir M = {M}. Lo simple es quedarse con los mas cercanos (naranja). HNSW usa una "
              f"heuristica (verde): acepta un candidato solo si esta mas cerca de q que de los ya aceptados. Resultado: en vez de "
              f"{M} enlaces apuntando al mismo grupo, hay menos pero en direcciones distintas, incluyendo enlaces hacia otros "
              f"grupos. Esos enlaces son los que impiden que el grafo quede en islas y que el paseo se atasque. "
              f"Mueve q hasta un hueco entre dos grupos para ver la diferencia.",
              draw_select, aux_select),
        Frame("el grafo terminado: capas y enlaces",
              f"Tras insertar los {n} puntos el grafo tiene {n_layers} capas. En gris la capa 0 (calles: todos los puntos, enlaces "
              f"cortos); en color las capas superiores (autopistas: pocos puntos, enlaces largos). Cada punto tiene a lo sumo M "
              f"enlaces por capa (2M en la capa 0). El punto de entrada es el de nivel mas alto. La memoria es lineal: "
              f"~2M enlaces por punto, mucho mas que un indice de celdas, y es el precio de HNSW.",
              draw_layers, aux_layers),
        Frame("consulta, paso 1: bajar por las capas",
              f"Se empieza en el punto de entrada (cuadrado blanco, capa {idx.top}) y se hace paseo voraz: se salta al vecino "
              f"mas cercano a q mientras alguno mejore. Cuando no mejora, se baja a la capa de abajo CON ESE PUNTO como "
              f"nueva entrada. En las capas altas los saltos son largos (cruzan el mapa); en las bajas, cortos y finos. Al llegar "
              f"a la capa 0 (rombo azul) ya estamos cerca de q, sin haber mirado casi ningun punto.",
              draw_descent, aux_descent),
        Frame(f"consulta, paso 2: busqueda en haz en la capa 0 (inicio)",
              f"En la capa 0 no se sigue un solo camino sino ef = {max(ef, K_NEIGHBORS)} candidatos a la vez. Se mantiene W = los ef mejores vistos (aros "
              f"verdes) y se va EXPANDIENDO el mejor candidato aun no explorado: se miran sus vecinos (lineas azules), y los que "
              f"mejoran W entran. Puntos azul palido = vistos (su distancia ya se calculo); azul intenso = expandidos. "
              f"Paso {k1} de {T}.",
              draw_beam(k1), aux_beam(k1)),
        Frame("consulta, paso 2: la busqueda en haz (a mitad)",
              f"Paso {k2} de {T}. W se va llenando de puntos cada vez mas cercanos a q y la exploracion se concentra alrededor "
              f"de la estrella. Las curvas de la derecha muestran lo que ocurre: el mejor de W y el peor de W bajan. "
              f"La busqueda terminara cuando el mejor candidato pendiente sea peor que el peor de W: es decir, cuando explorar "
              f"mas no puede mejorar W.",
              draw_beam(k2), aux_beam(k2)),
        Frame("consulta, paso 2: la busqueda en haz (fin)",
              f"Paso {T} de {T}: ya no quedan candidatos que mejoren W y se detiene. La respuesta son los {K_NEIGHBORS} mejores de W. "
              f"Solo se visitaron los puntos azul palido, un pequeno entorno de q. Compara con los {n} que habria que mirar "
              f"con busqueda exacta.",
              draw_beam(T), aux_beam(T)),
        Frame("resultado de la consulta",
              f"Resultado: se calcularon {res['ndist']} distancias (el {res['ndist'] / n:.0%} de {n}) con recall {rec:.2f}. A diferencia de LSH "
              f"o IVF, no se reordenan candidatos de una particion: la lista de candidatos W ya viene ordenada por la propia "
              f"busqueda. Mueve la consulta con el clic y prueba ef = 1: con ef pequeno la busqueda se parece al paseo voraz del "
              f"primer paso y puede perder vecinos.",
              draw_final, aux_final),
        Frame("el mando del compromiso: ef",
              f"ef regula cuantos candidatos se mantienen abiertos. Con ef pequeno la busqueda es rapida pero puede quedarse en "
              f"un minimo local; con ef grande es mas lenta y casi exacta. La curva promedia 40 consultas en este mismo grafo. "
              f"Es la gran ventaja de HNSW: ef se puede cambiar EN CADA CONSULTA sin reconstruir nada. (M y ef_construction, en "
              f"cambio, se fijan al construir el indice.)",
              draw_final, aux_ef),
        Frame("el compromiso, medido",
              "En 32 dimensiones: cada curva es un M y cada marca un ef. HNSW suele dar el mejor recall por distancia calculada "
              "de todos los metodos de este paquete, y por eso es el estandar actual. Su coste esta en otro sitio: mas memoria "
              "(los enlaces), una construccion lenta y la dificultad de actualizar o borrar puntos sin romper el grafo.",
              draw_final, aux_tradeoff),
    ]


PARAMS = {
    "M": Param("M (enlaces/punto)", 2, 10, 4),
    "efC": Param("ef_construction", 4, 60, 16),
    "ef": Param("ef (busqueda)", 1, 40, 8),
    "seed": Param("semilla", 0, 30, 1),
}

if __name__ == "__main__":
    cli("6. HNSW: grafos navegables jerarquicos", build, PARAMS)
