"""
1. LSH (Locality-Sensitive Hashing): "si dos puntos estan cerca, probablemente caen en el mismo cajon".

    python3 -m ann.lsh

La idea: una funcion de hash normal reparte los puntos al azar. Una funcion LSH esta disenada para que puntos
cercanos CHOQUEN (caigan en el mismo cajon) con mas probabilidad que puntos lejanos. Asi, en vez de mirar los n puntos,
solo se miran los que comparten cajon con la consulta.

Version aqui: LSH para distancia euclidea con proyecciones aleatorias (la familia "p-estable", E2LSH):

        h(x) = floor( (a . x + b) / w )          a = direccion aleatoria (unitaria),  b = desfase aleatorio en [0, w)

Geometricamente, h corta el espacio en FRANJAS paralelas de ancho w, perpendiculares a la direccion a.
  * k funciones juntas (AND)  -> el cajon es la interseccion de k franjas: celdas pequenas, pocos candidatos.
  * L tablas independientes (OR) -> se unen los candidatos de L particiones distintas: sube la probabilidad de no perder
                                    a un vecino verdadero.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .common import (COLORS, Frame, Param, annotate_box, cli, exact_knn, highd_benchmark, line_through, make_blobs,
                     mark_points, recall_at_k, scatter, sqdist)

K_NEIGHBORS = 5


# --------------------------------------------------------------------------------------------------------------------
# El algoritmo
# --------------------------------------------------------------------------------------------------------------------

class E2LSH:
    """L tablas hash; cada una combina k funciones h(x) = floor((a.x + b)/w). El cajon de x en una tabla es la tupla de k enteros."""

    def __init__(self, X: np.ndarray, n_tables: int, n_hashes: int, w: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        n, d = X.shape
        self.X, self.L, self.k, self.w = X, n_tables, n_hashes, w
        A = rng.normal(size=(n_tables, n_hashes, d))
        self.A = A / np.linalg.norm(A, axis=2, keepdims=True)          # direcciones unitarias
        self.B = rng.uniform(0, w, size=(n_tables, n_hashes))           # desfases
        self.codes = np.stack([self.hash(X, t) for t in range(n_tables)])    # (L, n, k): cajon de cada punto en cada tabla
        self.tables = []
        for t in range(n_tables):                                       # tabla = diccionario  cajon -> lista de puntos
            table = {}
            for i, c in enumerate(self.codes[t]):
                table.setdefault(c.tobytes(), []).append(i)
            self.tables.append(table)

    def hash(self, P: np.ndarray, t: int) -> np.ndarray:
        """Cajones de los puntos P en la tabla t: matriz (len(P), k) de enteros."""
        P = np.atleast_2d(P)
        return np.floor((P @ self.A[t].T + self.B[t]) / self.w).astype(np.int64)

    def bucket(self, q: np.ndarray, t: int) -> list:
        """Los puntos que comparten cajon con q en la tabla t."""
        return self.tables[t].get(self.hash(q, t)[0].tobytes(), [])

    def candidates(self, q: np.ndarray, n_tables: int = None) -> np.ndarray:
        """Union de los cajones de q en las primeras `n_tables` tablas."""
        out = set()
        for t in range(self.L if n_tables is None else n_tables):
            out.update(self.bucket(q, t))
        return np.array(sorted(out), dtype=int)

    def search(self, q: np.ndarray, k: int, n_tables: int = None):
        """Candidatos por colision + ordenacion exacta SOLO entre ellos. Devuelve (ids, n_candidatos)."""
        cand = self.candidates(q, n_tables)
        if len(cand) == 0:
            return cand, 0
        d = sqdist(q[None, :], self.X[cand])[0]
        return cand[np.argsort(d)[:k]], len(cand)


# --------------------------------------------------------------------------------------------------------------------
# Medicion en alta dimension (el compromiso real)
# --------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=8)
def lsh_tradeoff(n_hashes: int, w: float = 2.0, max_tables: int = 24, seed: int = 0):
    """Para L = 1..max_tables: (fraccion de puntos examinados, recall@10 medio) en el banco de 32 dimensiones."""
    X, Q, truth = highd_benchmark()
    lsh = E2LSH(X, max_tables, n_hashes, w, seed)
    cum = [set() for _ in range(len(Q))]
    frac, rec = [], []
    for t in range(max_tables):
        f, r = [], []
        for qi, q in enumerate(Q):
            cum[qi].update(lsh.bucket(q, t))
            cand = np.fromiter(cum[qi], dtype=int)
            f.append(len(cand) / len(X))
            if len(cand):
                found = cand[np.argsort(sqdist(q[None, :], X[cand])[0])[:10]]
                r.append(recall_at_k(found, truth[qi]))
            else:
                r.append(0.0)
        frac.append(float(np.mean(f)))
        rec.append(float(np.mean(r)))
    return frac, rec


# --------------------------------------------------------------------------------------------------------------------
# El recorrido
# --------------------------------------------------------------------------------------------------------------------

def draw_strips(ax, lsh, t, j, color, lw=1.0, alpha=1.0):
    """Las fronteras de las franjas de la funcion j de la tabla t: rectas  a.x + b = m*w."""
    a, b, w = lsh.A[t, j], lsh.B[t, j], lsh.w
    lo = np.floor((np.array([0, 0]) @ a + b) / w) - 2
    for m in range(int(lo), int(lo) + 40):
        off = m * w - b
        if -1.5 < off < 2.5:
            line_through(ax, a, off, color=color, lw=lw, alpha=alpha, zorder=1)


def build(p, q):
    X = make_blobs()
    n = len(X)
    k, L, w = p["k"], p["L"], p["w"]
    lsh = E2LSH(X, L, k, w, seed=p["seed"])
    truth, _ = exact_knn(X, q, K_NEIGHBORS)

    # --- paso 1/2: una sola funcion -----------------------------------------------------------------------------------
    h0 = lsh.hash(X, 0)[:, 0]
    hq0 = lsh.hash(q, 0)[0, 0]
    same1 = np.where(h0 == hq0)[0]

    def draw_one(ax):
        draw_strips(ax, lsh, 0, 0, "0.3")
        scatter(ax, X, color=[COLORS[int(c) % len(COLORS)] for c in h0], s=22)
        mark_points(ax, X, truth, color="black", s=110)

    def aux_projection(ax):
        proj = X @ lsh.A[0, 0] + lsh.B[0, 0]
        rng = np.random.default_rng(0)
        y = rng.uniform(-1, 1, n)
        ax.scatter(proj, y, c=[COLORS[int(c) % len(COLORS)] for c in h0], s=14, linewidths=0)
        ax.scatter(proj[truth], y[truth], s=90, facecolors="none", edgecolors="black", linewidths=1.8)
        qp = q @ lsh.A[0, 0] + lsh.B[0, 0]
        ax.scatter([qp], [0], marker="*", s=300, c="#d62728", edgecolors="black", zorder=5)
        for m in range(int(np.floor(proj.min() / w)), int(np.ceil(proj.max() / w)) + 1):
            ax.axvline(m * w, color="0.3", lw=0.8)
        ax.set_yticks([])
        ax.set_xlabel("proyeccion  a·x + b   (cada franja = un cajon)")
        ax.set_title("Los puntos proyectados sobre la direccion a", fontsize=10)

    def draw_collide1(ax):
        draw_strips(ax, lsh, 0, 0, "0.3")
        scatter(ax, X, color="0.85")
        scatter(ax, X[same1], color="#1f77b4", s=22)
        mark_points(ax, X, truth, color="black", s=110)

    found1 = len(set(truth) & set(same1))

    def aux_collide1(ax):
        annotate_box(ax, [f"puntos en el cajon de q: {len(same1)} de {n}",
                          f"vecinos verdaderos dentro: {found1} de {K_NEIGHBORS}",
                          "",
                          f"fraccion examinada: {len(same1) / n:.0%}"])

    # --- paso 3: k funciones (AND) ------------------------------------------------------------------------------------
    cand_t0 = np.array(lsh.bucket(q, 0), dtype=int)
    found_t0 = len(set(truth) & set(cand_t0))

    def draw_and(ax):
        for j in range(k):
            draw_strips(ax, lsh, 0, j, COLORS[j % 10], lw=0.9, alpha=0.8)
        scatter(ax, X, color="0.85")
        scatter(ax, X[cand_t0], color="#1f77b4", s=24)
        mark_points(ax, X, truth, color="black", s=110)

    def aux_and(ax):
        annotate_box(ax, [f"k = {k} funciones en AND (la tabla 0)",
                          f"puntos en el cajon de q: {len(cand_t0)} de {n}",
                          f"vecinos verdaderos dentro: {found_t0} de {K_NEIGHBORS}",
                          "",
                          "k mayor -> celdas mas pequenas:",
                          "  menos candidatos (mas rapido)",
                          "  pero se escapan mas vecinos"])

    # --- paso 4: L tablas (OR) ----------------------------------------------------------------------------------------
    cand_all = lsh.candidates(q)
    cum_c, cum_f = [], []
    acc = set()
    for t in range(L):
        acc.update(lsh.bucket(q, t))
        cum_c.append(len(acc))
        cum_f.append(len(acc & set(truth)))

    def draw_or(ax):
        for t in range(L):
            for j in range(k):
                draw_strips(ax, lsh, t, j, COLORS[t % 10], lw=0.6, alpha=0.55)
        scatter(ax, X, color="0.85")
        scatter(ax, X[cand_all], color="#1f77b4", s=24)
        mark_points(ax, X, truth, color="black", s=110)

    def aux_or(ax):
        xs = np.arange(1, L + 1)
        ax.plot(xs, np.array(cum_c) / n, "o-", color="#1f77b4", label="fraccion de puntos examinados")
        ax.plot(xs, np.array(cum_f) / K_NEIGHBORS, "s-", color="#2ca02c", label="vecinos verdaderos encontrados")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("numero de tablas usadas")
        ax.xaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(integer=True))
        ax.set_title("Cada tabla nueva anade candidatos... y vecinos", fontsize=10)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)

    # --- paso 5: reordenar con distancia exacta -----------------------------------------------------------------------
    ids, n_cand = lsh.search(q, K_NEIGHBORS)
    rec = recall_at_k(ids, truth)

    def draw_rank(ax):
        scatter(ax, X, color="0.88")
        scatter(ax, X[cand_all], color="#1f77b4", s=22)
        mark_points(ax, X, truth, color="black", s=130, label="verdad")
        mark_points(ax, X, ids, color="#2ca02c", s=70, label="encontrados")

    def aux_rank(ax):
        annotate_box(ax, [f"candidatos por colision: {n_cand}",
                          f"distancias exactas calculadas: {n_cand}  (de {n})",
                          f"ahorro: {1 - n_cand / n:.0%}",
                          "",
                          f"recall@{K_NEIGHBORS} de esta consulta: {rec:.2f}",
                          "",
                          "Observacion: el hash solo PROPONE candidatos;",
                          "la respuesta final se ordena con la distancia",
                          "exacta, asi no hay falsos positivos en lo",
                          "devuelto, solo vecinos perdidos (falsos negativos)."])

    # --- paso 6: el compromiso, medido --------------------------------------------------------------------------------
    def aux_tradeoff(ax):
        for kk, col in zip((4, 6, 8), ("#1f77b4", "#ff7f0e", "#2ca02c")):
            fr, rc = lsh_tradeoff(kk)
            ax.plot(fr, rc, "o-", ms=3, color=col, label=f"k={kk} funciones por tabla")
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="mirar al azar")
        ax.set_xlabel("fraccion de puntos examinados (coste)")
        ax.set_ylabel("recall@10")
        ax.set_title("Medido en 4000 puntos de 32 dimensiones; cada marca = anadir una tabla", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    return [
        Frame("una funcion de hash = franjas",
              "Una funcion LSH elige una direccion aleatoria a (la recta oscura marca los cortes) y corta el espacio en franjas "
              "paralelas de ancho w. Cada franja es un cajon (un color). Los puntos con aspecto cercano suelen caer en la misma "
              "franja, pero las fronteras caen donde caen: dos puntos casi iguales pueden quedar a lados distintos. "
              "Aro negro = vecinos verdaderos de q. El panel derecho muestra lo mismo en 1D.",
              draw_one, aux_projection),
        Frame("buscar = mirar el cajon de q",
              f"Solo examinamos los puntos que caen en la franja de q (azules). Hay {len(same1)} de {n}. Observa dos fallos: "
              f"muchos puntos lejanos tambien estan en la franja (la franja es larga y estrecha), y algun vecino verdadero puede "
              f"haber quedado fuera. Una sola proyeccion es una pista muy debil.",
              draw_collide1, aux_collide1),
        Frame(f"k = {k} funciones a la vez (AND)",
              f"Usamos k funciones y exigimos coincidir en TODAS: el cajon es una celda (interseccion de k franjas). Las celdas "
              f"son pequenas, asi que hay pocos candidatos y casi todos estan cerca de q. A cambio, cuantas mas funciones se "
              f"exigen, mas probable es perder un vecino real que cayo al otro lado de algun corte. Prueba a subir k con el deslizador.",
              draw_and, aux_and),
        Frame(f"L = {L} tablas independientes (OR)",
              f"Para recuperar los vecinos perdidos repetimos el truco con L tablas, cada una con sus k funciones elegidas al azar "
              f"(un color por tabla), y unimos los candidatos. Un vecino se pierde solo si se escapa en TODAS las tablas: si "
              f"en una tabla se encuentra con probabilidad p, en L tablas es 1 - (1-p)^L. Sube L y mira como suben las dos curvas.",
              draw_or, aux_or),
        Frame("reordenar con la distancia exacta",
              f"El hash solo propone candidatos. Despues se calcula la distancia exacta a esos {n_cand} puntos y se devuelven "
              f"los {K_NEIGHBORS} mejores. Resultado: recall {rec:.2f} examinando el {n_cand / n:.0%} de los datos. "
              f"Haz clic en otro lugar del grafico para ver como cambia segun la zona (en las zonas densas hay mas candidatos). "
              f"Mueve la semilla para sortear otras funciones aleatorias.",
              draw_rank, aux_rank),
        Frame("el compromiso, medido",
              "En 2D es dificil notar el compromiso; en 32 dimensiones se mide. Cada curva es un k; cada marca, una tabla mas. "
              "Se ve la regla general: mas coste = mas recall. k grande da cajones muy selectivos (pocos candidatos, pero "
              "necesita muchas tablas para llegar a recall alto); k pequeno da cajones grandes. Los parametros w, k y L se "
              "ajustan por tanteo segun el recall que se necesite. Defecto de LSH: necesita MUCHAS tablas = mucha memoria.",
              draw_rank, aux_tradeoff),
    ]


PARAMS = {
    "k": Param("k (funciones/tabla)", 1, 6, 2),
    "L": Param("L (tablas)", 1, 10, 3),
    "w": Param("w (ancho franja)", 0.05, 0.5, 0.15, 0.01),
    "seed": Param("semilla", 0, 30, 1),
}

if __name__ == "__main__":
    cli("1. LSH: hashing sensible a la localidad", build, PARAMS)
