"""
5. PQ (Product Quantization): "guardar cada vector como unos pocos numeros pequenos, y medir distancias con una tabla".

    python3 -m ann.pq

IVF y LSH reducen CUANTAS distancias se calculan. PQ ataca otro problema: la MEMORIA. Con mil millones de vectores de
128 floats (512 bytes cada uno) los datos no caben en RAM. La idea:

    1. se TROCEA cada vector en m subvectores,
    2. en cada trozo se aprende con k-means un pequeno diccionario de ksub centroides,
    3. cada subvector se sustituye por el INDICE de su centroide mas cercano  ->  el vector entero queda en m bytes.

Los m diccionarios, combinados, representan ksub^m vectores distintos aunque solo se guarden m * ksub. Y la distancia de una
consulta a un vector comprimido se calcula SIN descomprimir: se precalcula una tabla (distancia del trozo de q a cada
centroide) y la distancia aproximada es la suma de m entradas de la tabla (ADC, asymmetric distance computation).

En la ventana el espacio es 2D, asi que m = 2: el "trozo" 1 es la coordenada x y el trozo 2 es la coordenada y. Es un caso
mini, pero la mecanica es exactamente la misma que con 128 dimensiones y m = 16.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .common import (COLORS, Frame, Param, annotate_box, cli, exact_knn, highd_benchmark, kmeans, make_blobs, mark_points,
                     recall_at_k, scatter, sqdist)

K_NEIGHBORS = 5


# --------------------------------------------------------------------------------------------------------------------
# El algoritmo
# --------------------------------------------------------------------------------------------------------------------

class PQ:
    """m diccionarios (uno por trozo del vector) y, por cada punto, m indices: su codigo."""

    def __init__(self, X: np.ndarray, m: int, ksub: int, seed: int = 0, iters: int = 20):
        self.X, self.m, self.ksub = X, m, ksub
        self.slices = np.array_split(np.arange(X.shape[1]), m)                       # que dimensiones tiene cada trozo
        self.books = [kmeans(X[:, s], ksub, iters, seed).centers for s in self.slices]    # m matrices (ksub, dim_trozo)
        self.codes = self.encode(X)

    def encode(self, P: np.ndarray) -> np.ndarray:
        """El codigo de cada fila de P: en cada trozo, el indice del centroide mas cercano. Matriz (len(P), m)."""
        P = np.atleast_2d(P)
        return np.stack([sqdist(P[:, s], B).argmin(1) for s, B in zip(self.slices, self.books)], axis=1)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruccion: pegar los centroides que indica el codigo. Es una aproximacion del vector original."""
        return np.hstack([B[codes[:, j]] for j, B in enumerate(self.books)])

    def table(self, q: np.ndarray) -> np.ndarray:
        """Tabla ADC (m, ksub): la distancia^2 del trozo j de q al centroide c del diccionario j."""
        return np.stack([sqdist(q[None, s], B)[0] for s, B in zip(self.slices, self.books)])

    def adc(self, q: np.ndarray) -> np.ndarray:
        """Distancia^2 aproximada de q a TODOS los puntos: suma de m entradas de la tabla por punto, sin tocar los vectores."""
        T = self.table(q)
        return T[np.arange(self.m)[None, :], self.codes].sum(1)

    def search(self, q: np.ndarray, k: int, rerank: int):
        """
        Ordena todos los puntos por distancia aproximada, se queda con los `rerank` primeros y los reordena con la distancia
        exacta. Devuelve (ids, lista corta).
        """
        short = np.argsort(self.adc(q))[:max(rerank, k)]
        d = sqdist(q[None, :], self.X[short])[0]
        return short[np.argsort(d)[:k]], short


# --------------------------------------------------------------------------------------------------------------------
# Medicion en alta dimension (el compromiso real)
# --------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=4)
def pq_index(m: int, ksub: int = 256):
    X, _, _ = highd_benchmark()
    return PQ(X, m, ksub, seed=0, iters=10)


@lru_cache(maxsize=4)
def pq_tradeoff(m: int, ksub: int = 256):
    """Para varios R: recall@10 medio al reordenar con distancia exacta los R mejores segun la distancia aproximada."""
    X, Q, truth = highd_benchmark()
    pq = pq_index(m, ksub)
    rs = (10, 20, 40, 80, 160, 320)
    rec = []
    for r in rs:
        rec.append(float(np.mean([recall_at_k(pq.search(q, 10, r)[0], t) for q, t in zip(Q, truth)])))
    err = float(((X - pq.decode(pq.codes)) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())      # error relativo a la varianza
    return rs, rec, err


# --------------------------------------------------------------------------------------------------------------------
# El recorrido
# --------------------------------------------------------------------------------------------------------------------

def build(p, q):
    X = make_blobs()
    n = len(X)
    K, R = p["ksub"], p["R"]
    pq = PQ(X, 2, K, seed=p["seed"])
    bx, by = pq.books[0][:, 0], pq.books[1][:, 0]                      # centroides del trozo x y del trozo y
    recon = pq.decode(pq.codes)
    err = np.sqrt(((X - recon) ** 2).sum(1))
    truth, _ = exact_knn(X, q, K_NEIGHBORS)
    c0, c1 = COLORS[0], COLORS[1]

    def grid_lines(ax, alpha=0.7, lw=1.2):
        for v in bx:
            ax.axvline(v, color=c0, lw=lw, alpha=alpha, zorder=1)
        for v in by:
            ax.axhline(v, color=c1, lw=lw, alpha=alpha, zorder=1)

    def all_centers():
        return np.array([[a, b] for a in bx for b in by])

    # --- paso 1: el problema es la memoria ----------------------------------------------------------------------------
    def draw_plain(ax):
        scatter(ax, X, color="0.6", s=20)

    def aux_memory(ax):
        annotate_box(ax, ["Un vector de d floats ocupa 4*d bytes.",
                          "",
                          "  d = 128:    512 bytes",
                          "  1.000 millones de vectores:  512 GB",
                          "",
                          "Con PQ (m = 16 trozos, 1 byte por trozo):",
                          "  16 bytes por vector",
                          "  1.000 millones de vectores:   16 GB",
                          "  -> 32 veces menos memoria",
                          "",
                          f"Aqui (2D): {n} puntos = 2 floats (8 bytes) cada uno;",
                          f"con PQ, 2 indices de {int(np.ceil(np.log2(K)))} bits."])

    # --- paso 2: trocear -----------------------------------------------------------------------------------------------
    def draw_split(ax):
        scatter(ax, X, color="0.75", s=20)
        ax.scatter(X[:, 0], np.full(n, -0.012), marker="|", s=40, c=c0, linewidths=1, zorder=3, clip_on=False)
        ax.scatter(np.full(n, -0.012), X[:, 1], marker="_", s=40, c=c1, linewidths=1, zorder=3, clip_on=False)

    def aux_split(ax):
        ax.hist(X[:, 0], bins=30, range=(0, 1), color=c0, alpha=0.6, label="trozo 1 = coordenada x")
        ax.hist(X[:, 1], bins=30, range=(0, 1), color=c1, alpha=0.6, label="trozo 2 = coordenada y")
        ax.set_xlabel("valor del subvector")
        ax.set_ylabel("cuantos puntos")
        ax.set_title("Cada trozo del vector se estudia por separado", fontsize=10)
        ax.legend(fontsize=9)

    # --- paso 3: un diccionario por trozo -----------------------------------------------------------------------------
    def draw_books(ax):
        grid_lines(ax)
        scatter(ax, X, color="0.7", s=18)

    def aux_books(ax):
        rng = np.random.default_rng(0)
        for row, (col, vals, book, name) in enumerate(((c1, X[:, 1], by, "trozo 2 (y)"), (c0, X[:, 0], bx, "trozo 1 (x)"))):
            ax.scatter(vals, row + rng.uniform(-0.18, 0.18, n), s=8, c=col, alpha=0.4, linewidths=0)
            ax.vlines(book, row - 0.32, row + 0.32, colors="black", lw=2.5)
            ax.scatter(book, np.full(len(book), row), marker="D", s=50, c=col, edgecolors="black", zorder=5)
            for j, v in enumerate(book):
                ax.text(v, row + 0.38, str(j), ha="center", fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["trozo 2 (y)", "trozo 1 (x)"])
        ax.set_ylim(-0.6, 1.7)
        ax.set_xlabel("valor del subvector (los numeros son el indice de cada centroide)")
        ax.set_title(f"Un k-means de {K} centroides por cada trozo", fontsize=10)

    # --- paso 4: codificar -----------------------------------------------------------------------------------------------
    show = np.argsort(sqdist(q[None, :], X)[0])[:3]                    # tres puntos cerca de q, para el ejemplo

    def draw_encode(ax):
        grid_lines(ax, alpha=0.35, lw=1.0)
        scatter(ax, X, color="0.7", s=16)
        for i in range(n):
            ax.plot([X[i, 0], recon[i, 0]], [X[i, 1], recon[i, 1]], color="0.55", lw=0.5, zorder=2)
        ax.scatter(recon[:, 0], recon[:, 1], s=9, c="#d62728", zorder=4)
        mark_points(ax, X, show, color="black", s=120)

    def aux_encode(ax):
        lines = ["Cada punto se sustituye por el cruce de lineas",
                 "(el punto rojo) del par de centroides mas cercano:", ""]
        for i in show:
            ci = pq.codes[i]
            lines.append(f"punto {i}: ({X[i, 0]:.2f}, {X[i, 1]:.2f})")
            lines.append(f"   codigo ({ci[0]}, {ci[1]}) -> ({recon[i, 0]:.2f}, {recon[i, 1]:.2f})")
        lines += ["", f"error medio de reconstruccion: {err.mean():.3f}", f"(el espacio mide 1; ksub = {K})"]
        annotate_box(ax, lines, fontsize=10)

    # --- paso 5: el producto cartesiano -----------------------------------------------------------------------------
    used = {(int(a), int(b)) for a, b in pq.codes}

    def draw_product(ax):
        grid_lines(ax, alpha=0.3, lw=1.0)
        scatter(ax, X, color="0.85", s=14)
        for a in range(K):
            for b in range(K):
                ok = (a, b) in used
                ax.scatter([bx[a]], [by[b]], s=70 if ok else 40, facecolors="#d62728" if ok else "none", edgecolors="#d62728",
                           linewidths=1.5, zorder=5)

    def aux_product(ax):
        annotate_box(ax, [f"Se guardan {2 * K} centroides ({K} + {K}).",
                          f"Pero se pueden representar {K}x{K} = {K * K} puntos:",
                          "los cruces de las lineas (circulos).",
                          "",
                          f"Se usan {len(used)} de {K * K} (rellenos); los",
                          "huecos son combinaciones que ningun punto",
                          "necesito.",
                          "",
                          "Con m = 8 trozos y ksub = 256:",
                          "  se guardan  8 x 256 = 2.048 centroides",
                          "  y se representan 256^8 = 1,8e19 vectores.",
                          "Un diccionario de esa talla seria imposible de",
                          "entrenar ni de guardar; el producto lo evita."])

    # --- paso 6: la consulta: tablas -------------------------------------------------------------------------------
    T = pq.table(q)

    def draw_table(ax):
        grid_lines(ax, alpha=0.5, lw=1.2)
        scatter(ax, X, color="0.82", s=14)
        for j, v in enumerate(bx):
            ax.plot([q[0], v], [q[1], q[1]], color=c0, lw=2, zorder=4)
            ax.text(v, 1.012, f"{j}", color=c0, ha="center", fontsize=9, fontweight="bold", clip_on=False)
        for j, v in enumerate(by):
            ax.plot([q[0], q[0]], [q[1], v], color=c1, lw=2, zorder=4)
            ax.text(1.012, v, f"{j}", color=c1, va="center", fontsize=9, fontweight="bold", clip_on=False)

    def aux_table(ax):
        im = ax.imshow(T, cmap="Blues_r", aspect="auto")
        for r in range(2):
            for c in range(K):
                ax.text(c, r, f"{T[r, c]:.3f}", ha="center", va="center", fontsize=8 if K <= 8 else 6,
                        color="white" if T[r, c] < np.median(T) else "black")
        ax.set_xticks(range(K))
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["trozo 1 (x)", "trozo 2 (y)"])
        ax.set_xlabel("centroide")
        ax.set_title("Tabla ADC: distancia^2 del trozo de q a cada centroide", fontsize=10)

    # --- paso 7: distancia aproximada = sumar entradas de la tabla --------------------------------------------------
    approx = pq.adc(q)
    d_true = sqdist(q[None, :], X)[0]
    top_adc = np.argsort(approx)[:K_NEIGHBORS]
    rec_adc = recall_at_k(top_adc, truth)

    def draw_adc(ax):
        grid_lines(ax, alpha=0.2, lw=1.0)
        sc = ax.scatter(X[:, 0], X[:, 1], c=np.sqrt(approx), cmap="viridis_r", s=22, linewidths=0, zorder=2)
        mark_points(ax, X, truth, color="black", s=130, label="verdad")
        mark_points(ax, X, top_adc, color="#d62728", s=70, label="mejores por ADC")

    def aux_adc(ax):
        ax.scatter(np.sqrt(d_true), np.sqrt(approx), s=10, c="0.6", linewidths=0)
        ax.scatter(np.sqrt(d_true[truth]), np.sqrt(approx[truth]), s=80, facecolors="none", edgecolors="black", linewidths=1.6)
        hi = float(np.sqrt(d_true).max())
        ax.plot([0, hi], [0, hi], "k:", lw=1)
        ax.set_xlabel("distancia exacta a q")
        ax.set_ylabel("distancia aproximada (ADC)")
        ax.set_title("Aproximada vs. exacta (aros: vecinos verdaderos)", fontsize=10)
        ax.grid(alpha=0.3)

    # --- paso 8: reordenar ----------------------------------------------------------------------------------------------
    ids, short = pq.search(q, K_NEIGHBORS, R)
    rec = recall_at_k(ids, truth)

    def draw_rerank(ax):
        scatter(ax, X, color="0.88", s=16)
        scatter(ax, X[short], color="#1f77b4", s=24)
        mark_points(ax, X, truth, color="black", s=130, label="verdad")
        mark_points(ax, X, ids, color="#2ca02c", s=70, label="encontrados")

    def aux_rerank(ax):
        annotate_box(ax, [f"1) ADC sobre los {n} puntos:  {n} x 2 lecturas de tabla",
                          f"   (baratas: ninguna multiplicacion por dimension)",
                          f"   recall@{K_NEIGHBORS} solo con ADC: {rec_adc:.2f}",
                          "",
                          f"2) lista corta: los R = {max(R, K_NEIGHBORS)} mejores segun ADC (azul)",
                          f"   distancias EXACTAS: {len(short)}  (de {n})",
                          f"   recall@{K_NEIGHBORS} tras reordenar: {rec:.2f}",
                          "",
                          "La aproximacion cuesta precision; reordenar un",
                          "poco con los vectores originales la recupera.",
                          "Hace falta tener guardados los originales (en",
                          "disco, por ejemplo): en RAM solo van los codigos."])

    # --- paso 9: el compromiso, medido ---------------------------------------------------------------------------------
    def aux_tradeoff(ax):
        for m_, col in ((4, "#1f77b4"), (8, "#ff7f0e"), (16, "#2ca02c")):
            rs, rc, e = pq_tradeoff(m_)
            ax.plot(rs, rc, "o-", ms=4, color=col, label=f"m={m_}: {m_} bytes/vector, error de reconstruccion {e:.0%} de la varianza")
        ax.set_xscale("log")
        ax.set_xlabel("R: distancias exactas calculadas tras la lista corta")
        ax.set_ylabel("recall@10")
        ax.set_title("4000 puntos, 32 dim. (128 bytes/vector sin comprimir), ksub=256", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    return [
        Frame("el problema: la memoria",
              "LSH e IVF reducen el numero de distancias que se calculan, pero siguen guardando los vectores completos. Cuando "
              "hay mil millones de vectores eso no cabe en RAM. PQ comprime los vectores: no busca menos, busca con "
              "una version aproximada y diminuta de cada vector. La ventana es 2D (m = 2 trozos: x e y), pero lo que se ve "
              "es exactamente lo que ocurre con vectores de 128 dimensiones.",
              draw_plain, aux_memory),
        Frame("trocear el vector",
              "Primer paso: cortar el vector en m trozos de igual tamano. Aqui, un vector de 2 numeros se corta en dos trozos "
              "de 1 numero (la coordenada x y la coordenada y; las marcas de los bordes son cada trozo proyectado). Con 128 "
              "dimensiones y m = 16 serian 16 trozos de 8 dimensiones. A partir de aqui cada trozo vive en su propio "
              "espacio, mucho mas pequeno, donde agrupar es facil.",
              draw_split, aux_split),
        Frame(f"un diccionario de {K} centroides por trozo",
              f"En cada trozo se ejecuta k-means con ksub = {K} centroides. Resultado: m diccionarios independientes (aqui, "
              f"las lineas azules son los {K} centroides del trozo x y las naranjas los {K} del trozo y). Cada centroide tiene un "
              f"indice (0..{K - 1}). El panel derecho muestra lo mismo en 1D: los puntos y los centroides sobre cada eje.",
              draw_books, aux_books),
        Frame("codificar: guardar solo indices",
              f"Para codificar un punto se busca, en cada trozo, el centroide mas cercano y se guarda su INDICE. El punto "
              f"(x, y) pasa a ser un par de enteros (i, j), que ocupan {int(np.ceil(np.log2(K))) * 2} bits en lugar de 64. "
              f"Es una compresion CON perdida: el punto se reconstruye como el cruce de lineas del par elegido (puntos rojos). "
              f"Las rayitas grises unen cada punto con su reconstruccion. Sube ksub: las rayitas se acortan, pero el codigo "
              f"crece.",
              draw_encode, aux_encode),
        Frame("el producto cartesiano de diccionarios",
              f"Aqui esta el truco que da nombre a PQ: se guardan solo {2 * K} centroides, pero combinando uno de cada diccionario "
              f"se obtienen {K * K} reconstrucciones posibles (los circulos). Un k-means normal necesitaria {K * K} centroides "
              f"para esa resolucion. Con muchos trozos la diferencia es astronomica: 8 trozos de 256 centroides guardan 2.048 "
              f"vectores y representan 256^8. Por eso PQ comprime tanto sin entrenar un diccionario enorme.",
              draw_product, aux_product),
        Frame("consulta, paso 1: una tabla por trozo",
              f"Ahora la consulta. No se descomprime ningun punto. Se corta q en trozos igual que los datos y, para cada trozo, "
              f"se calcula la distancia al cuadrado a cada uno de sus {K} centroides (las lineas de colores). Resultado: una "
              f"tabla de 2 x {K} numeros (m x ksub). Calcularla cuesta m*ksub distancias de trozo, y es independiente de "
              f"cuantos puntos haya.",
              draw_table, aux_table),
        Frame("consulta, paso 2: sumar entradas de la tabla",
              f"La distancia aproximada de q a un punto es la suma de m entradas de la tabla: se mira el codigo (i, j) del punto "
              f"y se suma T[1][i] + T[2][j]. Es la distancia a la reconstruccion, pero sin reconstruir nada (ADC: solo q "
              f"esta sin comprimir; por eso 'asimetrica'). El color es esa distancia aproximada. Con ADC solo, los "
              f"mejores {K_NEIGHBORS} tienen recall {rec_adc:.2f}; el error de reconstruccion los confunde. A la derecha: aproximada "
              f"contra exacta.",
              draw_adc, aux_adc),
        Frame(f"reordenar los R = {max(R, K_NEIGHBORS)} mejores con distancia exacta",
              f"Para arreglar la perdida se usa ADC como filtro barato: se toman los R mejores (azules) y solo a ellos se les "
              f"calcula la distancia exacta con los vectores originales. Recall: {rec:.2f} (antes {rec_adc:.2f}). "
              f"Es el mismo esquema 'candidatos + reordenar' de LSH e IVF, pero los candidatos se eligen con la distancia "
              f"comprimida. Mueve R con el deslizador: R = {n} equivale a busqueda exacta.",
              draw_rerank, aux_rerank),
        Frame("el compromiso, medido",
              "En 32 dimensiones: cada curva es un m (numero de trozos = bytes por vector; el original ocupa 128). Menos "
              "trozos comprime mas pero aproxima peor y necesita reordenar una lista corta mas grande para llegar al mismo "
              "recall. En la practica PQ se combina con IVF (IVF-PQ): IVF elige las celdas y dentro de ellas se compara con "
              "ADC sobre los codigos, asi se reducen a la vez la memoria y el numero de distancias.",
              draw_rerank, aux_tradeoff),
    ]


PARAMS = {
    "ksub": Param("ksub (centroides/trozo)", 2, 16, 4),
    "R": Param("R (lista corta)", 5, 100, 20, 5),
    "seed": Param("semilla", 0, 30, 1),
}

if __name__ == "__main__":
    cli("5. PQ: cuantizacion de producto", build, PARAMS)
