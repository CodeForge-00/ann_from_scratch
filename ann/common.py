"""
Piezas comunes a todos los ejemplos:

  * datos: nubes de puntos en 2D (para ver) y un banco de pruebas en 32D (para medir),
  * la busqueda exacta (la "verdad" contra la que se mide cada algoritmo) y el recall,
  * k-means escrito a mano (lo usan IVF y PQ),
  * `Stepper`: la ventana interactiva paso a paso que comparten todos los ejemplos.

Nada de aqui es un algoritmo ANN; son los cimientos.
"""
from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------------------------------------

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


def make_blobs(n: int = 400, n_clusters: int = 7, spread: float = 0.06, noise: float = 0.08, seed: int = 0) -> np.ndarray:
    """Nube 2D en [0,1]^2: n_clusters grupos gaussianos y un poco de ruido uniforme. Es lo que se dibuja en pantalla."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.12, 0.88, size=(n_clusters, 2))
    n_noise = int(n * noise)
    labels = rng.integers(0, n_clusters, size=n - n_noise)
    pts = centers[labels] + rng.normal(0, spread, size=(n - n_noise, 2))
    pts = np.vstack([pts, rng.uniform(0, 1, size=(n_noise, 2))])
    return np.clip(pts, 0.0, 1.0)


@lru_cache(maxsize=4)
def highd_benchmark(n: int = 4000, d: int = 32, n_queries: int = 100, k: int = 10, seed: int = 1):
    """
    Banco de pruebas en alta dimension: mezcla de 30 gaussianas en R^d. Devuelve (X, Q, truth) donde truth[i] son los
    k vecinos exactos de Q[i]. Aqui es donde se MIDE el compromiso entre velocidad y precision: en 2D todo es facil.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, size=(30, d))
    X = centers[rng.integers(0, 30, size=n)] + rng.normal(0, 0.4, size=(n, d))
    Q = centers[rng.integers(0, 30, size=n_queries)] + rng.normal(0, 0.4, size=(n_queries, d))
    truth = np.array([exact_knn(X, q, k)[0] for q in Q])
    return X, Q, truth


# --------------------------------------------------------------------------------------------------------------------
# Busqueda exacta y recall
# --------------------------------------------------------------------------------------------------------------------

def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matriz de distancias euclideas al cuadrado entre las filas de A y las de B:  |a-b|^2 = |a|^2 - 2 a.b + |b|^2."""
    d = (A * A).sum(1)[:, None] - 2.0 * A @ B.T + (B * B).sum(1)[None, :]
    return np.maximum(d, 0.0)


def exact_knn(X: np.ndarray, q: np.ndarray, k: int):
    """Los k vecinos exactos de q: calcula la distancia a TODOS los puntos y ordena. Devuelve (indices, distancias)."""
    d = np.sqrt(sqdist(q[None, :], X)[0])
    idx = np.argsort(d)[:k]
    return idx, d[idx]


def recall_at_k(found, truth_row, k: Optional[int] = None) -> float:
    """Que fraccion de los k vecinos verdaderos aparece en lo encontrado. 1.0 = respuesta exacta."""
    truth_row = list(truth_row)[: k or len(truth_row)]
    return len(set(list(found)[: len(truth_row)]) & set(truth_row)) / len(truth_row)


# --------------------------------------------------------------------------------------------------------------------
# k-means (Lloyd) a mano
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class KMeansResult:
    centers: np.ndarray
    labels: np.ndarray
    init: np.ndarray                # centroides iniciales
    steps: list                     # por iteracion: (etiquetas tras asignar, centroides tras actualizar)
    inertia: float


def kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> KMeansResult:
    """
    k-means con inicializacion k-means++. Cada iteracion: (1) asignar cada punto a su centroide mas cercano,
    (2) mover cada centroide a la media de sus puntos. Se guarda cada iteracion para poder dibujarla.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(k, n)
    C = X[[rng.integers(n)]].copy()
    d2 = sqdist(X, C)[:, 0]                                # distancia^2 de cada punto a su centro elegido mas cercano
    while len(C) < k:                                      # k-means++: el siguiente centro, lejos de los ya elegidos
        total = d2.sum()
        p = d2 / total if total > 0 else np.full(n, 1.0 / n)
        C = np.vstack([C, X[rng.choice(n, p=p)]])
        d2 = np.minimum(d2, sqdist(X, C[-1:])[:, 0])
    init = C.copy()
    steps = []
    labels = None
    for _ in range(iters):
        new_labels = sqdist(X, C).argmin(1)                # (1) asignar
        newC = C.copy()
        for j in range(k):                                 # (2) actualizar
            members = X[new_labels == j]
            if len(members):
                newC[j] = members.mean(0)
        steps.append((new_labels, newC.copy()))
        converged = labels is not None and np.array_equal(new_labels, labels)
        labels, C = new_labels, newC
        if converged:
            break
    inertia = float(sqdist(X, C).min(1).sum())
    return KMeansResult(C, labels, init, steps, inertia)


# --------------------------------------------------------------------------------------------------------------------
# Dibujo comun
# --------------------------------------------------------------------------------------------------------------------

def scatter(ax, X, color="0.72", s=16, **kw):
    return ax.scatter(X[:, 0], X[:, 1], s=s, c=color, linewidths=0, zorder=2, **kw)


def mark_query(ax, q, label=True):
    ax.scatter([q[0]], [q[1]], marker="*", s=330, c="#d62728", edgecolors="black", linewidths=0.8, zorder=10,
               label="consulta q" if label else None)


def mark_points(ax, X, idx, color="#2ca02c", s=60, label=None, ring=True):
    idx = np.asarray(idx, dtype=int)
    if len(idx):
        ax.scatter(X[idx, 0], X[idx, 1], s=s, facecolors="none" if ring else color, edgecolors=color, linewidths=2.0,
                   zorder=6, label=label)


def line_through(ax, normal, offset, **kw):
    """Dibuja la recta {x : normal . x = offset} (en 2D), recortada a la ventana."""
    n = np.asarray(normal, dtype=float)
    nn = n @ n
    p0 = offset * n / nn
    v = np.array([-n[1], n[0]]) / np.sqrt(nn)
    ax.plot([p0[0] - 4 * v[0], p0[0] + 4 * v[0]], [p0[1] - 4 * v[1], p0[1] + 4 * v[1]], **kw)


def clip_polygon(poly: np.ndarray, normal, offset, keep_positive: bool) -> np.ndarray:
    """Recorta un poligono convexo con el semiplano  normal.x - offset >= 0  (o <= 0). Sutherland-Hodgman para un lado."""
    sign = 1.0 if keep_positive else -1.0
    out = []
    m = len(poly)
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        da, db = sign * (a @ normal - offset), sign * (b @ normal - offset)
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):                         # el segmento a-b cruza la recta: anadir el punto de corte
            t = da / (da - db)
            out.append(a + t * (b - a))
    return np.array(out)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def voronoi_cells(centers: np.ndarray, box: np.ndarray = UNIT_SQUARE) -> List[np.ndarray]:
    """
    Celdas de Voronoi de unos centros, recortadas a `box`: la celda de c_i es lo que queda tras quedarse, para cada otro
    centro c_j, con el lado de la mediatriz de c_i y c_j donde c_i esta mas cerca.
    """
    cells = []
    for i, ci in enumerate(centers):
        poly = box
        for j, cj in enumerate(centers):
            if j == i or len(poly) < 3:
                continue
            normal = cj - ci
            poly = clip_polygon(poly, normal, (cj @ cj - ci @ ci) / 2.0, keep_positive=False)
        cells.append(poly)
    return cells


# --------------------------------------------------------------------------------------------------------------------
# La ventana paso a paso
# --------------------------------------------------------------------------------------------------------------------

@dataclass
class Frame:
    """Un paso del recorrido: un titulo, una explicacion y como dibujar el panel principal y el auxiliar."""
    title: str
    text: str
    draw: Optional[Callable] = None            # draw(ax): panel principal (la nube 2D)
    aux: Optional[Callable] = None             # aux(ax): panel de la derecha (graficas, tablas, numeros)


@dataclass
class Param:
    label: str
    lo: float
    hi: float
    init: float
    step: float = 1.0


class Stepper:
    """
    Ventana con dos paneles, una explicacion, botones Anterior/Siguiente y deslizadores.

      * flechas izquierda/derecha (o espacio): paso anterior / siguiente
      * clic en el panel de la izquierda: mueve la consulta q y se recalcula todo
      * deslizadores: parametros del algoritmo; al moverlos se recalcula todo

    `build(params, q)` devuelve la lista de `Frame`. Se vuelve a llamar cada vez que cambia algo, y la ventana se queda
    en el mismo paso.
    """

    def __init__(self, name: str, build: Callable, params: Optional[Dict[str, Param]] = None, q0=(0.62, 0.45)):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        for key in ("keymap.back", "keymap.forward", "keymap.save", "keymap.pan", "keymap.zoom", "keymap.fullscreen"):
            plt.rcParams[key] = []                         # que las flechas y el espacio sean solo nuestros
        self.plt = plt
        self.name, self.build = name, build
        self.params = params or {}
        self.q = np.array(q0, dtype=float)
        self.frames: List[Frame] = []
        self.i = 0

        self.fig = plt.figure(figsize=(14.5, 8.6))
        self.fig.suptitle(name, fontsize=14, fontweight="bold", x=0.05, ha="left", y=0.995)
        self.ax = self.fig.add_axes([0.05, 0.36, 0.45, 0.55])
        self.aux = self.fig.add_axes([0.58, 0.36, 0.39, 0.55])
        self.caption = self.fig.text(0.05, 0.305, "", va="top", ha="left", fontsize=11, linespacing=1.35)
        self.status = self.fig.text(0.05, 0.075, "", va="center", ha="left", fontsize=10, color="0.35")
        self.fig.text(0.05, 0.012, "flechas ←/→ o espacio: pasos     clic en el grafico: mover la consulta (★)     "
                                   "deslizadores: parametros", fontsize=9, color="0.45")

        self.btn_prev = Button(self.fig.add_axes([0.05, 0.095, 0.10, 0.045]), "◀ Anterior")
        self.btn_next = Button(self.fig.add_axes([0.16, 0.095, 0.10, 0.045]), "Siguiente ▶")
        self.btn_first = Button(self.fig.add_axes([0.27, 0.095, 0.07, 0.045]), "Inicio")
        self.btn_prev.on_clicked(lambda _e: self.go(self.i - 1))
        self.btn_next.on_clicked(lambda _e: self.go(self.i + 1))
        self.btn_first.on_clicked(lambda _e: self.go(0))

        self.sliders = {}
        for row, (key, p) in enumerate(self.params.items()):
            sax = self.fig.add_axes([0.76, 0.235 - 0.043 * row, 0.20, 0.026])
            is_int = float(p.step).is_integer()
            s = Slider(sax, p.label, p.lo, p.hi, valinit=p.init, valstep=p.step, valfmt="%d" if is_int else "%.2f")
            s.on_changed(lambda _v: self.rebuild())
            self.sliders[key] = (s, is_int)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.rebuild()

    # ---- estado -----------------------------------------------------------------------------------------------------
    def values(self) -> dict:
        return {k: (int(round(s.val)) if is_int else float(s.val)) for k, (s, is_int) in self.sliders.items()}

    def rebuild(self):
        self.frames = self.build(self.values(), self.q.copy())
        self.i = min(self.i, len(self.frames) - 1)
        self.show()

    def go(self, i: int):
        self.i = max(0, min(len(self.frames) - 1, i))
        self.show()

    def show(self, save_path: Optional[str] = None):
        f = self.frames[self.i]
        ax, aux = self.ax, self.aux
        for a in (ax, aux):
            a.clear()
            a.set_axis_on()
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Paso {self.i + 1} de {len(self.frames)}: {f.title}", fontsize=12, loc="left")
        if f.draw:
            f.draw(ax)
        mark_query(ax, self.q, label=False)
        if f.aux:
            f.aux(aux)
        else:
            aux.axis("off")
        self.caption.set_text(textwrap.fill(f.text, width=104, replace_whitespace=False))
        vals = ", ".join(f"{k}={v:g}" for k, v in self.values().items())
        self.status.set_text(f"Paso {self.i + 1}/{len(self.frames)}     {vals}")
        if save_path:
            self.fig.savefig(save_path, dpi=70)
        self.fig.canvas.draw_idle()

    # ---- eventos ----------------------------------------------------------------------------------------------------
    def _on_key(self, e):
        if e.key in ("right", " ", "enter"):
            self.go(self.i + 1)
        elif e.key in ("left", "backspace"):
            self.go(self.i - 1)
        elif e.key == "home":
            self.go(0)
        elif e.key == "end":
            self.go(len(self.frames) - 1)

    def _on_click(self, e):
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if e.inaxes is self.ax and e.button == 1 and not (toolbar and getattr(toolbar, "mode", "")):
            self.q = np.array([min(max(e.xdata, 0.0), 1.0), min(max(e.ydata, 0.0), 1.0)])
            self.rebuild()

    # ---- salida a ficheros (para comprobar sin pantalla) --------------------------------------------------------------
    def save_all(self, folder: str):
        import os
        os.makedirs(folder, exist_ok=True)
        for i in range(len(self.frames)):
            self.i = i
            self.show(save_path=os.path.join(folder, f"paso_{i + 1:02d}.png"))
        return len(self.frames)


def cli(name: str, build: Callable, params: Optional[Dict[str, Param]] = None, q0=(0.62, 0.45)):
    """Punto de entrada comun: `python3 -m ann.<algoritmo>` abre la ventana; con --save DIR guarda cada paso como PNG."""
    parser = argparse.ArgumentParser(description=name)
    parser.add_argument("--save", metavar="DIR", help="no abrir ventana: guardar cada paso como imagen PNG en DIR")
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    if args.save:
        plt.switch_backend("Agg")
    st = Stepper(name, build, params, q0)
    if args.save:
        print(f"{st.save_all(args.save)} pasos guardados en {args.save}/")
    else:
        plt.show()
    return st


def annotate_box(ax, lines: List[str], loc=(0.03, 0.97), fontsize=11):
    """Texto en el panel auxiliar (cuando el paso necesita numeros en vez de una grafica)."""
    ax.axis("off")
    ax.text(loc[0], loc[1], "\n".join(lines), va="top", ha="left", fontsize=fontsize, family="monospace",
            transform=ax.transAxes, linespacing=1.5)
