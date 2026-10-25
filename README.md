# ANN desde cero

Algoritmos de **busqueda aproximada de vecinos mas cercanos** (ANN, *approximate nearest neighbors*) escritos a mano con
NumPy, sin librerias de indices. Cada algoritmo viene con una **ventana interactiva paso a paso**: no se ve solo el
resultado final, se recorre el funcionamiento un paso cada vez, y se puede mover la consulta y cambiar los parametros para
ver como reacciona.

## El problema

Dado un conjunto de `n` puntos y una consulta `q`, encontrar los `k` puntos mas cercanos a `q`. La respuesta exacta
(medir la distancia a los `n` puntos) cuesta `O(n·d)` por consulta, y a escala de millones de vectores es demasiado.
ANN renuncia a la garantia de acertar siempre a cambio de mirar solo una fraccion de los puntos. La vara de medir es el
**recall@k**: que fraccion de los `k` vecinos verdaderos aparece en lo encontrado.

## Los algoritmos

| # | Modulo | Algoritmo | Idea |
|---|---|---|---|
| 0 | `ann.exact` | Busqueda exacta | Distancia a todos los puntos. Punto de partida, maldicion de la dimensionalidad y recall. |
| 1 | `ann.lsh` | LSH (E2LSH) | Proyecciones aleatorias en franjas: los puntos cercanos tienden a caer en el mismo cajon. `k` funciones en AND, `L` tablas en OR. |
| 2 | `ann.kdtree` | KD-tree | Cortes por la mediana alineados con los ejes. Exacto en pocas dimensiones; aproximado parando tras un presupuesto de hojas. |
| 3 | `ann.rptree` | Random Projection Trees (Annoy) | Cortes oblicuos (la mediatriz de dos puntos al azar), un bosque de arboles y una cola de prioridad compartida. |
| 4 | `ann.ivf` | IVF | Celdas aprendidas con k-means y una lista de puntos por celda. Solo se visitan las `nprobe` celdas mas cercanas. |
| 5 | `ann.pq` | PQ (cuantizacion de producto) | Trocea el vector, un diccionario por trozo, y guarda solo indices. Distancias con una tabla (ADC). Reduce la **memoria**. |
| 6 | `ann.hnsw` | HNSW | Grafo de vecinos en capas. Se baja por las capas con un paseo voraz y se busca en haz en la capa 0. |

Por familias: **particionar** el espacio (KD-tree, RP-trees, IVF), **hashing** (LSH), **comprimir** (PQ) y **caminar por un
grafo** (HNSW). Salvo PQ, todos siguen el esquema *proponer candidatos y reordenar con la distancia exacta*.

## Como usarlo

Necesita `numpy` y `matplotlib`.

```bash
cd ann_from_scratch
python3 -m ann.exact     # o lsh, kdtree, rptree, ivf, pq, hnsw
```

Cada comando abre una ventana con dos paneles (la nube 2D y un panel auxiliar con graficas o numeros), una explicacion del
paso actual y deslizadores.

| Control | Efecto |
|---|---|
| flechas ←/→, espacio, botones | paso anterior / siguiente (boton `Inicio`, y teclas `Home` / `End` para saltar) |
| clic en el panel izquierdo | mueve la consulta ★ y se recalcula todo |
| deslizadores | parametros del algoritmo (se recalcula y se queda en el mismo paso) |

Para generar los pasos como imagenes, sin abrir ventana:

```bash
python3 -m ann.hnsw --save salida/hnsw      # paso_01.png, paso_02.png, ...
```

### Parametros de cada ventana

| Modulo | Deslizadores |
|---|---|
| `exact` | `k` |
| `lsh` | `k` (funciones por tabla), `L` (tablas), `w` (ancho de franja), semilla |
| `kdtree` | puntos por hoja, presupuesto de hojas |
| `rptree` | `T` (arboles), puntos por hoja, `search_k`, semilla |
| `ivf` | `nlist` (celdas), `nprobe` (celdas visitadas), semilla |
| `pq` | `ksub` (centroides por trozo), `R` (lista corta a reordenar), semilla |
| `hnsw` | `M` (enlaces por punto), `ef_construction`, `ef` (anchura del haz en la busqueda), semilla |

## Dos escalas

Cada ventana usa dos conjuntos de datos con propositos distintos:

* **2D** (`make_blobs`, 400 puntos): lo que se **ve**. Es donde se dibujan las celdas, los cortes, los grafos y los caminos.
* **32D** (`highd_benchmark`, 4000 puntos): lo que se **mide**. El ultimo paso de cada ventana muestra la curva de
  compromiso coste/recall real. En 2D todo es facil, y es en alta dimension donde se nota la diferencia entre metodos.

El banco de 32D es una mezcla de 30 gaussianas bien separadas, asi que es relativamente facil: varias curvas saturan pronto.

## Estructura

```
ann/
  common.py   datos, busqueda exacta, recall, k-means, voronoi_cells, y la ventana Stepper/Frame/Param/cli
  exact.py    0. busqueda exacta
  lsh.py      1. LSH
  kdtree.py   2. KD-tree
  rptree.py   3. Random Projection Trees / Annoy
  ivf.py      4. IVF
  pq.py       5. PQ
  hnsw.py     6. HNSW
tests/
  test_ivf_pq_hnsw.py
```

Todos los modulos de algoritmo siguen el mismo patron:

1. una **clase del algoritmo** (`E2LSH`, `KDTree`, `RPForest`, `IVF`, `PQ`, `HNSW`) con `search(q, k, ...)`, sin dependencias de dibujo;
2. una funcion `build(params, q)` que devuelve la lista de `Frame` (titulo, explicacion y como dibujar los dos paneles);
3. un diccionario `PARAMS` con los deslizadores y una llamada a `cli(...)` al final.

Para agregar un algoritmo basta con seguir ese patron: la ventana, los botones y el modo `--save` ya los da `common.py`.

## Pruebas

```bash
python3 -m unittest discover -s tests -t .
```

Las pruebas de IVF, PQ y HNSW comprueban, entre otras cosas, que IVF con `nprobe = nlist` es exacto, que la distancia ADC
de PQ coincide con la distancia a la reconstruccion, que HNSW respeta el maximo de enlaces por capa y alcanza buen recall, y
que cada ventana construye y dibuja todos sus pasos.

## Que no es esto

Es didactico, no rapido. Todo esta en NumPy puro y pensado para leerse: no hay SIMD, ni indices en disco, ni
actualizaciones o borrados. Para usar ANN en produccion, usa FAISS, hnswlib o Annoy; aqui se muestra por que funcionan.
