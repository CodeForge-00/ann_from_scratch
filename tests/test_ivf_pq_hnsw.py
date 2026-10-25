"""
Pruebas de IVF, PQ y HNSW. Ejecutar desde la raiz:  python3 -m unittest discover -s tests -t .
"""
import unittest

import matplotlib

matplotlib.use("Agg")

import numpy as np

from ann import hnsw, ivf, pq
from ann.common import Stepper, exact_knn, kmeans, recall_at_k, sqdist, voronoi_cells


def dataset(n=600, d=8, seed=3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (12, d))
    X = centers[rng.integers(0, 12, n)] + rng.normal(0, 0.3, (n, d))
    Q = centers[rng.integers(0, 12, 30)] + rng.normal(0, 0.3, (30, d))
    return X, Q


def mean_recall(search, X, Q, k=10):
    return float(np.mean([recall_at_k(search(q), exact_knn(X, q, k)[0]) for q in Q]))


class CommonTest(unittest.TestCase):
    def test_kmeans_incremental_init_gives_k_distinct_centers(self):
        X, _ = dataset()
        km = kmeans(X, 10, seed=0)
        self.assertEqual(len(np.unique(km.init, axis=0)), 10)

    def test_voronoi_cells_cover_the_square(self):
        from ann.common import polygon_area
        C = np.random.default_rng(0).uniform(0, 1, (7, 2))
        self.assertAlmostEqual(sum(polygon_area(c) for c in voronoi_cells(C)), 1.0, places=9)


class IVFTest(unittest.TestCase):
    def test_lists_partition_the_points(self):
        X, _ = dataset()
        index = ivf.IVF(X, 12)
        ids = np.sort(np.concatenate(index.lists))
        self.assertTrue(np.array_equal(ids, np.arange(len(X))))

    def test_probing_every_cell_is_exact(self):
        X, Q = dataset()
        index = ivf.IVF(X, 12)
        self.assertEqual(mean_recall(lambda q: index.search(q, 10, 12)[0], X, Q), 1.0)

    def test_more_probes_never_hurt_and_cost_more(self):
        X, Q = dataset()
        index = ivf.IVF(X, 12)
        recalls = [mean_recall(lambda q: index.search(q, 10, p)[0], X, Q) for p in (1, 3, 6)]
        self.assertLessEqual(recalls[0], recalls[1])
        self.assertLessEqual(recalls[1], recalls[2])
        self.assertLess(index.search(Q[0], 10, 1)[1], index.search(Q[0], 10, 6)[1])

    def test_returned_neighbors_are_sorted_by_exact_distance(self):
        X, Q = dataset()
        ids = ivf.IVF(X, 12).search(Q[0], 10, 4)[0]
        d = np.sqrt(sqdist(Q[0][None, :], X[ids])[0])
        self.assertTrue(np.all(np.diff(d) >= 0))


class PQTest(unittest.TestCase):
    def test_decode_error_shrinks_with_more_centroids(self):
        X, _ = dataset()
        errs = []
        for ksub in (4, 16, 64):
            index = pq.PQ(X, 4, ksub)
            errs.append(float(((X - index.decode(index.codes)) ** 2).sum()))
        self.assertGreater(errs[0], errs[1])
        self.assertGreater(errs[1], errs[2])

    def test_adc_equals_distance_to_the_reconstruction(self):
        X, Q = dataset()
        index = pq.PQ(X, 4, 16)
        recon = index.decode(index.codes)
        for q in Q[:5]:
            self.assertTrue(np.allclose(index.adc(q), sqdist(q[None, :], recon)[0]))

    def test_codes_use_one_index_per_slice(self):
        X, _ = dataset()
        index = pq.PQ(X, 4, 16)
        self.assertEqual(index.codes.shape, (len(X), 4))
        self.assertTrue(index.codes.max() < 16)

    def test_reranking_everything_is_exact(self):
        X, Q = dataset()
        index = pq.PQ(X, 4, 16)
        self.assertEqual(mean_recall(lambda q: index.search(q, 10, len(X))[0], X, Q), 1.0)

    def test_reranking_improves_on_adc_alone(self):
        X, Q = dataset()
        index = pq.PQ(X, 2, 8)
        adc_only = mean_recall(lambda q: np.argsort(index.adc(q))[:10], X, Q)
        reranked = mean_recall(lambda q: index.search(q, 10, 80)[0], X, Q)
        self.assertGreater(reranked, adc_only)


class HNSWTest(unittest.TestCase):
    def setUp(self):
        self.X, self.Q = dataset()
        self.index = hnsw.HNSW(self.X, M=8, ef_construction=40, seed=0)

    def test_degree_bounds_and_layer_membership(self):
        idx = self.index
        for i, layers in enumerate(idx.adj):
            self.assertEqual(len(layers), idx.levels[i] + 1)
            for l, nb in enumerate(layers):
                self.assertLessEqual(len(nb), idx.Mmax0 if l == 0 else idx.M)
                self.assertNotIn(i, nb)
                for j in nb:
                    self.assertGreaterEqual(idx.levels[j], l)         # un enlace en la capa l solo une puntos que estan en ella

    def test_entry_point_has_the_top_level(self):
        self.assertEqual(self.index.levels[self.index.entry], self.index.levels.max())
        self.assertEqual(self.index.top, self.index.levels.max())

    def test_every_point_is_reachable_in_layer_0(self):
        seen, stack = {self.index.entry}, [self.index.entry]
        while stack:
            for j in self.index.adj[stack.pop()][0]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        self.assertGreater(len(seen), 0.98 * len(self.X))

    def test_high_ef_gives_high_recall_with_few_distances(self):
        res = [self.index.search(q, 10, 64) for q in self.Q]
        recall = float(np.mean([recall_at_k(r["ids"], exact_knn(self.X, q, 10)[0]) for r, q in zip(res, self.Q)]))
        self.assertGreater(recall, 0.95)
        self.assertLess(np.mean([r["ndist"] for r in res]), len(self.X))

    def test_larger_ef_never_lowers_recall_much(self):
        low = mean_recall(lambda q: self.index.search(q, 10, 10)["ids"], self.X, self.Q)
        high = mean_recall(lambda q: self.index.search(q, 10, 100)["ids"], self.X, self.Q)
        self.assertGreaterEqual(high, low)

    def test_ef_1_is_the_greedy_walk(self):
        q = self.Q[0]
        tr = []
        W, _ = self.index.search_layer(q, [self.index.entry], 1, 0, tr)
        d = [float(np.linalg.norm(self.X[t["node"]] - q)) for t in tr]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(d, d[1:])))
        self.assertEqual(W[0][1], tr[-1]["W"][0])

    def test_heuristic_never_picks_more_than_m_and_keeps_the_closest(self):
        W, simple, heur = self.index.plan_insert(self.Q[0])
        self.assertLessEqual(len(heur), self.index.M)
        self.assertEqual(heur[0], simple[0])


class WindowsTest(unittest.TestCase):
    """Cada ventana se construye y cada paso se dibuja (sin pantalla)."""

    def check(self, module, params):
        st = Stepper("prueba", module.build, params)
        self.assertGreater(len(st.frames), 5)
        for i in range(len(st.frames)):
            st.go(i)
        st.q = np.array([0.2, 0.8])                                    # tambien con otra consulta
        st.rebuild()

    def test_ivf(self):
        self.check(ivf, ivf.PARAMS)

    def test_pq(self):
        self.check(pq, pq.PARAMS)

    def test_hnsw(self):
        self.check(hnsw, hnsw.PARAMS)


if __name__ == "__main__":
    unittest.main()
