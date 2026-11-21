"""Sampling strategies that pick tiles by position, image content or features.

Each sampler works on one slide: it gets the patient bag, that slide's rows (in
raster order), how many to keep and the settings, and returns the rows to keep in
raster order (the `Sampler` signature of `sample.py`). A strategy needs one of
three kinds of input (`STRATEGY_INPUT` in pipeline/config.py), which decides where
in the pipeline it can run:

    positions  tile positions only         spatial
    images     the tile images (cache)     yottixel, entropy, texture
    features   tile features                conch_kmeans (after extraction)

Several strategies cluster tiles and keep the tile nearest each cluster centre
(`representatives`).
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans

from stamp.preprocessing.data.bags import Bag
from stamp.preprocessing.pipeline.config import SamplingConfig
from stamp.preprocessing.slides.tile_cache import TileReader

Rows = npt.NDArray[np.int64]
GLCM_LEVELS = 16
GLCM_OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1))
"""(dy, dx) at distance 1: 0, 45, 90 and 135 degrees."""


# --- spatial -----------------------------------------------------------------


def spatial_grid(
    bag: Bag, rows: Rows, n: int, settings: SamplingConfig, rng: np.random.Generator
) -> Rows:
    """Divide the tissue into a grid of square cells (`spatial_cell_tiles` tiles
    per side) and take random tiles from each cell in proportion to the number of
    tiles it holds, so the sample covers the tissue evenly."""
    if n >= len(rows):
        return rows
    cell_um = settings.spatial_cell_tiles * bag.tile_size_um
    cells = np.floor(bag.coords_slide[rows] / cell_um).astype(np.int64)
    _, cell_of = np.unique(cells, axis=0, return_inverse=True)
    cell_of = cell_of.reshape(-1)
    sizes = np.bincount(cell_of)
    quota = proportional_quota(sizes, n)

    chosen = [
        rng.choice(np.flatnonzero(cell_of == c), size=int(k), replace=False)
        for c, k in enumerate(quota)
        if k > 0
    ]
    return np.sort(rows[np.concatenate(chosen)])


def proportional_quota(sizes: npt.NDArray[np.int64], n: int) -> npt.NDArray[np.int64]:
    """Split n among groups in proportion to their sizes (largest remainder).
    Needs n <= sizes.sum(); no group gets more than it has."""
    share = n * sizes / sizes.sum()
    quota = np.floor(share).astype(np.int64)
    remainder = n - int(quota.sum())
    by_fraction = np.argsort(-(share - quota), kind="stable")
    quota[by_fraction[:remainder]] += 1
    return quota


# --- image based -------------------------------------------------------------


def entropy(
    bag: Bag, rows: Rows, n: int, settings: SamplingConfig, rng: np.random.Generator
) -> Rows:
    """The n tiles with the highest grey-level entropy (the most varied tiles)."""
    if n >= len(rows):
        return rows
    scores = np.array(read_tiles(bag, rows, grey_entropy))
    return top_n(rows, scores, n, rng)


def grey_entropy(tile: Image.Image) -> float:
    """Shannon entropy (bits) of the tile's 256-bin grey-level histogram."""
    counts = np.bincount(np.asarray(tile.convert("L")).reshape(-1), minlength=256)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def texture(
    bag: Bag, rows: Rows, n: int, settings: SamplingConfig, rng: np.random.Generator
) -> Rows:
    """Cluster the tiles by GLCM texture statistics and keep the tile nearest each
    cluster centre, so the sample covers the slide's range of textures."""
    if n >= len(rows):
        return rows
    features = np.array(read_tiles(bag, rows, glcm_features))
    spread = features.std(axis=0)
    features = (features - features.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    return rows[representatives(features, n, rng)]


def glcm_features(tile: Image.Image) -> npt.NDArray[np.float64]:
    """Contrast, dissimilarity, homogeneity, energy and correlation of the tile's
    grey-level co-occurrence matrix, averaged over four directions."""
    grey = np.asarray(tile.convert("L").reduce(2))
    levels = (grey.astype(np.int64) * GLCM_LEVELS) // 256
    per_offset = [_glcm_statistics(_glcm(levels, dy, dx)) for dy, dx in GLCM_OFFSETS]
    return np.mean(per_offset, axis=0)


def _glcm(levels: npt.NDArray[np.int64], dy: int, dx: int) -> npt.NDArray[np.float64]:
    """Symmetric, normalised co-occurrence matrix for the pixel offset (dy, dx)."""
    height, width = levels.shape
    first = levels[: height - dy, max(0, -dx) : width - max(0, dx)]
    second = levels[dy:, max(0, dx) : width - max(0, -dx)]
    pairs = np.bincount(
        (first * GLCM_LEVELS + second).reshape(-1), minlength=GLCM_LEVELS**2
    ).reshape(GLCM_LEVELS, GLCM_LEVELS)
    matrix = (pairs + pairs.T).astype(np.float64)
    return matrix / matrix.sum()


def _glcm_statistics(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    i, j = np.indices(p.shape)
    mean_i, mean_j = (p * i).sum(), (p * j).sum()
    std_i = np.sqrt((p * (i - mean_i) ** 2).sum())
    std_j = np.sqrt((p * (j - mean_j) ** 2).sum())
    if std_i * std_j > 0:
        correlation = (p * (i - mean_i) * (j - mean_j)).sum() / (std_i * std_j)
    else:  # a flat tile: every pixel the same grey level
        correlation = 1.0
    return np.array(
        [
            (p * (i - j) ** 2).sum(),
            (p * np.abs(i - j)).sum(),
            (p / (1.0 + (i - j) ** 2)).sum(),
            np.sqrt((p**2).sum()),
            correlation,
        ]
    )


def yottixel(
    bag: Bag, rows: Rows, n: int, settings: SamplingConfig, rng: np.random.Generator
) -> Rows:
    """Yottixel's mosaic: cluster the tiles by RGB histogram (`color_clusters`
    clusters), share the quota among the clusters in proportion to their size, and
    within each cluster cluster the tile positions and keep the tile nearest each
    spatial centre. Keeps both colour variety and spatial coverage."""
    if n >= len(rows):
        return rows
    histograms = np.array(read_tiles(bag, rows, rgb_histogram))
    n_colours = min(settings.color_clusters, len(rows))
    colour_of = _kmeans(histograms, n_colours, rng).labels_
    sizes = np.bincount(colour_of, minlength=n_colours)
    quota = proportional_quota(sizes, n)

    positions = bag.coords_slide[rows]
    chosen = []
    for colour, k in enumerate(quota):
        members = np.flatnonzero(colour_of == colour)
        if k > 0:
            chosen.append(members[representatives(positions[members], int(k), rng)])
    return np.sort(rows[np.concatenate(chosen)])


def rgb_histogram(tile: Image.Image, bins: int = 8) -> npt.NDArray[np.float64]:
    """Normalised per-channel histograms of the tile, concatenated (3 * bins)."""
    pixels = np.asarray(tile.convert("RGB").reduce(2)).reshape(-1, 3).astype(np.int64)
    histogram = np.concatenate(
        [np.bincount(pixels[:, c] * bins // 256, minlength=bins) for c in range(3)]
    )
    return histogram / histogram.sum()


# --- feature based -----------------------------------------------------------


def feature_kmeans(
    bag: Bag, rows: Rows, n: int, settings: SamplingConfig, rng: np.random.Generator
) -> Rows:
    """Cluster the slide's tile embeddings into n clusters and keep the tile
    nearest each centre. Embeddings are L2-normalised first, so the clustering
    follows direction (cosine), as CONCH embeddings are usually compared."""
    if n >= len(rows):
        return rows
    if bag.feats is None:
        raise ValueError("this strategy clusters tile features; the bag has none")
    feats = bag.feats[rows].astype(np.float32)
    feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    return rows[representatives(feats, n, rng, minibatch=True)]


# --- shared ------------------------------------------------------------------


def top_n(
    rows: Rows, scores: npt.NDArray[np.floating], n: int, rng: np.random.Generator
) -> Rows:
    """The n rows with the highest scores; ties are broken at random."""
    shuffled = rng.permutation(len(rows))
    best = shuffled[np.argsort(-scores[shuffled], kind="stable")[:n]]
    return np.sort(rows[best])


def representatives(
    points: npt.NDArray[np.floating],
    n: int,
    rng: np.random.Generator,
    *,
    minibatch: bool = False,
) -> npt.NDArray[np.int64]:
    """Indices of n distinct points: cluster into n groups and take the point
    nearest each centre (sorted). If clusters come out empty (duplicate points),
    the rest are filled with random unchosen points."""
    model = _kmeans(points, n, rng, minibatch=minibatch)
    labels = model.labels_
    distance = np.linalg.norm(points - model.cluster_centers_[labels], axis=1)
    order = np.lexsort((distance, labels))  # by cluster, then by distance
    first_of_cluster = np.unique(labels[order], return_index=True)[1]
    chosen = order[first_of_cluster]
    if len(chosen) < n:
        rest = np.setdiff1d(np.arange(len(points)), chosen)
        chosen = np.concatenate(
            [chosen, rng.choice(rest, size=n - len(chosen), replace=False)]
        )
    return np.sort(chosen)


def _kmeans(
    points: npt.NDArray[np.floating],
    n_clusters: int,
    rng: np.random.Generator,
    *,
    minibatch: bool = False,
) -> KMeans | MiniBatchKMeans:
    """K-means seeded from `rng`. Mini-batch for high-dimensional embeddings,
    where full k-means with thousands of clusters is too slow."""
    seed = int(rng.integers(2**31 - 1))
    if minibatch:
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=1,
            batch_size=4096,
            max_iter=30,
            random_state=seed,
        ).fit(points)
    return KMeans(n_clusters=n_clusters, n_init=1, random_state=seed).fit(points)


def read_tiles[T](bag: Bag, rows: Rows, measure: Callable[[Image.Image], T]) -> list[T]:
    """`measure` applied to the cached image of each row (one slide's rows)."""
    slide = bag.slides[int(bag.slide_idx[rows[0]])]
    if slide.cache_file is None:
        raise ValueError(
            f"{slide.slide_id}: this strategy needs the tile images, but this "
            "slide's features were imported without a tile cache"
        )
    reader = TileReader()
    try:
        return [
            measure(reader.read(slide.cache_file, (x, y)))
            for x, y in bag.coords_slide[rows].tolist()
        ]
    finally:
        reader.close()
