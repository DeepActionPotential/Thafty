import os
import faiss
import numpy as np
import pickle
from typing import List, Tuple, Optional
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class FAISSManager:
    """
    Manages a FAISS IndexFlatIP+IDMap for fast cosine‐similarity search on normalized vectors,
    always returning similarity scores in [0,1], and supporting batch adds with path lookup.
    Now supports saving/loading bundled with image metadata as numpy arrays.
    """

    def __init__(self, dimension: int, index_path: str = None):
        """
        Initialize or load a FAISS IndexFlatIP wrapped in an IDMap.

        Args:
            dimension (int): Dimensionality of the vectors.
            index_path (str, optional): If provided and exists, load index + metadata from this file.
        """
        self.dimension      = dimension
        self.index_path     = index_path
        self.id_to_path     = {}
        self._next_id       = 0
        # list of np.ndarray images as metadata
        self.metadata_images: List[np.ndarray] = []

        if index_path and os.path.exists(index_path):
            imgs = self.load(index_path)
            self.metadata_images = imgs or []
        else:
            self.reset()

    def reset(self):
        """Reset to a fresh IndexFlatIP+IDMap and clear metadata."""
        flat = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(flat)
        self.id_to_path = {}
        self._next_id   = 0
        self.metadata_images = []
        print(f"⚙️  Reset: new IndexFlatIP+IDMap (dim={self.dimension})")

    def load(self, path: str = None) -> List[np.ndarray]:
        """
        Load FAISS index (with IDMap) and metadata images from a pickle file.

        Returns:
            List[np.ndarray]: The list of metadata images.
        """
        p = path or self.index_path
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f"No FAISS bundle at '{p}'")
        with open(p, 'rb') as f:
            bundle = pickle.load(f)
        # restore FAISS index
        self.index = faiss.deserialize_index(bundle['faiss_index'])
        self.index_path = p
        # restore state
        self.id_to_path     = bundle.get('id_to_path', {})
        self._next_id       = bundle.get('next_id', self.index.ntotal)
        print(f"✅ Loaded index (ntotal={self.index.ntotal}) and metadata from '{p}'")
        return bundle.get('metadata_images', [])

    def save(self, path: str = None):
        """
        Save the current FAISS index and metadata images to a single pickle file.

        Args:
            path (str, optional): Destination filepath.
        """
        p = path or self.index_path
        if not p:
            raise ValueError("No path specified for saving the FAISS bundle")
        bundle = {
            'faiss_index': faiss.serialize_index(self.index),
            'id_to_path': self.id_to_path,
            'next_id': self._next_id,
            'metadata_images': self.metadata_images,
        }
        with open(p, 'wb') as f:
            pickle.dump(bundle, f)
        self.index_path = p
        print(f"💾 Saved index (ntotal={self.index.ntotal}) and metadata to '{p}'")

    def _reshape(self, array: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(array, dtype="float32")
        if arr.ndim == 1:
            if arr.shape[0] != self.dimension:
                raise ValueError(f"{name} vector has shape {arr.shape}, expected ({self.dimension},)")
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.dimension:
            raise ValueError(f"{name} must have shape (N, {self.dimension}), got {arr.shape}")
        return arr

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def add(self, vectors: np.ndarray, ids: np.ndarray = None):
        """
        Add vectors (and optional custom IDs) to the index.
        Automatically L2-normalizes for cosine similarity.
        """
        vecs = self._reshape(vectors, "Input")
        vecs = self._l2_normalize(vecs)

        if ids is None:
            n = vecs.shape[0]
            ids = np.arange(self._next_id, self._next_id + n, dtype="int64")
            self._next_id += n
        else:
            ids = np.asarray(ids, dtype="int64")
            if ids.ndim == 0:
                ids = ids.reshape(1,)
            if ids.shape[0] != vecs.shape[0]:
                raise ValueError("IDs must match number of vectors")

        self.index.add_with_ids(vecs, ids)
        print(f"➕ Added {vecs.shape[0]} vectors with IDs {ids.tolist()} (ntotal={self.index.ntotal})")

    def search(self, queries: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query the index for nearest neighbors.
        Returns:
          - sims   : cosine similarities in [0,1]
          - ids    : custom IDs
        """
        q = self._reshape(queries, "Query")
        q = self._l2_normalize(q)
        sims, ids = self.index.search(q, k)
        sims = np.clip(sims, 0.0, 1.0)
        sims[ids == -1] = 0.0
        return sims, ids

    def add_images(
        self,
        image_info: List[Tuple[str, str]],
        embedder
    ):
        """
        Batch-embed and index a list of (image_path, _) tuples.
        Records numpy-array images in metadata_images and path mapping.
        """
        paths, _ = zip(*image_info)
        # embed all images
        vecs = [embedder.embed(p) for p in paths]
        vecs = np.stack(vecs, axis=0)

        # assign IDs
        N = vecs.shape[0]
        ids = np.arange(self._next_id, self._next_id + N, dtype="int64")
        self._next_id += N

        # bulk add
        self.add(vecs, ids=ids)

        # record id→path and load images as np.ndarray
        for fid, p in zip(ids, paths):
            self.id_to_path[int(fid)] = p
            img = Image.open(p)
            arr = np.array(img)
            self.metadata_images.append(arr)
        print(f"✅ Bulk-added {N} images, IDs {ids.tolist()} and stored metadata arrays")

        return self.metadata_images

    def search_with_paths(self, query_vec: np.ndarray, k: int = 1) -> List[Tuple[str, float]]:
        """
        Search and return a list of (image_path, similarity) for top-k.
        """
        sims, ids = self.search(query_vec, k=k)
        results = []
        for sim, fid in zip(sims[0], ids[0]):
            if fid == -1:
                continue
            path = self.id_to_path.get(int(fid), "<unknown>")
            results.append((path, float(sim)))
        return results
