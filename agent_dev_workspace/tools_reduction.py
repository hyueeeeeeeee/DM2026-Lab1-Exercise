"""Tools for dimensionality reduction and label processing."""

from typing import Any, Dict, List, Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory function creating dimensionality reduction and label processing tools bound to session."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Reduces the dimensionality of the document-term matrix and plots the coordinates.

        Performs dimensionality reduction using PCA, t-SNE, or UMAP directly on session.feature_matrix.
        Creates a scatter plot colored by document categories set on session.pending_figure,
        and saves coordinates to session.artifacts.

        Args:
            method: Reduction algorithm to use ("pca", "tsne", or "umap"). Default is "pca".
            n_components: Target number of dimensions. Default is 2.
            perplexity: Perplexity parameter for t-SNE. Default is 30.0.
            n_neighbors: Number of neighbors for UMAP. Default is 15.
            random_state: Seed for reproducible dimensionality reduction. Default is 42.

        Returns:
            Dict containing summary of reduction result including result_id, method,
            n_components, coordinates list (preview of up to 20 rows), and explained_variance_ratio if PCA.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature_matrix found in session. Run build_dtm_tool first."}

        X = session.feature_matrix

        # Get category labels for coloring
        if session.labels is not None:
            labels = session.labels
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].tolist()
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"].tolist()
        else:
            labels = ["unlabeled"] * X.shape[0]

        method_clean = method.lower()
        explained_variance_ratio = None

        if method_clean == "pca":
            X_dense = X.toarray() if issparse(X) else np.asarray(X)
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X_dense)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method_clean == "tsne":
            X_dense = X.toarray() if issparse(X) else np.asarray(X)
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X_dense)
        elif method_clean == "umap":
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
        else:
            return {"status": "error", "message": f"Unknown method: {method}. Choose from 'pca', 'tsne', or 'umap'."}

        # Save full coordinates array in artifacts
        artifact_key = f"reduced_{method_clean}"
        session.artifacts[artifact_key] = coords

        # Generate scatter plot as required
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=labels,
            style=labels,
            s=100,
            ax=ax,
        )
        ax.set_title(f"Dimensionality Reduction ({method_clean.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.tight_layout()

        # Set plot on session.pending_figure per pipeline requirements
        session.pending_figure = fig

        result_id = session.next_result_id(f"reduce_{method_clean}")

        # Limit summary coordinates preview to at most 20 rows to keep summary concise
        coords_preview = coords[:20].tolist()

        summary = {
            "result_id": result_id,
            "method": method_clean,
            "n_components": n_components,
            "n_documents": int(coords.shape[0]),
            "coordinates": coords_preview,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result("reduce_dimensions_tool", {"method": method, "n_components": n_components}, summary)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes document category labels into binary indicator vectors.

        Reads category labels from session.dataframe or session.labels, converts them into a 2D
        one-hot encoded binary matrix (e.g. catA -> [1, 0], catB -> [0, 1]), stores the result in
        session.artifacts["binarized_labels"], and returns a summary.

        Returns:
            Dict containing result_id, list of categories (column names), n_samples, matrix shape,
            and a preview of up to 20 row encodings.
        """
        # Retrieve target labels from session
        labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"]
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"]
        elif session.labels is not None:
            labels = pd.Series(session.labels)

        if labels is None:
            return {"status": "error", "message": "No category labels found in session. Load a dataset first."}

        # Perform one-hot encoding using pandas.get_dummies
        encoded_df = pd.get_dummies(labels, dtype=int)
        categories = list(encoded_df.columns)
        encoded_matrix = encoded_df.values.tolist()

        # Save to session artifacts for downstream tasks
        session.artifacts["binarized_labels"] = encoded_matrix
        session.artifacts["binarized_label_names"] = categories

        result_id = session.next_result_id("binarize_labels")

        # Preview up to 20 rows to keep summary small for LLM
        sample_encodings = encoded_matrix[:20]

        summary = {
            "result_id": result_id,
            "categories": [str(c) for c in categories],
            "n_samples": len(encoded_matrix),
            "shape": [len(encoded_matrix), len(categories)],
            "sample_encodings": sample_encodings,
        }

        session.store_result("binarize_labels_tool", {}, summary)

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
