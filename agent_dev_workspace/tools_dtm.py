from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
    ) -> Dict[str, Any]:
        """Builds a document-term matrix from session.dataframe["text"] using CountVectorizer.

        Args:
            max_features: Maximum number of features (terms) to build into the vocabulary. Defaults to None.
            min_df: Minimum document frequency for terms to be kept (int count or float ratio). Defaults to 1.
            max_df: Maximum document frequency for terms to be kept (int count or float ratio). Defaults to 1.0.
            ngram_range: List of two integers [min_n, max_n] for n-gram bounds. Defaults to [1, 1].

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, n_documents, n_features, non_zero, total_elements, sparsity_pct, and feature_names_sample.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}
        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "DataFrame missing 'text' column."}

        # Convert empty/NaN text values to empty strings so CountVectorizer handles them cleanly
        text_series = session.dataframe["text"].fillna("").astype(str)

        # Convert n-gram range parameter to tuple for scikit-learn
        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
        )

        try:
            dtm = vectorizer.fit_transform(text_series)
        except ValueError as e:
            # e.g., empty vocabulary if min_df/max_df is too strict
            return {"status": "error", "message": str(e)}

        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Sync labels/categories if category column exists
        cat_col = None
        if "category_name" in session.dataframe.columns:
            cat_col = "category_name"
        elif "category" in session.dataframe.columns:
            cat_col = "category"

        if cat_col is not None:
            labels = session.dataframe[cat_col].astype(str).values
            session.set_labels(labels)

        # Sparsity calculation: nnz, total_elements, sparsity_pct
        n_documents, n_features = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = n_documents * n_features
        sparsity_pct = (
            float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0
        )

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "n_features": n_features,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "feature_names_sample": feature_names[:10],
        }

        session.store_result(
            "build_dtm_tool",
            {
                "max_features": max_features,
                "min_df": min_df,
                "max_df": max_df,
                "ngram_range": ngram_range,
            },
            summary,
        )
        return summary

    @tool
    def term_frequency_tool(top_n: Optional[int] = 20) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in session.feature_matrix.

        Args:
            top_n: Optional maximum number of top terms to return in top_terms list. Defaults to 20.

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, n_terms, and top_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix in session. Please run build_dtm_tool first."
            }

        # Sum total occurrence count for each term across all documents (axis 0 sum)
        if issparse(session.feature_matrix):
            counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        else:
            counts = np.asarray(session.feature_matrix).sum(axis=0).flatten()

        feature_names = session.feature_names
        term_freq_dict = {str(term): int(count) for term, count in zip(feature_names, counts)}

        # Create sorted list of [term, frequency] descending
        sorted_pairs = sorted(term_freq_dict.items(), key=lambda x: x[1], reverse=True)
        top_k = top_n if top_n is not None else len(sorted_pairs)
        top_terms = sorted_pairs[:top_k]

        full_df = pd.DataFrame(sorted_pairs, columns=["term", "frequency"])

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "n_terms": len(feature_names),
            "top_terms": top_terms,
        }

        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=full_df,
        )
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a heatmap visualization of a slice of the global document-term matrix.

        Args:
            n_terms: Number of terms (columns) to include from the beginning of session.feature_names. Defaults to 20.
            n_documents: Number of documents (rows) to include from session.feature_matrix. Defaults to 20.

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, n_documents, n_terms, matrix_slice, terms, and document_ids.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix in session. Please run build_dtm_tool first."
            }

        total_docs, total_terms = session.feature_matrix.shape
        actual_docs = min(n_documents, total_docs)
        actual_terms = min(n_terms, total_terms)

        matrix_slice_sparse = session.feature_matrix[:actual_docs, :actual_terms]
        if issparse(matrix_slice_sparse):
            matrix_dense = matrix_slice_sparse.toarray()
        else:
            matrix_dense = np.asarray(matrix_slice_sparse)

        slice_terms = session.feature_names[:actual_terms]
        doc_ids = [f"Doc_{i}" for i in range(actual_docs)]

        # Generate plot and attach to session.pending_figure
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.5), max(4, actual_docs * 0.4)))
        sns.heatmap(
            matrix_dense,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=slice_terms,
            yticklabels=doc_ids,
            ax=ax,
        )
        ax.set_title(f"DTM Slice ({actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "matrix_slice": matrix_dense.tolist(),
            "terms": slice_terms,
            "document_ids": doc_ids,
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
