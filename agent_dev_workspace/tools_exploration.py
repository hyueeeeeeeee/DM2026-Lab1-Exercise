import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index1: int, doc_index2: int) -> dict:
        """Compute the cosine similarity between two documents in session.feature_matrix.

        Args:
            doc_index1 (int): The 0-based row index of the first document.
            doc_index2 (int): The 0-based row index of the second document.

        Returns:
            dict: Summary containing result_id, doc_index1, doc_index2, and similarity score.
        """
        if session.feature_matrix is None:
            return {"error": "session.feature_matrix is None. Run build_dtm_tool first."}

        n_rows = session.feature_matrix.shape[0]
        if not (0 <= doc_index1 < n_rows) or not (0 <= doc_index2 < n_rows):
            return {"error": f"Document indices must be between 0 and {n_rows - 1}."}

        row1 = session.feature_matrix[doc_index1]
        row2 = session.feature_matrix[doc_index2]

        if issparse(session.feature_matrix):
            norm1 = float(np.sqrt(row1.multiply(row1).sum()))
            norm2 = float(np.sqrt(row2.multiply(row2).sum()))
            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                dot = float(row1.dot(row2.T).toarray()[0, 0])
                sim = dot / (norm1 * norm2)
        else:
            r1 = np.asarray(row1).ravel()
            r2 = np.asarray(row2).ravel()
            norm1 = float(np.linalg.norm(r1))
            norm2 = float(np.linalg.norm(r2))
            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                sim = float(np.dot(r1, r2) / (norm1 * norm2))

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index1": doc_index1,
            "doc_index2": doc_index2,
            "similarity": sim
        }

        session.store_result("cosine_similarity_tool", {"doc_index1": doc_index1, "doc_index2": doc_index2}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> dict:
        """Compute feature-vs-feature correlation matrix over top variance terms and plot a heatmap.

        Args:
            top_n (int, optional): Number of top-variance terms to include. Defaults to 20.

        Returns:
            dict: Summary with result_id, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix is None. Run build_dtm_tool first."}

        X = session.feature_matrix
        
        # Calculate term variances without densifying the whole matrix
        if issparse(X):
            mean = np.array(X.mean(axis=0)).ravel()
            mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
            variances = mean_sq - (mean ** 2)
        else:
            variances = np.var(np.asarray(X), axis=0)
        
        # Select top_n terms by variance
        n_terms = len(session.feature_names)
        actual_k = min(top_n, n_terms)
        top_indices = np.argsort(variances)[::-1][:actual_k]
        
        selected_terms = [session.feature_names[i] for i in top_indices]

        # Slice ONLY the top_k columns before densifying
        sub_matrix = X[:, top_indices]
        sub_dtm = sub_matrix.toarray() if issparse(sub_matrix) else np.asarray(sub_matrix)

        # Compute Pearson correlation matrix across columns (rowvar=False)
        corr_matrix = np.corrcoef(sub_dtm, rowvar=False)
        # Handle 1x1 case cleanly
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[float(corr_matrix)]])

        corr_rounded = np.round(corr_matrix, 4).tolist()

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, xticklabels=selected_terms, yticklabels=selected_terms, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title(f"Feature Correlation Matrix (Top {actual_k} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "terms": selected_terms,
            "correlation_matrix": corr_rounded
        }

        session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
