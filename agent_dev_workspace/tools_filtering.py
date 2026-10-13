from typing import Dict, Any, List
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Filter terms from the document-term matrix based on variance threshold.

        Args:
            threshold: Minimum variance threshold. Terms with variance <= threshold are flagged for removal.

        Returns:
            Dict containing result_id, total_features, retained_features_count, removed_features_count, and previews sorted by variance.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        matrix = session.feature_matrix

        # Sparse-friendly variance calculation: Var(X) = E[X^2] - (E[X])^2
        if sp.issparse(matrix):
            mean = np.asarray(matrix.mean(axis=0)).ravel()
            mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
            variances = mean_sq - np.square(mean)
        else:
            variances = np.var(matrix, axis=0)

        feature_names = np.array(session.feature_names)

        # Build full_report DataFrame sorted descending by variance
        report_df = pd.DataFrame({
            "feature": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Separate retained vs removed
        retained_df = report_df[report_df["variance"] > threshold]
        removed_df = report_df[report_df["variance"] <= threshold]

        # Top 20 retained by highest variance, bottom 20 removed by lowest variance
        retained_terms_preview = retained_df["feature"].head(20).tolist()
        removed_terms_preview = removed_df["feature"].tail(20).tolist()

        result_id = session.next_result_id("variance_filter")

        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_features": len(feature_names),
            "retained_features_count": len(retained_df),
            "removed_features_count": len(removed_df),
            "retained_terms_preview": retained_terms_preview,
            "removed_terms_preview": removed_terms_preview
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Compute Pearson correlation between term counts and a binary target class indicator.

        Args:
            target_class: The category label to correlate against (1 if category matches target_class, 0 otherwise).
            threshold: Minimum correlation magnitude (|pearson_r| >= threshold) to retain a feature.

        Returns:
            Dict containing result_id, target_class, threshold, feature counts, and top correlated term previews.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "No dataframe with category_name found in session. Please load dataset first."}

        # Create binary target vector y
        labels = session.dataframe["category_name"].values
        y = (labels == target_class).astype(float)

        if np.all(y == 0):
            return {"error": f"Target class '{target_class}' not found in session dataframe labels."}

        matrix = session.feature_matrix
        feature_names = np.array(session.feature_names)
        n_features = matrix.shape[1]

        # Vectorized sparse-friendly Pearson calculation across ALL features
        y_std = np.std(y)
        if y_std == 0:
            pearson_r = np.zeros(n_features)
        else:
            y_centered = y - np.mean(y)
            if sp.issparse(matrix):
                x_mean = np.asarray(matrix.mean(axis=0)).ravel()
                x_mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
                x_std = np.sqrt(np.maximum(0, x_mean_sq - np.square(x_mean)))

                cov = np.asarray(matrix.T.dot(y_centered)).ravel() / len(y)

                with np.errstate(divide='ignore', invalid='ignore'):
                    pearson_r = np.where(x_std > 0, cov / (x_std * y_std), 0.0)
            else:
                dense = np.asarray(matrix)
                x_std = np.std(dense, axis=0)
                cov = np.mean((dense - np.mean(dense, axis=0)) * y_centered[:, None], axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    pearson_r = np.where(x_std > 0, cov / (x_std * y_std), 0.0)

        # Build full_report DataFrame sorted descending by pearson_r (contract requirement)
        report_df = pd.DataFrame({
            "feature": feature_names,
            "pearson_r": pearson_r
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        # Filter by absolute threshold |r| >= threshold
        retained_df = report_df[report_df["pearson_r"].abs() >= threshold]
        removed_df = report_df[report_df["pearson_r"].abs() < threshold]

        result_id = session.next_result_id("pearson_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_features": n_features,
            "retained_features_count": len(retained_df),
            "removed_features_count": len(removed_df),
            "top_positive_correlated_preview": report_df["feature"].head(10).tolist(),
            "top_negative_correlated_preview": report_df["feature"].tail(10).tolist()
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0, max_terms: int = 1500) -> Dict[str, Any]:
        """Compute Spearman rank correlation between term counts and a binary target class indicator.

        Args:
            target_class: The category label to correlate against (1 if category matches target_class, 0 otherwise).
            threshold: Minimum correlation magnitude (|spearman_r| >= threshold) to retain a feature.
            max_terms: Maximum top-variance terms to pre-filter before correlation analysis (default 1500).

        Returns:
            Dict containing result_id, target_class, threshold, feature counts, and top correlated term previews.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found in session. Please run build_dtm_tool first."}

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "No dataframe with category_name found in session. Please load dataset first."}

        # Create binary target vector y
        labels = session.dataframe["category_name"].values
        y = (labels == target_class).astype(float)

        if np.all(y == 0):
            return {"error": f"Target class '{target_class}' not found in session dataframe labels."}

        matrix = session.feature_matrix
        feature_names = np.array(session.feature_names)
        n_features = matrix.shape[1]

        # Pre-filter to top max_terms by variance for Spearman correlation if dataset exceeds max_terms
        is_prefiltered = n_features > max_terms
        if is_prefiltered:
            if sp.issparse(matrix):
                mean = np.asarray(matrix.mean(axis=0)).ravel()
                mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
                vars_arr = mean_sq - np.square(mean)
            else:
                vars_arr = np.var(matrix, axis=0)

            top_indices = np.argsort(vars_arr)[::-1][:max_terms]
            sub_matrix = matrix[:, top_indices]
            sub_feature_names = feature_names[top_indices]
        else:
            sub_matrix = matrix
            sub_feature_names = feature_names

        # Convert target vector y to ranks
        y_ranks = stats.rankdata(y)
        y_rank_std = np.std(y_ranks)

        if y_rank_std == 0:
            spearman_r = np.zeros(len(sub_feature_names))
        else:
            y_rank_centered = y_ranks - np.mean(y_ranks)
            spearman_r = []

            # Column-by-column rank computation on sub_matrix
            for i in range(len(sub_feature_names)):
                if sp.issparse(sub_matrix):
                    x = sub_matrix[:, i].toarray().ravel()
                else:
                    x = np.asarray(sub_matrix)[:, i]

                if np.std(x) == 0:
                    rho = 0.0
                else:
                    x_ranks = stats.rankdata(x)
                    x_rank_std = np.std(x_ranks)
                    if x_rank_std == 0:
                        rho = 0.0
                    else:
                        cov = np.mean((x_ranks - np.mean(x_ranks)) * y_rank_centered)
                        rho = cov / (x_rank_std * y_rank_std)
                spearman_r.append(rho)

            spearman_r = np.array(spearman_r)

        # Build full_report DataFrame sorted descending by spearman_r (contract requirement)
        report_df = pd.DataFrame({
            "feature": sub_feature_names,
            "spearman_r": spearman_r
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        # Filter by absolute threshold |rho| >= threshold
        retained_df = report_df[report_df["spearman_r"].abs() >= threshold]
        removed_df = report_df[report_df["spearman_r"].abs() < threshold]

        result_id = session.next_result_id("spearman_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_features": n_features,
            "analyzed_features_count": len(sub_feature_names),
            "is_prefiltered_to_top_variance": is_prefiltered,
            "max_terms": max_terms,
            "retained_features_count": len(retained_df),
            "removed_features_count": len(removed_df),
            "top_positive_correlated_preview": report_df["feature"].head(10).tolist(),
            "top_negative_correlated_preview": report_df["feature"].tail(10).tolist(),
            "note": f"Evaluated on top {len(sub_feature_names)} terms by variance." if is_prefiltered else f"Evaluated across all {n_features} terms."
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold, "max_terms": max_terms},
            summary=summary,
            full_report=report_df
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
