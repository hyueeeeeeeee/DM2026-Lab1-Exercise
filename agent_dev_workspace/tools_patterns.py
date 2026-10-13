import os
import tempfile
import pandas as pd
import numpy as np
from scipy import sparse
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: int = 1,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Mines frequent term patterns for a single category using PAMI pattern mining algorithms.

        Args:
            category_name: The category label string to mine patterns for.
            filtering_method: Vocabulary filtering approach before mining ('variance', 'tfidf', or 'term_frequency').
            algorithm: Mining algorithm to run ('fpgrowth', 'topk', or 'maxfpgrowth'). Defaults to 'fpgrowth'.
            min_sup: Minimum support threshold (integer count). Used for 'fpgrowth' and 'maxfpgrowth'. Defaults to 1.
            k: Number of top patterns to keep when using 'topk' algorithm. Defaults to 10.

        Returns:
            Dict containing the status, parameters, top 20 highest-support mined patterns, and summary metadata.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No DataFrame loaded in session."}

        # Filter dataframe to target category
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        # Filter out empty texts
        texts = cat_df["text"].dropna().tolist()
        texts = [t for t in texts if isinstance(t, str) and t.strip()]

        if not texts:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no valid text rows for this category",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 1: Category-local CountVectorizer with English stop words
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts)  # scipy sparse CSR matrix
        except ValueError:
            # Vocabulary was empty after stop word removal
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived stop word removal",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names = np.array(vectorizer.get_feature_names_out())

        if dtm.shape[1] == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 2: Apply vocabulary filtering method operating directly on sparse matrices
        if filtering_method in ["variance", "term_frequency"]:
            if filtering_method == "variance":
                mean = np.array(dtm.mean(axis=0)).ravel()
                dtm_sq = dtm.copy()
                dtm_sq.data **= 2
                mean_sq = np.array(dtm_sq.mean(axis=0)).ravel()
                scores = mean_sq - (mean ** 2)
            else:
                scores = np.array(dtm.sum(axis=0)).ravel()

            low = np.percentile(scores, 5)
            high = np.percentile(scores, 95)
            keep_mask = (scores > low) & (scores < high)
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(dtm)  # sparse CSR matrix
            mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).ravel()
            p20 = np.percentile(mean_tfidf, 20)
            keep_mask = mean_tfidf >= p20
        else:
            return {"error": f"Unknown filtering_method '{filtering_method}'."}

        kept_features = feature_names[keep_mask]
        kept_indices = np.where(keep_mask)[0]

        if len(kept_features) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 3: Construct DataFrame of kept features and convert with DF2DB matching Master notebook
        kept_dtm = dtm[:, kept_indices].toarray()
        filter_df = pd.DataFrame(kept_dtm, columns=kept_features)

        temp_dir = tempfile.mkdtemp()
        db_file = os.path.join(temp_dir, "transactional.db")

        from PAMI.extras.convert import DF2DB
        obj = DF2DB.DF2DB(filter_df)
        obj.convert2TransactionalDatabase(db_file, '>=', 1)

        # Step 4: Mine patterns using PAMI
        patterns = []
        try:
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                obj_mine = FPGrowth.FPGrowth(db_file, minSup=min_sup)
                obj_mine.startMine()
                df_patterns = obj_mine.getPatternsAsDataFrame()
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                obj_mine = FAE.FAE(db_file, k=k)
                obj_mine.startMine()
                df_patterns = obj_mine.getPatternsAsDataFrame()
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                obj_mine = MaxFPGrowth.MaxFPGrowth(db_file, minSup=min_sup)
                obj_mine.startMine()
                df_patterns = obj_mine.getPatternsAsDataFrame()
            else:
                return {"error": f"Unknown algorithm '{algorithm}'."}

            if df_patterns is not None and not df_patterns.empty:
                col_names = [c.lower() for c in df_patterns.columns]
                df_patterns.columns = col_names
                pat_col = [c for c in df_patterns.columns if "pattern" in c or "itemset" in c][0]
                sup_col = [c for c in df_patterns.columns if "support" in c or "freq" in c][0]

                for _, row in df_patterns.iterrows():
                    patterns.append({
                        "pattern": str(row[pat_col]),
                        "support": int(row[sup_col]),
                    })
        finally:
            if os.path.exists(db_file):
                os.remove(db_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

        # Sort patterns by support descending so top 20 are the highest-support patterns
        patterns.sort(key=lambda x: x["support"], reverse=True)

        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns[:20],  # Top 20 highest-support patterns
            "n_patterns": len(patterns),
        }

        # Full report contains the complete DataFrame sorted by support
        full_report = pd.DataFrame(patterns) if patterns else pd.DataFrame(columns=["pattern", "support"])

        # Store in session state
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns
        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
            full_report=full_report,
        )

        return summary

    return [mine_patterns_tool]
