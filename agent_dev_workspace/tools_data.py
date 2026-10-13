import os
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def tokenize_tool(lowercase: bool = True) -> Dict[str, Any]:
        """Tokenizes each document's text in session.dataframe into unigrams.

        Args:
            lowercase: Whether to lowercase document text before tokenization. Defaults to True.

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, n_documents, total_tokens, and sample_tokens.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}
        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "DataFrame missing 'text' column."}

        # Download NLTK punkt tokenizer data if needed
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        tokens_per_doc: List[List[str]] = []
        for text in session.dataframe["text"]:
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                tokens_per_doc.append([])
            else:
                processed_text = text.lower() if lowercase else text
                tokens = nltk.word_tokenize(processed_text)
                tokens_per_doc.append(tokens)

        # Store in session
        session.dataframe["unigrams"] = tokens_per_doc
        session.artifacts["unigrams"] = tokens_per_doc

        result_id = session.next_result_id("tokenize")
        # Truncate preview to first 5 tokens of first 3 documents to avoid huge payloads
        sample_tokens = [t[:5] for t in tokens_per_doc[:3]]

        summary = {
            "result_id": result_id,
            "n_documents": len(tokens_per_doc),
            "total_tokens": sum(len(t) for t in tokens_per_doc),
            "sample_tokens": sample_tokens
        }

        session.store_result("tokenize_tool", {"lowercase": lowercase}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: Optional[str] = None) -> Dict[str, Any]:
        """Lists files and folders in the workspace root or a specified subdirectory.

        Args:
            subdirectory: Optional subdirectory path relative to workspace root (e.g., 'newdataset').

        Returns:
            Dict[str, Any]: Dictionary containing result_id, subdirectory, directories list, and files list.
        """
        base_dir = os.getcwd()
        target_dir = os.path.join(base_dir, subdirectory) if subdirectory else base_dir

        if not os.path.exists(target_dir):
            return {"status": "error", "message": f"Directory '{target_dir}' does not exist."}
        if not os.path.isdir(target_dir):
            return {"status": "error", "message": f"Path '{target_dir}' is not a directory."}

        directories = []
        files = []

        for item in sorted(os.listdir(target_dir)):
            # Skip hidden files/folders like .git, .pytest_cache
            if item.startswith("."):
                continue
            item_path = os.path.join(target_dir, item)
            if os.path.isdir(item_path):
                directories.append(item)
            else:
                files.append(item)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory or ".",
            "directories": directories,
            "files": files
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool() -> Dict[str, Any]:
        """Inspects session.dataframe for data quality anomalies including empty documents, single-word documents, and extreme length outliers using IQR.

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, total_documents, empty_count, single_word_count, length_outlier_count, total_flagged, flagged_indices, and flagged_preview.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}
        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "DataFrame missing 'text' column."}

        df = session.dataframe
        total_documents = len(df)

        # 1. Identify empty/missing text rows
        empty_mask = df["text"].apply(lambda x: pd.isna(x) or not isinstance(x, str) or not x.strip())
        empty_indices = df[empty_mask].index.tolist()

        # 2. Analyze non-empty rows for word count and character length
        valid_df = df[~empty_mask].copy()
        valid_df["word_count"] = valid_df["text"].apply(lambda x: len(str(x).split()))
        valid_df["char_len"] = valid_df["text"].apply(lambda x: len(str(x)))

        # Single-word documents
        single_word_mask = valid_df["word_count"] == 1
        single_word_indices = valid_df[single_word_mask].index.tolist()

        # Extreme length outliers using robust IQR bounds (3x IQR)
        length_outlier_indices: List[int] = []
        if len(valid_df) >= 3:
            q1 = valid_df["char_len"].quantile(0.25)
            q3 = valid_df["char_len"].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = max(0.0, q1 - 3.0 * iqr)
                upper_bound = q3 + 3.0 * iqr
                outlier_mask = (valid_df["char_len"] < lower_bound) | (valid_df["char_len"] > upper_bound)
                length_outlier_indices = valid_df[outlier_mask].index.tolist()

        # Combine all flagged indices
        all_flagged_indices = sorted(list(set(empty_indices + single_word_indices + length_outlier_indices)))

        # Build detailed records for full_report (untruncated)
        full_flagged_records = []
        for idx in all_flagged_indices:
            row = df.loc[idx]
            raw_text = row.get("text", "")
            text_str = str(raw_text) if pd.notna(raw_text) else ""

            record = {
                "index": int(idx),
                "is_empty": idx in empty_indices,
                "is_single_word": idx in single_word_indices,
                "is_length_outlier": idx in length_outlier_indices,
                "word_count": len(text_str.split()) if pd.notna(raw_text) else 0,
                "char_length": len(text_str) if pd.notna(raw_text) else 0,
                "full_text": text_str
            }
            if "category_name" in row:
                record["category_name"] = str(row["category_name"])
            elif "category" in row:
                record["category"] = str(row["category"])

            full_flagged_records.append(record)

        full_report = pd.DataFrame(full_flagged_records) if full_flagged_records else pd.DataFrame()

        # Build preview limited to top 5 flagged documents (truncated text)
        flagged_preview = []
        for rec in full_flagged_records[:5]:
            rec_preview = rec.copy()
            full_txt = rec_preview.pop("full_text", "")
            rec_preview["text_preview"] = full_txt[:100] + "..." if len(full_txt) > 100 else full_txt
            flagged_preview.append(rec_preview)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_documents": total_documents,
            "empty_count": len(empty_indices),
            "single_word_count": len(single_word_indices),
            "length_outlier_count": len(length_outlier_indices),
            "total_flagged": len(all_flagged_indices),
            "flagged_indices": all_flagged_indices[:10],
            "flagged_preview": flagged_preview
        }

        session.store_result("inspect_data_tool", {}, summary, full_report=full_report)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks session.dataframe for missing, NaN, empty, or whitespace-only text values.

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, n_documents, missing_count, and missing_indices.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}
        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "DataFrame missing 'text' column."}

        missing_indices: List[int] = []
        for idx, text in enumerate(session.dataframe["text"]):
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                missing_indices.append(idx)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "n_documents": len(session.dataframe),
            "missing_count": len(missing_indices),
            "missing_indices": missing_indices[:10]  # Truncated preview for large outputs
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks session.dataframe for duplicate text documents and optionally drops all copies.

        Args:
            drop: If True, drops ALL copies of duplicate documents (keep=False) and resets dataframe index. Defaults to False.

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, n_documents_before, duplicate_count, total_duplicated_rows, dropped, and n_documents_after.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}
        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "DataFrame missing 'text' column."}

        n_documents_before = len(session.dataframe)

        # Redundant extra copies (excluding original occurrence)
        duplicate_count = int(session.dataframe.duplicated(subset=["text"], keep="first").sum())

        # Total rows involved in duplication (marked when using keep=False)
        total_duplicated_rows = int(session.dataframe.duplicated(subset=["text"], keep=False).sum())

        if drop and total_duplicated_rows > 0:
            session.dataframe.drop_duplicates(subset=["text"], keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)

        n_documents_after = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "n_documents_before": n_documents_before,
            "duplicate_count": duplicate_count,
            "total_duplicated_rows": total_duplicated_rows,
            "dropped": drop,
            "n_documents_after": n_documents_after
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = 42,
        replace: bool = False
    ) -> Dict[str, Any]:
        """Subsamples rows from session.dataframe without modifying the working dataframe.

        Args:
            n: Explicit number of rows to sample. Must be specified if frac is not provided.
            frac: Explicit fraction of total rows to sample (e.g. 0.5 for 50%). Must be specified if n is not provided.
            random_state: Seed for random generator to ensure reproducible sampling. If omitted by user, defaults to 42.
            replace: Whether to sample with replacement. Defaults to False.

        Returns:
            Dict[str, Any]: Summary containing result_id, total_documents, sample_size, sampled_indices, and sample_preview.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}

        if n is None and frac is None:
            return {
                "status": "error",
                "message": "Please explicitly specify either 'n' (number of rows) or 'frac' (fraction of rows) to sample."
            }

        effective_random_state = 42 if random_state is None else random_state

        try:
            sampled_df = session.dataframe.sample(
                n=n,
                frac=frac,
                random_state=effective_random_state,
                replace=replace
            )
        except Exception as e:
            return {"status": "error", "message": str(e)}

        sampled_indices = sampled_df.index.tolist()
        result_id = session.next_result_id("sample_data")

        # Cap preview to top 5 sampled rows and truncate long text values to 100 chars
        sample_preview = sampled_df.head(5).to_dict(orient="records")
        for record in sample_preview:
            if "text" in record and isinstance(record["text"], str) and len(record["text"]) > 100:
                record["text"] = record["text"][:100] + "..."

        summary = {
            "result_id": result_id,
            "total_documents": len(session.dataframe),
            "sample_size": len(sampled_df),
            "sampled_indices": sampled_indices,
            "sample_preview": sample_preview
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "random_state": effective_random_state, "replace": replace},
            summary
        )
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes summary statistics on text length (overall and per category) and builds a box plot.

        Returns:
            Dict[str, Any]: Summary dictionary containing result_id, overall_stats, and per_category_stats.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}
        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "DataFrame missing 'text' column."}

        # Calculate character length for each document
        lengths = session.dataframe["text"].apply(lambda x: len(x) if isinstance(x, str) else 0)
        session.dataframe["text_length"] = lengths

        overall_desc = lengths.describe().to_dict()

        per_category_stats = {}
        category_col = "category_name" if "category_name" in session.dataframe.columns else "category"
        if category_col in session.dataframe.columns:
            grouped = session.dataframe.groupby(category_col)["text_length"]
            for cat, group in grouped:
                per_category_stats[str(cat)] = group.describe().to_dict()

        # Build boxplot
        fig, ax = plt.subplots(figsize=(6, 4))
        if category_col in session.dataframe.columns:
            sns.boxplot(data=session.dataframe, x=category_col, y="text_length", ax=ax)
            ax.set_title("Text Length by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y=session.dataframe["text_length"], ax=ax)
            ax.set_title("Text Length Distribution")
        ax.set_ylabel("Text Length (chars)")
        plt.tight_layout()

        # Store figure in session for visualizer
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_desc,
            "per_category_stats": per_category_stats
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        tokenize_tool,
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool
    ]
