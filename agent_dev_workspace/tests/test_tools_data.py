import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_tokenize_tool_fixture():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate tokenize_tool directly from tools_data
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"lowercase": True})

    assert res["n_documents"] == 8
    assert res["total_tokens"] == 27
    assert res["sample_tokens"] == [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"]
    ]

    tokens = session.dataframe["unigrams"].tolist()

    # Known answer #10 in TEST_FIXTURE.md
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []  # empty row
    ]

    assert tokens == expected_tokens


def test_tokenize_tool_no_lowercase():
    session = SessionState()

    session.dataframe = pd.DataFrame({"text": ["Hello World", "Data Mining"]})

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    # Test lowercase=False preserves original capitalization
    res_no_lower = tokenize_tool.invoke({"lowercase": False})
    assert session.dataframe["unigrams"].tolist() == [["Hello", "World"], ["Data", "Mining"]]

    # Test lowercase=True converts to lower case
    res_lower = tokenize_tool.invoke({"lowercase": True})
    assert session.dataframe["unigrams"].tolist() == [["hello", "world"], ["data", "mining"]]


def test_list_files_tool_fixture():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    # Known answer #11 in TEST_FIXTURE.md
    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_list_files_tool_nonexistent_directory():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "nonexistent_folder_xyz_123"})

    assert res["status"] == "error"
    assert "does not exist" in res["message"]


def test_inspect_data_tool_fixture():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({})

    assert res["total_documents"] == 8
    assert res["empty_count"] == 1
    assert res["single_word_count"] == 0
    assert res["length_outlier_count"] == 0
    assert res["total_flagged"] == 1
    assert res["flagged_indices"] == [7]
    assert len(res["flagged_preview"]) == 1
    assert res["flagged_preview"][0]["is_empty"] is True


def test_inspect_data_tool_custom_cases():
    session = SessionState()

    # Create dataset with varying document lengths (from ~22 to ~165 chars)
    # to yield a positive, realistic IQR
    varying_normal_texts = [
        "short text " * (i + 2) for i in range(15)
    ]
    texts = varying_normal_texts + ["Yes", "", "x " * 2000]  # single-word (15), empty (16), extreme outlier (17)
    session.dataframe = pd.DataFrame({"text": texts, "category_name": ["catA"] * len(texts)})

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({})

    assert res["total_documents"] == 18
    assert res["empty_count"] == 1
    assert res["single_word_count"] == 1
    assert res["length_outlier_count"] == 1
    assert res["total_flagged"] == 3
    assert res["flagged_indices"] == [15, 16, 17]


def test_check_missing_tool_fixture():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})

    # Known Answer #2 in TEST_FIXTURE.md: exactly 1 row has missing text (row 7)
    assert res["n_documents"] == 8
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]

    # Verify result stored in real session.results_store
    assert res["result_id"] in session.results_store


def test_check_missing_tool_edge_cases():
    session = SessionState()

    # Testing NaN, None, empty string, whitespace string, non-string, valid text
    session.dataframe = pd.DataFrame({
        "text": ["valid text", None, "", "   ", pd.NA, 12345, "another valid text"]
    })

    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})

    assert res["n_documents"] == 7
    assert res["missing_count"] == 5
    assert res["missing_indices"] == [1, 2, 3, 4, 5]
    assert res["result_id"] in session.results_store


def test_check_duplicates_tool_fixture():
    # Fresh session for drop=False check
    session1 = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool1 = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool1.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools1 = make_tools(session1)
    check_duplicates_tool1 = next(t for t in data_tools1 if t.name == "check_duplicates_tool")

    # Known Answer #3 in TEST_FIXTURE.md without dropping: duplicate_count == 1, total_duplicated_rows == 2
    res_check = check_duplicates_tool1.invoke({"drop": False})
    assert res_check["n_documents_before"] == 8
    assert res_check["duplicate_count"] == 1
    assert res_check["total_duplicated_rows"] == 2
    assert res_check["n_documents_after"] == 8

    # Fresh session for drop=True check to avoid state contamination
    session2 = SessionState()
    load_dataset_tool2, _ = load_workspace_tools("premade_tools", session2)
    next(t for t in load_dataset_tool2 if t.name == "load_dataset_tool").invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools2 = make_tools(session2)
    check_duplicates_tool2 = next(t for t in data_tools2 if t.name == "check_duplicates_tool")

    # Known Answer #3 with keep=False dropping: leaves exactly 6 rows (both copies dropped)
    res_drop = check_duplicates_tool2.invoke({"drop": True})
    assert res_drop["n_documents_before"] == 8
    assert res_drop["duplicate_count"] == 1
    assert res_drop["total_duplicated_rows"] == 2
    assert res_drop["n_documents_after"] == 6
    assert len(session2.dataframe) == 6

    # Verify index was properly reset (contiguous 0..5 index)
    assert list(session2.dataframe.index) == list(range(6))


def test_sample_data_tool_known_answer_13():
    """Tests sample_data_tool against TEST_FIXTURE.md Known Answer 13:
    n=4, random_state=42 returns exactly rows [1, 5, 0, 7].
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture CSV (8 rows)
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert "status" not in res or res.get("status") != "error"
    assert res["total_documents"] == 8
    assert res["sample_size"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]


def test_sample_data_tool_edge_cases():
    """Tests sample_data_tool edge cases:
    1. Unloaded session (session.dataframe is None)
    2. Missing both n and frac parameters
    3. Sampling with frac
    4. Omitting random_state (defaults to 42)
    """
    # 1. Unloaded dataframe error check
    session_empty = SessionState()
    tools_empty = make_tools(session_empty)
    sample_tool_empty = next(t for t in tools_empty if t.name == "sample_data_tool")

    res_empty = sample_tool_empty.invoke({"n": 2})
    assert res_empty["status"] == "error"
    assert "No dataframe loaded" in res_empty["message"]

    # Load fixture into fresh session
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    sample_tool = next(t for t in tools if t.name == "sample_data_tool")

    # 2. Missing both n and frac
    res_no_params = sample_tool.invoke({})
    assert res_no_params["status"] == "error"
    assert "explicitly specify" in res_no_params["message"].lower()

    # 3. Sampling with frac=0.5 (4 out of 8 rows)
    res_frac = sample_tool.invoke({"frac": 0.5, "random_state": 42})
    assert res_frac["sample_size"] == 4
    assert res_frac["sampled_indices"] == [1, 5, 0, 7]

    # 4. Omitting random_state (should default to 42 and match Known Answer 13)
    res_default_seed = sample_tool.invoke({"n": 4})
    assert res_default_seed["sampled_indices"] == [1, 5, 0, 7]

    # Check non-destructive behavior: working dataframe length is still 8
    assert len(session.dataframe) == 8


def test_describe_data_tool_known_answer_18():
    """Tests describe_data_tool against TEST_FIXTURE.md Known Answer 18."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})

    assert res["overall_stats"]["count"] == 8
    assert abs(res["overall_stats"]["mean"] - 19.875) < 1e-3
    assert abs(res["overall_stats"]["min"] - 0) < 1e-3
    assert abs(res["overall_stats"]["max"] - 30) < 1e-3

    # Check category stats
    catA_stats = res["per_category_stats"]["catA"]
    assert catA_stats["count"] == 4
    assert abs(catA_stats["mean"] - 17.25) < 1e-3

    catB_stats = res["per_category_stats"]["catB"]
    assert catB_stats["count"] == 4
    assert abs(catB_stats["mean"] - 22.5) < 1e-3

    # Check pending figure was set
    assert session.pending_figure is not None
