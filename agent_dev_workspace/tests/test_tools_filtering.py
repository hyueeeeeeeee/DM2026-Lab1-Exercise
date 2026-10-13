import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools

def test_variance_filter_tool():
    # Create single session
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load test fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using CountVectorizer defaults on session
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vectorizer.get_feature_names_out().tolist()

    # Instantiate student's tools
    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Call tool with threshold 0.15 (TEST_FIXTURE.md #5)
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Assertions according to TEST_FIXTURE.md #5
    assert result["total_features"] == 5
    assert result["retained_features_count"] == 4
    assert result["removed_features_count"] == 1
    assert "always" not in result["retained_terms_preview"]
    assert "always" in result["removed_terms_preview"]

    # Verify full_report stored in session.results_store
    result_id = result["result_id"]
    stored_result = session.results_store[result_id]
    report_df = stored_result["full_report"]

    # Check columns
    assert list(report_df.columns[:2]) == ["feature", "variance"]

    # Check exact variance values with numerical tolerance
    var_dict = dict(zip(report_df["feature"], report_df["variance"]))
    assert np.isclose(var_dict["alpha"], 1.1875, atol=1e-3)
    assert np.isclose(var_dict["always"], 0.109375, atol=1e-3)
    assert np.isclose(var_dict["beta"], 0.234375, atol=1e-3)
    assert np.isclose(var_dict["delta"], 0.25, atol=1e-3)
    assert np.isclose(var_dict["gamma"], 1.109375, atol=1e-3)

def test_pearson_filter_tool():
    # Fresh session for isolated state
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vectorizer.get_feature_names_out().tolist()

    filtering_tools = make_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Call with target_class="catB" (TEST_FIXTURE.md #6)
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert result["total_features"] == 5

    # Check full_report format and values
    result_id = result["result_id"]
    stored_result = session.results_store[result_id]
    report_df = stored_result["full_report"]

    assert list(report_df.columns[:2]) == ["feature", "pearson_r"]

    # Assert Pearson correlation values against known answer #6
    # alpha: -0.6882, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.8307
    pearson_dict = dict(zip(report_df["feature"], report_df["pearson_r"]))
    assert np.isclose(pearson_dict["alpha"], -0.6882, atol=1e-3)
    assert np.isclose(pearson_dict["always"], 0.3780, atol=1e-3)
    assert np.isclose(pearson_dict["beta"], -0.7746, atol=1e-3)
    assert np.isclose(pearson_dict["delta"], 1.0000, atol=1e-3)
    assert np.isclose(pearson_dict["gamma"], 0.8307, atol=1e-3)

def test_spearman_filter_tool():
    # Fresh session for isolated state
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vectorizer.get_feature_names_out().tolist()

    filtering_tools = make_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Call with target_class="catB" (TEST_FIXTURE.md #6)
    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert result["total_features"] == 5

    # Check full_report format and values
    result_id = result["result_id"]
    stored_result = session.results_store[result_id]
    report_df = stored_result["full_report"]

    assert list(report_df.columns[:2]) == ["feature", "spearman_r"]

    # Assert Spearman correlation values against known answer #6
    # alpha: -0.7500, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.9363
    spearman_dict = dict(zip(report_df["feature"], report_df["spearman_r"]))
    assert np.isclose(spearman_dict["alpha"], -0.7500, atol=1e-3)
    assert np.isclose(spearman_dict["always"], 0.3780, atol=1e-3)
    assert np.isclose(spearman_dict["beta"], -0.7746, atol=1e-3)
    assert np.isclose(spearman_dict["delta"], 1.0000, atol=1e-3)
    assert np.isclose(spearman_dict["gamma"], 0.9363, atol=1e-3)

def test_spearman_filter_tool_max_terms_prefiltering():
    # Fresh session to test max_terms pre-filtering
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
    session.feature_names = vectorizer.get_feature_names_out().tolist()

    filtering_tools = make_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Call with max_terms=2 to test pre-filtering to top 2 variance terms (alpha and gamma)
    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0, "max_terms": 2})

    assert result["total_features"] == 5
    assert result["analyzed_features_count"] == 2
    assert result["is_prefiltered_to_top_variance"] is True
    assert result["max_terms"] == 2

    # Verify only top 2 variance terms (alpha and gamma) were analyzed in full_report
    stored_result = session.results_store[result["result_id"]]
    report_df = stored_result["full_report"]
    assert len(report_df) == 2
    assert set(report_df["feature"]) == {"alpha", "gamma"}
