import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


@pytest.fixture
def session_and_tools():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    return {
        "session": session,
        "load_dataset_tool": load_dataset_tool,
        "build_dtm_tool": build_dtm_tool,
        "term_frequency_tool": term_frequency_tool,
        "dtm_heatmap_tool": dtm_heatmap_tool,
    }


def test_build_dtm_fixture(session_and_tools):
    """Tests build_dtm_tool against Known Answer #4 in TEST_FIXTURE.md using sample_fixture.csv."""
    session = session_and_tools["session"]
    load_dataset_tool = session_and_tools["load_dataset_tool"]
    build_dtm_tool = session_and_tools["build_dtm_tool"]

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    result = build_dtm_tool.invoke({})

    # Verify return values match Known Answer #4
    assert result["n_documents"] == 8
    assert result["n_features"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Check session state updates
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]

    # Verify count_vectorizer artifact is stored
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None


def test_build_dtm_empty_vocabulary(session_and_tools):
    """Tests that build_dtm_tool handles overly strict parameters without crashing."""
    session = session_and_tools["session"]
    load_dataset_tool = session_and_tools["load_dataset_tool"]
    build_dtm_tool = session_and_tools["build_dtm_tool"]

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # min_df = 100 on an 8-doc dataset produces an empty vocabulary
    result = build_dtm_tool.invoke({"min_df": 100})

    assert result["status"] == "error"
    assert "message" in result


def test_term_frequency_fixture(session_and_tools):
    """Tests term_frequency_tool against Known Answer #14 in TEST_FIXTURE.md using sample_fixture.csv."""
    session = session_and_tools["session"]
    load_dataset_tool = session_and_tools["load_dataset_tool"]
    build_dtm_tool = session_and_tools["build_dtm_tool"]
    term_frequency_tool = session_and_tools["term_frequency_tool"]

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM first
    build_dtm_tool.invoke({})

    # Run term frequency tool
    result = term_frequency_tool.invoke({})

    assert result["n_terms"] == 5

    # Check top_terms returned in summary
    top_terms_dict = dict(result["top_terms"])
    assert top_terms_dict["always"] == 7
    assert top_terms_dict["gamma"] == 7
    assert top_terms_dict["alpha"] == 6
    assert top_terms_dict["delta"] == 4
    assert top_terms_dict["beta"] == 3

    # Verify full report in stored results
    stored_result = session.results_store[result["result_id"]]
    full_df = stored_result["full_report"]
    tf_full_dict = dict(zip(full_df["term"], full_df["frequency"]))
    assert tf_full_dict["alpha"] == 6
    assert tf_full_dict["always"] == 7
    assert tf_full_dict["beta"] == 3
    assert tf_full_dict["delta"] == 4
    assert tf_full_dict["gamma"] == 7


def test_dtm_heatmap_known_answer_21(session_and_tools):
    """Tests dtm_heatmap_tool against Known Answer #21 in TEST_FIXTURE.md using sample_fixture.csv."""
    session = session_and_tools["session"]
    load_dataset_tool = session_and_tools["load_dataset_tool"]
    build_dtm_tool = session_and_tools["build_dtm_tool"]
    dtm_heatmap_tool = session_and_tools["dtm_heatmap_tool"]

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})

    # Test full matrix request (defaults n_terms=20, n_documents=20)
    result = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert result["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    # Test smaller slice request
    slice_result = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert slice_result["n_documents"] == 3
    assert slice_result["n_terms"] == 2
    assert slice_result["terms"] == ["alpha", "always"]
    assert slice_result["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
