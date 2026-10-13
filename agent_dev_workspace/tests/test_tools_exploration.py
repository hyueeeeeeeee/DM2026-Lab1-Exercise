import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def setup_test_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM to populate session.feature_matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session

def test_cosine_similarity_tool():
    session = setup_test_session()
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    
    # Known answer #16 test cases from TEST_FIXTURE.md
    
    # 1. Doc 0 vs 1 (both catA, near-identical text): 0.9847
    res_0_1 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 1})
    assert np.isclose(res_0_1["similarity"], 0.9847, atol=1e-3)
    
    # 2. Doc 0 vs 3 (catA vs catB, share only "always"): 0.0909
    res_0_3 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 3})
    assert np.isclose(res_0_3["similarity"], 0.0909, atol=1e-3)
    
    # 3. Doc 3 vs 4 (both catB, near-identical text): 0.9847
    res_3_4 = cosine_sim_tool.invoke({"doc_index1": 3, "doc_index2": 4})
    assert np.isclose(res_3_4["similarity"], 0.9847, atol=1e-3)
    
    # 4. Doc 0 vs 7 (catA vs empty-text row): 0.0
    res_0_7 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 7})
    assert np.isclose(res_0_7["similarity"], 0.0, atol=1e-5)

def test_feature_correlation_matrix_tool():
    session = setup_test_session()
    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({"top_n": 20})
    
    # Known answer #19 in TEST_FIXTURE.md
    # Terms in descending order of variance: alpha, gamma, delta, beta, always
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # Check correlation matrix against known answer #19
    expected_matrix = [
        [1.0000, -0.5718, -0.6882,  0.8885, 0.2601],
        [-0.5718, 1.0000,  0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307,  1.0000, -0.7746, 0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000, 0.2928],
        [ 0.2601, 0.3140,  0.3780,  0.2928, 1.0000]
    ]

    actual_matrix = res["correlation_matrix"]
    assert np.allclose(actual_matrix, expected_matrix, atol=1e-3)

    # Check that pending_figure was generated
    assert session.pending_figure is not None
