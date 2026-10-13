import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


@pytest.fixture
def test_setup():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    return session, mine_patterns_tool


def test_mine_patterns_fpgrowth_tfidf(test_setup):
    """Test mine_patterns_tool with tfidf filtering and fpgrowth algorithm (Known Answer #17)."""
    session, mine_patterns_tool = test_setup

    # catA -> keeps alpha, support 3 for min_sup in 1-3, 0 patterns at min_sup=4
    for min_s in [1, 2, 3]:
        res_catA = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": min_s
        })
        assert res_catA["category_name"] == "catA"
        assert len(res_catA["patterns"]) == 1
        assert "alpha" in res_catA["patterns"][0]["pattern"]
        assert res_catA["patterns"][0]["support"] == 3

    res_catA_4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catA_4["patterns"] == []

    # catB -> keeps gamma, support 4 for min_sup in 1-4
    for min_s in [1, 2, 3, 4]:
        res_catB = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": min_s
        })
        assert res_catB["category_name"] == "catB"
        assert len(res_catB["patterns"]) == 1
        assert "gamma" in res_catB["patterns"][0]["pattern"]
        assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_zero_terms_survive(test_setup):
    """Test mine_patterns_tool when variance or term_frequency filtering leaves 0 terms (Known Answer #17)."""
    session, mine_patterns_tool = test_setup

    for method in ["variance", "term_frequency"]:
        for cat in ["catA", "catB"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert res["patterns"] == []
            assert "no terms survived filtering" in res["note"]


def test_mine_patterns_topk_all(test_setup):
    """Test topk algorithm across k=1, 2, 3 for both categories (Known Answer #20)."""
    session, mine_patterns_tool = test_setup

    for k_val in [1, 2, 3]:
        # catA -> alpha support 3
        res_A = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert len(res_A["patterns"]) == 1
        assert "alpha" in res_A["patterns"][0]["pattern"]
        assert res_A["patterns"][0]["support"] == 3

        # catB -> gamma support 4
        res_B = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert len(res_B["patterns"]) == 1
        assert "gamma" in res_B["patterns"][0]["pattern"]
        assert res_B["patterns"][0]["support"] == 4


def test_mine_patterns_maxfpgrowth_all(test_setup):
    """Test maxfpgrowth algorithm across min_sup=1 and 4 for both categories (Known Answer #20)."""
    session, mine_patterns_tool = test_setup

    for min_s in [1, 4]:
        for cat in ["catA", "catB"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": min_s
            })
            assert res["patterns"] == []
