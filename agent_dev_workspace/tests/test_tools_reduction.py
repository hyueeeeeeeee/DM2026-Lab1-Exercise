"""Tests for tools_reduction.py using sample_fixture.csv."""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools


def test_reduce_dimensions_pca():
    """Test reduce_dimensions_tool with PCA method matching TEST_FIXTURE.md #7."""
    session = SessionState()

    # Load fixture dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using CountVectorizer matching standard pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"])
    session.feature_names = vectorizer.get_feature_names_out().tolist()
    session.set_labels(session.dataframe["category_name"].tolist())

    # Get reduce_dimensions_tool bound to this session
    tools = make_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "pca", "n_components": 2})

    assert res["method"] == "pca"
    assert res["n_components"] == 2
    assert res["n_documents"] == 8
    assert "explained_variance_ratio" in res

    # Known answer #7 expected PCA coordinates
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])

    expected_evr = [0.7532, 0.1965]

    coords = np.array(res["coordinates"])
    np.testing.assert_allclose(coords, expected_coords, atol=1e-3)
    np.testing.assert_allclose(res["explained_variance_ratio"], expected_evr, atol=1e-3)

    # Verify pending figure plot was created
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne():
    """Test reduce_dimensions_tool with t-SNE method."""
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
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"])
    session.feature_names = vectorizer.get_feature_names_out().tolist()
    session.set_labels(session.dataframe["category_name"].tolist())

    tools = make_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    # Run 1
    res1 = reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })

    # Run 2 with same random state
    res2 = reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })

    assert res1["method"] == "tsne"
    assert res1["n_documents"] == 8
    coords1 = np.array(res1["coordinates"])
    coords2 = np.array(res2["coordinates"])

    assert coords1.shape == (8, 2)
    # Reproducibility check: same random state produces exact same coordinates
    np.testing.assert_allclose(coords1, coords2, atol=1e-5)

    # Duplicate row check: row 5 and row 6 in sample_fixture.csv are identical ("always gamma delta")
    np.testing.assert_allclose(coords1[5], coords1[6], atol=1e-3)

    assert session.pending_figure is not None


def test_reduce_dimensions_umap():
    """Test reduce_dimensions_tool with UMAP method matching TEST_FIXTURE.md #9."""
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
    session.feature_matrix = vectorizer.fit_transform(session.dataframe["text"])
    session.feature_names = vectorizer.get_feature_names_out().tolist()
    session.set_labels(session.dataframe["category_name"].tolist())

    tools = make_tools(session)
    reduce_tool = next(t for t in tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })

    assert res["method"] == "umap"

    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])

    coords = np.array(res["coordinates"])
    np.testing.assert_allclose(coords, expected_coords, atol=1e-2)


def test_binarize_labels_tool():
    """Test binarize_labels_tool on sample_fixture.csv."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})

    assert res["n_samples"] == 8
    assert res["shape"] == [8, 2]
    assert res["categories"] == ["catA", "catB"]

    # First 3 and last document are catA ([1, 0]), docs 3..6 are catB ([0, 1])
    encodings = res["sample_encodings"]
    assert encodings[0] == [1, 0]  # catA
    assert encodings[3] == [0, 1]  # catB
    assert encodings[7] == [1, 0]  # catA (empty text row has label catA in fixture)

    # Verify artifacts store
    assert "binarized_labels" in session.artifacts
    assert session.artifacts["binarized_labels"] == encodings
    assert session.artifacts["binarized_label_names"] == ["catA", "catB"]
