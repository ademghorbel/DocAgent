"""Tag mapping for the two benchmark loaders (the Hugging Face download itself runs on Kaggle)."""
import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mmlongbench_tags():
    m = load("prepare_mmlongbench")
    assert m.tags_for(["Table"], [3], "21%") == ["table"]
    assert m.tags_for(["Chart", "Pure-text (Plain-text)"], [2, 5], "x") == ["figure", "multi-page", "text"]
    assert m.tags_for(["Generalized-text (Layout)"], [], "Not answerable") == ["unanswerable"]
    assert m._as_list("[19, 20]") == [19, 20] and m._as_list([1]) == [1]
    assert m._as_list("['Table', 'Chart']") == ["Table", "Chart"]


def test_vidore_tags():
    m = load("prepare_vidore")
    assert m.tags_for(["Table", "Text"], 3) == ["multi-page", "table", "text"]
    assert m.tags_for(["Image"], 1) == ["figure"]
    assert m.tags_for(None, 1) == []
