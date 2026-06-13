"""B1 regression tests: the slug must be incapable of path traversal or git ref
injection by construction. These run in the free per-push CI tier."""
import re
from main import _make_slug


def test_path_traversal_neutralized():
    assert _make_slug("../../etc/passwd") == "etc-passwd"

def test_git_ref_and_shell_injection_neutralized():
    assert _make_slug("feature/../../main; rm -rf /") == "feature-main-rm-rf"

def test_normal_topic():
    assert _make_slug("CatBoost & Friends, 2026 — A Review") == \
        "catboost-and-friends-2026-a-review"

def test_output_alphabet_is_closed():
    for nasty in ("a/b\\c..d e!f", "ловушка", "{{slug}}", "..", "a\x00b"):
        assert re.fullmatch(r"[a-z0-9-]*", _make_slug(nasty)), nasty

def test_length_cap():
    assert len(_make_slug("word " * 200)) <= 80

def test_degenerate_topic_yields_empty_for_caller_to_reject():
    assert _make_slug("———") == ""