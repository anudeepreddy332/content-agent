import agent.nodes as nodes

GRID = '<div class="articles-grid" id="articlesGrid">'
SKELETON = f'<html><body><section id="learning-log">{GRID}\n' \
           '                    <a href="old.html" class="article-card" data-category="concept-exploration">\n' \
           '                        <h3 class="article-title">Old</h3>\n                    </a>\n' \
           '</section></body></html>'

class _Log:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass

def test_index_insert_newest_first(tmp_path):
    idx = tmp_path / "index.html"; idx.write_text(SKELETON)
    ok, action = nodes._update_index_html(tmp_path, "gradient-descent.html", "Gradient Descent",
                                          "concept-exploration", "An excerpt.", _Log(), "r")
    html = idx.read_text()
    assert ok and action == "inserted"
    # new card appears before the old one
    assert html.index("gradient-descent.html") < html.index("old.html")
    assert html.count('class="article-card"') == 2

def test_index_republish_updates_not_duplicates(tmp_path):
    idx = tmp_path / "index.html"; idx.write_text(SKELETON)
    nodes._update_index_html(tmp_path, "gradient-descent.html", "GD v1", "concept-exploration", "v1", _Log(), "r")
    ok, action = nodes._update_index_html(tmp_path, "gradient-descent.html", "GD v2", "concept-exploration", "v2", _Log(), "r")
    html = idx.read_text()
    assert ok and action == "updated"
    assert html.count("gradient-descent.html") == 1      # no duplicate
    assert "GD v2" in html and "GD v1" not in html       # updated in place

def test_index_missing_anchor_is_nonfatal(tmp_path):
    idx = tmp_path / "index.html"; idx.write_text("<html><body>no grid</body></html>")
    ok, msg = nodes._update_index_html(tmp_path, "x.html", "X", "concept-exploration", "e", _Log(), "r")
    assert ok is False