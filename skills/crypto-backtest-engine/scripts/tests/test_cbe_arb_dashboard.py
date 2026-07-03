"""Tests for triangle universe building and the HTML dashboard."""

from __future__ import annotations

from cbe_arb_dashboard import render_dashboard, triangle_stats, write_dashboard
from cbe_arbitrage import (
    DEFAULT_ALT_UNIVERSE,
    Triangle,
    build_triangles,
    universe_symbols,
)


class TestBuildTriangles:
    def test_basic_construction(self):
        tris = build_triangles(["SOL"], ["BTC"], "USDT")
        assert len(tris) == 1
        assert tris[0] == Triangle("BTCUSDT", "SOLBTC", "SOLUSDT")

    def test_multiple_bridges_and_alts(self):
        tris = build_triangles(["SOL", "XRP"], ["BTC", "ETH"], "USDT")
        assert len(tris) == 4  # 2 alts x 2 bridges

    def test_skips_degenerate(self):
        # BTC as both alt and bridge, plus quote collisions -> skipped
        tris = build_triangles(["BTC", "USDT", "SOL"], ["BTC"], "USDT")
        assert all(t.symbols() != ("BTCBTC", "BTCBTC", "BTCUSDT") for t in tris)
        assert len(tris) == 1  # only SOL survives
        assert tris[0].leg_b == "SOLBTC"

    def test_dedup(self):
        tris = build_triangles(["SOL", "SOL"], ["BTC"], "USDT")
        assert len(tris) == 1

    def test_default_universe_nonempty(self):
        tris = build_triangles(DEFAULT_ALT_UNIVERSE, ["BTC", "ETH"], "USDT")
        assert len(tris) > 20
        # every triangle's symbols are unique-per-leg and quote-terminated
        for t in tris:
            assert t.leg_a.endswith("USDT")
            assert t.leg_c.endswith("USDT")

    def test_universe_symbols_sorted_unique(self):
        tris = build_triangles(["SOL"], ["BTC"], "USDT")
        syms = universe_symbols(tris)
        assert syms == sorted(syms)
        assert syms == ["BTCUSDT", "SOLBTC", "SOLUSDT"]


def _executions():
    return [
        {
            "ts": 0,
            "kind": "triangular",
            "description": "A->B->C (forward)",
            "edge_bps": 5.0,
            "deployed": 100.0,
            "profit": 0.05,
        },
        {
            "ts": 1,
            "kind": "triangular",
            "description": "A->B->C (forward)",
            "edge_bps": 8.0,
            "deployed": 200.0,
            "profit": 0.16,
        },
        {
            "ts": 2,
            "kind": "triangular",
            "description": "X->Y->Z (backward)",
            "edge_bps": 3.0,
            "deployed": 50.0,
            "profit": 0.015,
        },
    ]


class TestTriangleStats:
    def test_aggregation_and_sort(self):
        stats = triangle_stats(_executions())
        assert len(stats) == 2
        # A->B->C has more profit -> ranked first
        assert stats[0].description == "A->B->C (forward)"
        assert stats[0].hits == 2
        assert stats[0].total_profit == 0.05 + 0.16
        assert stats[0].best_edge_bps == 8.0
        assert stats[0].max_notional_seen == 200.0

    def test_empty(self):
        assert triangle_stats([]) == []


class TestRenderDashboard:
    def test_contains_core_sections(self):
        timeline = [(0, 0.0), (1, 0.05), (2, 0.21), (3, 0.225)]
        html = render_dashboard(timeline, _executions(), 0.225, {"title": "Test Board"})
        assert "<!doctype html>" in html
        assert "Test Board" in html
        assert "PAPER MODE" in html
        assert "A-&gt;B-&gt;C (forward)" in html  # html-escaped arrows
        assert "<svg" in html  # equity curve rendered
        assert "Triangle leaderboard" in html

    def test_refresh_meta_when_looping(self):
        html = render_dashboard([(0, 0.0), (1, 1.0)], _executions(), 1.0, {"refresh_seconds": 5})
        assert 'http-equiv="refresh"' in html
        assert 'content="5"' in html

    def test_no_refresh_meta_when_absent(self):
        html = render_dashboard([(0, 0.0), (1, 1.0)], _executions(), 1.0, {})
        assert "http-equiv" not in html

    def test_equity_curve_needs_two_points(self):
        html = render_dashboard([(0, 0.0)], [], 0.0, {})
        assert "appears after 2+ polls" in html

    def test_negative_profit_styling(self):
        html = render_dashboard([(0, 0.0), (1, -5.0)], [], -5.0, {})
        assert 'polyline class="neg"' in html

    def test_empty_executions_safe(self):
        html = render_dashboard([(0, 0.0), (1, 0.0)], [], 0.0, {})
        assert "No opportunities yet" in html
        assert "No fills yet" in html

    def test_html_escapes_untrusted_description(self):
        evil = [
            {
                "ts": 0,
                "kind": "triangular",
                "description": "<script>alert(1)</script>",
                "edge_bps": 1.0,
                "deployed": 1.0,
                "profit": 0.1,
            }
        ]
        html = render_dashboard([(0, 0.0), (1, 0.1)], evil, 0.1, {})
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html


class TestWriteDashboard:
    def test_writes_file(self, tmp_path):
        path = tmp_path / "sub" / "board.html"
        write_dashboard(str(path), [(0, 0.0), (1, 0.5)], _executions(), 0.5, {"title": "Board"})
        assert path.exists()
        assert "<!doctype html>" in path.read_text()


class TestCliUniverseAndDashboard:
    def test_universe_mode_writes_dashboard(self, tmp_path, capsys):
        from run_arbitrage import main

        board = tmp_path / "board.html"
        rc = main(
            [
                "--synthetic",
                "--assets",
                "SOL,XRP",
                "--bridges",
                "BTC,ETH",
                "--polls",
                "15",
                "--min-edge-bps",
                "1",
                "--dashboard",
                str(board),
            ]
        )
        assert rc == 0
        assert board.exists()
        doc = board.read_text()
        assert "Crypto Arbitrage Scanner" in doc
        out = capsys.readouterr().out
        assert "scanning 4 triangle(s)" in out

    def test_assets_default_keyword(self, capsys):
        from run_arbitrage import main

        rc = main(["--synthetic", "--assets", "default", "--polls", "3"])
        assert rc == 0
        assert "scanning" in capsys.readouterr().out

    def test_empty_assets_rejected(self):
        # only the quote asset -> no valid triangle
        import pytest
        from run_arbitrage import main

        with pytest.raises(SystemExit, match="no valid triangles"):
            main(["--synthetic", "--assets", "USDT", "--bridges", "USDT"])
