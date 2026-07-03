"""Self-contained HTML dashboard for the arbitrage scanner.

Renders a single static HTML file (no external assets, no JS libraries) with:
  * a header summarizing the session,
  * an inline-SVG paper-equity curve,
  * a per-triangle leaderboard (which pairs produced the most/biggest edges),
  * a table of the most recent opportunities.

When the scanner runs with ``--loop --dashboard PATH`` the file is rewritten
every poll and carries a ``<meta refresh>`` so an open browser tab updates
itself — a live "screen" without any GUI framework.
"""

from __future__ import annotations

import html
from dataclasses import dataclass


@dataclass
class TriangleStat:
    description: str
    hits: int
    total_profit: float
    best_edge_bps: float
    max_notional_seen: float


def triangle_stats(executions: list) -> list[TriangleStat]:
    """Aggregate a ledger's executions into a per-triangle leaderboard."""
    acc: dict = {}
    for ex in executions:
        key = ex["description"]
        s = acc.get(key)
        if s is None:
            acc[key] = TriangleStat(key, 1, ex["profit"], ex["edge_bps"], ex["deployed"])
        else:
            s.hits += 1
            s.total_profit += ex["profit"]
            s.best_edge_bps = max(s.best_edge_bps, ex["edge_bps"])
            s.max_notional_seen = max(s.max_notional_seen, ex["deployed"])
    return sorted(acc.values(), key=lambda s: s.total_profit, reverse=True)


def _svg_equity_curve(timeline: list, width: int = 720, height: int = 220) -> str:
    """Inline SVG line chart of cumulative paper profit over polls."""
    if len(timeline) < 2:
        return '<p class="muted">Equity curve appears after 2+ polls.</p>'
    xs = [pt[0] for pt in timeline]
    ys = [pt[1] for pt in timeline]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad = 28.0
    span_x = max(x_max - x_min, 1)
    span_y = max(y_max - y_min, 1e-9)

    def px(x: float) -> float:
        return pad + (x - x_min) / span_x * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - y_min) / span_y * (height - 2 * pad)

    points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in timeline)
    zero_line = ""
    if y_min <= 0 <= y_max:
        zy = py(0.0)
        zero_line = (
            f'<line x1="{pad}" y1="{zy:.1f}" x2="{width - pad}" y2="{zy:.1f}" class="zero"/>'
        )
    last = timeline[-1][1]
    stroke = "pos" if last >= 0 else "neg"
    return (
        f'<svg viewBox="0 0 {width} {height}" class="equity" '
        f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="paper equity curve">'
        f"{zero_line}"
        f'<polyline class="{stroke}" points="{points}"/>'
        f'<text x="{pad}" y="16" class="axis">cumulative paper profit</text>'
        f'<text x="{width - pad}" y="{py(last):.1f}" class="axis end">{last:+.2f}</text>'
        f"</svg>"
    )


def _rows(stats: list[TriangleStat], limit: int) -> str:
    if not stats:
        return '<tr><td colspan="5" class="muted">No opportunities yet.</td></tr>'
    out = []
    for s in stats[:limit]:
        cls = "pos" if s.total_profit >= 0 else "neg"
        out.append(
            f"<tr><td>{html.escape(s.description)}</td>"
            f"<td>{s.hits}</td>"
            f'<td class="{cls}">{s.total_profit:+.4f}</td>'
            f"<td>{s.best_edge_bps:.2f}</td>"
            f"<td>{s.max_notional_seen:,.2f}</td></tr>"
        )
    return "\n".join(out)


def _recent_rows(executions: list, limit: int) -> str:
    if not executions:
        return '<tr><td colspan="4" class="muted">No fills yet.</td></tr>'
    out = []
    for ex in reversed(executions[-limit:]):
        out.append(
            f"<tr><td>{ex['ts']}</td>"
            f"<td>{html.escape(ex['description'])}</td>"
            f"<td>{ex['edge_bps']:.2f}</td>"
            f'<td class="pos">{ex["profit"]:+.4f}</td></tr>'
        )
    return "\n".join(out)


def render_dashboard(
    timeline: list,
    executions: list,
    realized_profit: float,
    meta: dict,
) -> str:
    """Return a full self-contained HTML document string."""
    stats = triangle_stats(executions)
    refresh = meta.get("refresh_seconds")
    refresh_tag = f'<meta http-equiv="refresh" content="{int(refresh)}">' if refresh else ""
    title = html.escape(meta.get("title", "Arbitrage Scanner"))
    subtitle = html.escape(meta.get("subtitle", ""))
    polls = meta.get("polls", 0)
    hit_total = len(executions)
    profit_cls = "pos" if realized_profit >= 0 else "neg"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh_tag}
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 24px;
         background: #0d1117; color: #e6edf3; }}
  @media (prefers-color-scheme: light) {{ body {{ background: #f6f8fa; color: #1f2328; }} }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .subtitle {{ color: #8b949e; font-size: 13px; margin-bottom: 20px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px;
          padding: 12px 16px; min-width: 130px; }}
  @media (prefers-color-scheme: light) {{ .card {{ background:#fff; border-color:#d0d7de; }} }}
  .card .label {{ color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }}
  .card .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  .pos {{ color: #3fb950; }}
  .neg {{ color: #f85149; }}
  .muted {{ color: #8b949e; }}
  svg.equity {{ width: 100%; max-width: 720px; height: auto; background: #161b22;
               border: 1px solid #30363d; border-radius: 10px; }}
  @media (prefers-color-scheme: light) {{ svg.equity {{ background:#fff; border-color:#d0d7de; }} }}
  svg.equity polyline {{ fill: none; stroke-width: 2; }}
  svg.equity polyline.pos {{ stroke: #3fb950; }}
  svg.equity polyline.neg {{ stroke: #f85149; }}
  svg.equity .zero {{ stroke: #6e7681; stroke-dasharray: 4 4; stroke-width: 1; }}
  svg.equity .axis {{ fill: #8b949e; font-size: 11px; }}
  svg.equity .axis.end {{ text-anchor: end; }}
  h2 {{ font-size: 14px; margin: 24px 0 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #30363d; }}
  @media (prefers-color-scheme: light) {{ th, td {{ border-color: #d0d7de; }} }}
  th {{ color: #8b949e; font-weight: 500; }}
  td:nth-child(n+2), th:nth-child(n+2) {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .wrap {{ overflow-x: auto; }}
  .banner {{ background:#1f2937; border:1px solid #30363d; border-radius:8px; padding:8px 12px;
            font-size:12px; color:#8b949e; margin-bottom:16px; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="subtitle">{subtitle}</div>
  <div class="banner">PAPER MODE — simulated fills only, no real orders are ever placed.</div>
  <div class="cards">
    <div class="card"><div class="label">Paper profit</div>
      <div class="value {profit_cls}">{realized_profit:+.2f}</div></div>
    <div class="card"><div class="label">Opportunities</div>
      <div class="value">{hit_total}</div></div>
    <div class="card"><div class="label">Polls</div>
      <div class="value">{polls}</div></div>
    <div class="card"><div class="label">Triangles tracked</div>
      <div class="value">{len(stats)}</div></div>
  </div>

  {_svg_equity_curve(timeline)}

  <h2>Triangle leaderboard (by paper profit)</h2>
  <div class="wrap"><table>
    <tr><th>Triangle</th><th>Hits</th><th>Profit</th><th>Best edge (bps)</th><th>Max size seen</th></tr>
    {_rows(stats, meta.get("top", 15))}
  </table></div>

  <h2>Recent opportunities</h2>
  <div class="wrap"><table>
    <tr><th>Poll</th><th>Triangle</th><th>Edge (bps)</th><th>Paper P&amp;L</th></tr>
    {_recent_rows(executions, meta.get("recent", 15))}
  </table></div>
</body>
</html>
"""


def write_dashboard(
    path: str, timeline: list, executions: list, realized_profit: float, meta: dict
) -> None:
    from pathlib import Path

    doc = render_dashboard(timeline, executions, realized_profit, meta)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
