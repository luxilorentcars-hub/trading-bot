"""Market-neutral arbitrage detectors (paper/scanner only).

Two detectors, one interface — both return :class:`ArbOpportunity`:

  * **Triangular arbitrage (Binance spot)** — for a triangle of pairs
    (X/QUOTE, Y/X, Y/QUOTE), a round trip QUOTE -> X -> Y -> QUOTE (or the
    reverse) should multiply capital by exactly 1.0 in an efficient market.
    When the product of executable prices, net of taker fees on each leg,
    exceeds 1.0, the loop locks in a risk-free edge.

  * **Prediction-market YES/NO arbitrage (Polymarket-style)** — YES + NO
    resolves to exactly $1. Whenever ``ask(YES) + ask(NO) < $1`` buying both
    legs locks in the difference regardless of the outcome.

This module is deliberately execution-free: it detects, sizes against
top-of-book depth, and paper-simulates PnL. Real triangular execution needs
websocket books and near-atomic legs (partial fills break the loop); real
Polymarket execution needs the signed CLOB order stack and is subject to
jurisdiction restrictions. Treat the output as signals, not fills.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

REQUEST_TIMEOUT_SECONDS = 30

BINANCE_BOOK_TICKER_URL = "https://api.binance.com/api/v3/ticker/bookTicker"
POLYMARKET_BOOK_URL = "https://clob.polymarket.com/book"


@dataclass(frozen=True)
class BookTop:
    """Top of book for one instrument."""

    symbol: str
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float

    def validate(self) -> None:
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError(f"{self.symbol}: non-positive bid/ask")
        if self.bid > self.ask:
            raise ValueError(f"{self.symbol}: crossed book (bid {self.bid} > ask {self.ask})")


@dataclass
class ArbOpportunity:
    kind: str  # "triangular" or "yes_no"
    description: str
    edge_bps: float  # net edge per unit of capital, in basis points
    max_notional: float  # top-of-book size cap, in quote currency (or $)
    expected_profit: float  # edge applied to max_notional
    legs: list = field(default_factory=list)  # human-readable leg descriptions

    @property
    def is_actionable(self) -> bool:
        return self.edge_bps > 0 and self.max_notional > 0


# ---------------------------------------------------------------------------
# Triangular arbitrage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Triangle:
    """Three spot pairs with fixed semantics.

    ``leg_a`` = X/QUOTE (e.g. BTCUSDT), ``leg_b`` = Y/X (e.g. ETHBTC),
    ``leg_c`` = Y/QUOTE (e.g. ETHUSDT). QUOTE is the accounting currency.
    """

    leg_a: str
    leg_b: str
    leg_c: str

    def symbols(self) -> tuple:
        return (self.leg_a, self.leg_b, self.leg_c)


# A pragmatic default universe of liquid alts to bridge through BTC/ETH.
# Alt triangles misprice more often than BTC/ETH but have thinner books —
# the scanner sizes by depth, so tiny opportunities surface as tiny sizes.
DEFAULT_ALT_UNIVERSE = [
    "SOL",
    "XRP",
    "ADA",
    "DOGE",
    "AVAX",
    "LINK",
    "DOT",
    "MATIC",
    "LTC",
    "TRX",
    "ATOM",
    "NEAR",
    "APT",
    "ARB",
]
DEFAULT_BRIDGES = ["BTC", "ETH", "BNB"]


def build_triangles(alts: list[str], bridges: list[str], quote: str = "USDT") -> list[Triangle]:
    """Generate triangles that bridge each alt through each bridge vs quote.

    For alt Y, bridge X and quote Q, the loop is Q -> X -> Y -> Q with legs
    X/Q (leg_a), Y/X (leg_b), Y/Q (leg_c). Degenerate cases (alt == bridge)
    are skipped. Returns de-duplicated triangles.
    """
    seen: set = set()
    triangles: list[Triangle] = []
    q = quote.upper()
    for bridge in bridges:
        b = bridge.upper()
        for alt in alts:
            a = alt.upper()
            if a == b or a == q or b == q:
                continue
            tri = Triangle(f"{b}{q}", f"{a}{b}", f"{a}{q}")
            if tri.symbols() not in seen:
                seen.add(tri.symbols())
                triangles.append(tri)
    return triangles


def universe_symbols(triangles: list[Triangle]) -> list[str]:
    """Unique symbols needed to price a set of triangles (sorted)."""
    return sorted({s for tri in triangles for s in tri.symbols()})


def triangular_edges(
    triangle: Triangle,
    book_a: BookTop,
    book_b: BookTop,
    book_c: BookTop,
    fee_bps: float = 10.0,
) -> list[ArbOpportunity]:
    """Evaluate both directions of a triangle against top-of-book prices.

    Direction *forward*  : QUOTE -> X -> Y -> QUOTE (buy A, buy B, sell C)
    Direction *backward* : QUOTE -> Y -> X -> QUOTE (buy C, sell B, sell A)

    Fees are charged on each of the three legs. ``max_notional`` is the
    largest QUOTE amount executable against the quoted top-of-book sizes.
    """
    for book in (book_a, book_b, book_c):
        book.validate()
    fee = fee_bps / 10_000.0
    net = (1.0 - fee) ** 3
    results: list[ArbOpportunity] = []

    # -- forward: buy X with QUOTE, buy Y with X, sell Y for QUOTE
    gross_fwd = book_c.bid / (book_a.ask * book_b.ask)
    edge_fwd = gross_fwd * net - 1.0
    max_quote_fwd = min(
        book_a.ask_qty * book_a.ask,  # X available to buy
        book_b.ask_qty * book_b.ask * book_a.ask,  # Y available to buy, in QUOTE
        book_c.bid_qty * book_b.ask * book_a.ask,  # Y sellable, in QUOTE at entry
    )
    results.append(
        ArbOpportunity(
            kind="triangular",
            description=f"{triangle.leg_a}->{triangle.leg_b}->{triangle.leg_c} (forward)",
            edge_bps=edge_fwd * 10_000.0,
            max_notional=max_quote_fwd,
            expected_profit=edge_fwd * max_quote_fwd,
            legs=[
                f"BUY  {triangle.leg_a} @ {book_a.ask}",
                f"BUY  {triangle.leg_b} @ {book_b.ask}",
                f"SELL {triangle.leg_c} @ {book_c.bid}",
            ],
        )
    )

    # -- backward: buy Y with QUOTE, sell Y for X, sell X for QUOTE
    gross_bwd = (book_b.bid * book_a.bid) / book_c.ask
    edge_bwd = gross_bwd * net - 1.0
    max_quote_bwd = min(
        book_c.ask_qty * book_c.ask,  # Y available to buy, in QUOTE
        book_b.bid_qty * book_c.ask,  # Y sellable for X, in QUOTE at entry
        book_a.bid_qty * book_a.bid / max(gross_bwd, 1e-12),  # X sellable, back-converted
    )
    results.append(
        ArbOpportunity(
            kind="triangular",
            description=f"{triangle.leg_c}->{triangle.leg_b}->{triangle.leg_a} (backward)",
            edge_bps=edge_bwd * 10_000.0,
            max_notional=max_quote_bwd,
            expected_profit=edge_bwd * max_quote_bwd,
            legs=[
                f"BUY  {triangle.leg_c} @ {book_c.ask}",
                f"SELL {triangle.leg_b} @ {book_b.bid}",
                f"SELL {triangle.leg_a} @ {book_a.bid}",
            ],
        )
    )
    return results


def scan_triangles(
    triangles: list[Triangle],
    books: dict,
    fee_bps: float = 10.0,
    min_edge_bps: float = 1.0,
) -> list[ArbOpportunity]:
    """Scan many triangles against a ``{symbol: BookTop}`` snapshot.

    Returns only actionable opportunities with edge >= ``min_edge_bps``,
    sorted by expected profit (best first). Triangles with missing books
    are skipped silently — a stale snapshot must not crash the scanner.
    """
    found: list[ArbOpportunity] = []
    for tri in triangles:
        symbols = tri.symbols()
        if any(s not in books for s in symbols):
            continue
        for opp in triangular_edges(
            tri, books[symbols[0]], books[symbols[1]], books[symbols[2]], fee_bps
        ):
            if opp.is_actionable and opp.edge_bps >= min_edge_bps:
                found.append(opp)
    found.sort(key=lambda o: o.expected_profit, reverse=True)
    return found


def fetch_binance_book_tops(symbols: list[str], session: object = None) -> dict:
    """Fetch top-of-book for the given symbols from Binance's public API."""
    import requests

    http = session if session is not None else requests
    joined = "[" + ",".join(f'"{s.upper()}"' for s in symbols) + "]"
    response = http.get(
        BINANCE_BOOK_TICKER_URL, params={"symbols": joined}, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    books = {}
    for row in response.json():
        books[row["symbol"]] = BookTop(
            symbol=row["symbol"],
            bid=float(row["bidPrice"]),
            bid_qty=float(row["bidQty"]),
            ask=float(row["askPrice"]),
            ask_qty=float(row["askQty"]),
        )
    return books


# ---------------------------------------------------------------------------
# Prediction-market YES/NO arbitrage
# ---------------------------------------------------------------------------


def yes_no_edge(
    yes_book: BookTop,
    no_book: BookTop,
    fee_bps: float = 0.0,
    market: str = "market",
) -> ArbOpportunity:
    """Buy-both arbitrage on a binary market.

    Buying 1 YES share and 1 NO share costs ``ask(YES) + ask(NO)`` and pays
    exactly $1 at resolution whichever way it settles. Positive edge when the
    combined cost (plus fees) is below $1. ``max_notional`` is capped by the
    smaller top-of-book size (shares are $1-denominated at resolution).
    """
    yes_book.validate()
    no_book.validate()
    cost = yes_book.ask + no_book.ask
    fee = fee_bps / 10_000.0
    net_payout = 1.0 * (1.0 - fee)
    edge = net_payout - cost  # per share-pair, in dollars
    pairs = min(yes_book.ask_qty, no_book.ask_qty)
    max_notional = pairs * cost if cost > 0 else 0.0
    edge_bps = (edge / cost) * 10_000.0 if cost > 0 else 0.0
    return ArbOpportunity(
        kind="yes_no",
        description=f"{market}: buy YES@{yes_book.ask} + NO@{no_book.ask}",
        edge_bps=edge_bps,
        max_notional=max_notional,
        expected_profit=edge * pairs,
        legs=[
            f"BUY YES {pairs:.2f} @ {yes_book.ask}",
            f"BUY NO  {pairs:.2f} @ {no_book.ask}",
            "HOLD to resolution -> $1.00/pair",
        ],
    )


def fetch_polymarket_book_top(token_id: str, session: object = None) -> BookTop:
    """Fetch top-of-book for one Polymarket CLOB token (public endpoint).

    Note: Polymarket availability is jurisdiction-dependent; this fetcher is
    provided for analysis in places where access is lawful.
    """
    import requests

    http = session if session is not None else requests
    response = http.get(
        POLYMARKET_BOOK_URL, params={"token_id": token_id}, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    data = response.json()
    bids = data.get("bids") or []
    asks = data.get("asks") or []
    if not bids or not asks:
        raise ValueError(f"token {token_id}: empty book")
    best_bid = max(bids, key=lambda x: float(x["price"]))
    best_ask = min(asks, key=lambda x: float(x["price"]))
    return BookTop(
        symbol=token_id,
        bid=float(best_bid["price"]),
        bid_qty=float(best_bid["size"]),
        ask=float(best_ask["price"]),
        ask_qty=float(best_ask["size"]),
    )


# ---------------------------------------------------------------------------
# Paper ledger
# ---------------------------------------------------------------------------


@dataclass
class PaperArbLedger:
    """Accumulates simulated arbitrage executions (capped per-trade capital)."""

    capital_per_trade: float = 1_000.0
    realized_profit: float = 0.0
    executions: list = field(default_factory=list)

    def execute(self, opp: ArbOpportunity, now_ms: int | None = None) -> float:
        """Paper-execute an opportunity at its computed edge; returns profit."""
        if not opp.is_actionable:
            return 0.0
        deployed = min(self.capital_per_trade, opp.max_notional)
        profit = deployed * opp.edge_bps / 10_000.0
        self.realized_profit += profit
        self.executions.append(
            {
                "ts": now_ms if now_ms is not None else int(time.time() * 1000),
                "kind": opp.kind,
                "description": opp.description,
                "edge_bps": round(opp.edge_bps, 3),
                "deployed": round(deployed, 2),
                "profit": round(profit, 4),
            }
        )
        return profit


def format_opportunity(opp: ArbOpportunity) -> str:
    lines = [
        f"[{opp.kind}] {opp.description}",
        f"  edge: {opp.edge_bps:+.2f} bps | max size: {opp.max_notional:,.2f} "
        f"| expected: {opp.expected_profit:+,.4f}",
    ]
    lines.extend(f"  {leg}" for leg in opp.legs)
    return "\n".join(lines)
