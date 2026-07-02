"""Broker abstraction for live/paper trading.

Two implementations share one interface so the live trader is broker-agnostic:

  * ``PaperBroker`` — fully simulated fills against a mark price. Supports
    shorts. Used for dry-runs, forward-testing, and the test suite.
  * ``BinanceBroker`` — real Binance **spot** REST client (signed). Spot
    cannot short, so a SHORT target is treated as FLAT. Every real order
    path is guarded: it only runs when explicitly constructed with
    ``live=True`` and valid API credentials.

The live trader never calls exchange endpoints directly — it goes through
this interface, so swapping paper for real (or adding Alpaca/Kraken later)
touches only this file.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
from dataclasses import dataclass, field

from cbe_strategies import FLAT, LONG, SHORT

BINANCE_SPOT_BASE = "https://api.binance.com"
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 4

SIDE_BUY = "BUY"
SIDE_SELL = "SELL"


@dataclass
class Position:
    """Signed position: qty > 0 long, < 0 short, 0 flat."""

    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0

    @property
    def direction(self) -> int:
        if self.qty > 0:
            return LONG
        if self.qty < 0:
            return SHORT
        return FLAT


@dataclass
class OrderResult:
    symbol: str
    side: str
    qty: float
    price: float  # fill price (or mark, for market orders)
    notional: float
    status: str  # FILLED, DRY_RUN, REJECTED, SKIPPED
    reason: str = ""
    raw: dict = field(default_factory=dict)


class Broker:
    """Interface every broker implements."""

    supports_short: bool = False

    def get_equity(self, mark_price: float) -> float:
        """Account equity valued at ``mark_price`` (cash + position value)."""
        raise NotImplementedError

    def get_position(self, symbol: str) -> Position:
        raise NotImplementedError

    def market_order(self, symbol: str, side: str, qty: float, mark_price: float) -> OrderResult:
        """Submit a market order. ``mark_price`` is used for paper fills and
        notional accounting; real brokers fill at the live price."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Paper broker (simulated)
# ---------------------------------------------------------------------------


class PaperBroker(Broker):
    """In-memory simulated broker with fees, slippage, and short support."""

    supports_short = True

    def __init__(
        self,
        cash: float = 10_000.0,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ) -> None:
        self.cash = cash
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self._positions: dict[str, Position] = {}
        self.order_log: list[OrderResult] = []

    def get_position(self, symbol: str) -> Position:
        return self._positions.get(symbol, Position(symbol))

    def get_equity(self, mark_price: float) -> float:
        pos_value = sum(p.qty * mark_price for p in self._positions.values())
        return self.cash + pos_value

    def market_order(self, symbol: str, side: str, qty: float, mark_price: float) -> OrderResult:
        if qty <= 0:
            return OrderResult(symbol, side, 0.0, mark_price, 0.0, "SKIPPED", "non-positive qty")
        # adverse slippage: buys fill higher, sells lower
        trade_side = 1 if side == SIDE_BUY else -1
        fill = mark_price * (1.0 + trade_side * self.slippage_bps / 10_000.0)
        notional = qty * fill
        fee = notional * self.fee_bps / 10_000.0

        pos = self._positions.setdefault(symbol, Position(symbol))
        signed = qty if side == SIDE_BUY else -qty
        new_qty = pos.qty + signed
        # weighted avg price only when adding in the same direction
        if pos.qty == 0 or (pos.qty > 0) == (signed > 0):
            total = abs(pos.qty) + abs(signed)
            pos.avg_price = (
                (pos.avg_price * abs(pos.qty) + fill * abs(signed)) / total if total else fill
            )
        elif abs(signed) > abs(pos.qty):
            pos.avg_price = fill  # flipped through zero
        pos.qty = round(new_qty, 12)
        self.cash -= signed * fill + fee
        result = OrderResult(symbol, side, qty, fill, notional, "FILLED")
        self.order_log.append(result)
        return result


# ---------------------------------------------------------------------------
# Binance spot broker (real, guarded)
# ---------------------------------------------------------------------------


class BinanceBroker(Broker):
    """Real Binance **spot** REST client. No shorting (spot).

    Guard: constructing with ``live=True`` requires an API key/secret and is
    the only mode that sends order requests. With ``live=False`` (default),
    ``market_order`` returns a DRY_RUN result and never touches the network
    for order placement — but balance/price reads still work if credentials
    are provided.
    """

    supports_short = False

    def __init__(  # nosec B107 - empty default = no credentials (paper/dry-run), not a secret
        self,
        api_key: str = "",
        api_secret: str = "",
        live: bool = False,
        base_url: str = BINANCE_SPOT_BASE,
        quote_asset: str = "USDT",
        recv_window: int = 5000,
        session: object = None,
    ) -> None:
        if live and (not api_key or not api_secret):
            raise ValueError("live=True requires api_key and api_secret")
        self.api_key = api_key
        self.api_secret = api_secret
        self.live = live
        self.base_url = base_url.rstrip("/")
        self.quote_asset = quote_asset.upper()
        self.recv_window = recv_window
        self._session = session

    # -- HTTP plumbing --------------------------------------------------

    def _http(self):
        if self._session is not None:
            return self._session
        import requests

        return requests

    def _sign(self, params: dict) -> str:
        query = urllib.parse.urlencode(params)
        return hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

    def _signed_request(self, method: str, path: str, params: dict) -> dict:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self.recv_window
        params["signature"] = self._sign(params)
        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{self.base_url}{path}"
        http = self._http()
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = http.request(
                    method, url, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"Binance {method} {path} failed after {MAX_RETRIES} retries: {last_error}"
        )

    # -- Broker interface ----------------------------------------------

    def get_position(self, symbol: str) -> Position:
        """Spot 'position' = free balance of the base asset (e.g. BTC of BTCUSDT)."""
        base_asset = symbol.upper().replace(self.quote_asset, "")
        account = self._signed_request("GET", "/api/v3/account", {})
        for bal in account.get("balances", []):
            if bal.get("asset") == base_asset:
                return Position(symbol, qty=float(bal.get("free", 0.0)))
        return Position(symbol)

    def get_equity(self, mark_price: float) -> float:
        account = self._signed_request("GET", "/api/v3/account", {})
        cash = 0.0
        base_qty = 0.0
        for bal in account.get("balances", []):
            free = float(bal.get("free", 0.0)) + float(bal.get("locked", 0.0))
            if bal.get("asset") == self.quote_asset:
                cash += free
        return cash + base_qty * mark_price

    def market_order(self, symbol: str, side: str, qty: float, mark_price: float) -> OrderResult:
        if qty <= 0:
            return OrderResult(symbol, side, 0.0, mark_price, 0.0, "SKIPPED", "non-positive qty")
        notional = qty * mark_price
        if not self.live:
            return OrderResult(
                symbol,
                side,
                qty,
                mark_price,
                notional,
                "DRY_RUN",
                "live=False; order not sent",
            )
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "MARKET",
            "quantity": _trim_qty(qty),
        }
        raw = self._signed_request("POST", "/api/v3/order", params)
        fills = raw.get("fills", [])
        fill_price = (
            sum(float(f["price"]) * float(f["qty"]) for f in fills)
            / sum(float(f["qty"]) for f in fills)
            if fills
            else mark_price
        )
        executed = float(raw.get("executedQty", qty))
        return OrderResult(
            symbol,
            side,
            executed,
            fill_price,
            executed * fill_price,
            raw.get("status", "FILLED"),
            raw=raw,
        )


def _trim_qty(qty: float, decimals: int = 6) -> str:
    """Format quantity to a fixed precision string (Binance rejects overly
    precise quantities; real deployments should use per-symbol step size)."""
    return f"{qty:.{decimals}f}".rstrip("0").rstrip(".")
