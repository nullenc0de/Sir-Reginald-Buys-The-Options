#!/usr/bin/env python3
"""
ASYMMETRIC OPTIONS AGENT
========================
Discovers second-order trading opportunities from scratch.
All data and state managed through Alpaca API - no local storage.

Data: Alpaca (news, market data, options)
State: Alpaca (positions, orders, account)
Reasoning: Local Ollama or manual review

Flow:
1. Check current positions/exposure via Alpaca
2. Pull latest news
3. LLM identifies second-order effects → ticker → direction
4. Find cheap high-convexity options
5. Execute if risk limits allow

TECHNICAL NOTES (from audit):
- API uses v2 for contracts, v1beta1 for market data
- Free tier uses IEX data (~2.5% of volume) - wide spreads possible
- 0DTE options may have null Greeks (Black-Scholes singularity)
- Pagination required for large option chains (next_page_token)
"""

import os
import re
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# JSON REPAIR UTILITY (for LLM outputs)
# =============================================================================

def repair_json(broken_json: str) -> str:
    """
    Attempt to repair malformed JSON from LLM outputs.

    Common issues:
    - Trailing commas
    - Single quotes instead of double
    - Unquoted keys
    - Markdown code blocks
    """
    text = broken_json.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)

    # Replace single quotes with double (careful with apostrophes)
    # Only replace if it looks like JSON structure
    if "'" in text and '"' not in text:
        text = text.replace("'", '"')

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # Try to extract JSON object/array from surrounding text
    json_match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
    if json_match:
        text = json_match.group()

    return text


def safe_json_parse(text: str) -> Optional[Dict]:
    """Safely parse JSON with repair fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            repaired = repair_json(text)
            return json.loads(repaired)
        except:
            return None


# =============================================================================
# CATALYST STORAGE
# =============================================================================

CATALYST_FILE = os.path.expanduser("~/.asymmetric_catalysts.json")


def load_catalysts() -> Dict:
    """Load catalyst data from JSON file."""
    if os.path.exists(CATALYST_FILE):
        try:
            with open(CATALYST_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_catalysts(catalysts: Dict) -> None:
    """Save catalyst data to JSON file."""
    with open(CATALYST_FILE, 'w') as f:
        json.dump(catalysts, f, indent=2)


def add_catalyst(ticker: str, description: str, date: str) -> None:
    """Add or update a catalyst for a ticker."""
    catalysts = load_catalysts()
    catalysts[ticker.upper()] = {
        "description": description,
        "date": date,
        "added": datetime.now().isoformat()
    }
    save_catalysts(catalysts)
    print(f"✅ Saved catalyst: {ticker} - {description} ({date})")


def remove_catalyst(ticker: str) -> None:
    """Remove a catalyst for a ticker."""
    catalysts = load_catalysts()
    if ticker.upper() in catalysts:
        del catalysts[ticker.upper()]
        save_catalysts(catalysts)
        print(f"🗑️ Removed catalyst: {ticker}")


def list_catalysts() -> Dict:
    """List all stored catalysts."""
    catalysts = load_catalysts()
    if not catalysts:
        print("No catalysts stored. Add with: add_catalyst('TICKER', 'description', '2026-01-15')")
        return {}

    print("="*60)
    print("STORED CATALYSTS")
    print("="*60)
    for ticker, data in sorted(catalysts.items(), key=lambda x: x[1].get("date", "")):
        print(f"  {ticker}: {data.get('description', 'Unknown')} ({data.get('date', 'No date')})")
    print("="*60)
    return catalysts


def get_catalyst_for_symbol(symbol: str) -> Optional[Dict]:
    """Get catalyst info for an option symbol (extracts underlying ticker)."""
    catalysts = load_catalysts()
    for ticker in catalysts.keys():
        if ticker in symbol.upper():
            return {"ticker": ticker, **catalysts[ticker]}
    return None


def rebuild_catalysts_from_orders(api_key: str, api_secret: str,
                                   merge: bool = True) -> Dict:
    """
    Rebuild catalyst data from Alpaca order history.

    This recovers catalyst DATES from client_order_id fields.
    Descriptions will be set to "Recovered from orders" - update manually.

    Args:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        merge: If True, merge with existing catalysts. If False, replace.

    Returns:
        Dict of recovered catalysts
    """
    import requests

    BASE_URL = "https://paper-api.alpaca.markets"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
        "accept": "application/json"
    }

    print("="*60)
    print("REBUILDING CATALYSTS FROM ALPACA ORDER HISTORY")
    print("="*60)

    # Get all orders (including filled, canceled, etc.)
    resp = requests.get(
        f"{BASE_URL}/v2/orders",
        headers=headers,
        params={"status": "all", "limit": 500}
    )

    if resp.status_code != 200:
        print(f"Error fetching orders: {resp.status_code}")
        return {}

    orders = resp.json()
    print(f"Found {len(orders)} orders in history")

    # Parse client_order_ids for catalyst data
    recovered = {}
    for order in orders:
        client_id = order.get("client_order_id", "")
        if client_id.startswith("CAT|"):
            parts = client_id.split("|")
            if len(parts) >= 3:
                ticker = parts[1]
                date = parts[2]
                if ticker not in recovered:
                    recovered[ticker] = {
                        "description": "Recovered from orders - update description",
                        "date": date,
                        "recovered": True
                    }
                    print(f"  ✅ Recovered: {ticker} - {date}")

    if not recovered:
        print("  No catalyst data found in order history")
        print("  (Only orders placed after this update have encoded catalysts)")
        return {}

    # Merge or replace
    if merge:
        existing = load_catalysts()
        for ticker, data in recovered.items():
            if ticker not in existing:
                existing[ticker] = data
            else:
                # Keep existing description, update date if different
                if existing[ticker].get("date") != data["date"]:
                    print(f"  ⚠️  {ticker}: Keeping existing date {existing[ticker].get('date')}")
        save_catalysts(existing)
        print(f"\nMerged {len(recovered)} recovered catalysts with existing data")
        return existing
    else:
        save_catalysts(recovered)
        print(f"\nReplaced catalyst file with {len(recovered)} recovered catalysts")
        return recovered


def backup_catalysts(path: str = None) -> str:
    """Backup catalyst file to specified path."""
    import shutil
    from datetime import datetime

    if path is None:
        path = os.path.expanduser(f"~/.asymmetric_catalysts_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    if os.path.exists(CATALYST_FILE):
        shutil.copy(CATALYST_FILE, path)
        print(f"✅ Backed up to: {path}")
        return path
    else:
        print("⚠️  No catalyst file to backup")
        return ""


# =============================================================================
# ALPACA CLIENT
# =============================================================================

class Alpaca:
    """
    Unified Alpaca client.

    Paper vs Live is determined by the API key itself:
    - Paper keys work with paper-api.alpaca.markets
    - Live keys work with api.alpaca.markets

    The client auto-detects based on a test call.

    IMPORTANT DATA TIER NOTES:
    - Free tier: IEX data only (~2.5% of market volume)
    - Paid tier: Full SIP/OPRA data (NBBO)
    - Free tier spreads may be wider than actual market
    """

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
            "accept": "application/json"
        }
        self.data_url = "https://data.alpaca.markets"

        # Rate limit tracking
        self._rate_limit_remaining = 200
        self._rate_limit_reset = 0

        # Auto-detect paper vs live and check account
        self.base_url = self._detect_environment()
        self._check_account_capabilities()

    def _detect_environment(self) -> str:
        """Try paper first, fall back to live"""
        for url in ["https://paper-api.alpaca.markets", "https://api.alpaca.markets"]:
            try:
                resp = requests.get(f"{url}/v2/account", headers=self.headers, timeout=10)
                if resp.status_code == 200:
                    env = "PAPER" if "paper" in url else "LIVE"
                    logger.info(f"Connected to Alpaca ({env})")
                    return url
            except:
                continue
        raise ConnectionError("Cannot connect to Alpaca. Check API keys.")

    def _check_account_capabilities(self):
        """Check account tier and options trading level."""
        try:
            account = self.get_account()

            # Check options trading level
            options_level = account.get("options_trading_level")
            if options_level:
                logger.info(f"Options Trading Level: {options_level}")
                if int(options_level) < 2:
                    logger.warning("Account is Level 1 - only covered calls allowed!")

            # Warn about data tier (free = IEX only)
            # Note: Can't directly query tier, but we log the warning
            logger.warning(
                "DATA QUALITY NOTE: If on free tier, quotes are IEX-only "
                "(~2.5% of volume). Spreads may be wider than NBBO."
            )
        except Exception as e:
            logger.warning(f"Could not check account capabilities: {e}")

    def _handle_rate_limit(self, response: requests.Response):
        """Track rate limits from response headers."""
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")

        if remaining:
            self._rate_limit_remaining = int(remaining)
        if reset:
            self._rate_limit_reset = int(reset)

        # If we're close to limit, pause
        if self._rate_limit_remaining < 10:
            sleep_time = max(1, self._rate_limit_reset - int(time.time()))
            logger.warning(f"Rate limit low ({self._rate_limit_remaining}), sleeping {sleep_time}s")
            time.sleep(sleep_time)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make request with rate limit handling."""
        resp = requests.request(method, url, headers=self.headers, **kwargs)
        self._handle_rate_limit(resp)

        # Handle 429 Too Many Requests
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 60))
            logger.warning(f"Rate limited! Sleeping {retry_after}s")
            time.sleep(retry_after)
            return self._request(method, url, **kwargs)

        return resp

    # -------------------------------------------------------------------------
    # Account & Positions (for risk management)
    # -------------------------------------------------------------------------

    def get_account(self) -> Dict:
        resp = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def get_positions(self) -> List[Dict]:
        resp = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def get_position(self, symbol: str) -> Optional[Dict]:
        resp = requests.get(f"{self.base_url}/v2/positions/{symbol}", headers=self.headers)
        if resp.status_code == 200:
            return resp.json()
        return None

    def get_orders(self, status: str = "open") -> List[Dict]:
        resp = requests.get(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            params={"status": status}
        )
        resp.raise_for_status()
        return resp.json()

    # -------------------------------------------------------------------------
    # News
    # -------------------------------------------------------------------------

    def get_news(self, limit: int = 50) -> List[Dict]:
        resp = requests.get(
            f"{self.data_url}/v1beta1/news",
            headers=self.headers,
            params={"limit": limit}
        )
        if resp.status_code == 200:
            return resp.json().get("news", [])
        return []

    # -------------------------------------------------------------------------
    # Market Data
    # -------------------------------------------------------------------------

    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        resp = requests.get(
            f"{self.data_url}/v2/stocks/{symbol}/quotes/latest",
            headers=self.headers
        )
        if resp.status_code == 200:
            q = resp.json().get("quote", {})
            bid = float(q.get("bp", 0) or 0)
            ask = float(q.get("ap", 0) or 0)
            if bid and ask:
                return (bid + ask) / 2
            return bid or ask
        return 0

    def is_tradeable(self, symbol: str) -> bool:
        resp = requests.get(f"{self.base_url}/v2/assets/{symbol}", headers=self.headers)
        if resp.status_code == 200:
            asset = resp.json()
            return asset.get("tradable", False) and asset.get("status") == "active"
        return False

    # -------------------------------------------------------------------------
    # Options
    # -------------------------------------------------------------------------

    def get_options(
        self,
        underlying: str,
        expiration_gte: str = None,
        expiration_lte: str = None,
        strike_gte: float = None,
        strike_lte: float = None,
        option_type: str = None,
        limit: int = 100,
        paginate: bool = False
    ) -> List[Dict]:
        """
        Get options contracts.

        Args:
            underlying: Stock symbol
            expiration_gte: Minimum expiration date (YYYY-MM-DD)
            expiration_lte: Maximum expiration date (YYYY-MM-DD)
            strike_gte: Minimum strike price
            strike_lte: Maximum strike price
            option_type: "call" or "put"
            limit: Max contracts per request (max 1000)
            paginate: If True, fetch all pages (for large chains)

        Note: API uses v2 for contracts. Default limit is low, so
        specify larger limit or paginate=True for full chains.
        """
        params = {"underlying_symbols": underlying, "limit": min(limit, 1000)}
        if expiration_gte:
            params["expiration_date_gte"] = expiration_gte
        if expiration_lte:
            params["expiration_date_lte"] = expiration_lte
        if strike_gte:
            params["strike_price_gte"] = str(strike_gte)
        if strike_lte:
            params["strike_price_lte"] = str(strike_lte)
        if option_type:
            params["type"] = option_type

        all_contracts = []

        while True:
            resp = self._request(
                "GET",
                f"{self.base_url}/v2/options/contracts",
                params=params
            )

            if resp.status_code != 200:
                break

            data = resp.json()
            contracts = data.get("option_contracts", [])
            all_contracts.extend(contracts)

            # Check for pagination
            next_page = data.get("next_page_token")
            if paginate and next_page:
                params["page_token"] = next_page
            else:
                break

        return all_contracts

    def get_option_quotes(self, symbols: List[str]) -> Dict:
        """
        Get quotes for options.

        Note: API uses v1beta1 for market data.
        Batches requests in chunks of 100 symbols.
        """
        if not symbols:
            return {}

        all_quotes = {}

        # Batch in chunks of 100 (API limit)
        for i in range(0, len(symbols), 100):
            chunk = symbols[i:i+100]
            resp = self._request(
                "GET",
                f"{self.data_url}/v1beta1/options/quotes/latest",
                params={"symbols": ",".join(chunk)}
            )
            if resp.status_code == 200:
                quotes = resp.json().get("quotes", {})
                all_quotes.update(quotes)

        return all_quotes

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def buy_option(self, symbol: str, qty: int, limit_price: float,
                   client_order_id: str = None) -> Dict:
        """Submit limit order to buy option

        Args:
            symbol: Option symbol
            qty: Number of contracts
            limit_price: Limit price per contract
            client_order_id: Optional custom ID (max 48 chars) - used to store catalyst date
        """
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy",
            "type": "limit",
            "limit_price": str(round(limit_price, 2)),
            "time_in_force": "day"
        }
        if client_order_id:
            order_data["client_order_id"] = client_order_id[:48]  # Alpaca limit

        resp = requests.post(
            f"{self.base_url}/v2/orders",
            headers={**self.headers, "content-type": "application/json"},
            json=order_data
        )
        resp.raise_for_status()
        return resp.json()

    def close_position(self, symbol: str) -> Dict:
        """Close a position"""
        resp = requests.delete(
            f"{self.base_url}/v2/positions/{symbol}",
            headers=self.headers
        )
        resp.raise_for_status()
        return resp.json()


# =============================================================================
# RISK MANAGER - Uses Alpaca positions to manage exposure
# =============================================================================

class RiskManager:
    """
    Manages risk by checking Alpaca positions/account.
    No local state - everything from API.
    """

    def __init__(self, alpaca: Alpaca, max_portfolio_risk: float = 0.20, max_single_trade: float = 0.05):
        """
        Args:
            max_portfolio_risk: Max total options exposure as % of equity (default 20%)
            max_single_trade: Max single trade as % of equity (default 5%)
        """
        self.alpaca = alpaca
        self.max_portfolio_risk = max_portfolio_risk
        self.max_single_trade = max_single_trade

    def get_exposure(self) -> Dict:
        """Calculate current options exposure from Alpaca positions"""
        account = self.alpaca.get_account()
        equity = float(account.get("equity", 0))
        positions = self.alpaca.get_positions()

        options_value = 0
        options_positions = []

        for pos in positions:
            symbol = pos.get("symbol", "")
            # Options symbols are longer and have specific format
            if len(symbol) > 10 or pos.get("asset_class") == "option":
                market_value = abs(float(pos.get("market_value", 0)))
                options_value += market_value
                options_positions.append({
                    "symbol": symbol,
                    "qty": pos.get("qty"),
                    "market_value": market_value,
                    "unrealized_pl": pos.get("unrealized_pl"),
                    "unrealized_plpc": pos.get("unrealized_plpc")
                })

        return {
            "equity": equity,
            "options_value": options_value,
            "options_pct": options_value / equity if equity > 0 else 0,
            "positions": options_positions,
            "position_count": len(options_positions)
        }

    def can_trade(self, trade_cost: float) -> tuple[bool, str]:
        """Check if a trade is allowed under risk limits"""
        exposure = self.get_exposure()
        equity = exposure["equity"]

        if equity <= 0:
            return False, "No equity"

        # Check single trade limit
        trade_pct = trade_cost / equity
        if trade_pct > self.max_single_trade:
            return False, f"Trade {trade_pct:.1%} exceeds max {self.max_single_trade:.1%}"

        # Check total portfolio limit
        new_total = exposure["options_pct"] + trade_pct
        if new_total > self.max_portfolio_risk:
            return False, f"Would exceed portfolio limit ({new_total:.1%} > {self.max_portfolio_risk:.1%})"

        return True, "OK"

    def get_available_risk(self) -> float:
        """How much more can we allocate to options?"""
        exposure = self.get_exposure()
        equity = exposure["equity"]
        current_pct = exposure["options_pct"]
        remaining_pct = max(0, self.max_portfolio_risk - current_pct)
        return equity * remaining_pct

    def print_status(self):
        """Print current risk status"""
        exposure = self.get_exposure()

        print(f"\n{'='*60}")
        print("PORTFOLIO RISK STATUS")
        print(f"{'='*60}")
        print(f"Equity: ${exposure['equity']:,.2f}")
        print(f"Options Exposure: ${exposure['options_value']:,.2f} ({exposure['options_pct']:.1%})")
        print(f"Risk Limit: {self.max_portfolio_risk:.0%}")
        print(f"Available: ${self.get_available_risk():,.2f}")

        if exposure["positions"]:
            print(f"\nOpen Positions ({exposure['position_count']}):")
            for pos in exposure["positions"]:
                pl = float(pos.get("unrealized_pl", 0) or 0)
                pl_str = f"+${pl:,.2f}" if pl >= 0 else f"-${abs(pl):,.2f}"
                print(f"  • {pos['symbol']}: ${pos['market_value']:,.2f} ({pl_str})")


# =============================================================================
# LLM INTERFACE
# =============================================================================

class LLM(ABC):
    @abstractmethod
    def analyze(self, news: List[Dict]) -> List[Dict]:
        """Returns list of opportunities"""
        pass


class OllamaLLM(LLM):
    """Local Ollama"""

    def __init__(self, model: str = "llama3.2", url: str = "http://localhost:11434", research: str = ""):
        self.model = model
        self.url = url
        self.external_research = research  # Pre-loaded research from file or Gemini etc
        self._check()

    def _check(self):
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError()
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"Ollama ready. Models: {models}")
        except:
            raise ConnectionError(f"Ollama not running at {self.url}. Run: ollama serve")

    def _query(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def analyze(self, news: List[Dict]) -> List[Dict]:
        if not news and not self.external_research:
            return []

        # Format news
        news_text = "\n".join([
            f"{i}. [{', '.join(a.get('symbols', [])) or 'GENERAL'}] {a.get('headline', '')}"
            for i, a in enumerate(news[:20], 1)
        ]) if news else "No news available."

        # Include external research if provided
        research_section = ""
        if self.external_research:
            research_section = f"""
MACRO RESEARCH (from external analysis):
{self.external_research}
"""

        prompt = f"""You are a macro trader finding second-order effects.

CONCEPT: Markets price PRIMARY events but miss SECOND-ORDER consequences.
Example: COVID lockdowns (primary) → can't visit family → send flowers instead (second-order) → Long 1-800-Flowers = 300% gain
{research_section}
TODAY'S NEWS:
{news_text}

Find second-order opportunities. Output ONLY valid JSON lines:
{{"primary_event": "what happened", "second_order_effect": "non-obvious consequence", "ticker": "SYMBOL", "direction": "call", "confidence": 0.75, "reasoning": "why", "days": 30}}

Rules:
- US stocks only (not ETFs except SPY/QQQ/IWM)
- Confidence 0.6-1.0
- Think contrarian - what is everyone MISSING?
- 0-5 opportunities max. Quality over quantity.

JSON only, no other text:"""

        try:
            response = self._query(prompt)
            opportunities = []

            for line in response.strip().split("\n"):
                line = line.strip()
                if line.startswith("{") or "{" in line:
                    # Use safe JSON parser with repair fallback
                    opp = safe_json_parse(line)
                    if opp and isinstance(opp, dict):
                        if opp.get("confidence", 0) >= 0.6:
                            opp["ticker"] = opp.get("ticker", "").upper()
                            opp["direction"] = opp.get("direction", "call").lower()
                            opportunities.append(opp)

            return opportunities
        except Exception as e:
            logger.error(f"Ollama failed: {e}")
            return []


class ManualLLM(LLM):
    """Manual mode - you provide opportunities"""

    def analyze(self, news: List[Dict]) -> List[Dict]:
        print(f"\n{'='*70}")
        print("NEWS FOR REVIEW")
        print(f"{'='*70}")

        for i, a in enumerate(news[:25], 1):
            print(f"\n{i}. {a.get('headline', '')[:80]}")
            syms = a.get('symbols', [])
            if syms:
                print(f"   Tickers: {', '.join(syms)}")

        print(f"\n{'='*70}")
        print("Enter opportunities (or press Enter to skip):")
        print('Format: {"ticker": "XYZ", "direction": "call", "confidence": 0.8, "days": 30, "reasoning": "why"}')
        print(f"{'='*70}")

        opportunities = []
        while True:
            try:
                line = input("> ").strip()
                if not line:
                    break
                opp = json.loads(line)
                opp["ticker"] = opp.get("ticker", "").upper()
                opportunities.append(opp)
                print(f"  Added: {opp['ticker']} {opp.get('direction', 'call')}")
            except json.JSONDecodeError:
                print("  Invalid JSON, try again")
            except EOFError:
                break

        return opportunities


class ClaudeCodeLLM(LLM):
    """
    Claude Code mode - returns news as structured data for Claude to analyze.
    Used when running in a Claude Code session where Claude IS the reasoning engine.

    Flow:
    1. Agent fetches news, returns it structured
    2. Claude (in conversation) analyzes and identifies second-order effects
    3. Claude calls agent.add_opportunities() with findings
    4. Agent finds options and executes
    """

    def __init__(self):
        self.pending_news = []
        self.opportunities = []
        self.research = []  # Store research findings

    def analyze(self, news: List[Dict]) -> List[Dict]:
        """Store news and return any pre-loaded opportunities"""
        self.pending_news = news

        # Return opportunities if they were pre-loaded
        result = self.opportunities.copy()
        self.opportunities = []
        return result

    def get_news_for_analysis(self) -> List[Dict]:
        """Get news in a clean format for Claude to analyze"""
        return [
            {
                "headline": a.get("headline", ""),
                "summary": a.get("summary", "")[:500],
                "symbols": a.get("symbols", []),
                "source": a.get("source", ""),
                "created": a.get("created_at", "")[:19]
            }
            for a in self.pending_news
        ]

    def add_opportunity(self, ticker: str, direction: str = "call",
                        confidence: float = 0.75, days: int = 30,
                        primary_event: str = "", second_order: str = "",
                        reasoning: str = ""):
        """Add an opportunity identified by Claude"""
        self.opportunities.append({
            "ticker": ticker.upper(),
            "direction": direction.lower(),
            "confidence": confidence,
            "days": days,
            "primary_event": primary_event,
            "second_order_effect": second_order,
            "reasoning": reasoning
        })


# =============================================================================
# OPTION FINDER
# =============================================================================

@dataclass
class Option:
    symbol: str
    underlying: str
    strike: float
    expiration: str
    type: str
    bid: float
    ask: float
    mid: float
    spread_pct: float
    oi: int
    otm_pct: float
    score: float


class OptionFinder:
    """Finds best options for a thesis"""

    def __init__(self, alpaca: Alpaca):
        self.alpaca = alpaca

    def find(self, ticker: str, direction: str, days: int, confidence: float) -> List[Option]:
        """Find suitable options"""

        price = self.alpaca.get_price(ticker)
        if price == 0:
            logger.warning(f"Can't get price for {ticker}")
            return []

        logger.info(f"Finding {direction}s for {ticker} @ ${price:.2f}")

        # Expiration window
        exp_min = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        exp_max = (datetime.now() + timedelta(days=days + 21)).strftime("%Y-%m-%d")

        # Strike range (5-15% OTM)
        if direction == "call":
            strike_min, strike_max = price * 1.03, price * 1.18
        else:
            strike_min, strike_max = price * 0.82, price * 0.97

        contracts = self.alpaca.get_options(
            underlying=ticker,
            expiration_gte=exp_min,
            expiration_lte=exp_max,
            strike_gte=strike_min if direction == "call" else None,
            strike_lte=strike_max,
            option_type=direction
        )

        if not contracts:
            return []

        # Get quotes
        symbols = [c["symbol"] for c in contracts if c.get("symbol")]
        quotes = self.alpaca.get_option_quotes(symbols)

        options = []
        for c in contracts:
            sym = c.get("symbol")
            q = quotes.get(sym, {})

            bid = float(q.get("bp", 0) or 0)
            ask = float(q.get("ap", 0) or 0)
            if not (bid or ask):
                continue

            mid = (bid + ask) / 2 if bid and ask else (bid or ask)
            spread = (ask - bid) / mid if mid else 1

            strike = float(c.get("strike_price", 0))
            oi = int(c.get("open_interest", 0) or 0)

            if direction == "call":
                otm = (strike - price) / price
            else:
                otm = (price - strike) / price

            # Score
            score = 0
            if 0.05 <= otm <= 0.12: score += 30
            elif 0.03 <= otm <= 0.15: score += 15
            if spread < 0.15: score += 25
            elif spread < 0.25: score += 10
            if oi >= 50: score += 20
            elif oi >= 10: score += 10
            if mid < 2: score += 15
            elif mid < 5: score += 5

            score *= confidence

            if score > 0:
                options.append(Option(
                    symbol=sym,
                    underlying=ticker,
                    strike=strike,
                    expiration=c.get("expiration_date", ""),
                    type=direction,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    spread_pct=spread,
                    oi=oi,
                    otm_pct=otm,
                    score=score
                ))

        options.sort(key=lambda x: x.score, reverse=True)
        return options[:5]


# =============================================================================
# MAIN AGENT
# =============================================================================

class Agent:
    """
    Main agent. All state in Alpaca, no local storage.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        llm: LLM = None,
        max_portfolio_risk: float = 0.20,
        max_single_trade: float = 0.05,
        min_confidence: float = 0.65
    ):
        self.alpaca = Alpaca(api_key, api_secret)
        self.risk = RiskManager(self.alpaca, max_portfolio_risk, max_single_trade)
        self.finder = OptionFinder(self.alpaca)
        self.llm = llm or ManualLLM()
        self.min_confidence = min_confidence

    def scan(self) -> List[Dict]:
        """Scan for opportunities"""

        print("""
╔═══════════════════════════════════════════════════════════════╗
║           ASYMMETRIC OPTIONS AGENT                            ║
║   "The market prices the disaster, not the adaptation"        ║
╚═══════════════════════════════════════════════════════════════╝
        """)

        # Show current risk status
        self.risk.print_status()

        # Get news
        logger.info("\nFetching news...")
        news = self.alpaca.get_news(limit=50)
        logger.info(f"Got {len(news)} articles")

        # Analyze
        logger.info("Analyzing for second-order effects...")
        opportunities = self.llm.analyze(news)
        logger.info(f"Found {len(opportunities)} potential opportunities")

        # Find options for each
        results = []
        for opp in opportunities:
            ticker = opp.get("ticker", "")
            if not ticker:
                continue

            if not self.alpaca.is_tradeable(ticker):
                logger.warning(f"{ticker} not tradeable")
                continue

            options = self.finder.find(
                ticker=ticker,
                direction=opp.get("direction", "call"),
                days=opp.get("days", 30),
                confidence=opp.get("confidence", 0.7)
            )

            if options:
                best = options[0]
                opp["options"] = options
                opp["best"] = best
                results.append(opp)

                logger.info(f"\n✓ {ticker} {opp.get('direction', 'call').upper()}")
                logger.info(f"  {opp.get('second_order_effect', opp.get('reasoning', ''))[:60]}")
                logger.info(f"  Best: {best.symbol} @ ${best.mid:.2f}")

        return results

    def execute(self, opportunities: List[Dict]) -> List[Dict]:
        """Execute trades for opportunities that pass risk checks"""

        executed = []

        for opp in opportunities:
            if opp.get("confidence", 0) < self.min_confidence:
                logger.info(f"Skipping {opp['ticker']}: confidence {opp['confidence']:.0%} < {self.min_confidence:.0%}")
                continue

            best: Option = opp.get("best")
            if not best:
                continue

            # Calculate position size
            available = self.risk.get_available_risk()
            cost_per_contract = best.mid * 100

            if cost_per_contract <= 0:
                continue

            qty = min(10, max(1, int(available * 0.5 / cost_per_contract)))  # Use up to 50% of available
            total_cost = qty * cost_per_contract

            # Risk check
            can_trade, reason = self.risk.can_trade(total_cost)
            if not can_trade:
                logger.warning(f"Risk block for {opp['ticker']}: {reason}")
                continue

            # Execute
            logger.info(f"\n{'='*50}")
            logger.info(f"EXECUTING: {opp['ticker']}")
            logger.info(f"Option: {best.symbol}")
            logger.info(f"Qty: {qty} @ ${best.mid:.2f} = ${total_cost:.2f}")
            logger.info(f"{'='*50}")

            try:
                order = self.alpaca.buy_option(best.symbol, qty, best.mid)
                logger.info(f"Order {order.get('id')}: {order.get('status')}")
                executed.append({
                    "ticker": opp["ticker"],
                    "option": best.symbol,
                    "qty": qty,
                    "price": best.mid,
                    "order_id": order.get("id"),
                    "status": order.get("status")
                })
            except Exception as e:
                logger.error(f"Order failed: {e}")

        return executed

    def run(self, execute: bool = False):
        """Full run"""
        opportunities = self.scan()

        if execute and opportunities:
            executed = self.execute(opportunities)
            logger.info(f"\nExecuted {len(executed)} trades")
        else:
            logger.info(f"\nFound {len(opportunities)} opportunities (execution disabled)")

        # Final status
        self.risk.print_status()

        return opportunities


# =============================================================================
# CLAUDE CODE HELPERS - Use these when running in a Claude Code session
# =============================================================================

def get_100x_framework() -> Dict:
    """
    Framework for finding 100x binary event plays.

    The mechanics of 100x returns:
    1. Deep OTM options (Delta < 0.10) = maximum leverage
    2. Binary catalyst = forces instant repricing
    3. Low float + high short interest = liquidity vacuum / squeeze
    4. Gamma expansion = Delta snaps from 0.05 to 1.00 on gap up

    Returns framework for finding these setups.
    """
    today = datetime.now()

    return {
        "theory": {
            "gamma_squeeze": """
                When a stock gaps up 50%+ overnight, deep OTM call Delta doesn't move linearly.
                It SNAPS from 0.05 to 1.00 instantly. A $0.10 option becomes $10.00.
                This requires: binary event + liquidity vacuum + forced covering.
            """,
            "vega_paradox": """
                Normally IV crushes after events. But in squeezes, IV EXPANDS because
                market makers scramble to hedge. This supercharges returns.
            """,
            "the_100x_math": """
                - $0.10 option → $10.00 = 100x
                - Requires stock to gap through strike with velocity
                - Market makers can't hedge gradually = forced repricing
            """
        },

        "ideal_setup": {
            "float": "LOW - Limited shares available for trading",
            "short_interest": "HIGH (>15%) - Guaranteed buyers if price rises",
            "catalyst": "BINARY - Single moment that forces re-evaluation",
            "option_delta": "<0.10 - Deep OTM for maximum leverage",
            "time_to_expiry": "30-60 days - Captures event + buffer for delays"
        },

        "binary_event_types": {
            "fda_pdufa": {
                "description": "FDA drug approval decisions - statutory deadline",
                "why_binary": "Approved = stock doubles/triples. Rejected = stock craters.",
                "search_queries": [
                    f"FDA PDUFA dates {today.strftime('%B %Y')}",
                    f"biotech FDA approval calendar {today.year}",
                    "FDA advisory committee meetings upcoming",
                    "biotech catalyst calendar PDUFA"
                ],
                "signals": {
                    "bullish": ["No AdCom required", "Priority Review", "Breakthrough designation"],
                    "bearish": ["AdCom scheduled", "Complete Response Letter history", "Manufacturing issues"]
                },
                "key_sites": ["biopharmcatalyst.com", "fdatracker.com"]
            },

            "earnings_squeeze": {
                "description": "Earnings + high short interest = squeeze potential",
                "why_binary": "Beat + guidance raise on heavily shorted stock = forced covering",
                "search_queries": [
                    f"high short interest stocks earnings {today.strftime('%B %Y')}",
                    "most shorted stocks earnings this week",
                    f"short squeeze candidates {today.year}"
                ],
                "ideal_setup": "Short interest >20%, beat expectations, raise guidance"
            },

            "regulatory_binary": {
                "description": "Government decisions with defined timelines",
                "examples": [
                    "Crypto legislation votes (CLARITY Act, FIT21)",
                    "EPA/environmental permits",
                    "FTC merger approvals",
                    "Patent decisions"
                ],
                "search_queries": [
                    "crypto regulation vote date congress",
                    "pending merger approvals FTC",
                    f"regulatory decisions calendar {today.strftime('%B %Y')}"
                ]
            },

            "conference_catalyst": {
                "description": "Major announcements at industry conferences",
                "examples": [
                    "CES (Jan) - Tech/AI announcements",
                    "JPM Healthcare (Jan) - Biotech deals",
                    "GTC (Mar) - NVIDIA/AI",
                    "WWDC (Jun) - Apple"
                ],
                "search_queries": [
                    f"CES {today.year} announcements stocks",
                    f"JPM healthcare conference {today.year}",
                    "tech conference calendar stocks"
                ]
            },

            "m_and_a": {
                "description": "Acquisition speculation with defined timelines",
                "why_binary": "Deal closes = premium. Deal breaks = crash.",
                "search_queries": [
                    "pending acquisitions deadline",
                    "merger arbitrage spreads wide",
                    "takeover rumors biotech"
                ]
            }
        },

        "screening_criteria": {
            "must_have": [
                "Defined binary catalyst with SPECIFIC DATE",
                "Options available with strikes 30-50% OTM",
                "Stock price ideally <$50 (cheaper options)",
                "Event within 60 days"
            ],
            "prefer": [
                "Short interest >15%",
                "Float <50M shares",
                "No AdCom (for FDA plays)",
                "Positive precedent (similar drugs approved)",
                "Insider buying"
            ],
            "avoid": [
                "Stocks with only monthly options (illiquid)",
                "Events already heavily hyped",
                "Companies with dilution history post-catalyst"
            ]
        },

        "execution_rules": {
            "entry": "Buy 30-45 days before catalyst, deep OTM (Delta <0.10)",
            "position_size": "Only what you can lose 100% - this IS gambling",
            "exit_win": "Sell INTO the initial spike. Never hold overnight after news.",
            "exit_loss": "If stock drifts pre-catalyst, cut at 50% loss",
            "roll_strategy": "Winner profits fund next catalyst in basket",
            "dilution_warning": "Biotechs often announce offerings post-approval. Exit fast."
        },

        "current_searches": [
            f"PDUFA calendar {today.strftime('%B %Y')} FDA decisions",
            f"biotech catalyst calendar {today.strftime('%B %Y')}",
            f"high short interest earnings {today.strftime('%B %Y')}",
            f"short squeeze candidates {today.year}",
            "unusual options activity biotech",
            f"crypto legislation vote schedule {today.year}",
            f"CES {today.year} stock plays AI",
            f"JPM healthcare conference {today.year} biotech"
        ]
    }


def get_research_framework() -> Dict:
    """
    Get the dynamic research framework.

    This tells Claude WHAT TYPES of things to research - not specific topics.
    The actual searches should be based on current date/context.

    Returns framework with:
    - Research categories and what to look for
    - Current date context
    - Seasonal factors that apply RIGHT NOW
    """
    today = datetime.now()

    framework = {
        "economic_calendar": {
            "goal": "What earnings, Fed events, economic data releases are happening THIS WEEK?",
            "look_for": "Events that could trigger volatility or sector moves",
            "example_searches": [
                f"economic calendar {today.strftime('%B %d %Y')}",
                f"earnings releases this week {today.strftime('%B %Y')}",
                "fed fomc meeting schedule"
            ]
        },
        "geopolitical_now": {
            "goal": "What global events are happening RIGHT NOW that could be obscure catalysts?",
            "look_for": "Conflicts, negotiations, policy changes affecting specific industries",
            "example_searches": [
                "breaking geopolitical news today",
                "trade policy changes this week",
                "international conflict market impact"
            ]
        },
        "weather_anomalies": {
            "goal": "What extreme weather is occurring or forecasted?",
            "look_for": "Events impacting energy, agriculture, insurance, shipping, travel",
            "example_searches": [
                "extreme weather forecast US",
                "polar vortex forecast",
                "hurricane forecast atlantic",
                "drought conditions agriculture"
            ]
        },
        "viral_consumer_trends": {
            "goal": "What's going viral that changes consumer behavior?",
            "look_for": "TikTok trends, viral products, behavioral shifts (like COVID→gifting)",
            "example_searches": [
                "viral tiktok products trending",
                "consumer spending trends new",
                "what's selling out right now"
            ]
        },
        "unusual_options_activity": {
            "goal": "Where is smart money positioning?",
            "look_for": "Unusual volume, high IV, big OTM bets - someone knows something",
            "example_searches": [
                "unusual options activity today",
                "stocks high implied volatility",
                "large options trades this week"
            ]
        },
        "seasonal_patterns": {
            "goal": "What seasonal effects apply RIGHT NOW?",
            "look_for": "Tax effects, holiday spending, cyclical industry patterns",
            "example_searches": [
                f"January effect stocks {today.year}",
                "tax loss harvesting reversal stocks",
                "seasonal stock patterns this month"
            ]
        },
        "sector_flows": {
            "goal": "Where is money rotating TO and FROM?",
            "look_for": "Fund flows, sector ETF movements, institutional positioning",
            "example_searches": [
                "sector rotation this week",
                "etf fund flows",
                "institutional money movement"
            ]
        },
        "supply_chain": {
            "goal": "Any disruptions creating winners/losers?",
            "look_for": "Port issues, shipping delays, component shortages, logistics problems",
            "example_searches": [
                "supply chain disruption news",
                "port congestion delays",
                "shipping container shortage"
            ]
        }
    }

    # Current context
    context = {
        "today": today.strftime("%B %d, %Y"),
        "day_of_week": today.strftime("%A"),
        "this_week": f"week of {today.strftime('%B %d')}",
        "month": today.strftime("%B"),
        "year": str(today.year),
    }

    # What's seasonally relevant RIGHT NOW
    seasonal_factors = []

    # Late December / Early January
    if (today.month == 12 and today.day >= 20) or (today.month == 1 and today.day <= 15):
        seasonal_factors.append({
            "effect": "JANUARY EFFECT",
            "description": "Small caps and beaten-down stocks often rally as tax-loss selling ends",
            "beneficiaries": "Small cap value, stocks that dropped 30%+ in prior year"
        })
        seasonal_factors.append({
            "effect": "NEW YEAR RESOLUTION SPENDING",
            "description": "Surge in fitness, diet, organization, self-improvement spending",
            "beneficiaries": "Gyms (PLNT), fitness equipment (PTON), health food, planners/organization"
        })
        seasonal_factors.append({
            "effect": "TAX-LOSS HARVESTING REVERSAL",
            "description": "Stocks sold in December for tax losses often bounce in January",
            "beneficiaries": "Last year's biggest losers with solid fundamentals"
        })

    # Q4 Earnings preview (early January)
    if today.month == 1 and today.day >= 5:
        seasonal_factors.append({
            "effect": "EARNINGS SEASON PREVIEW",
            "description": "Banks kick off earnings mid-January, sets tone for market",
            "beneficiaries": "Depends on guidance - watch for beats/misses"
        })

    # Summer (May-June)
    if today.month in [5, 6]:
        seasonal_factors.append({
            "effect": "SUMMER DRIVING SEASON",
            "description": "Gas demand rises, travel picks up",
            "beneficiaries": "Refiners, travel/leisure, airlines"
        })

    # Hurricane season (June-November)
    if today.month in [6, 7, 8, 9, 10, 11]:
        seasonal_factors.append({
            "effect": "HURRICANE SEASON",
            "description": "Gulf storms can disrupt oil/gas production, boost rebuilding",
            "beneficiaries": "Home improvement (HD, LOW), generators, insurance (volatility)"
        })

    # Back to school (July-August)
    if today.month in [7, 8]:
        seasonal_factors.append({
            "effect": "BACK TO SCHOOL",
            "description": "Retail spike for supplies, clothes, electronics",
            "beneficiaries": "Target, Walmart, office supplies, laptops"
        })

    # Holiday shopping (Nov-Dec)
    if today.month in [11, 12]:
        seasonal_factors.append({
            "effect": "HOLIDAY SHOPPING SEASON",
            "description": "Retail makes or breaks on holiday sales",
            "beneficiaries": "E-commerce, retail, shipping/logistics (UPS, FDX)"
        })

    return {
        "framework": framework,
        "context": context,
        "seasonal_factors": seasonal_factors,
        "process": """
FOR EACH FRAMEWORK CATEGORY:
1. Do targeted web searches based on current date
2. Look for SPECIFIC, IMMEDIATE catalysts (not general trends)
3. For each catalyst, identify the SECOND-ORDER effect
4. Find the non-obvious beneficiary (not the obvious one)
5. Verify options exist on that ticker

KEY QUESTIONS:
- What is the market pricing in? (obvious first-order reaction)
- What is it MISSING? (second-order consequence)
- Who benefits that nobody is talking about?
"""
    }


def create_agent(api_key: str, api_secret: str) -> Agent:
    """Create agent with Claude Code LLM for use in Claude Code sessions"""
    return Agent(
        api_key=api_key,
        api_secret=api_secret,
        llm=ClaudeCodeLLM()
    )


def add_research(agent: Agent, category: str, findings: str, catalyst: str = "",
                 second_order: str = "", potential_ticker: str = ""):
    """
    Add research findings for a category.

    Args:
        category: Which framework category (e.g., "geopolitical_now")
        findings: What you found
        catalyst: The specific catalyst event
        second_order: The second-order effect you identified
        potential_ticker: Ticker that might benefit
    """
    if not hasattr(agent.llm, 'research'):
        agent.llm.research = []

    agent.llm.research.append({
        "category": category,
        "findings": findings,
        "catalyst": catalyst,
        "second_order": second_order,
        "potential_ticker": potential_ticker,
        "timestamp": datetime.now().isoformat()
    })


def get_research(agent: Agent) -> List[Dict]:
    """Get all research findings"""
    if hasattr(agent.llm, 'research'):
        return agent.llm.research
    return []


def clear_research(agent: Agent):
    """Clear research for fresh analysis"""
    if hasattr(agent.llm, 'research'):
        agent.llm.research = []
    if hasattr(agent.llm, 'opportunities'):
        agent.llm.opportunities = []


def fetch_news(agent: Agent) -> List[Dict]:
    """Fetch news and return for Claude to analyze"""
    news = agent.alpaca.get_news(limit=50)
    agent.llm.pending_news = news
    return agent.llm.get_news_for_analysis()


def add_opportunity(agent: Agent, ticker: str, direction: str = "call",
                   confidence: float = 0.75, days: int = 30,
                   reasoning: str = ""):
    """Add an opportunity that Claude identified"""
    agent.llm.add_opportunity(
        ticker=ticker,
        direction=direction,
        confidence=confidence,
        days=days,
        reasoning=reasoning
    )


def find_options(agent: Agent) -> List[Dict]:
    """Find options for all added opportunities"""
    results = []
    for opp in agent.llm.opportunities:
        ticker = opp.get("ticker")
        if not agent.alpaca.is_tradeable(ticker):
            print(f"⚠ {ticker} not tradeable")
            continue

        options = agent.finder.find(
            ticker=ticker,
            direction=opp.get("direction", "call"),
            days=opp.get("days", 30),
            confidence=opp.get("confidence", 0.7)
        )

        if options:
            opp["options"] = options
            opp["best"] = options[0]
            results.append(opp)

            best = options[0]
            print(f"\n✓ {ticker} {opp.get('direction', 'call').upper()}")
            print(f"  Best: {best.symbol} @ ${best.mid:.2f} ({best.otm_pct*100:.1f}% OTM)")
            print(f"  Spread: {best.spread_pct*100:.1f}% | OI: {best.oi}")

    return results


def execute_trades(agent: Agent, opportunities: List[Dict]) -> List[Dict]:
    """Execute trades for the opportunities"""
    return agent.execute(opportunities)


def show_status(agent: Agent):
    """Show current portfolio status"""
    agent.risk.print_status()


def find_lottery_tickets(agent: Agent, ticker: str, catalyst_date: str,
                         catalyst_description: str, min_otm: float = 0.25,
                         max_otm: float = 0.60) -> List[Dict]:
    """
    Find deep OTM "lottery ticket" options for binary events.

    These are options with:
    - Delta < 0.10 (deep OTM)
    - Cheap premiums ($0.10 - $2.00)
    - Expiration AFTER the catalyst
    - Maximum gamma potential

    Args:
        ticker: Stock symbol
        catalyst_date: Date of binary event (YYYY-MM-DD)
        catalyst_description: What the event is
        min_otm: Minimum OTM % (default 25%)
        max_otm: Maximum OTM % (default 60%)

    Returns:
        List of lottery ticket options with pricing
    """
    price = agent.alpaca.get_price(ticker)
    if price == 0:
        print(f"Cannot get price for {ticker}")
        return []

    print(f"\n{'='*60}")
    print(f"LOTTERY TICKET SCAN: {ticker}")
    print(f"{'='*60}")
    print(f"Current Price: ${price:.2f}")
    print(f"Catalyst: {catalyst_description}")
    print(f"Date: {catalyst_date}")
    print(f"Target: {min_otm*100:.0f}% - {max_otm*100:.0f}% OTM")

    # Find expiration after catalyst
    from datetime import datetime, timedelta
    catalyst = datetime.strptime(catalyst_date, "%Y-%m-%d")
    exp_min = (catalyst + timedelta(days=1)).strftime("%Y-%m-%d")
    exp_max = (catalyst + timedelta(days=45)).strftime("%Y-%m-%d")

    # Strike range for deep OTM
    strike_min = price * (1 + min_otm)
    strike_max = price * (1 + max_otm)

    contracts = agent.alpaca.get_options(
        underlying=ticker,
        expiration_gte=exp_min,
        expiration_lte=exp_max,
        strike_gte=strike_min,
        strike_lte=strike_max,
        option_type="call",
        limit=50
    )

    if not contracts:
        print(f"No options found in range. Trying closer strikes...")
        strike_min = price * 1.10
        strike_max = price * 1.40
        contracts = agent.alpaca.get_options(
            underlying=ticker,
            expiration_gte=exp_min,
            expiration_lte=exp_max,
            strike_gte=strike_min,
            strike_lte=strike_max,
            option_type="call",
            limit=50
        )

    if not contracts:
        print(f"No options available for {ticker}")
        return []

    # Get quotes
    symbols = [c["symbol"] for c in contracts]
    quotes = agent.alpaca.get_option_quotes(symbols)

    results = []
    print(f"\nFound {len(contracts)} contracts:")

    for c in contracts:
        sym = c.get("symbol")
        q = quotes.get(sym, {})

        bid = float(q.get("bp", 0) or 0)
        ask = float(q.get("ap", 0) or 0)
        if not (bid or ask):
            continue

        mid = (bid + ask) / 2 if bid and ask else (bid or ask)
        strike = float(c.get("strike_price", 0))
        exp = c.get("expiration_date", "")
        otm_pct = (strike - price) / price

        # Only want cheap options for lottery tickets
        if mid > 3.00:
            continue

        # Calculate potential return scenarios
        # If stock hits strike at expiry, option worth at least intrinsic
        # If stock goes 2x strike distance, option worth that much
        potential_2x = (strike * 1.5 - strike)  # Stock 50% above strike
        leverage = potential_2x / mid if mid > 0 else 0

        result = {
            "symbol": sym,
            "strike": strike,
            "expiration": exp,
            "otm_pct": otm_pct,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "cost_per_contract": mid * 100,
            "leverage_potential": leverage,
            "catalyst": catalyst_description,
            "catalyst_date": catalyst_date
        }
        results.append(result)

        print(f"\n  {sym}")
        print(f"    Strike: ${strike:.2f} ({otm_pct*100:.0f}% OTM)")
        print(f"    Bid/Ask: ${bid:.2f} / ${ask:.2f} (Mid: ${mid:.2f})")
        print(f"    Cost: ${mid*100:.2f} per contract")
        print(f"    Expiry: {exp}")

    # Sort by cheapest (maximum leverage)
    results.sort(key=lambda x: x["mid"])

    if results:
        best = results[0]
        print(f"\n{'='*60}")
        print(f"BEST LOTTERY TICKET:")
        print(f"  {best['symbol']}")
        print(f"  Cost: ${best['cost_per_contract']:.2f}")
        print(f"  OTM: {best['otm_pct']*100:.0f}%")
        print(f"  If stock hits strike + 50%: ~{best['leverage_potential']:.0f}x potential")
        print(f"{'='*60}")

    return results


def scan_binary_events(agent: Agent, events: List[Dict]) -> List[Dict]:
    """
    Scan multiple binary events and find lottery tickets for each.

    Args:
        events: List of dicts with keys:
            - ticker: Stock symbol
            - catalyst_date: YYYY-MM-DD
            - catalyst: Description
            - allocation: $ to allocate (optional)

    Returns:
        List of best options for each event
    """
    print("\n" + "="*70)
    print("BINARY EVENT BASKET SCANNER")
    print("="*70)

    basket = []

    for event in events:
        ticker = event.get("ticker")
        date = event.get("catalyst_date")
        catalyst = event.get("catalyst")
        alloc = event.get("allocation", 200)

        if not all([ticker, date, catalyst]):
            continue

        options = find_lottery_tickets(agent, ticker, date, catalyst)

        if options:
            best = options[0]
            qty = max(1, int(alloc / best["cost_per_contract"]))
            total_cost = qty * best["cost_per_contract"]

            basket.append({
                "ticker": ticker,
                "catalyst": catalyst,
                "catalyst_date": date,
                "option": best["symbol"],
                "strike": best["strike"],
                "otm_pct": best["otm_pct"],
                "price": best["mid"],
                "qty": qty,
                "total_cost": total_cost
            })

    print("\n" + "="*70)
    print("LOTTERY BASKET SUMMARY")
    print("="*70)

    total = 0
    for b in basket:
        print(f"\n{b['ticker']} - {b['catalyst']} ({b['catalyst_date']})")
        print(f"  Option: {b['option']}")
        print(f"  {b['qty']}x @ ${b['price']:.2f} = ${b['total_cost']:.2f}")
        print(f"  OTM: {b['otm_pct']*100:.0f}%")
        total += b["total_cost"]

    print(f"\nTOTAL BASKET: ${total:.2f}")

    return basket


def execute_lottery_basket(agent: Agent, basket: List[Dict]) -> List[Dict]:
    """
    Execute orders for a lottery ticket basket.

    Args:
        basket: Output from scan_binary_events()

    Returns:
        List of executed orders
    """
    import requests

    executed = []

    print("\n" + "="*70)
    print("EXECUTING LOTTERY BASKET")
    print("="*70)

    for b in basket:
        symbol = b["option"]
        qty = b["qty"]
        price = b["price"]

        # Get fresh quote
        quotes = agent.alpaca.get_option_quotes([symbol])
        q = quotes.get(symbol, {})
        ask = float(q.get("ap", 0) or price * 1.05)

        print(f"\n{b['ticker']}: {symbol}")
        print(f"  Qty: {qty} @ ${ask:.2f}")

        # Encode catalyst in client_order_id for Alpaca persistence
        # Format: CAT|TICKER|DATE (e.g., "CAT|APLD|2026-01-07")
        client_id = f"CAT|{b['ticker']}|{b['catalyst_date']}"

        try:
            order = agent.alpaca.buy_option(symbol, qty, ask, client_order_id=client_id)
            status = order.get("status", "unknown")
            print(f"  Status: {status}")
            print(f"  Catalyst encoded in order: {client_id}")

            # Save catalyst info for tracking (local cache + description)
            add_catalyst(b["ticker"], b["catalyst"], b["catalyst_date"])

            executed.append({
                "ticker": b["ticker"],
                "symbol": symbol,
                "qty": qty,
                "price": ask,
                "status": status,
                "catalyst": b["catalyst"],
                "catalyst_date": b["catalyst_date"]
            })
        except Exception as e:
            print(f"  FAILED: {e}")

    print("\n" + "="*70)
    print(f"EXECUTED: {len(executed)} / {len(basket)} orders")
    print("="*70)

    return executed


# =============================================================================
# PROFIT-TAKING SYSTEM
# =============================================================================

# =============================================================================
# SCALED EXIT STRATEGY
# =============================================================================
#
# Why scaled exits?
# - Fixed targets leave money on table if stock moons (10x+ moves)
# - No targets mean you round-trip gains when it crashes
# - Scaled exits capture BOTH scenarios
#
# The Strategy:
# - Tranche 1 (50%): Sell at 2x → Recovers 100% of investment
# - Tranche 2 (30%): Sell at 4x → Captures extended move
# - Tranche 3 (20%): Let ride with trailing stop → Catches moonshots
#
# Example with 10 contracts @ $1.00 ($1,000 investment):
#   If stock goes to 3x and crashes:
#     - Tranche 1: 5 contracts @ $2.00 = $1,000 (break even!)
#     - Tranche 2: Never hits, expire worthless
#     - Result: $1,000 (0% loss, protected!)
#
#   If stock goes to 10x:
#     - Tranche 1: 5 @ $2.00 = $1,000
#     - Tranche 2: 3 @ $4.00 = $1,200
#     - Tranche 3: 2 @ $10.00 = $2,000
#     - Result: $4,200 (320% gain vs 100% with fixed 2x target!)
#
# =============================================================================

# Scaled exit tranches (percentage of position, target multiplier)
SCALED_EXIT_TRANCHES = {
    # Tranche 1: Recover investment
    1: {"pct": 0.50, "mult": 2.0, "reason": "Recover cost basis"},
    # Tranche 2: Capture extended move
    2: {"pct": 0.30, "mult": 4.0, "reason": "Capture big move"},
    # Tranche 3: Moonshot (trailing stop or very high target)
    3: {"pct": 0.20, "mult": 10.0, "reason": "Moonshot potential"},
}

# Catalyst-specific adjustments to base multipliers
# These modify Tranche 1 target (Tranche 2 = 2x Tranche 1, Tranche 3 = 5x Tranche 1)
CATALYST_ADJUSTMENTS = {
    # FDA plays - slightly higher base due to binary nature
    "fda": 2.5,
    "pdufa": 2.5,
    "approval": 2.5,

    # Earnings plays - standard
    "earnings": 2.0,
    "er": 2.0,

    # Short squeeze plays - can run further, higher base
    "squeeze": 2.5,
    "short": 2.5,

    # Conference catalysts - usually shorter spikes
    "ces": 2.0,
    "conference": 2.0,
    "gtc": 2.0,

    # Weather plays - temporary spikes, take profits faster
    "weather": 1.75,
    "vortex": 1.75,
    "polar": 1.75,

    # Macro/geopolitical - can run longer
    "china": 2.0,
    "silver": 1.75,
    "macro": 1.75,
    "tariff": 2.0,

    # Default
    "default": 2.0
}

# Legacy single-target system (for backwards compatibility)
PROFIT_TARGETS = {
    "fda": 3.0,
    "pdufa": 3.0,
    "approval": 3.0,
    "earnings": 2.5,
    "er": 2.5,
    "squeeze": 2.5,
    "short": 2.5,
    "ces": 2.0,
    "conference": 2.0,
    "gtc": 2.0,
    "weather": 2.0,
    "vortex": 2.0,
    "polar": 2.0,
    "china": 1.75,
    "silver": 1.5,
    "macro": 1.5,
    "tariff": 1.75,
    "default": 2.0
}


def _detect_catalyst_type(symbol: str, notes: str = "") -> str:
    """Detect catalyst type from symbol and notes."""
    text = f"{symbol} {notes}".lower()

    # Check each catalyst type
    for catalyst_type in PROFIT_TARGETS.keys():
        if catalyst_type in text:
            return catalyst_type

    # Try to infer from common patterns
    if any(x in text for x in ["tvtx", "aqst", "fbio", "bio", "pharma", "drug"]):
        return "fda"
    if any(x in text for x in ["apld", "soun", "ai", "tech"]):
        return "earnings"
    if "ung" in text or "gas" in text:
        return "weather"
    if "slv" in text or "mp" in text or "rare" in text:
        return "china"

    return "default"


def get_profit_targets(agent: Agent, catalysts: Dict[str, str] = None) -> List[Dict]:
    """
    Calculate profit targets for all positions.

    Args:
        agent: The trading agent
        catalysts: Optional dict mapping symbols to catalyst descriptions
                   e.g., {"APLD260123C00035000": "Earnings + CES Jan 7"}

    Returns:
        List of positions with profit targets
    """
    positions = agent.alpaca.get_positions()

    if not positions:
        print("No positions to set profit targets for.")
        return []

    catalysts = catalysts or {}
    targets = []

    print("="*70)
    print("PROFIT TARGETS")
    print("="*70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│  LOTTERY TICKET PROFIT RULES                                        │
├─────────────────────────────────────────────────────────────────────┤
│  • FDA plays:     Sell at 3x (dilution risk post-approval)          │
│  • Earnings:      Sell at 2.5x (can reverse quickly)                │
│  • Squeeze:       Sell at 2.5x (shorts can reload)                  │
│  • Conference:    Sell at 2x (news fades fast)                      │
│  • Weather:       Sell at 2x (temporary spike)                      │
│  • Macro/China:   Sell at 1.5-1.75x (can run but lock gains)        │
│                                                                     │
│  NO STOP-LOSSES - prices fluctuate too much, risk selling early     │
│  Let losers expire, capture winners with profit targets             │
└─────────────────────────────────────────────────────────────────────┘
""")

    for p in positions:
        symbol = p.get("symbol", "")
        qty = int(float(p.get("qty", 0)))
        entry_price = float(p.get("avg_entry_price", 0))
        current_price = float(p.get("current_price", 0))
        cost_basis = float(p.get("cost_basis", 0))
        market_value = float(p.get("market_value", 0))
        pnl = float(p.get("unrealized_pl", 0))
        pnl_pct = float(p.get("unrealized_plpc", 0)) * 100

        # Get catalyst notes
        notes = catalysts.get(symbol, "")

        # Detect catalyst type and get multiplier
        catalyst_type = _detect_catalyst_type(symbol, notes)
        multiplier = PROFIT_TARGETS.get(catalyst_type, 2.0)

        # Calculate target
        target_price = round(entry_price * multiplier, 2)
        target_value = target_price * qty * 100
        potential_profit = target_value - cost_basis

        # Status emoji
        if current_price >= target_price:
            status = "🎯 TARGET HIT!"
        elif pnl >= 0:
            status = "🟢 Profitable"
        else:
            status = "🔴 Underwater"

        target_info = {
            "symbol": symbol,
            "qty": qty,
            "entry_price": entry_price,
            "current_price": current_price,
            "target_price": target_price,
            "multiplier": multiplier,
            "catalyst_type": catalyst_type,
            "cost_basis": cost_basis,
            "market_value": market_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "potential_profit": potential_profit,
            "status": status,
            "notes": notes
        }
        targets.append(target_info)

        print(f"\n{status} {symbol}")
        print(f"   Entry: ${entry_price:.2f} | Current: ${current_price:.2f} | Target: ${target_price:.2f} ({multiplier}x)")
        print(f"   Qty: {qty} | P/L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        print(f"   Catalyst: {catalyst_type.upper()} {notes}")
        if current_price >= target_price:
            print(f"   ⚡ SELL NOW - TARGET REACHED!")

    return targets


def monitor_positions(agent: Agent, catalysts: Dict[str, str] = None) -> Dict:
    """
    Monitor all positions against profit targets.

    Returns dict with:
        - targets_hit: positions at or above target
        - profitable: positions with gains but below target
        - underwater: positions with losses
        - action_required: positions that need attention
    """
    targets = get_profit_targets(agent, catalysts)

    result = {
        "targets_hit": [],
        "profitable": [],
        "underwater": [],
        "action_required": []
    }

    for t in targets:
        if t["current_price"] >= t["target_price"]:
            result["targets_hit"].append(t)
            result["action_required"].append(t)
        elif t["pnl"] >= 0:
            result["profitable"].append(t)
        else:
            result["underwater"].append(t)

    # Summary
    print("\n" + "="*70)
    print("MONITORING SUMMARY")
    print("="*70)
    print(f"🎯 Targets Hit:  {len(result['targets_hit'])} - SELL THESE NOW!")
    print(f"🟢 Profitable:   {len(result['profitable'])} - Hold for target")
    print(f"🔴 Underwater:   {len(result['underwater'])} - Hold for catalyst")

    if result["targets_hit"]:
        print("\n⚡ ACTION REQUIRED - SELL THESE:")
        for t in result["targets_hit"]:
            print(f"   {t['symbol']}: Current ${t['current_price']:.2f} >= Target ${t['target_price']:.2f}")

    return result


def place_profit_orders(agent: Agent, catalysts: Dict[str, str] = None,
                        dry_run: bool = True) -> List[Dict]:
    """
    Place limit sell orders at profit targets for all positions.

    Args:
        agent: The trading agent
        catalysts: Optional catalyst descriptions
        dry_run: If True, just show what would be placed (default True for safety)

    Returns:
        List of orders placed/proposed

    Note: Alpaca options only support 'day' orders, not GTC.
          These orders expire at market close and need to be re-placed daily.
    """
    targets = get_profit_targets(agent, catalysts)
    orders = []

    print("\n" + "="*70)
    print("PROFIT-TAKING SELL ORDERS" + (" (DRY RUN)" if dry_run else ""))
    print("="*70)

    if dry_run:
        print("""
⚠️  DRY RUN MODE - No orders will be placed
    Call with dry_run=False to actually place orders

⚠️  ALPACA LIMITATION: Options only support 'day' orders
    These orders expire at market close
    Must re-place each morning before catalyst days
""")

    for t in targets:
        symbol = t["symbol"]
        qty = t["qty"]
        target_price = t["target_price"]
        multiplier = t["multiplier"]

        print(f"\n{symbol}")
        print(f"   SELL {qty}x @ ${target_price:.2f} limit ({multiplier}x entry)")

        if not dry_run:
            try:
                order_data = {
                    "symbol": symbol,
                    "qty": str(qty),
                    "side": "sell",
                    "type": "limit",
                    "time_in_force": "day",
                    "limit_price": str(target_price)
                }

                resp = agent.alpaca._request(
                    "POST",
                    f"{agent.alpaca.base_url}/v2/orders",
                    json=order_data
                )

                if resp.status_code in [200, 201]:
                    result = resp.json()
                    print(f"   ✅ Order placed: {result.get('status')}")
                    orders.append({
                        "symbol": symbol,
                        "qty": qty,
                        "target": target_price,
                        "status": result.get("status"),
                        "order_id": result.get("id")
                    })
                else:
                    error = resp.json().get("message", resp.text)
                    print(f"   ❌ Failed: {error}")
                    orders.append({
                        "symbol": symbol,
                        "qty": qty,
                        "target": target_price,
                        "status": "failed",
                        "error": error
                    })
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            orders.append({
                "symbol": symbol,
                "qty": qty,
                "target": target_price,
                "status": "dry_run"
            })

    return orders


def get_daily_gameplan(agent: Agent, catalysts: Dict[str, Dict] = None) -> Dict:
    """
    Generate a daily action plan based on upcoming catalysts.

    Args:
        agent: The trading agent
        catalysts: Dict mapping symbols to catalyst info:
                   {"SYMBOL": {"description": "...", "date": "2026-01-07"}}

    Returns:
        Dict with today's actions
    """
    from datetime import datetime, timedelta

    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    this_week = today + timedelta(days=7)

    positions = agent.alpaca.get_positions()

    # Load catalysts from storage (or use provided override)
    catalysts = catalysts or load_catalysts()

    if not catalysts:
        print("⚠️  No catalysts stored. Add with: add_catalyst('TICKER', 'description', '2026-01-15')")

    print("="*70)
    print(f"DAILY GAMEPLAN - {today.strftime('%B %d, %Y')}")
    print("="*70)

    gameplan = {
        "today": [],
        "tomorrow": [],
        "this_week": [],
        "later": []
    }

    for p in positions:
        symbol = p.get("symbol", "")

        # Find underlying ticker
        underlying = None
        for ticker in catalysts.keys():
            if ticker in symbol:
                underlying = ticker
                break

        if not underlying:
            continue

        cat = catalysts.get(underlying, {})
        cat_desc = cat.get("description", "Unknown")
        cat_date_str = cat.get("date", "")

        try:
            cat_date = datetime.strptime(cat_date_str, "%Y-%m-%d").date()
        except:
            cat_date = None

        position_info = {
            "symbol": symbol,
            "underlying": underlying,
            "qty": p.get("qty"),
            "catalyst": cat_desc,
            "catalyst_date": cat_date_str,
            "entry_price": float(p.get("avg_entry_price", 0)),
            "current_price": float(p.get("current_price", 0))
        }

        if cat_date:
            if cat_date == today:
                gameplan["today"].append(position_info)
            elif cat_date == tomorrow:
                gameplan["tomorrow"].append(position_info)
            elif cat_date <= this_week:
                gameplan["this_week"].append(position_info)
            else:
                gameplan["later"].append(position_info)
        else:
            # Ongoing catalysts
            gameplan["today"].append(position_info)

    # Print gameplan
    if gameplan["today"]:
        print("\n🔴 TODAY'S CATALYSTS - MONITOR CLOSELY:")
        for p in gameplan["today"]:
            print(f"   {p['underlying']}: {p['catalyst']}")
            print(f"      Option: {p['symbol']}")
            print(f"      Action: Set sell order at target, watch for spike")

    if gameplan["tomorrow"]:
        print("\n🟡 TOMORROW'S CATALYSTS - PLACE SELL ORDERS TODAY:")
        for p in gameplan["tomorrow"]:
            print(f"   {p['underlying']}: {p['catalyst']} ({p['catalyst_date']})")
            print(f"      Option: {p['symbol']}")
            print(f"      Action: Place profit-taking sell order NOW")

    if gameplan["this_week"]:
        print("\n🟢 THIS WEEK'S CATALYSTS - PREPARE:")
        for p in gameplan["this_week"]:
            print(f"   {p['underlying']}: {p['catalyst']} ({p['catalyst_date']})")

    if gameplan["later"]:
        print("\n⚪ LATER CATALYSTS - HOLD:")
        for p in gameplan["later"]:
            print(f"   {p['underlying']}: {p['catalyst']} ({p['catalyst_date']})")

    # Daily checklist
    print("\n" + "="*70)
    print("DAILY CHECKLIST")
    print("="*70)
    print("""
□ Check pre-market for news on catalyst positions
□ Place profit-taking sell orders for today's catalysts
□ Review positions hitting targets (run monitor_positions())
□ Do NOT place stop-losses (prices fluctuate too much)
□ If target hit: SELL INTO THE SPIKE
□ If catalyst fails: Accept loss, move to next opportunity
""")

    return gameplan


def get_scaled_exit_plan(agent: Agent, catalysts: Dict[str, str] = None) -> List[Dict]:
    """
    Calculate a scaled exit plan for all positions.

    Instead of one target, creates 3 tranches:
    - Tranche 1 (50%): Sell at 2x - recovers cost basis
    - Tranche 2 (30%): Sell at 4x - captures big move
    - Tranche 3 (20%): Sell at 10x or let ride - moonshot

    Returns:
        List of exit plans with tranches for each position
    """
    positions = agent.alpaca.get_positions()

    if not positions:
        print("No positions to create exit plan for.")
        return []

    catalysts = catalysts or {}
    plans = []

    print("="*70)
    print("SCALED EXIT PLAN")
    print("="*70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    WHY SCALED EXITS?                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Problem: Fixed 2x target → miss 10x moonshots                      │
│  Problem: No target → round-trip gains when it crashes              │
│                                                                      │
│  Solution: SELL IN TRANCHES                                          │
│                                                                      │
│  Tranche 1 (50%): Sell at 2x                                        │
│    → Recovers 100% of investment                                     │
│    → You're now playing with "house money"                          │
│                                                                      │
│  Tranche 2 (30%): Sell at 4x                                        │
│    → Captures extended move                                          │
│    → Solid profit locked in                                          │
│                                                                      │
│  Tranche 3 (20%): Let ride to 10x+                                  │
│    → Catches moonshots (GME, etc.)                                   │
│    → Small position = limited downside                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
""")

    for p in positions:
        symbol = p.get("symbol", "")
        qty = int(float(p.get("qty", 0)))
        entry_price = float(p.get("avg_entry_price", 0))
        current_price = float(p.get("current_price", 0))
        cost_basis = float(p.get("cost_basis", 0))

        # Get catalyst type for adjustment
        notes = catalysts.get(symbol, "")
        catalyst_type = _detect_catalyst_type(symbol, notes)
        base_mult = CATALYST_ADJUSTMENTS.get(catalyst_type, 2.0)

        # Calculate tranches
        tranches = []
        remaining_qty = qty

        for tranche_num, tranche_config in SCALED_EXIT_TRANCHES.items():
            pct = tranche_config["pct"]
            mult = tranche_config["mult"]
            reason = tranche_config["reason"]

            # Adjust multiplier based on catalyst (scale proportionally)
            if tranche_num == 1:
                adjusted_mult = base_mult
            elif tranche_num == 2:
                adjusted_mult = base_mult * 2  # 2x the base
            else:
                adjusted_mult = base_mult * 5  # 5x the base for moonshot

            # Calculate quantity for this tranche
            tranche_qty = max(1, int(qty * pct))
            if tranche_num == 3:  # Last tranche gets remainder
                tranche_qty = remaining_qty

            remaining_qty -= tranche_qty
            if tranche_qty <= 0:
                continue

            target_price = round(entry_price * adjusted_mult, 2)
            target_value = target_price * tranche_qty * 100

            # Status
            if current_price >= target_price:
                status = "🎯 HIT"
            elif current_price >= target_price * 0.75:
                status = "🟡 CLOSE"
            else:
                status = "⏳ WAIT"

            tranches.append({
                "tranche": tranche_num,
                "qty": tranche_qty,
                "pct": pct,
                "target_mult": adjusted_mult,
                "target_price": target_price,
                "target_value": target_value,
                "reason": reason,
                "status": status
            })

        plan = {
            "symbol": symbol,
            "total_qty": qty,
            "entry_price": entry_price,
            "current_price": current_price,
            "cost_basis": cost_basis,
            "catalyst_type": catalyst_type,
            "tranches": tranches
        }
        plans.append(plan)

        # Print plan
        current_value = current_price * qty * 100
        current_mult = current_price / entry_price if entry_price else 0

        print(f"\n{'='*70}")
        print(f"{symbol}")
        print(f"{'='*70}")
        print(f"Entry: ${entry_price:.2f} | Current: ${current_price:.2f} ({current_mult:.1f}x)")
        print(f"Position: {qty} contracts | Cost: ${cost_basis:.2f} | Value: ${current_value:.2f}")
        print(f"Catalyst: {catalyst_type.upper()}")
        print(f"\nEXIT TRANCHES:")
        print(f"{'─'*70}")

        for t in tranches:
            print(f"  {t['status']} Tranche {t['tranche']}: SELL {t['qty']}x @ ${t['target_price']:.2f} ({t['target_mult']:.1f}x)")
            print(f"       → {t['reason']} | Value: ${t['target_value']:.2f}")

    # Summary
    print("\n" + "="*70)
    print("EXIT STRATEGY SUMMARY")
    print("="*70)
    print("""
WHEN CATALYST HITS:
1. Place Tranche 1 sell orders immediately (50% at 2x)
2. If Tranche 1 fills, place Tranche 2 orders (30% at 4x)
3. Let Tranche 3 ride - this is your moonshot lottery

IF IT MOONS (5x+):
- Tranche 1 already sold - you're playing with house money
- Tranche 2 captures the run
- Tranche 3 could be 10x, 20x, 100x

IF IT CRASHES AFTER SPIKE:
- Tranche 1 recovered your cost basis
- Tranches 2&3 expire worthless but you lost nothing!
""")

    return plans


def place_scaled_exit_orders(agent: Agent, symbol: str = None,
                             tranche: int = None, dry_run: bool = True) -> List[Dict]:
    """
    Place scaled exit orders for positions.

    Args:
        agent: The trading agent
        symbol: Specific symbol (or None for all)
        tranche: Specific tranche to place (1, 2, or 3), or None for all
        dry_run: If True, just show what would be placed

    Returns:
        List of orders placed/proposed
    """
    plans = get_scaled_exit_plan(agent)
    orders = []

    print("\n" + "="*70)
    print(f"PLACING SCALED EXIT ORDERS" + (" (DRY RUN)" if dry_run else ""))
    print("="*70)

    if dry_run:
        print("\n⚠️  DRY RUN - Call with dry_run=False to execute\n")

    for plan in plans:
        pos_symbol = plan["symbol"]

        # Filter by symbol if specified
        if symbol and symbol not in pos_symbol:
            continue

        print(f"\n{pos_symbol}:")

        for t in plan["tranches"]:
            # Filter by tranche if specified
            if tranche and t["tranche"] != tranche:
                continue

            qty = t["qty"]
            target = t["target_price"]

            print(f"  Tranche {t['tranche']}: SELL {qty}x @ ${target:.2f}")

            if not dry_run:
                try:
                    order_data = {
                        "symbol": pos_symbol,
                        "qty": str(qty),
                        "side": "sell",
                        "type": "limit",
                        "time_in_force": "day",
                        "limit_price": str(target)
                    }

                    resp = agent.alpaca._request(
                        "POST",
                        f"{agent.alpaca.base_url}/v2/orders",
                        json=order_data
                    )

                    if resp.status_code in [200, 201]:
                        result = resp.json()
                        print(f"    ✅ Placed: {result.get('status')}")
                        orders.append({
                            "symbol": pos_symbol,
                            "tranche": t["tranche"],
                            "qty": qty,
                            "target": target,
                            "status": "placed",
                            "order_id": result.get("id")
                        })
                    else:
                        error = resp.json().get("message", resp.text)
                        print(f"    ❌ Failed: {error}")
                        orders.append({
                            "symbol": pos_symbol,
                            "tranche": t["tranche"],
                            "qty": qty,
                            "target": target,
                            "status": "failed",
                            "error": error
                        })
                except Exception as e:
                    print(f"    ❌ Error: {e}")
            else:
                orders.append({
                    "symbol": pos_symbol,
                    "tranche": t["tranche"],
                    "qty": qty,
                    "target": target,
                    "status": "dry_run"
                })

    return orders


def moonshot_check(agent: Agent) -> List[Dict]:
    """
    Check if any positions are mooning (5x+) and need attention.

    This is for the "holy shit it's going crazy" scenario.

    Returns:
        List of positions that are mooning
    """
    positions = agent.alpaca.get_positions()
    mooning = []

    print("="*70)
    print("🚀 MOONSHOT CHECK")
    print("="*70)

    for p in positions:
        symbol = p.get("symbol", "")
        entry_price = float(p.get("avg_entry_price", 0))
        current_price = float(p.get("current_price", 0))
        qty = int(float(p.get("qty", 0)))

        if entry_price == 0:
            continue

        mult = current_price / entry_price
        pnl_pct = (mult - 1) * 100

        if mult >= 5:
            status = "🚀🚀🚀 MOONSHOT"
            mooning.append({
                "symbol": symbol,
                "entry": entry_price,
                "current": current_price,
                "mult": mult,
                "qty": qty,
                "action": "SELL SOME NOW - This is rare!"
            })
        elif mult >= 3:
            status = "🔥 ON FIRE"
            mooning.append({
                "symbol": symbol,
                "entry": entry_price,
                "current": current_price,
                "mult": mult,
                "qty": qty,
                "action": "Sell Tranche 1 if not already"
            })
        elif mult >= 2:
            status = "🟢 TARGET HIT"
        elif mult >= 1.5:
            status = "🟡 WARMING UP"
        else:
            status = "⏳ Waiting"

        print(f"\n{status} {symbol}")
        print(f"   Entry: ${entry_price:.2f} → Current: ${current_price:.2f}")
        print(f"   Multiple: {mult:.1f}x | P/L: {pnl_pct:+.0f}%")

    if mooning:
        print("\n" + "="*70)
        print("⚡ ACTION REQUIRED")
        print("="*70)
        for m in mooning:
            print(f"\n{m['symbol']}: {m['mult']:.1f}x")
            print(f"   {m['action']}")
    else:
        print("\nNo moonshots detected. Hold for catalysts.")

    return mooning


def take_profits(agent: Agent, symbol: str = None) -> List[Dict]:
    """
    Immediately sell positions that have hit profit targets.

    Args:
        agent: The trading agent
        symbol: Specific symbol to sell, or None for all targets hit

    Returns:
        List of executed sells
    """
    positions = agent.alpaca.get_positions()
    executed = []

    print("="*70)
    print("TAKING PROFITS")
    print("="*70)

    for p in positions:
        pos_symbol = p.get("symbol", "")

        # Filter to specific symbol if provided
        if symbol and symbol not in pos_symbol:
            continue

        qty = int(float(p.get("qty", 0)))
        entry_price = float(p.get("avg_entry_price", 0))
        current_price = float(p.get("current_price", 0))

        # Get target
        catalyst_type = _detect_catalyst_type(pos_symbol)
        multiplier = PROFIT_TARGETS.get(catalyst_type, 2.0)
        target_price = entry_price * multiplier

        # Check if at or above target
        if current_price >= target_price * 0.95:  # Within 5% of target
            print(f"\n{pos_symbol}")
            print(f"   Current: ${current_price:.2f} | Target: ${target_price:.2f}")
            print(f"   Selling {qty} contracts at market...")

            try:
                # Market sell to ensure execution
                order_data = {
                    "symbol": pos_symbol,
                    "qty": str(qty),
                    "side": "sell",
                    "type": "market",
                    "time_in_force": "day"
                }

                resp = agent.alpaca._request(
                    "POST",
                    f"{agent.alpaca.base_url}/v2/orders",
                    json=order_data
                )

                if resp.status_code in [200, 201]:
                    result = resp.json()
                    print(f"   ✅ SOLD: {result.get('status')}")
                    executed.append({
                        "symbol": pos_symbol,
                        "qty": qty,
                        "price": current_price,
                        "status": result.get("status")
                    })
                else:
                    error = resp.json().get("message", resp.text)
                    print(f"   ❌ Failed: {error}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            pct_to_target = (target_price - current_price) / target_price * 100
            print(f"\n{pos_symbol}: {pct_to_target:.0f}% below target - HOLD")

    if not executed:
        print("\nNo positions at profit target. Hold for catalyst.")

    return executed


# =============================================================================
# AUTONOMOUS DAILY SYSTEM
# =============================================================================

def daily_briefing(agent: Agent, auto_execute: bool = False,
                   scan_new: bool = True) -> Dict:
    """
    COMPLETE AUTONOMOUS DAILY BRIEFING

    Run this once per day for a complete picture:
    1. Portfolio status & P/L
    2. Moonshot check (anything spiking?)
    3. Catalyst calendar (what's happening?)
    4. Position health (scaled exit status)
    5. Risk check (within limits?)
    6. Action items (what to do)

    Args:
        agent: The trading agent
        auto_execute: If True, automatically place recommended orders
        scan_new: If True, show new opportunity suggestions

    Returns:
        Dict with all briefing data and action items
    """
    from datetime import datetime, timedelta

    today = datetime.now()

    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "        ASYMMETRIC OPTIONS AGENT - DAILY BRIEFING".center(68) + "█")
    print("█" + f"        {today.strftime('%A, %B %d, %Y %I:%M %p')}".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    briefing = {
        "timestamp": today.isoformat(),
        "portfolio": {},
        "positions": [],
        "moonshots": [],
        "catalysts": {"today": [], "tomorrow": [], "this_week": [], "later": []},
        "risk": {},
        "actions": []
    }

    # =========================================================================
    # 1. PORTFOLIO STATUS
    # =========================================================================
    print("\n" + "="*70)
    print("📊 PORTFOLIO STATUS")
    print("="*70)

    account = agent.alpaca.get_account()
    equity = float(account.get("equity", 0))
    buying_power = float(account.get("buying_power", 0))

    positions = agent.alpaca.get_positions()
    total_cost = sum(float(p.get("cost_basis", 0)) for p in positions)
    total_value = sum(float(p.get("market_value", 0)) for p in positions)
    total_pnl = sum(float(p.get("unrealized_pl", 0)) for p in positions)
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

    briefing["portfolio"] = {
        "equity": equity,
        "buying_power": buying_power,
        "positions_cost": total_cost,
        "positions_value": total_value,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "exposure_pct": (total_cost / equity * 100) if equity else 0
    }

    print(f"\n  Account Equity:    ${equity:,.2f}")
    print(f"  Buying Power:      ${buying_power:,.2f}")
    print(f"  Positions Cost:    ${total_cost:,.2f}")
    print(f"  Positions Value:   ${total_value:,.2f}")

    pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
    print(f"  Total P/L:         {pnl_emoji} ${total_pnl:+,.2f} ({total_pnl_pct:+.1f}%)")
    print(f"  Portfolio Risk:    {total_cost/equity*100:.1f}% of equity")

    # =========================================================================
    # 2. POSITION DETAILS & MOONSHOT CHECK
    # =========================================================================
    print("\n" + "="*70)
    print("📈 POSITIONS & MOONSHOT CHECK")
    print("="*70)

    # Load catalysts from storage
    catalyst_map = load_catalysts()

    for p in positions:
        symbol = p.get("symbol", "")
        qty = int(float(p.get("qty", 0)))
        entry = float(p.get("avg_entry_price", 0))
        current = float(p.get("current_price", 0))
        pnl = float(p.get("unrealized_pl", 0))
        pnl_pct = float(p.get("unrealized_plpc", 0)) * 100

        mult = current / entry if entry else 0

        # Find catalyst for this position
        cat_info = get_catalyst_for_symbol(symbol)
        underlying = cat_info.get("ticker", symbol[:4]) if cat_info else symbol[:4]  # Extract ticker from symbol
        cat_desc = cat_info.get("description", "Unknown") if cat_info else "Unknown"
        cat_date = cat_info.get("date", "") if cat_info else ""

        # Determine status
        if mult >= 3:
            status = "🚀🚀🚀 MOONSHOT"
            briefing["moonshots"].append({"symbol": symbol, "mult": mult})
            briefing["actions"].append({
                "priority": "HIGH",
                "action": f"SELL SOME {underlying} NOW - at {mult:.1f}x!",
                "reason": "Moonshot - take profits immediately"
            })
        elif mult >= 2:
            status = "🎯 TARGET HIT"
            briefing["actions"].append({
                "priority": "MEDIUM",
                "action": f"Sell Tranche 1 of {underlying}",
                "reason": f"Hit 2x target, currently at {mult:.1f}x"
            })
        elif mult >= 1.5:
            status = "🟡 WARMING UP"
        elif mult >= 1:
            status = "🟢 PROFITABLE"
        else:
            status = "🔴 UNDERWATER"

        pos_data = {
            "symbol": symbol,
            "underlying": underlying,
            "qty": qty,
            "entry": entry,
            "current": current,
            "mult": mult,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "catalyst": cat_desc,
            "catalyst_date": cat_date,
            "status": status
        }
        briefing["positions"].append(pos_data)

        print(f"\n  {status} {symbol}")
        print(f"     Entry: ${entry:.2f} → Current: ${current:.2f} ({mult:.1f}x)")
        print(f"     Qty: {qty} | P/L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        print(f"     Catalyst: {cat_desc} ({cat_date})")

    if not briefing["moonshots"]:
        print("\n  💤 No moonshots detected. Hold for catalysts.")

    # =========================================================================
    # 3. CATALYST CALENDAR
    # =========================================================================
    print("\n" + "="*70)
    print("📅 CATALYST CALENDAR")
    print("="*70)

    today_date = today.date()
    tomorrow_date = today_date + timedelta(days=1)
    week_end = today_date + timedelta(days=7)

    for pos in briefing["positions"]:
        cat_date_str = pos.get("catalyst_date", "")
        try:
            cat_date = datetime.strptime(cat_date_str, "%Y-%m-%d").date()

            if cat_date <= today_date:
                briefing["catalysts"]["today"].append(pos)
            elif cat_date == tomorrow_date:
                briefing["catalysts"]["tomorrow"].append(pos)
            elif cat_date <= week_end:
                briefing["catalysts"]["this_week"].append(pos)
            else:
                briefing["catalysts"]["later"].append(pos)
        except:
            briefing["catalysts"]["today"].append(pos)

    if briefing["catalysts"]["today"]:
        print("\n  🔴 TODAY/ACTIVE - MONITOR CLOSELY:")
        for pos in briefing["catalysts"]["today"]:
            print(f"     • {pos['underlying']}: {pos['catalyst']}")

    if briefing["catalysts"]["tomorrow"]:
        print("\n  🟡 TOMORROW - SET UP ORDERS TODAY:")
        for pos in briefing["catalysts"]["tomorrow"]:
            print(f"     • {pos['underlying']}: {pos['catalyst']}")
            briefing["actions"].append({
                "priority": "HIGH",
                "action": f"Place Tranche 1 sell order for {pos['underlying']}",
                "reason": "Catalyst tomorrow - be prepared"
            })

    if briefing["catalysts"]["this_week"]:
        print("\n  🟢 THIS WEEK:")
        for pos in briefing["catalysts"]["this_week"]:
            try:
                days = (datetime.strptime(pos['catalyst_date'], "%Y-%m-%d").date() - today_date).days
                print(f"     • {pos['underlying']}: {pos['catalyst']} ({days} days)")
            except:
                print(f"     • {pos['underlying']}: {pos['catalyst']}")

    if briefing["catalysts"]["later"]:
        print("\n  ⚪ LATER:")
        for pos in briefing["catalysts"]["later"]:
            print(f"     • {pos['underlying']}: {pos['catalyst']} ({pos['catalyst_date']})")

    # =========================================================================
    # 4. SCALED EXIT STATUS
    # =========================================================================
    print("\n" + "="*70)
    print("🎯 SCALED EXIT STATUS")
    print("="*70)

    for pos in briefing["positions"]:
        entry = pos["entry"]
        current = pos["current"]
        qty = pos["qty"]
        mult = pos["mult"]

        t1_target = entry * 2.0
        t1_pct = (current / t1_target * 100) if t1_target else 0

        t1_status = "✅" if mult >= 2 else f"⏳ {t1_pct:.0f}%"
        t2_status = "✅" if mult >= 4 else "⏳"
        t3_status = "✅" if mult >= 10 else "⏳"

        print(f"\n  {pos['underlying'] or pos['symbol'][:10]}: Currently {mult:.1f}x")
        print(f"     T1 (2x):  ${t1_target:.2f}  {t1_status}")
        print(f"     T2 (4x):  ${entry * 4:.2f}  {t2_status}")
        print(f"     T3 (10x): ${entry * 10:.2f} {t3_status}")

    # =========================================================================
    # 5. RISK CHECK
    # =========================================================================
    print("\n" + "="*70)
    print("⚠️  RISK CHECK")
    print("="*70)

    max_risk = 0.20
    current_risk = total_cost / equity if equity else 0

    if current_risk <= max_risk:
        print(f"\n  ✅ WITHIN LIMITS ({current_risk*100:.1f}% / {max_risk*100:.0f}% max)")
    else:
        print(f"\n  🚨 OVER LIMIT! ({current_risk*100:.1f}% / {max_risk*100:.0f}% max)")

    available = max(0, (max_risk - current_risk) * equity)
    print(f"  Available for new trades: ${available:,.2f}")

    briefing["risk"] = {
        "current": current_risk,
        "max": max_risk,
        "available": available,
        "ok": current_risk <= max_risk
    }

    # =========================================================================
    # 6. ACTION ITEMS
    # =========================================================================
    print("\n" + "="*70)
    print("📋 ACTION ITEMS")
    print("="*70)

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    briefing["actions"].sort(key=lambda x: priority_order.get(x["priority"], 99))

    if briefing["actions"]:
        for action in briefing["actions"]:
            p = action["priority"]
            emoji = "🔴" if p == "HIGH" else "🟡" if p == "MEDIUM" else "⚪"
            print(f"\n  {emoji} [{p}] {action['action']}")
            print(f"     └─ {action['reason']}")
    else:
        print("\n  ✅ No immediate actions. Hold for catalysts.")

    # =========================================================================
    # 7. COMMANDS
    # =========================================================================
    print("\n" + "="*70)
    print("⌨️  COMMANDS")
    print("="*70)
    print("""
  quick_check(agent)              - 30-second status
  get_scaled_exit_plan(agent)     - Detailed exit tranches
  place_scaled_exit_orders(agent, tranche=1, dry_run=False)
  moonshot_check(agent)           - Check for spikes
  end_of_day_review(agent)        - EOD summary
    """)

    print("█"*70 + "\n")

    return briefing


def quick_check(agent: Agent) -> Dict:
    """30-second check - just essentials."""
    print("\n" + "="*50)
    print("⚡ QUICK CHECK")
    print("="*50)

    positions = agent.alpaca.get_positions()
    alerts = []

    for p in positions:
        symbol = p.get("symbol", "")
        entry = float(p.get("avg_entry_price", 0))
        current = float(p.get("current_price", 0))
        mult = current / entry if entry else 0

        if mult >= 3:
            print(f"  🚀 {symbol[:25]}: {mult:.1f}x MOONSHOT!")
            alerts.append("moonshot")
        elif mult >= 2:
            print(f"  🎯 {symbol[:25]}: {mult:.1f}x TARGET")
            alerts.append("target")
        elif mult >= 1.5:
            print(f"  🟡 {symbol[:25]}: {mult:.1f}x warming")
        elif mult >= 1:
            print(f"  🟢 {symbol[:25]}: {mult:.1f}x profit")
        else:
            print(f"  🔴 {symbol[:25]}: {mult:.1f}x underwater")

    if alerts:
        print("\n  ⚡ ACTION NEEDED!")
    else:
        print("\n  ✅ All good")

    print("="*50 + "\n")
    return {"alerts": alerts}


def end_of_day_review(agent: Agent) -> Dict:
    """End of day summary."""
    from datetime import datetime, timedelta

    today = datetime.now()

    print("\n" + "="*70)
    print(f"🌙 END OF DAY - {today.strftime('%B %d, %Y')}")
    print("="*70)

    account = agent.alpaca.get_account()
    equity = float(account.get("equity", 0))

    positions = agent.alpaca.get_positions()
    total_pnl = sum(float(p.get("unrealized_pl", 0)) for p in positions)

    emoji = "🟢" if total_pnl >= 0 else "🔴"
    print(f"\n  {emoji} Day P/L: ${total_pnl:+,.2f}")
    print(f"  Account: ${equity:,.2f}")

    # Tomorrow check - build from stored catalysts
    tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    catalysts = load_catalysts()

    # Build date -> ticker/description map
    tomorrow_catalysts = []
    for ticker, data in catalysts.items():
        if data.get("date") == tomorrow:
            tomorrow_catalysts.append(f"{ticker}: {data.get('description', '')}")

    if tomorrow_catalysts:
        print(f"\n  ⚡ TOMORROW CATALYSTS:")
        for cat in tomorrow_catalysts:
            print(f"     • {cat}")
        print(f"     ➡️  Set up sell orders tonight!")
    else:
        print(f"\n  No major catalysts tomorrow")

    print("="*70 + "\n")
    return {"pnl": total_pnl, "equity": equity}


# =============================================================================
# AUTONOMOUS ACTION SYSTEM
# =============================================================================

def check_actions_needed(agent: Agent) -> Dict:
    """
    Check all positions and return required actions.

    Returns:
        Dict with:
            - moonshots: Positions at 3x+ (URGENT SELL)
            - targets_hit: Positions at 2x+ (Sell tranche 1)
            - warming: Positions at 1.5x+ (Watch closely)
            - catalysts_today: Positions with catalyst today
            - catalysts_tomorrow: Positions with catalyst tomorrow
    """
    positions = agent.alpaca.get_positions()
    catalysts = load_catalysts()

    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    result = {
        "moonshots": [],
        "targets_hit": [],
        "warming": [],
        "underwater": [],
        "catalysts_today": [],
        "catalysts_tomorrow": [],
        "all_positions": []
    }

    for p in positions:
        symbol = p.get("symbol", "")
        entry = float(p.get("avg_entry_price", 0))
        current = float(p.get("current_price", 0))
        qty = int(float(p.get("qty", 0)))
        pnl = float(p.get("unrealized_pl", 0))

        if entry == 0:
            continue

        mult = current / entry

        # Find catalyst info
        cat_info = get_catalyst_for_symbol(symbol)
        cat_date = cat_info.get("date", "") if cat_info else ""
        cat_desc = cat_info.get("description", "Unknown") if cat_info else "Unknown"

        pos_data = {
            "symbol": symbol,
            "qty": qty,
            "entry": entry,
            "current": current,
            "mult": mult,
            "pnl": pnl,
            "catalyst": cat_desc,
            "catalyst_date": cat_date,
            "sell_qty_t1": max(1, int(qty * 0.5)),  # 50% for tranche 1
            "sell_price": current
        }

        result["all_positions"].append(pos_data)

        # Categorize by multiple
        if mult >= 3:
            pos_data["action"] = "SELL NOW - MOONSHOT"
            pos_data["urgency"] = "CRITICAL"
            result["moonshots"].append(pos_data)
        elif mult >= 2:
            pos_data["action"] = f"Sell Tranche 1 ({pos_data['sell_qty_t1']} contracts)"
            pos_data["urgency"] = "HIGH"
            result["targets_hit"].append(pos_data)
        elif mult >= 1.5:
            pos_data["action"] = "Watch closely - approaching target"
            pos_data["urgency"] = "MEDIUM"
            result["warming"].append(pos_data)
        else:
            pos_data["action"] = "Hold for catalyst"
            pos_data["urgency"] = "LOW"
            result["underwater"].append(pos_data)

        # Check catalyst timing
        if cat_date == today:
            result["catalysts_today"].append(pos_data)
        elif cat_date == tomorrow:
            result["catalysts_tomorrow"].append(pos_data)

    return result


def get_research_tasks(agent: Agent) -> List[Dict]:
    """
    Generate specific research tasks based on frameworks and current context.

    Returns list of research tasks, each with:
        - category: Type of research
        - query: Specific search query to run
        - priority: HIGH/MEDIUM/LOW
        - reason: Why this research matters now
    """
    framework_100x = get_100x_framework()
    research_fw = get_research_framework()
    seasonal = research_fw.get("seasonal_factors", [])

    tasks = []

    # Get current searches from 100x framework
    current_searches = framework_100x.get("current_searches", [])
    for i, search in enumerate(current_searches[:4]):
        tasks.append({
            "category": "100x_framework",
            "query": search,
            "priority": "HIGH" if i < 2 else "MEDIUM",
            "reason": "Core binary event scan from 100x framework"
        })

    # Get binary event type searches
    binary_types = framework_100x.get("binary_event_types", {})
    for event_type, data in list(binary_types.items())[:3]:
        queries = data.get("search_queries", [])
        sites = data.get("key_sites", [])
        if queries:
            tasks.append({
                "category": event_type,
                "query": queries[0],
                "priority": "HIGH",
                "reason": data.get("description", ""),
                "sites": sites
            })

    # Add seasonal research if applicable
    if seasonal:
        for s in seasonal[:2]:
            tasks.append({
                "category": "seasonal",
                "query": f"{s['effect']} stocks {datetime.now().strftime('%B %Y')}",
                "priority": "MEDIUM",
                "reason": s.get("description", ""),
                "beneficiaries": s.get("beneficiaries", "")
            })

    # Add research framework categories
    fw_cats = research_fw.get("framework", {})
    for cat_name, cat_data in list(fw_cats.items())[:3]:
        examples = cat_data.get("example_searches", [])
        if examples:
            tasks.append({
                "category": cat_name,
                "query": examples[0],
                "priority": "MEDIUM",
                "reason": cat_data.get("goal", "")
            })

    return tasks


def autonomous_run(agent: Agent) -> Dict:
    """
    AUTONOMOUS AGENT EXECUTION

    This function:
    1. Checks current time and determines mode
    2. Checks all positions for required actions
    3. Generates research tasks if appropriate
    4. Returns structured data for Claude to act on

    Returns:
        Dict with:
            - mode: Current operating mode
            - time_info: Current time details
            - actions: Required position actions (exits, sells)
            - research: Research tasks to perform
            - portfolio: Current portfolio status
            - recommendations: What Claude should do next
    """
    from datetime import datetime, timedelta

    try:
        import pytz
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
    except:
        now = datetime.now()

    hour = now.hour
    minute = now.minute
    time_decimal = hour + minute/60
    weekday = now.weekday()

    # Determine mode
    if weekday >= 5:
        mode = "weekend"
    elif time_decimal < 9.5:
        mode = "premarket"
    elif time_decimal < 11:
        mode = "market_open"
    elif time_decimal < 14:
        mode = "midday"
    elif time_decimal < 15.5:
        mode = "afternoon"
    elif time_decimal < 16:
        mode = "power_hour"
    else:
        mode = "after_hours"

    # Check positions for actions
    actions = check_actions_needed(agent)

    # Get account info
    account = agent.alpaca.get_account()
    equity = float(account.get("equity", 0))
    buying_power = float(account.get("buying_power", 0))

    positions = agent.alpaca.get_positions()
    total_cost = sum(float(p.get("cost_basis", 0)) for p in positions)
    total_pnl = sum(float(p.get("unrealized_pl", 0)) for p in positions)

    available_capital = equity * 0.20 - total_cost
    can_add_positions = available_capital > 200

    # Determine if research mode
    research_modes = ["weekend", "premarket", "midday", "after_hours"]
    research_tasks = []
    if mode in research_modes and can_add_positions:
        research_tasks = get_research_tasks(agent)

    # Build recommendations
    recommendations = []

    # Urgent actions first
    if actions["moonshots"]:
        for pos in actions["moonshots"]:
            recommendations.append({
                "priority": "CRITICAL",
                "action": "SELL",
                "description": f"🚀 MOONSHOT: {pos['symbol']} at {pos['mult']:.1f}x - SELL IMMEDIATELY",
                "details": pos
            })

    if actions["targets_hit"]:
        for pos in actions["targets_hit"]:
            recommendations.append({
                "priority": "HIGH",
                "action": "SELL_TRANCHE_1",
                "description": f"🎯 TARGET HIT: {pos['symbol']} at {pos['mult']:.1f}x - Sell {pos['sell_qty_t1']} contracts",
                "details": pos
            })

    if actions["catalysts_today"]:
        for pos in actions["catalysts_today"]:
            recommendations.append({
                "priority": "HIGH",
                "action": "MONITOR",
                "description": f"⚠️ CATALYST TODAY: {pos['symbol']} - {pos['catalyst']}",
                "details": pos
            })

    if actions["catalysts_tomorrow"]:
        for pos in actions["catalysts_tomorrow"]:
            recommendations.append({
                "priority": "MEDIUM",
                "action": "PREPARE_ORDERS",
                "description": f"📅 CATALYST TOMORROW: {pos['symbol']} - Set up exit orders today",
                "details": pos
            })

    # Research recommendations
    if research_tasks and can_add_positions:
        recommendations.append({
            "priority": "MEDIUM",
            "action": "RESEARCH",
            "description": f"🔬 RESEARCH MODE: {len(research_tasks)} searches to run",
            "details": {"tasks": research_tasks[:5], "available_capital": available_capital}
        })

    return {
        "mode": mode,
        "time_info": {
            "time": now.strftime("%I:%M %p ET"),
            "day": now.strftime("%A"),
            "date": now.strftime("%B %d, %Y")
        },
        "portfolio": {
            "equity": equity,
            "buying_power": buying_power,
            "positions_cost": total_cost,
            "total_pnl": total_pnl,
            "available_for_new": available_capital,
            "can_add": can_add_positions,
            "position_count": len(positions)
        },
        "actions": actions,
        "research_tasks": research_tasks if can_add_positions else [],
        "recommendations": recommendations
    }


# =============================================================================
# SINGLE ENTRY POINT - Just run this!
# =============================================================================

def run(agent: Agent) -> Dict:
    """
    THE ONLY COMMAND NEW USERS NEED TO KNOW.

    Automatically runs the right check based on time of day:
    - Pre-market: What's coming today
    - Market open: Full daily briefing
    - Mid-day: Quick check
    - Power hour: Prep for close
    - After hours: End of day review

    Usage:
        agent = create_agent(API_KEY, API_SECRET)
        run(agent)

    That's it. Run this whenever you check in.
    """
    from datetime import datetime
    import pytz

    # Get Eastern Time (market time)
    try:
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
    except:
        # Fallback if pytz not installed
        now = datetime.now()

    hour = now.hour
    minute = now.minute
    time_decimal = hour + minute/60
    weekday = now.weekday()  # 0=Monday, 6=Sunday

    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "         ASYMMETRIC OPTIONS AGENT".center(68) + "█")
    print("█" + f"         {now.strftime('%A, %B %d %Y %I:%M %p ET')}".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    # Get account info for research suggestions
    account = agent.alpaca.get_account()
    equity = float(account.get("equity", 0))
    positions = agent.alpaca.get_positions()
    position_cost = sum(float(p.get("cost_basis", 0)) for p in positions)
    risk_pct = (position_cost / equity * 100) if equity > 0 else 0
    available_capital = equity * 0.20 - position_cost  # 20% max risk
    can_add_positions = available_capital > 200  # At least $200 for a new play

    # Get catalysts
    catalysts = load_catalysts()
    today_str = now.strftime("%Y-%m-%d")

    # Find catalysts this week we DON'T have positions in
    from datetime import timedelta
    week_from_now = (now + timedelta(days=7)).strftime("%Y-%m-%d")
    held_tickers = set()
    for p in positions:
        sym = p.get("symbol", "")
        for ticker in catalysts.keys():
            if ticker in sym:
                held_tickers.add(ticker)

    upcoming_unheld = []
    for ticker, data in catalysts.items():
        cat_date = data.get("date", "")
        if cat_date >= today_str and cat_date <= week_from_now and ticker not in held_tickers:
            upcoming_unheld.append((ticker, data))

    # Get frameworks for smart suggestions
    framework_100x = get_100x_framework()
    research_fw = get_research_framework()
    seasonal = research_fw.get("seasonal_factors", [])

    # Weekend
    if weekday >= 5:
        print("\n  📅 WEEKEND - Markets Closed")

        result = quick_check(agent)

        print("\n  ─" * 30)
        print("  🔬 WEEKEND RESEARCH SESSION")
        print("  ─" * 30)

        if can_add_positions:
            print(f"\n  💰 Available capital: ${available_capital:,.0f}")

            # Show seasonal factors
            if seasonal:
                print("\n  📅 WHAT'S SEASONALLY RELEVANT NOW:")
                for s in seasonal[:2]:
                    print(f"     • {s['effect']}: {s['description']}")

            # Use actual search queries from framework
            searches = framework_100x.get("current_searches", [])[:3]
            print("\n  🔍 RESEARCH THESE (from 100x framework):")
            for search in searches:
                print(f"     • \"{search}\"")

            print("\n  💡 Tell me:")
            print("     • \"Use the 100x framework to find plays\"")
            print("     • \"Scan for binary events this month\"")
            print("     • \"What's the research framework say to look for?\"")
        else:
            print(f"\n  ⚠️  Near risk limit ({risk_pct:.0f}% / 20% max)")
            print("     Wait for positions to close before adding more")

        print("\n  Also: \"Validate my current positions\" to check thesis")
        print()
        return result

    # Pre-market (before 9:30 AM ET)
    if time_decimal < 9.5:
        print("\n  🌅 PRE-MARKET")
        print("  ─" * 30)
        print("\n  Market opens at 9:30 AM ET")

        # Show catalysts for today
        today_catalysts = [(t, c) for t, c in catalysts.items() if c.get("date") == today_str]

        if today_catalysts:
            print("\n  🔴 CATALYSTS TODAY:")
            for ticker, data in today_catalysts:
                in_portfolio = "✓" if ticker in held_tickers else "✗"
                print(f"     {in_portfolio} {ticker}: {data.get('description', 'Unknown')}")
            print("\n  ⚠️  Watch these closely at open!")
        else:
            print("\n  No catalysts today - standard monitoring day")

        # Research prompt for pre-market - USE FRAMEWORKS
        if can_add_positions:
            print("\n  ─" * 30)
            print(f"  💰 ${available_capital:,.0f} available for new positions")

            # Show seasonal factors from research framework
            if seasonal:
                print("\n  📅 SEASONAL FACTORS ACTIVE NOW:")
                for s in seasonal[:2]:
                    print(f"     • {s['effect']}: {s['description']}")

            # Show dynamic search queries from 100x framework
            searches = framework_100x.get("current_searches", [])[:3]
            if searches:
                print("\n  🔍 PRE-MARKET RESEARCH (from 100x framework):")
                for search in searches:
                    print(f"     • \"{search}\"")

            # Show binary event types to scan
            binary_types = framework_100x.get("binary_event_types", {})
            if binary_types:
                print("\n  🎯 BINARY EVENT CATEGORIES TO SCAN:")
                for event_type, data in list(binary_types.items())[:3]:
                    print(f"     • {event_type.replace('_', ' ').title()}: {data.get('description', '')[:50]}...")

        print("\n  Run again after 9:30 AM for full briefing.")
        print()
        return {"mode": "premarket", "catalysts_today": len(today_catalysts)}

    # Market open / Morning (9:30 AM - 11:00 AM)
    elif time_decimal < 11:
        print("\n  🔔 MARKET OPEN - Full Briefing")
        print("  ─" * 30)

        result = daily_briefing(agent, auto_execute=False, scan_new=False)

        # Add research prompt if we have capacity
        if can_add_positions:
            print("\n  ─" * 30)
            print(f"  💰 CAPACITY: ${available_capital:,.0f} available")
            if upcoming_unheld:
                print("\n  📅 Upcoming catalysts you're NOT in:")
                for ticker, data in upcoming_unheld[:3]:
                    print(f"     • {ticker}: {data.get('description', '')} ({data.get('date', '')})")
                print("\n  Say: \"Add [TICKER] to the basket\" to research options")
            else:
                print("\n  Say: \"Find new lottery tickets\" to scan for opportunities")

        return result

    # Mid-day (11:00 AM - 2:00 PM)
    elif time_decimal < 14:
        print("\n  ☀️ MID-DAY CHECK")
        print("  ─" * 30)

        result = quick_check(agent)

        # Check for any positions near targets
        hot_positions = []
        for p in positions:
            entry = float(p.get("avg_entry_price", 0))
            current = float(p.get("current_price", 0))
            if entry > 0:
                mult = current / entry
                if mult >= 1.8:
                    hot_positions.append((p.get("symbol", ""), mult))

        if hot_positions:
            print("\n  🎯 APPROACHING TARGETS:")
            for sym, mult in hot_positions:
                print(f"     • {sym}: {mult:.1f}x (T1 target: 2.0x)")
            print("\n  Consider: \"Show scaled exit plan\"")

        # Mid-day is good research time - USE FRAMEWORKS
        print("\n  ─" * 30)
        print("  ☕ MID-DAY = RESEARCH TIME")

        if can_add_positions:
            print(f"\n  💰 ${available_capital:,.0f} available for new plays")

            # Show what's seasonally relevant from framework
            if seasonal:
                print(f"\n  📅 SEASONAL EDGE: {seasonal[0]['effect']}")
                print(f"     {seasonal[0]['description']}")
                if seasonal[0].get('beneficiaries'):
                    print(f"     Beneficiaries: {seasonal[0]['beneficiaries'][:60]}...")

            # Get binary event types with their search queries from framework
            binary_types = framework_100x.get("binary_event_types", {})
            print(f"\n  🎯 BINARY EVENT CATEGORIES (from 100x framework):")
            for event_type, data in list(binary_types.items())[:3]:
                print(f"     • {event_type.replace('_', ' ').title()}")
                # Show actual search queries from framework
                queries = data.get("search_queries", [])
                if queries:
                    print(f"       Search: \"{queries[0]}\"")

            # Show research framework categories with their goals
            print(f"\n  🔬 RESEARCH FRAMEWORK CATEGORIES:")
            fw_cats = research_fw.get("framework", {})
            for cat_name, cat_data in list(fw_cats.items())[:3]:
                print(f"     • {cat_name.replace('_', ' ').title()}")
                example = cat_data.get("example_searches", [""])[0]
                if example:
                    print(f"       Search: \"{example}\"")

        else:
            print(f"\n  Portfolio at {risk_pct:.0f}% risk - monitoring existing positions")

        print()
        return result

    # Afternoon (2:00 PM - 3:30 PM)
    elif time_decimal < 15.5:
        print("\n  🌤️ AFTERNOON CHECK")
        print("  ─" * 30)

        result = quick_check(agent)

        print("\n  ⏰ ~1-2 hours until close")
        print("     • Review any positions you want to adjust")
        print("     • Options orders must be placed before 4:00 PM ET")

        # Check for tomorrow's catalysts from stored data
        tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        tomorrow_catalysts = [(t, c) for t, c in catalysts.items() if c.get("date") == tomorrow]
        if tomorrow_catalysts:
            print("\n  📅 TOMORROW'S CATALYSTS - Set up orders today:")
            for ticker, data in tomorrow_catalysts:
                in_portfolio = "✓" if ticker in held_tickers else "✗"
                print(f"     {in_portfolio} {ticker}: {data.get('description', 'Unknown')}")

        # Light research prompt - USE FRAMEWORK
        if can_add_positions:
            print(f"\n  💰 ${available_capital:,.0f} available - last chance to enter today")

            # Show one high-priority binary event type from framework
            binary_types = framework_100x.get("binary_event_types", {})
            if binary_types:
                # Pick first event type and show its search query
                event_type, data = list(binary_types.items())[0]
                queries = data.get("search_queries", [])
                if queries:
                    print(f"\n  🎯 Quick scan idea: \"{queries[0]}\"")

        print()
        return result

    # Power Hour (3:30 PM - 4:00 PM)
    elif time_decimal < 16:
        print("\n  ⚡ POWER HOUR")
        print("  ─" * 30)
        print("  Last 30 minutes! Focus on EXECUTION, not research.")
        print()

        result = quick_check(agent)

        # Show execution rules from framework
        exec_rules = framework_100x.get("execution_rules", {})
        if exec_rules:
            print("\n  📋 EXECUTION RULES (from 100x framework):")
            exit_win = exec_rules.get("exit_win", "")
            if exit_win:
                print(f"     • Exit (win): {exit_win[:60]}...")
            dilution = exec_rules.get("dilution_warning", "")
            if dilution:
                print(f"     • Warning: {dilution[:60]}...")

        print("\n  ⚠️  DECISION TIME:")
        print("     • Any sells to place before close?")
        print("     • Any positions spiking that need attention?")
        print("\n  Say: \"Show scaled exit plan\" to review targets")
        print()
        return result

    # After hours (4:00 PM+)
    else:
        print("\n  🌙 AFTER HOURS - End of Day Review")
        print("  ─" * 30)

        result = end_of_day_review(agent)

        # After hours is prime research time - USE FRAMEWORKS EXTENSIVELY
        print("\n  ─" * 30)
        print("  🔬 EVENING RESEARCH SESSION")
        print("  ─" * 30)

        if can_add_positions:
            print(f"\n  💰 ${available_capital:,.0f} available for tomorrow")

            if upcoming_unheld:
                print("\n  📅 This week's catalysts you could add:")
                for ticker, data in upcoming_unheld[:3]:
                    print(f"     • {ticker}: {data.get('description', '')} ({data.get('date', '')})")

            # Show seasonal factors with beneficiaries
            if seasonal:
                print(f"\n  📅 SEASONAL FACTORS ACTIVE:")
                for s in seasonal[:2]:
                    print(f"     • {s['effect']}: {s['description']}")
                    if s.get('beneficiaries'):
                        print(f"       Beneficiaries: {s['beneficiaries'][:70]}...")

            # Show current search queries from 100x framework
            searches = framework_100x.get("current_searches", [])
            if searches:
                print(f"\n  🎯 100x FRAMEWORK - TONIGHT'S SEARCHES:")
                for search in searches[:4]:
                    print(f"     • \"{search}\"")

            # Show binary event types with their specific search queries
            binary_types = framework_100x.get("binary_event_types", {})
            if binary_types:
                print(f"\n  🔬 BINARY EVENT DEEP DIVES:")
                for event_type, data in list(binary_types.items())[:3]:
                    queries = data.get("search_queries", [])
                    sites = data.get("key_sites", [])
                    print(f"     • {event_type.replace('_', ' ').title()}")
                    if queries:
                        print(f"       Search: \"{queries[0]}\"")
                    if sites:
                        print(f"       Sites: {', '.join(sites[:2])}")

            # Show research framework categories with example searches
            print(f"\n  🔍 RESEARCH FRAMEWORK - TONIGHT'S CATEGORIES:")
            fw_cats = research_fw.get("framework", {})
            for cat_name, cat_data in list(fw_cats.items())[:4]:
                examples = cat_data.get("example_searches", [])
                print(f"     • {cat_name.replace('_', ' ').title()}")
                print(f"       Goal: {cat_data.get('goal', '')[:55]}...")
                if examples:
                    print(f"       Search: \"{examples[0]}\"")

            # Show screening criteria from 100x framework
            screening = framework_100x.get("screening_criteria", {})
            if screening:
                must_have = screening.get("must_have", [])[:2]
                if must_have:
                    print(f"\n  ✅ SCREENING CRITERIA (must have):")
                    for criteria in must_have:
                        print(f"     • {criteria}")

        else:
            print(f"\n  Portfolio at {risk_pct:.0f}% capacity")
            print("  Focus: Monitor existing positions for exits")

            # Still show execution rules for managing positions
            exec_rules = framework_100x.get("execution_rules", {})
            if exec_rules:
                print(f"\n  📋 EXIT REMINDERS (from 100x framework):")
                for rule_name, rule_text in list(exec_rules.items())[:2]:
                    print(f"     • {rule_name.replace('_', ' ').title()}: {rule_text[:50]}...")

        print()
        return result


def help_me(agent: Agent = None):
    """
    Quick help for new users.
    """
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ASYMMETRIC OPTIONS AGENT - HELP                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  JUST STARTING? Run this:                                            ║
║  ─────────────────────────────────────────────────────────────────   ║
║    agent = create_agent(API_KEY, API_SECRET)                         ║
║    run(agent)                                                        ║
║                                                                      ║
║  That's it! run(agent) does the right thing based on time of day.    ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  COMMON COMMANDS (just tell Claude in plain English):                ║
║  ─────────────────────────────────────────────────────────────────   ║
║                                                                      ║
║  "Run the agent"              → Time-appropriate check               ║
║  "Quick check"                → 30-second status                     ║
║  "Full briefing"              → Comprehensive daily report           ║
║  "Show exit plan"             → Scaled exit targets                  ║
║  "Find new plays"             → Research opportunities               ║
║  "List catalysts"             → Show all tracked events              ║
║  "APLD is mooning, sell!"     → Execute exit tranches                ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  DAILY ROUTINE:                                                      ║
║  ─────────────────────────────────────────────────────────────────   ║
║                                                                      ║
║  Morning (9:30 AM)   → run(agent) → Full briefing                    ║
║  Mid-day (12:00 PM)  → run(agent) → Quick check                      ║
║  Afternoon (3:00 PM) → run(agent) → Pre-close check                  ║
║  After hours         → run(agent) → EOD review                       ║
║                                                                      ║
║  On catalyst days: Check more frequently!                            ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  KEY CONCEPTS:                                                       ║
║  ─────────────────────────────────────────────────────────────────   ║
║                                                                      ║
║  Lottery Tickets = Deep OTM options on binary events                 ║
║  Catalyst = The event that triggers the move (FDA, earnings, etc)    ║
║  Scaled Exit = Sell 50% at 2x, 30% at 4x, 20% at 10x                 ║
║  Moonshot = Position up 3x+ (SELL IMMEDIATELY)                       ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  REMEMBER:                                                           ║
║  ─────────────────────────────────────────────────────────────────   ║
║                                                                      ║
║  • 80-90% of lottery tickets expire worthless                        ║
║  • One 10x winner pays for many losers                               ║
║  • NO stop-losses (prices fluctuate too much)                        ║
║  • Sell INTO the spike, don't wait for the peak                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Asymmetric Options Agent")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-secret", required=True)
    parser.add_argument("--execute", action="store_true", help="Execute trades")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama")
    parser.add_argument("--model", default="llama3.2", help="Ollama model")
    parser.add_argument("--research", type=str, help="Path to research file (for Ollama mode)")
    parser.add_argument("--max-risk", type=float, default=0.20, help="Max portfolio risk %")
    parser.add_argument("--max-trade", type=float, default=0.05, help="Max single trade %")
    parser.add_argument("--min-confidence", type=float, default=0.65, help="Min confidence")

    args = parser.parse_args()

    # Load research if provided
    research_text = ""
    if args.research:
        try:
            with open(args.research, 'r') as f:
                research_text = f.read()
            logger.info(f"Loaded research from {args.research} ({len(research_text)} chars)")
        except Exception as e:
            logger.warning(f"Could not load research file: {e}")

    # Setup LLM
    if args.ollama:
        try:
            llm = OllamaLLM(model=args.model, research=research_text)
        except ConnectionError as e:
            logger.error(str(e))
            logger.info("Falling back to manual mode")
            llm = ManualLLM()
    else:
        llm = ManualLLM()

    agent = Agent(
        api_key=args.api_key,
        api_secret=args.api_secret,
        llm=llm,
        max_portfolio_risk=args.max_risk,
        max_single_trade=args.max_trade,
        min_confidence=args.min_confidence
    )

    agent.run(execute=args.execute)


if __name__ == "__main__":
    main()
