"""
Stock Trading Agent using Alpaca-py.

Features (MVP):
1. Load Alpaca API credentials from environment variables:
     ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER (optional, 'true' for paper trading)
2. Provide an interactive CLI for:
     • account  - display account summary (cash, equity, buying power)
     • quote SYM - show latest trade & quote for symbol
     • buy  SYM QTY - market buy order
     • sell SYM QTY - market sell order
     • positions - list open positions
     • help / ? - show commands
     • exit - quit
3. Wrap Alpaca REST calls via `alpaca-py` SDK.
4. Basic error handling & confirmation prompts.

This keeps initial scope small yet useful; additional natural-language
parsing and strategy modules can be added later.
"""
from __future__ import annotations

import os
import sys
import readline
from decimal import Decimal, InvalidOperation
from typing import Any, Dict

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import re
import requests
import atexit 
from Agents.helpers.symbol_fetcher import SymbolFetcher
from Agents.helpers.conditional_trading_agent import ConditionalTradingAgent
from Agents.helpers.color_utils import error, success, warning, info, highlight, bold

load_dotenv()


def _init_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit(error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in environment."))
    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    return TradingClient(key, secret, paper=paper)


def _init_data_client() -> StockHistoricalDataClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    return StockHistoricalDataClient(key, secret)


class TradingAgentCLI:
    """Interactive CLI around Alpaca TradingClient."""

    def __init__(self) -> None:
        self.trading = _init_trading_client()
        self.data = _init_data_client()
        
        # Initialize symbol fetcher with Alpaca credentials
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_SECRET_KEY")
        paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        self.symbol_fetcher = SymbolFetcher(api_key, api_secret, paper=paper)
        
        # Load/update company symbols from Alpaca API
        self.company_symbols = self._load_company_symbols()
        
        # Initialize conditional trading agent
        self.conditional_agent = ConditionalTradingAgent(
            trading_client=self.trading,
            data_client=self.data,
            symbol_fetcher=self.symbol_fetcher
        )
        
        # Configure readline for better command line experience
        self._setup_readline()

    def _load_company_symbols(self) -> Dict[str, str]:
        """Load company name to symbol mapping from Alpaca API."""
        try:
            print("Loading symbol mappings from Alpaca API...")
            # This will either load from cache or fetch fresh data from Alpaca
            symbols = self.symbol_fetcher.update_symbols(force_refresh=False)
            print(f"Loaded {len(symbols)} symbol mappings")
            return symbols
        except Exception as e:
            print(warning(f"Could not load symbols from Alpaca: {e}"))
            # Fallback to basic mapping if API fails
            return {
                'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'amazon': 'AMZN',
                'tesla': 'TSLA', 'meta': 'META', 'facebook': 'META', 'nvidia': 'NVDA',
                'netflix': 'NFLX', 'disney': 'DIS', 'coca cola': 'KO', 'coke': 'KO',
                'mcdonalds': 'MCD', 'starbucks': 'SBUX', 'walmart': 'WMT', 'target': 'TGT',
                'home depot': 'HD', 'lowes': 'LOW', 'bank of america': 'BAC', 'jpmorgan': 'JPM',
                'goldman sachs': 'GS', 'wells fargo': 'WFC', 'visa': 'V', 'mastercard': 'MA',
                'paypal': 'PYPL', 'salesforce': 'CRM', 'adobe': 'ADBE', 'intel': 'INTC',
                'amd': 'AMD', 'qualcomm': 'QCOM', 'cisco': 'CSCO', 'oracle': 'ORCL',
                'ibm': 'IBM', 'at&t': 'T', 'verizon': 'VZ', 'comcast': 'CMCSA',
                'spotify': 'SPOT', 'uber': 'UBER', 'lyft': 'LYFT', 'airbnb': 'ABNB',
                'zoom': 'ZM', 'snapchat': 'SNAP', 'pinterest': 'PINS', 'square': 'SQ',
                'robinhood': 'HOOD', 'coinbase': 'COIN', 'bitcoin': 'BTCUSD', 'ethereum': 'ETHUSD',
                'iphone': 'AAPL', 'iphones': 'AAPL', 'mac': 'AAPL', 'macbook': 'AAPL', 'ipad': 'AAPL',
                'windows': 'MSFT', 'xbox': 'MSFT', 'office': 'MSFT', 'alphabet': 'GOOGL',
                'youtube': 'GOOGL', 'aws': 'AMZN', 'spacex': 'TSLA', 'instagram': 'META',
                'twitter': 'META', 'slack': 'CRM', 'block': 'SQ', 'crypto': 'BTCUSD'
            }

    def _setup_readline(self) -> None:
        """Configure readline for command history and arrow key navigation."""
        try:
            # Enable command history
            readline.set_history_length(1000)
            
            # Try to load existing history file
            history_file = os.path.expanduser("~/.alpaca_agent_history")
            try:
                readline.read_history_file(history_file)
            except FileNotFoundError:
                pass  # No history file exists yet
            
            # Save history on exit
            atexit.register(readline.write_history_file, history_file)
            
        except Exception as e:
            # Readline might not be available on all systems
            print(warning(f"Command history not available: {e}"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _show_account(self) -> None:
        try:
            acct = self.trading.get_account()
            print("\n===== Account Summary =====")
            print(f"Cash         : {acct.cash}")
            print(f"Portfolio Val: {acct.portfolio_value}")
            print(f"Equity       : {acct.equity}")
            print(f"Buying Power : {acct.buying_power}\n")
        except Exception as e:
            print(error(f"Error getting account info: {e}"))

    def _get_buying_power(self) -> Decimal:
        """Get current buying power as Decimal."""
        try:
            acct = self.trading.get_account()
            return Decimal(str(acct.buying_power))
        except Exception as e:
            print(warning(f"Could not retrieve buying power: {e}"))
            return Decimal('0')

    def _show_positions(self, symbol: str = None) -> None:
        try:
            if symbol:
                # Show position for specific symbol
                try:
                    position = self.trading.get_open_position(symbol.upper())
                    print(f"\n===== Position for {symbol.upper()} =====")
                    print(f"Quantity     : {position.qty}")
                    print(f"Side         : {position.side}")
                    print(f"Market Value : ${position.market_value}")
                    print(f"Cost Basis   : ${position.cost_basis}")
                    print(f"Avg Price    : ${position.avg_entry_price}")
                    print(f"Unrealized P/L: ${position.unrealized_pl} ({position.unrealized_plpc}%)")
                    print(f"Today's P/L  : ${position.unrealized_intraday_pl} ({position.unrealized_intraday_plpc}%)\n")
                except Exception:
                    print(f"No position found for {symbol.upper()}")
            else:
                # Show all positions
                positions = self.trading.get_all_positions()
                if not positions:
                    print("No open positions.")
                    return
                print("\n===== Open Positions =====")
                for p in positions:
                    print(f"{p.symbol:5} {p.side:4} {p.qty:>6} @ {p.avg_entry_price} P/L: {p.unrealized_pl}")
                print()
        except Exception as e:
            print(error(f"Error getting positions: {e}"))

    def _show_quote(self, symbol: str) -> None:
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol.upper())
            q = self.data.get_stock_latest_quote(request)[symbol.upper()]
            t_req = StockLatestTradeRequest(symbol_or_symbols=symbol.upper())
            t = self.data.get_stock_latest_trade(t_req)[symbol.upper()]
            print(f"\n{symbol.upper()} Quote - Bid {q.bid_price} x {q.bid_size} | Ask {q.ask_price} x {q.ask_size}")
            print(f"Last Trade   - Price {t.price} Size {t.size} at {t.timestamp}\n")
        except Exception as e:
            print(error(f"Error getting quote for {symbol.upper()}: {e}"))

    def _market_order(self, side: OrderSide, symbol: str, qty: Decimal) -> None:
        # Determine if this is a crypto symbol and set appropriate time_in_force
        symbol_upper = symbol.upper()
        is_crypto = symbol_upper.endswith('USD') and len(symbol_upper) > 3
        
        # For crypto, use GTC (Good Till Canceled), for stocks use DAY
        time_in_force = TimeInForce.GTC if is_crypto else TimeInForce.DAY
        
        # Get current quote for confirmation
        current_price = "Unknown"
        estimated_cost = None
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol_upper)
            quote = self.data.get_stock_latest_quote(request)[symbol_upper]
            if side == OrderSide.BUY:
                current_price = f"${quote.ask_price}"
                estimated_cost = qty * Decimal(str(quote.ask_price))
            else:
                current_price = f"${quote.bid_price}"
                estimated_cost = qty * Decimal(str(quote.bid_price))
        except:
            pass
        
        # Get buying power for validation (only for buy orders)
        buying_power_warning = ""
        if side == OrderSide.BUY and estimated_cost:
            buying_power = self._get_buying_power()
            if buying_power > 0 and estimated_cost > buying_power:
                buying_power_warning = f"\n{warning('WARNING:')} Estimated cost (${estimated_cost:,.2f}) exceeds buying power (${buying_power:,.2f})"
        
        # Show order confirmation
        order_type = "crypto" if is_crypto else "stock"
        print(f"\n{info('Order Confirmation:')}")
        print(f"   Type: Market {side.value.upper()}")
        print(f"   Symbol: {symbol_upper}")
        print(f"   Quantity: {qty}")
        print(f"   Current Price: {current_price}")
        if estimated_cost:
            print(f"   Estimated Cost: ${estimated_cost:,.2f}")
        print(f"   Time in Force: {time_in_force.value}")
        print(f"   Asset Type: {order_type}")
        if buying_power_warning:
            print(buying_power_warning)
        
        confirm = input(f"\n{highlight('Confirm')} {side.value.upper()} {qty} shares of {symbol_upper} at market price? [y/N]: ").strip().lower()
        
        if confirm != 'y':
            print(error("Order cancelled."))
            return
        
        order = MarketOrderRequest(
            symbol=symbol_upper,
            qty=str(qty),
            side=side,
            time_in_force=time_in_force,
        )
        try:
            o = self.trading.submit_order(order)
            print(success(f"Submitted {side.value.upper()} order: {o.qty} {o.symbol} (id {o.id})"))
        except Exception as e:
            print(error(f"Order failed: {e}"))

    def _limit_order(self, side: OrderSide, symbol: str, qty: Decimal, price: Decimal) -> None:
        """Submit a limit order."""
        symbol_upper = symbol.upper()
        is_crypto = symbol_upper.endswith('USD') and len(symbol_upper) > 3
        time_in_force = TimeInForce.GTC if is_crypto else TimeInForce.DAY
        
        # Get current quote for reference
        current_price = "Unknown"
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol_upper)
            quote = self.data.get_stock_latest_quote(request)[symbol_upper]
            if side == OrderSide.BUY:
                current_price = f"${quote.ask_price}"
            else:
                current_price = f"${quote.bid_price}"
        except:
            pass
        
        # Calculate potential order value
        order_value = qty * price
        order_type = "crypto" if is_crypto else "stock"
        
        # Get buying power for validation (only for buy orders)
        buying_power_warning = ""
        if side == OrderSide.BUY:
            buying_power = self._get_buying_power()
            if buying_power > 0 and order_value > buying_power:
                buying_power_warning = f"\n{warning('WARNING:')} Order value (${order_value:,.2f}) exceeds buying power (${buying_power:,.2f})"
        
        # Show order confirmation
        print(f"\n{info('Order Confirmation:')}")
        print(f"   Type: Limit {side.value.upper()}")
        print(f"   Symbol: {symbol_upper}")
        print(f"   Quantity: {qty}")
        print(f"   Limit Price: ${price}")
        print(f"   Current Price: {current_price}")
        print(f"   Order Value: ${order_value:,.2f}")
        print(f"   Time in Force: {time_in_force.value}")
        print(f"   Asset Type: {order_type}")
        if buying_power_warning:
            print(buying_power_warning)
        
        confirm = input(f"\n{highlight('Confirm')} {side.value.upper()} {qty} shares of {symbol_upper} at limit ${price}? [y/N]: ").strip().lower()
        
        if confirm != 'y':
            print(error("Order cancelled."))
            return
        
        order = LimitOrderRequest(
            symbol=symbol_upper,
            qty=str(qty),
            side=side,
            time_in_force=time_in_force,
            limit_price=str(price)
        )
        try:
            o = self.trading.submit_order(order)
            print(success(f"Submitted {side.value.upper()} limit order: {o.qty} {o.symbol} @ ${price} (id {o.id})"))
        except Exception as e:
            print(error(f"Limit order failed: {e}"))

    def _stop_order(self, side: OrderSide, symbol: str, qty: Decimal, stop_price: Decimal) -> None:
        """Submit a stop order."""
        symbol_upper = symbol.upper()
        is_crypto = symbol_upper.endswith('USD') and len(symbol_upper) > 3
        time_in_force = TimeInForce.GTC if is_crypto else TimeInForce.DAY
        
        # Get current quote for reference
        current_price = "Unknown"
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol_upper)
            quote = self.data.get_stock_latest_quote(request)[symbol_upper]
            if side == OrderSide.BUY:
                current_price = f"${quote.ask_price}"
            else:
                current_price = f"${quote.bid_price}"
        except:
            pass
        
        # Calculate potential order value (estimate)
        order_value = qty * stop_price
        order_type = "crypto" if is_crypto else "stock"
        
        # Get buying power for validation (only for buy orders)
        buying_power_warning = ""
        if side == OrderSide.BUY:
            buying_power = self._get_buying_power()
            if buying_power > 0 and order_value > buying_power:
                buying_power_warning = f"\n{warning('WARNING:')} Estimated order value (${order_value:,.2f}) exceeds buying power (${buying_power:,.2f})"
        
        # Show order confirmation
        print(f"\n{info('Order Confirmation:')}")
        print(f"   Type: Stop {side.value.upper()}")
        print(f"   Symbol: {symbol_upper}")
        print(f"   Quantity: {qty}")
        print(f"   Stop Price: ${stop_price}")
        print(f"   Current Price: {current_price}")
        print(f"   Estimated Value: ${order_value:,.2f}")
        print(f"   Time in Force: {time_in_force.value}")
        print(f"   Asset Type: {order_type}")
        if buying_power_warning:
            print(buying_power_warning)
        
        confirm = input(f"\n{highlight('Confirm')} {side.value.upper()} {qty} shares of {symbol_upper} with stop at ${stop_price}? [y/N]: ").strip().lower()
        
        if confirm != 'y':
            print(error("Order cancelled."))
            return
        
        order = StopOrderRequest(
            symbol=symbol_upper,
            qty=str(qty),
            side=side,
            time_in_force=time_in_force,
            stop_price=str(stop_price)
        )
        try:
            o = self.trading.submit_order(order)
            print(success(f"Submitted {side.value.upper()} stop order: {o.qty} {o.symbol} stop @ ${stop_price} (id {o.id})"))
        except Exception as e:
            print(error(f"Stop order failed: {e}"))

    def _show_orders(self) -> None:
        """Show all open orders."""
        try:
            request = GetOrdersRequest(status="open")
            orders = self.trading.get_orders(request)
            if not orders:
                print("No open orders.")
                return
            print("\n===== Open Orders =====")
            # Store orders for easy reference by index
            self._current_orders = orders
            for i, order in enumerate(orders):
                order_type = order.order_type.value if hasattr(order, 'order_type') else 'market'
                price_info = f" @ ${order.limit_price}" if hasattr(order, 'limit_price') and order.limit_price else ""
                stop_info = f" stop @ ${order.stop_price}" if hasattr(order, 'stop_price') and order.stop_price else ""
                order_id_str = str(order.id)[:8]  # Short ID for display
                print(f"[{i+1}] {order_id_str} {order.symbol:5} {order.side.value:4} {order.qty} {order_type}{price_info}{stop_info}")
            print("\nTo cancel: use 'cancel [number]' or 'cancel [full_order_id]'")
            print()
        except Exception as e:
            print(error(f"Error getting orders: {e}"))

    def _cancel_order(self, order_identifier: str) -> None:
        """Cancel an order by ID or index number."""
        try:
            # Check if it's a number (index)
            if order_identifier.isdigit():
                index = int(order_identifier) - 1
                if hasattr(self, '_current_orders') and 0 <= index < len(self._current_orders):
                    order = self._current_orders[index]
                    self.trading.cancel_order_by_id(str(order.id))
                    print(success(f"Cancelled order {str(order.id)[:8]} ({order.symbol} {order.side.value} {order.qty})"))
                else:
                    print(error("Invalid order number. Use 'orders' to see current orders."))
            else:
                # Try to use as full UUID
                self.trading.cancel_order_by_id(order_identifier)
                print(success(f"Cancelled order {order_identifier}"))
        except Exception as e:
            print(error(f"Error cancelling order: {e}"))
            if "badly formed hexadecimal UUID string" in str(e):
                print(info("Tip: Use 'orders' to see order numbers, then 'cancel [number]'"))

    def _cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            # Get all open orders
            request = GetOrdersRequest(status="open")
            orders = self.trading.get_orders(request)
            
            if not orders:
                print(info("No open orders to cancel."))
                return
            
            # Show orders that will be cancelled
            print(f"\n{info(f'Found {len(orders)} open orders:')}")
            for i, order in enumerate(orders, 1):
                order_type = order.order_type.value if hasattr(order, 'order_type') else 'market'
                price_info = f" @ ${order.limit_price}" if hasattr(order, 'limit_price') and order.limit_price else ""
                stop_info = f" stop @ ${order.stop_price}" if hasattr(order, 'stop_price') and order.stop_price else ""
                order_id_str = str(order.id)[:8]  # Short ID for display
                print(f"   {i}. {order_id_str} {order.symbol:5} {order.side.value:4} {order.qty} {order_type}{price_info}{stop_info}")
            
            # Confirm cancellation
            confirm = input(f"\n{highlight('Cancel ALL')} {len(orders)} orders? [y/N]: ").strip().lower()
            
            if confirm != 'y':
                print(error("Bulk cancellation aborted."))
                return
            
            # Cancel all orders
            cancelled_count = 0
            failed_count = 0
            
            print(f"\n{info('Cancelling')} {len(orders)} orders...")
            
            for order in orders:
                try:
                    self.trading.cancel_order_by_id(str(order.id))
                    print(success(f"Cancelled {str(order.id)[:8]} ({order.symbol} {order.side.value} {order.qty})"))
                    cancelled_count += 1
                except Exception as e:
                    print(error(f"Failed to cancel {str(order.id)[:8]}: {e}"))
                    failed_count += 1
            
            # Summary
            print(f"\n{info('Cancellation Summary:')}")
            print(f"   {success('Successfully cancelled:')} {cancelled_count}")
            if failed_count > 0:
                print(f"   {error('Failed to cancel:')} {failed_count}")
            print(f"   {info('Total processed:')} {cancelled_count + failed_count}")
            
        except Exception as e:
            print(error(f"Error during bulk cancellation: {e}"))

    def _interactive_cancel_menu(self) -> None:
        """Interactive menu for cancelling orders."""
        try:
            # Get all open orders
            request = GetOrdersRequest(status="open")
            orders = self.trading.get_orders(request)
            
            if not orders:
                print(info("No open orders to cancel."))
                return
            
            print(f"\n{info(f'Cancel Orders Menu - {len(orders)} open orders:')}")
            print("=" * 50)
            
            # Show orders with numbers
            for i, order in enumerate(orders, 1):
                order_type = order.order_type.value if hasattr(order, 'order_type') else 'market'
                price_info = f" @ ${order.limit_price}" if hasattr(order, 'limit_price') and order.limit_price else ""
                stop_info = f" stop @ ${order.stop_price}" if hasattr(order, 'stop_price') and order.stop_price else ""
                order_id_str = str(order.id)[:8]  # Short ID for display
                print(f"   {i}. {order_id_str} {order.symbol:5} {order.side.value:4} {order.qty} {order_type}{price_info}{stop_info}")
            
            print(f"\nOptions:")
            print(f"   • Enter number (1-{len(orders)}) to cancel specific order")
            print(f"   • Type 'all' to cancel all orders")
            print(f"   • Type 'exit' to return to main menu")
            
            while True:
                choice = input(f"\ncancel> ").strip().lower()
                
                if choice in ['exit', 'quit', 'back']:
                    break
                elif choice == 'all':
                    self._cancel_all_orders()
                    break
                elif choice.isdigit():
                    order_num = int(choice)
                    if 1 <= order_num <= len(orders):
                        self._cancel_order(str(order_num))
                        # Refresh the orders list and continue
                        request = GetOrdersRequest(status="open")
                        orders = self.trading.get_orders(request)
                        if not orders:
                            print(info("No more open orders."))
                            break
                    else:
                        print(error(f"Invalid order number. Choose 1-{len(orders)}"))
                else:
                    print(error("Invalid choice. Enter a number, 'all', or 'exit'"))
            
        except Exception as e:
            print(error(f"Error in cancel menu: {e}"))

    # ------------------------------------------------------------------
    # Enhanced FinancialModelingPrep helpers
    # ------------------------------------------------------------------
    def _get_company_info(self, symbol: str) -> Dict[str, Any] | None:
        """Get company profile information."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            print(warning("FMP_API_KEY not set - limited functionality"))
            return None
            
        url = f"https://financialmodelingprep.com/api/v3/profile/{symbol.upper()}?apikey={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data[0] if isinstance(data, list) and data else None
        except Exception as e:
            print(warning(f"Could not fetch company info: {e}"))
            return None

    def _get_financial_metrics(self, symbol: str) -> Dict[str, Any] | None:
        """Get key financial metrics."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            return None
            
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol.upper()}?apikey={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data[0] if isinstance(data, list) and data else None
        except Exception as e:
            print(warning(f"Could not fetch financial metrics: {e}"))
            return None

    def _show_company_profile(self, symbol: str) -> None:
        """Display comprehensive company information."""
        print(f"\n===== {symbol.upper()} Company Profile =====")
        
        # Get quote first
        self._show_quote(symbol)
        
        # Get company info
        company = self._get_company_info(symbol)
        if company:
            print(f"Company      : {company.get('companyName', 'N/A')}")
            print(f"Sector       : {company.get('sector', 'N/A')}")
            print(f"Industry     : {company.get('industry', 'N/A')}")
            print(f"Market Cap   : ${company.get('mktCap', 0):,.0f}")
            print(f"Description  : {company.get('description', 'N/A')[:200]}...")
        
        # Get financial metrics
        metrics = self._get_financial_metrics(symbol)
        if metrics:
            print(f"\n===== Key Metrics =====")
            print(f"P/E Ratio    : {metrics.get('peRatio', 'N/A')}")
            print(f"EPS          : ${metrics.get('eps', 'N/A')}")
            print(f"ROE          : {metrics.get('roe', 'N/A')}")
            print(f"Debt/Equity  : {metrics.get('debtToEquity', 'N/A')}")
        print()

    def _screen_stocks(self, criteria: str) -> None:
        """Screen stocks based on criteria."""
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            print(warning("FMP_API_KEY required for stock screening"))
            return
            
        if "gainer" in criteria.lower():
            url = f"https://financialmodelingprep.com/api/v3/gainers?apikey={api_key}"
        elif "loser" in criteria.lower():
            url = f"https://financialmodelingprep.com/api/v3/losers?apikey={api_key}"
        elif "active" in criteria.lower():
            url = f"https://financialmodelingprep.com/api/v3/actives?apikey={api_key}"
        else:
            print("Available screens: gainers, losers, actives")
            return
            
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                print(f"\n===== {criteria.title()} =====")
                for i, stock in enumerate(data[:10]):  # Top 10
                    # Safely convert price and percentage to float, handle string values
                    try:
                        price = float(stock.get('price', 0))
                        change_pct = float(stock.get('changesPercentage', 0))
                        ticker = stock.get('ticker', 'N/A')
                        print(f"{i+1:2}. {ticker:5} ${price:>8.2f} {change_pct:>+7.2f}%")
                    except (ValueError, TypeError):
                        # Fallback for invalid data
                        ticker = stock.get('ticker', 'N/A')
                        price = stock.get('price', 'N/A')
                        change_pct = stock.get('changesPercentage', 'N/A')
                        print(f"{i+1:2}. {ticker:5} ${str(price):>8} {str(change_pct):>+7}%")
                print()
        except Exception as e:
            print(error(f"Error screening stocks: {e}"))

    def _get_company_display_name(self, symbol: str) -> str:
        """Get a human-readable company name for display purposes."""
        try:
            # Try to get company info first
            company_info = self._get_company_info(symbol)
            if company_info and company_info.get('companyName'):
                return company_info['companyName']
            
            # Fallback to searching our symbol mappings in reverse
            for company_name, sym in self.company_symbols.items():
                if sym == symbol and len(company_name) > 3:  # Prefer longer names
                    return company_name.title()
            
            # Last resort: just return the symbol
            return symbol
        except:
            return symbol

    # ------------------------------------------------------------------
    # Enhanced natural language processing
    # ------------------------------------------------------------------
    def _handle_nl_prompt(self, prompt: str) -> None:
        """Enhanced natural language command parser."""
        lower = prompt.lower()
        
        # Check for conditional trading patterns first
        conditional_keywords = ['if', 'when', 'maintain', 'rebalance', 'top', 'gainers', 'losers', 
                               'divided equally', 'equal value', 'equal allocation', 'condition']
        
        if any(keyword in lower for keyword in conditional_keywords):
            print(info("Detected conditional trading command, analyzing..."))
            trading_plan = self.conditional_agent.parse_conditional_command(prompt)
            
            if trading_plan:
                success = self.conditional_agent.execute_plan(trading_plan, confirm=True)
                if success:
                    print(success("Conditional trading plan executed successfully"))
                else:
                    print(error("Conditional trading plan execution failed"))
                return
            else:
                print(error("Could not parse conditional trading command"))
                print(info("Try commands like:"))
                print("   • 'buy AAPL if price is greater than $120'")
                print("   • 'buy the top 5 gainers divided equally amongst $50,000'")
                print("   • 'maintain equal value for all positions in the portfolio'")
                return
        
        # Extract potential symbols, quantities, and prices
        # Improved symbol pattern - look for 1-5 uppercase letters that are likely stock symbols
        symbol_pattern = r'\b[A-Z]{1,5}\b'
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        
        # Find all potential symbols
        all_symbols = re.findall(symbol_pattern, prompt.upper())
        # Filter out common English words that might be mistaken for symbols
        common_words = {'GIVE', 'ME', 'GET', 'BUY', 'SELL', 'THE', 'AND', 'OR', 'FOR', 'TO', 'OF', 'IN', 'ON', 'AT', 'BY', 'UP', 'DO', 'GO', 'NO', 'SO', 'MY', 'WE', 'HE', 'SHE', 'IT', 'BE', 'IS', 'ARE', 'WAS', 'WERE', 'BEEN', 'HAVE', 'HAS', 'HAD', 'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'MIGHT', 'MUST', 'CAN', 'ABOUT', 'WHAT', 'WHO', 'WHERE', 'WHEN', 'WHY', 'HOW', 'WHICH', 'THAT', 'THIS', 'THESE', 'THOSE', 'SOME', 'ANY', 'ALL', 'EACH', 'EVERY', 'BOTH', 'EITHER', 'NEITHER', 'MORE', 'MOST', 'LESS', 'LEAST', 'MUCH', 'MANY', 'FEW', 'LITTLE', 'GOOD', 'BEST', 'BETTER', 'BAD', 'WORST', 'WORSE', 'NEW', 'OLD', 'FIRST', 'LAST', 'NEXT', 'SAME', 'OTHER', 'ANOTHER', 'SUCH', 'LIKE', 'JUST', 'ONLY', 'ALSO', 'EVEN', 'STILL', 'ALREADY', 'YET', 'SOON', 'NOW', 'THEN', 'HERE', 'THERE', 'WHERE', 'VERY', 'TOO', 'QUITE', 'RATHER', 'ALMOST', 'NEARLY', 'HARDLY', 'BARELY', 'EXACTLY', 'PROBABLY', 'PERHAPS', 'MAYBE', 'CERTAINLY', 'DEFINITELY', 'SURELY', 'REALLY', 'TRULY', 'ACTUALLY', 'INDEED', 'CLEARLY', 'OBVIOUSLY', 'APPARENTLY', 'FORTUNATELY', 'UNFORTUNATELY', 'GENERALLY', 'USUALLY', 'NORMALLY', 'TYPICALLY', 'BASICALLY', 'ESSENTIALLY', 'MAINLY', 'MOSTLY', 'LARGELY', 'PARTLY', 'PARTLY', 'COMPLETELY', 'TOTALLY', 'ENTIRELY', 'ABSOLUTELY', 'PERFECTLY', 'EXACTLY', 'PRECISELY', 'SPECIFICALLY', 'PARTICULARLY', 'ESPECIALLY', 'NAMELY', 'INCLUDING', 'EXCLUDING', 'EXCEPT', 'BESIDES', 'APART', 'ASIDE', 'ALONG', 'ACROSS', 'THROUGH', 'THROUGHOUT', 'WITHIN', 'WITHOUT', 'INSIDE', 'OUTSIDE', 'ABOVE', 'BELOW', 'UNDER', 'OVER', 'BEHIND', 'FRONT', 'BACK', 'LEFT', 'RIGHT', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'TOP', 'BOTTOM', 'SIDE', 'EDGE', 'CENTER', 'MIDDLE', 'AROUND', 'BETWEEN', 'AMONG', 'AGAINST', 'TOWARD', 'TOWARDS', 'AWAY', 'NEAR', 'CLOSE', 'FAR', 'DISTANT', 'BEFORE', 'AFTER', 'DURING', 'SINCE', 'UNTIL', 'WHILE', 'ALTHOUGH', 'THOUGH', 'UNLESS', 'BECAUSE', 'SINCE', 'ORDER', 'ORDERS', 'LIMIT', 'MARKET', 'STOP', 'STOCK', 'STOCKS', 'SHARE', 'SHARES', 'PRICE', 'PRICES', 'INFO', 'INFORMATION', 'PROFILE', 'COMPANY', 'SCREEN', 'FIND', 'SHOW', 'DISPLAY', 'LIST', 'CANCEL', 'POSITION', 'POSITIONS', 'ACCOUNT', 'QUOTE', 'QUOTES', 'GAINER', 'GAINERS', 'LOSER', 'LOSERS', 'ACTIVE', 'ACTIVES', 'MOVED', 'MOST'}
        
        symbols = [s for s in all_symbols if s not in common_words]
        numbers = [Decimal(n) for n in re.findall(number_pattern, prompt)]
        
        # Check for company names in the prompt using dynamic symbol fetcher
        detected_company_symbol = None
        potential_matches = []
        
        # First check our loaded symbols cache with improved matching
        # Sort by length (longest first) to prioritize specific matches over general ones
        sorted_companies = sorted(self.company_symbols.items(), key=lambda x: len(x[0]), reverse=True)
        
        for company, symbol in sorted_companies:
            # Use word boundaries to avoid partial matches like "sh" in "shares"
            pattern = r'\b' + re.escape(company) + r'\b'
            if re.search(pattern, lower, re.IGNORECASE):
                # Check if this is an exact match or just the first good match
                if not detected_company_symbol:
                    detected_company_symbol = symbol
                    matched_company = company
                # Collect similar matches for potential disambiguation
                if len(potential_matches) < 10:  # Limit to 10 options
                    potential_matches.append((company, symbol))
        
        # If we found multiple potential matches, let user choose
        if len(potential_matches) > 1:
            # Check if the matches are for different symbols (ambiguous)
            unique_symbols = set(symbol for _, symbol in potential_matches)
            if len(unique_symbols) > 1:
                print(f"\n{highlight('Found multiple possible matches:')}")
                
                # Group by symbol and show company info
                symbol_groups = {}
                for company, symbol in potential_matches[:10]:
                    if symbol not in symbol_groups:
                        symbol_groups[symbol] = []
                    symbol_groups[symbol].append(company)
                
                options = []
                for i, (symbol, companies) in enumerate(symbol_groups.items(), 1):
                    display_name = self._get_company_display_name(symbol)
                    print(f"  {i}. {symbol} - {display_name}")
                    options.append(symbol)
                
                choice = input(f"\nWhich stock did you mean? (enter number 1-{len(options)}, symbol, or 0 to cancel): ").strip()
                
                if choice == "0":
                    print(error("Order cancelled."))
                    return
                elif choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(options):
                        detected_company_symbol = options[choice_num - 1]
                        print(highlight(f"Selected {detected_company_symbol}"))
                    else:
                        print(error("Invalid selection."))
                        return
                elif choice.upper() in unique_symbols:
                    detected_company_symbol = choice.upper()
                    print(highlight(f"Selected {detected_company_symbol}"))
                else:
                    print(error("Invalid selection."))
                    return
            else:
                # All matches point to the same symbol, use the first one
                print(highlight(f"Found symbol {detected_company_symbol} for '{matched_company}'"))
        elif detected_company_symbol:
            print(highlight(f"Found symbol {detected_company_symbol} for '{matched_company}'"))
        
        # If not found in cache, try dynamic search with Alpaca API
        if not detected_company_symbol:
            # First try to identify common company descriptions
            company_descriptions = {
                'iphone': 'AAPL',
                'apple': 'AAPL',
                'iphones': 'AAPL',
                'mac': 'AAPL',
                'macbook': 'AAPL',
                'ipad': 'AAPL',
                'iphone maker': 'AAPL',
                'iphone company': 'AAPL',
                'company that makes iphones': 'AAPL',
                'company that makes iphone': 'AAPL',
                'microsoft': 'MSFT',
                'windows': 'MSFT',
                'xbox': 'MSFT',
                'office': 'MSFT',
                'google': 'GOOGL',
                'alphabet': 'GOOGL',
                'youtube': 'GOOGL',
                'amazon': 'AMZN',
                'aws': 'AMZN',
                'tesla': 'TSLA',
                'spacex': 'TSLA',  # Tesla CEO also runs SpaceX
                'meta': 'META',
                'facebook': 'META',
                'instagram': 'META',
                'nvidia': 'NVDA',
                'netflix': 'NFLX',
                'disney': 'DIS',
                'coca cola': 'KO',
                'coca-cola': 'KO',
                'coke': 'KO',
                'mcdonalds': 'MCD',
                'starbucks': 'SBUX',
                'walmart': 'WMT',
                'target': 'TGT',
                'home depot': 'HD',
                'lowes': 'LOW',
                'bank of america': 'BAC',
                'jpmorgan': 'JPM',
                'jp morgan': 'JPM',
                'goldman sachs': 'GS',
                'wells fargo': 'WFC',
                'visa': 'V',
                'mastercard': 'MA',
                'paypal': 'PYPL',
                'salesforce': 'CRM',
                'adobe': 'ADBE',
                'intel': 'INTC',
                'amd': 'AMD',
                'qualcomm': 'QCOM',
                'cisco': 'CSCO',
                'oracle': 'ORCL',
                'ibm': 'IBM',
                'at&t': 'T',
                'verizon': 'VZ',
                'comcast': 'CMCSA',
                'netflix': 'NFLX',
                'spotify': 'SPOT',
                'uber': 'UBER',
                'lyft': 'LYFT',
                'airbnb': 'ABNB',
                'zoom': 'ZM',
                'slack': 'CRM',  # Now part of Salesforce
                'twitter': 'META',  # Now X, but no public stock
                'snapchat': 'SNAP',
                'pinterest': 'PINS',
                'square': 'SQ',
                'block': 'SQ',  # Square renamed to Block
                'robinhood': 'HOOD',
                'coinbase': 'COIN',
                'bitcoin': 'BTCUSD',
                'ethereum': 'ETHUSD',
                'crypto': 'BTCUSD'
            }
            
            # Try to match company descriptions first
            for desc, symbol in company_descriptions.items():
                if desc in lower:
                    detected_company_symbol = symbol
                    print(highlight(f"Found symbol {symbol} for '{desc}'"))
                    break
            
            # If no description match, try individual words
            if not detected_company_symbol:
                # Extract potential company names (words that aren't numbers or common words)
                words = re.findall(r'\b[a-zA-Z]+\b', lower)
                company_words = [w for w in words if w not in ['buy', 'sell', 'purchase', 'order', 'limit', 'market', 'stop', 'stock', 'share', 'shares', 'at', 'the', 'and', 'or', 'for', 'to', 'of', 'in', 'on', 'with', 'company', 'that', 'makes', 'makes', 'shares', 'shares']]
                
                # Try to find multiple matches from API search
                api_matches = []
                for word in company_words:
                    try:
                        symbol = self.symbol_fetcher.search_symbol(word)
                        if symbol and (word, symbol) not in api_matches:
                            api_matches.append((word, symbol))
                            if len(api_matches) >= 5:  # Limit API searches
                                break
                    except Exception as e:
                        continue  # Continue searching other words
            
            if len(api_matches) == 1:
                detected_company_symbol = api_matches[0][1]
                print(highlight(f"Found symbol {detected_company_symbol} for '{api_matches[0][0]}'"))
            elif len(api_matches) > 1:
                print(f"\n{highlight('Found multiple possible matches from search:')}")
                unique_symbols = set(symbol for _, symbol in api_matches)
                
                if len(unique_symbols) > 1:
                    for i, (word, symbol) in enumerate(api_matches, 1):
                        display_name = self._get_company_display_name(symbol)
                        print(f"  {i}. {symbol} - {display_name}")
                    
                    choice = input(f"\nWhich stock did you mean? (enter number 1-{len(api_matches)}, symbol, or 0 to cancel): ").strip()
                    
                    if choice == "0":
                        print(error("Order cancelled."))
                        return
                    elif choice.isdigit():
                        choice_num = int(choice)
                        if 1 <= choice_num <= len(api_matches):
                            detected_company_symbol = api_matches[choice_num - 1][1]
                            print(highlight(f"Selected {detected_company_symbol}"))
                        else:
                            print(error("Invalid selection."))
                            return
                    elif choice.upper() in unique_symbols:
                        detected_company_symbol = choice.upper()
                        print(highlight(f"Selected {detected_company_symbol}"))
                    else:
                        print(error("Invalid selection."))
                        return
                else:
                    # All API matches point to same symbol
                    detected_company_symbol = api_matches[0][1]
                    print(highlight(f"Found symbol {detected_company_symbol} for '{api_matches[0][0]}'"))
        
        # If we found a company name, add it to our symbols list
        if detected_company_symbol and detected_company_symbol not in symbols:
            symbols.insert(0, detected_company_symbol)
        
        # Buy/Sell with natural language
        if ("buy" in lower or "purchase" in lower):
            symbol = symbols[0] if symbols else None
            qty = numbers[0] if numbers else None
            
            if not symbol:
                print(error("I couldn't identify which stock you want to buy."))
                print("Try: 'buy AAPL 100' or 'buy 100 Apple stock'")
                return
            
            if "at market" in lower or "market order" in lower:
                if not qty:
                    qty_input = input(f"How many shares of {symbol}? ").strip()
                    try:
                        qty = Decimal(qty_input)
                    except:
                        print("Invalid quantity.")
                        return
                self._market_order(OrderSide.BUY, symbol, qty)
                
            elif "limit" in lower and len(numbers) >= 2:
                qty = numbers[0]
                price = numbers[1]
                self._limit_order(OrderSide.BUY, symbol, qty, price)
                
            elif "moved the most" in lower or "top gainer" in lower:
                symbol = self._top_gainer()
                if symbol:
                    qty = numbers[0] if numbers else Decimal(input("Quantity: "))
                    confirm = input(f"Buy {qty} shares of {symbol}? [y/N]: ").strip().lower()
                    if confirm == "y":
                        self._market_order(OrderSide.BUY, symbol, qty)
                return
            else:
                # Default to market order
                if not qty:
                    qty_input = input(f"How many shares of {symbol}? ").strip()
                    try:
                        qty = Decimal(qty_input)
                    except:
                        print("Invalid quantity.")
                        return
                self._market_order(OrderSide.BUY, symbol, qty)
                
        elif ("sell" in lower or "short" in lower):
            symbol = symbols[0] if symbols else None
            qty = numbers[0] if numbers else None
            
            if not symbol:
                print(error("I couldn't identify which stock you want to sell."))
                print("Try: 'sell AAPL 100' or 'sell 100 Apple stock'")
                return
                
            if not qty:
                qty_input = input(f"How many shares of {symbol}? ").strip()
                try:
                    qty = Decimal(qty_input)
                except:
                    print("Invalid quantity.")
                    return
            
            if "at market" in lower:
                self._market_order(OrderSide.SELL, symbol, qty)
            elif "limit" in lower and len(numbers) >= 2:
                price = numbers[1]
                self._limit_order(OrderSide.SELL, symbol, qty, price)
            else:
                self._market_order(OrderSide.SELL, symbol, qty)
                
        # Information requests
        elif ("info" in lower or "profile" in lower or "about" in lower) and symbols:
            self._show_company_profile(symbols[0])
            
        elif ("quote" in lower or "price" in lower) and symbols:
            self._show_quote(symbols[0])
            
        elif "position" in lower and symbols:
            self._show_positions(symbols[0])
            
        # Market screening
        elif "screen" in lower or "find" in lower:
            if "gainer" in lower:
                self._screen_stocks("gainers")
            elif "loser" in lower:
                self._screen_stocks("losers")
            elif "active" in lower:
                self._screen_stocks("actives")
            else:
                print("Try: 'screen gainers', 'screen losers', or 'screen actives'")
                
        # Order management
        elif "cancel" in lower:
            if "all" in lower:
                # Cancel all orders
                self._cancel_all_orders()
            elif numbers:
                # If there's a number, try to cancel by index
                self._cancel_order(str(int(numbers[0])))
            elif "order" in lower:
                self._show_orders()
                order_id = input("Enter order number or full ID to cancel: ").strip()
                if order_id:
                    self._cancel_order(order_id)
            else:
                self._interactive_cancel_menu()
                    
        # Fallback
        else:
            print("I understand commands like:")
            print("• 'buy AAPL 100' or 'buy AAPL' (will ask for quantity)")
            print("• 'buy 100 Apple stock' or 'buy Apple stock'")
            print("• 'sell 50 TSLA limit 200'")
            print("• 'info about NVDA'")
            print("• 'screen gainers'")
            print("• 'position MSFT'")
            print("Or use traditional commands - type 'help' for list.")

    def loop(self) -> None:
        print(info("Alpaca Trading Agent. Type 'help' for commands."))
        self._print_help()  # Display help on startup
        while True:
            try:
                cmd = input("alpaca> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not cmd:
                continue
            
            try:
                parts = cmd.split()
                action = parts[0].lower()

                if action in {"exit", "quit"}:
                    break
                elif action in {"help", "?"}:
                    self._print_help()
                elif action == "account":
                    self._show_account()
                elif action == "positions":
                    if len(parts) == 2:
                        self._show_positions(parts[1])  # Show position for specific symbol
                    else:
                        self._show_positions()  # Show all positions
                elif action == "orders":
                    self._show_orders()
                elif action == "cancel" and len(parts) == 2:
                    if parts[1].lower() == "all":
                        self._cancel_all_orders()
                    else:
                        self._cancel_order(parts[1])
                elif action == "cancel" and len(parts) == 1:
                    # Show interactive cancel menu
                    self._interactive_cancel_menu()
                elif action == "quote" and len(parts) == 2:
                    self._show_quote(parts[1])
                elif action == "info" and len(parts) == 2:
                    self._show_company_profile(parts[1])
                elif action == "screen" and len(parts) == 2:
                    self._screen_stocks(parts[1])
                elif action == "refresh":
                    self._refresh_symbols()
                elif action == "conditional":
                    # Interactive conditional trading mode
                    self._interactive_conditional_mode()
                elif action == "monitor":
                    # Monitor conditions (placeholder for future enhancement)
                    print(info("Condition monitoring not yet implemented"))
                elif action == "buy" and len(parts) == 2:
                    # Handle "buy SYMBOL" without quantity
                    symbol = parts[1]
                    qty_input = input(f"How many shares of {symbol.upper()}? ").strip()
                    try:
                        qty = Decimal(qty_input)
                        self._market_order(OrderSide.BUY, symbol, qty)
                    except (ValueError, InvalidOperation):
                        print("Invalid quantity. Please enter a number.")
                elif action == "buy" and len(parts) == 3:
                    self._market_order(OrderSide.BUY, parts[1], Decimal(parts[2]))
                elif action == "buy" and len(parts) == 5 and parts[3].lower() == "limit":
                    # buy SYMBOL QTY limit PRICE
                    self._limit_order(OrderSide.BUY, parts[1], Decimal(parts[2]), Decimal(parts[4]))
                elif action == "sell" and len(parts) == 2:
                    # Handle "sell SYMBOL" without quantity
                    symbol = parts[1]
                    qty_input = input(f"How many shares of {symbol.upper()}? ").strip()
                    try:
                        qty = Decimal(qty_input)
                        self._market_order(OrderSide.SELL, symbol, qty)
                    except (ValueError, InvalidOperation):
                        print("Invalid quantity. Please enter a number.")
                elif action == "sell" and len(parts) == 3:
                    self._market_order(OrderSide.SELL, parts[1], Decimal(parts[2]))
                elif action == "sell" and len(parts) == 5 and parts[3].lower() == "limit":
                    # sell SYMBOL QTY limit PRICE
                    self._limit_order(OrderSide.SELL, parts[1], Decimal(parts[2]), Decimal(parts[4]))
                else:
                    # Try natural language handler
                    self._handle_nl_prompt(cmd)
            except Exception as e:
                print(error(f"Error processing command '{cmd}': {e}"))
                print("Type 'help' for available commands.")

    def _refresh_symbols(self) -> None:
        """Refresh symbol mappings from Alpaca API."""
        try:
            print(info("Refreshing symbol mappings from Alpaca API..."))
            symbols = self.symbol_fetcher.update_symbols(force_refresh=True)
            self.company_symbols = symbols
            print(success(f"Successfully refreshed {len(symbols)} symbol mappings"))
        except Exception as e:
            print(error(f"Error refreshing symbols: {e}"))

    def _print_help(self) -> None:
        print(
            """
Commands:
  account                 Show account summary
  positions [SYMBOL]      List open positions (all or specific symbol)
  orders                  List open orders
  cancel                  Interactive cancel menu
  cancel ORDER_NUM        Cancel an order by number (from 'orders' list)
  cancel ORDER_ID         Cancel an order by full UUID
  cancel all              Cancel ALL open orders
  quote SYMBOL            Show latest quote & trade for symbol
  info SYMBOL             Show company profile and metrics
  screen TYPE             Screen stocks (gainers/losers/actives)
  refresh                 Update symbol mappings from Alpaca API
  conditional             Interactive conditional trading mode
  
Trading Commands:
  buy SYMBOL QTY          Submit market buy order
  buy SYMBOL QTY limit PRICE    Submit limit buy order
  sell SYMBOL QTY         Submit market sell order  
  sell SYMBOL QTY limit PRICE   Submit limit sell order
  
Conditional Trading Examples:
  'buy AAPL if price is greater than $120'
  'buy the top 5 gainers divided equally amongst $50,000'
  'maintain equal value for all positions in the portfolio'
  'sell all positions if SPY drops below $400'
  
Natural Language Examples:
  'buy 100 AAPL at market'
  'sell 50 TSLA limit 200'
  'info about NVDA'
  'screen gainers'
  'position MSFT'
  
  help / ?                Show this message
  exit / quit             Quit agent

Note: Crypto symbols ending in 'USD' (e.g., BTCUSD, ETHUSD) use GTC time-in-force.
Set FMP_API_KEY environment variable for enhanced market data.
Set ANTHROPIC_API_KEY for advanced conditional trading with LLM parsing.
"""
        )

    def _top_gainer(self) -> str | None:
        """Return symbol of top gainer of the day using FMP endpoint."""
        api_key = os.getenv("FMP_API_KEY")
        url = "https://financialmodelingprep.com/api/v3/gainers"
        if api_key:
            url += f"?apikey={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]["ticker"]
        except Exception as e:
            print(warning(f"Could not fetch gainers list: {e}"))
        return None

    # ------------------------------------------------------------------
    # Interactive conditional trading mode
    # ------------------------------------------------------------------
    def _interactive_conditional_mode(self) -> None:
        """Interactive mode for building conditional trading commands"""
        print(f"\n{info('Conditional Trading Mode')}")
        print("=" * 50)
        print("Build sophisticated trading conditions with natural language.")
        print("Type 'examples' to see sample commands, 'back' to return to main menu.\n")
        
        while True:
            try:
                cmd = input("conditional> ").strip()
                if not cmd:
                    continue
                    
                if cmd.lower() in ['back', 'exit', 'quit']:
                    break
                elif cmd.lower() == 'examples':
                    self._show_conditional_examples()
                    continue
                
                # Parse and execute conditional command
                print(info("Analyzing conditional command..."))
                trading_plan = self.conditional_agent.parse_conditional_command(cmd)
                
                if trading_plan:
                    success = self.conditional_agent.execute_plan(trading_plan, confirm=True)
                    if success:
                        print(success("Conditional trading plan executed successfully\n"))
                    else:
                        print(error("Conditional trading plan execution failed\n"))
                else:
                    print(error("Could not parse conditional trading command"))
                    print(info("Type 'examples' for sample commands\n"))
                    
            except (EOFError, KeyboardInterrupt):
                break
        
        print(info("Returning to main trading menu..."))
    
    def _show_conditional_examples(self) -> None:
        """Show examples of conditional trading commands"""
        print(f"""
{info('Conditional Trading Examples:')}

Price-Based Conditions:
  • buy AAPL if price is greater than $120
  • sell TSLA if price drops below $200
  • buy 100 NVDA when price exceeds $800

Portfolio Management:
  • maintain equal value for all positions in the portfolio  
  • rebalance portfolio to equal allocations
  • sell all positions if total portfolio drops 10%

Market-Based Strategies:
  • buy the top 5 gainers divided equally amongst $50,000
  • buy top 3 most active stocks with $10,000 each
  • sell all positions if SPY drops below $400

Complex Conditions:
  • buy AAPL if it's up more than 5% and volume is above average
  • allocate $100,000 equally among all FAANG stocks
  • buy any stock that gaps up more than 10% at market open

Tips:
  • Be specific with dollar amounts and quantities
  • Include clear trigger conditions
  • Consider risk management in your conditions
""")

    def _top_gainer(self) -> str | None:
        """Return symbol of top gainer of the day using FMP endpoint."""
        api_key = os.getenv("FMP_API_KEY")
        url = "https://financialmodelingprep.com/api/v3/gainers"
        if api_key:
            url += f"?apikey={api_key}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data:
                return data[0]["ticker"]
        except Exception as e:
            print(warning(f"Could not fetch gainers list: {e}"))
        return None


def main() -> None:
    """Main function to run the trading agent"""
    TradingAgentCLI().loop()


if __name__ == "__main__":
    main()
