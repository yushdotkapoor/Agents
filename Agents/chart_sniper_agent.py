"""
Enhanced ChartSniper Agent - Analyze charts with vision AI and execute trades
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
import re
import argparse
import warnings

# Suppress PyTorch warnings on macOS MPS
warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_DATA_AVAILABLE = True
except ImportError:
    ALPACA_DATA_AVAILABLE = False

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    ALPACA_TRADING_AVAILABLE = True
except ImportError:
    ALPACA_TRADING_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_error(message):
    """Print error message in red"""
    print(f"{Colors.RED}{message}{Colors.RESET}")

def print_warning(message):
    """Print warning message in yellow"""
    print(f"{Colors.YELLOW}{message}{Colors.RESET}")

def print_success(message):
    """Print success message in green"""
    print(f"{Colors.GREEN}{message}{Colors.RESET}")

def print_info(message):
    """Print info message in cyan"""
    print(f"{Colors.CYAN}{message}{Colors.RESET}")

def print_header(message):
    """Print header message in bold"""
    print(f"{Colors.BOLD}{message}{Colors.RESET}")

@dataclass
class ChartAnalysis:
    """Represents a chart analysis result"""
    extracted_text: str
    detected_symbols: List[str]
    detected_prices: List[float]
    detected_patterns: List[str]
    confidence_score: float
    timestamp: datetime

@dataclass
class TradingDecision:
    """Represents a trading decision from chart analysis"""
    action: str  # 'buy', 'sell', 'none', 'watch'
    symbol: str
    quantity: Optional[int] = None
    order_type: str = 'market'  # 'market', 'limit', 'stop'
    price: Optional[float] = None
    confidence: float = 0.0
    reasoning: str = ""
    risk_level: str = "medium"  # 'low', 'medium', 'high'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class ChartSniperAgent:
    """Enhanced Chart Sniper Agent with CLI interface"""
    
    def __init__(self, trading_client=None, data_client=None, debug=False):
        self.trading_client = trading_client
        self.data_client = data_client
        self.debug = debug or os.getenv("CHARTSNIPER_DEBUG", "").lower() == "true"
        
        # Initialize Alpaca clients
        self._init_alpaca_clients()
        
        # Initialize OCR
        self.ocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                # Configure PyTorch for better compatibility with Apple Silicon
                import torch
                if torch.backends.mps.is_available():
                    # Disable memory pinning for MPS to avoid warnings
                    torch.multiprocessing.set_sharing_strategy('file_system')
                
                # Suppress specific warnings during EasyOCR initialization
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*pin_memory.*")
                    warnings.filterwarnings("ignore", category=UserWarning)
                    self.ocr_reader = easyocr.Reader(['en'])
                
                if self.debug:
                    print_success("EasyOCR initialized")
            except Exception as e:
                print_warning(f"Could not initialize EasyOCR: {e}")
        else:
            if self.debug:
                print_warning("EasyOCR not available - install with: pip install easyocr")
        
        # Initialize LLM
        self.llm_source = os.getenv("LLM_SOURCE", "anthropic").lower()
        self.llm_client = None
        
        if self.llm_source == "ollama":
            self._init_ollama()
        else:
            self._init_anthropic()
        
        # Analysis history
        self.analysis_history = []
        self.decision_history = []
    
    def _init_alpaca_clients(self):
        """Initialize Alpaca trading and data clients"""
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        
        if self.debug:
            print_info(f"API Key: {api_key[:8] if api_key else 'None'}...")
            print_info(f"Secret Key: {secret_key[:8] if secret_key else 'None'}...")
            print_info(f"Paper Trading: {paper}")
        
        if not self.trading_client and ALPACA_TRADING_AVAILABLE:
            try:
                self.trading_client = TradingClient(api_key, secret_key, paper=paper)
                if self.debug:
                    print_success("Alpaca trading client initialized")
            except Exception as e:
                print_error(f"Failed to initialize Alpaca trading client: {e}")
        
        # Initialize data client
        if not self.data_client and ALPACA_DATA_AVAILABLE:
            try:
                self.data_client = StockHistoricalDataClient(api_key, secret_key)
                if self.debug:
                    print_success("Alpaca data client initialized")
            except Exception as e:
                print_error(f"Failed to initialize Alpaca data client: {e}")
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        if ANTHROPIC_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.llm_client = Anthropic(api_key=api_key)
                self.llm_source = "anthropic"
                if self.debug:
                    print_success("Using Anthropic Claude for chart analysis")
            else:
                print_warning("ANTHROPIC_API_KEY not set - chart analysis will be limited")
        else:
            if self.debug:
                print_warning("Anthropic not available - install with: pip install anthropic")
    
    def _init_ollama(self):
        """Initialize Ollama client"""
        if OLLAMA_AVAILABLE:
            try:
                models = ollama.list()
                if models:
                    self.llm_client = ollama
                    if self.debug:
                        print_success("Using Ollama for chart analysis")
                else:
                    if self.debug:
                        print_warning("No Ollama models found - falling back to Anthropic")
                    self._init_anthropic()
            except Exception as e:
                if self.debug:
                    print_warning(f"Could not connect to Ollama ({e}) - falling back to Anthropic")
                self._init_anthropic()
        else:
            if self.debug:
                print_warning("Ollama not available - install with: pip install ollama")
            self._init_anthropic()
    
    def analyze_chart(self, image_path: str) -> Optional[ChartAnalysis]:
        """Extract text and analyze chart from image"""
        if not self.ocr_reader:
            print_error("OCR reader not available")
            return None
        
        try:
            if self.debug:
                print(f"Analyzing chart: {image_path}")
            
            # Extract text using OCR
            results = self.ocr_reader.readtext(image_path)
            extracted_text = ' '.join([res[1] for res in results])
            
            if self.debug:
                print_info(f"Extracted text: {extracted_text}")
            
            # Detect symbols (basic pattern matching)
            symbols = self._detect_symbols(extracted_text)
            
            # Detect prices
            prices = self._detect_prices(extracted_text)
            
            # Detect patterns
            patterns = self._detect_patterns(extracted_text)
            
            # Calculate confidence based on what we found
            confidence = self._calculate_confidence(symbols, prices, patterns, extracted_text)
            
            analysis = ChartAnalysis(
                extracted_text=extracted_text,
                detected_symbols=symbols,
                detected_prices=prices,
                detected_patterns=patterns,
                confidence_score=confidence,
                timestamp=datetime.now()
            )
            
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            print_error(f"Error analyzing chart: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _detect_symbols(self, text: str) -> List[str]:
        """Detect stock symbols in text"""
        # Look for common stock symbol patterns
        patterns = [
            r'\\$([A-Z]{1,5})',  # $AAPL format
            r'\\b([A-Z]{2,5})\\b',  # AAPL format
            r'([A-Z]+)\\s*(?:USD|\\$)',  # AAPL USD format
        ]
        
        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            symbols.update(matches)
        
        # Filter out common false positives
        false_positives = {'USD', 'BUY', 'SELL', 'HIGH', 'LOW', 'OPEN', 'CLOSE', 'VOL', 'RSI', 'MACD'}
        symbols = [s for s in symbols if s not in false_positives and len(s) >= 2]
        
        return list(symbols)
    
    def _detect_prices(self, text: str) -> List[float]:
        """Detect price values in text"""
        # Look for price patterns
        price_patterns = [
            r'\\$([0-9,]+\\.?[0-9]*)',  # $123.45
            r'([0-9,]+\\.[0-9]{2})\\s*(?:USD|\\$)?',  # 123.45 USD
            r'Price[:\\s]+([0-9,]+\\.?[0-9]*)',  # Price: 123.45
        ]
        
        prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    price = float(match.replace(',', ''))
                    if 0.01 <= price <= 100000:  # Reasonable price range
                        prices.append(price)
                except ValueError:
                    continue
        
        return sorted(list(set(prices)))
    
    def _detect_patterns(self, text: str) -> List[str]:
        """Detect trading patterns and signals in text"""
        pattern_keywords = {
            'bullish': ['bullish', 'bull', 'breakout', 'break out', 'uptrend', 'rising', 'support'],
            'bearish': ['bearish', 'bear', 'breakdown', 'break down', 'downtrend', 'falling', 'resistance'],
            'reversal': ['reversal', 'reverse', 'turn', 'bounce', 'pivot'],
            'consolidation': ['consolidation', 'sideways', 'range', 'flat', 'consolidating'],
            'volume': ['volume', 'vol', 'high volume', 'low volume'],
            'momentum': ['momentum', 'rsi', 'macd', 'stochastic', 'oversold', 'overbought']
        }
        
        detected = []
        text_lower = text.lower()
        
        for pattern_type, keywords in pattern_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected.append(f"{pattern_type}:{keyword}")
        
        return detected
    
    def _calculate_confidence(self, symbols: List[str], prices: List[float], 
                            patterns: List[str], text: str) -> float:
        """Calculate confidence score for the analysis"""
        score = 0.0
        
        # Symbol detection
        if symbols:
            score += 0.3
        
        # Price detection
        if prices:
            score += 0.2
        
        # Pattern detection
        if patterns:
            score += 0.2
        
        # Text quality (length and recognizable terms)
        trading_terms = ['price', 'chart', 'trade', 'buy', 'sell', 'volume', 'trend']
        term_count = sum(1 for term in trading_terms if term.lower() in text.lower())
        score += min(0.3, term_count * 0.05)
        
        return min(1.0, score)
    
    def generate_chart_from_symbol(self, symbol: str, days: int = 30) -> Optional[str]:
        """Generate a chart for a stock symbol and return the image path"""
        if not MATPLOTLIB_AVAILABLE:
            print_error("Chart generation requires matplotlib")
            return None
        
        if not self.data_client:
            print_error("Chart generation requires Alpaca data client")
            print("Tip: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
            print("Tip: Ensure your Alpaca account has access to historical data")
            return None
        
        try:
            from datetime import datetime, timedelta
            
            if self.debug:
                print(f"Fetching {days} days of data for {symbol}...")
            
            # Calculate date range - use pre-formatted date strings
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days + 5)
            
            # Format as simple YYYY-MM-DD strings (what Alpaca expects)
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            if self.debug:
                print_info(f"Date range: {start_date_str} to {end_date_str}")
                print_info(f"Using symbol: {symbol.upper()}")
            
            # Create request for historical data using pre-formatted date strings
            request = StockBarsRequest(
                symbol_or_symbols=[symbol.upper()],
                timeframe=TimeFrame.Day,
                start=start_date_str,  # Pre-formatted date string
                end=end_date_str       # Pre-formatted date string
            )
            
            if self.debug:
                print_info(f"Request parameters: symbols={[symbol.upper()]}, timeframe=Day, start={start_date_str}, end={end_date_str}")
            
            # Fetch data
            bars = self.data_client.get_stock_bars(request)
            
            if self.debug:
                print_info(f"API Response type: {type(bars)}")
                if hasattr(bars, 'df'):
                    print_info(f"DataFrame shape: {bars.df.shape}")
                    print_info(f"DataFrame columns: {bars.df.columns.tolist()}")
            
            if hasattr(bars, 'df') and not bars.df.empty:
                chart_path = self._create_chart(bars.df, symbol.upper())
                if self.debug:
                    print_success(f"Chart generated: {chart_path}")
                return chart_path
            else:
                print_error(f"No data found for symbol {symbol}")
                if self.debug:
                    print_info(f"Response: {bars}")
                return None
                
        except Exception as e:
            print_error(f"Error generating chart: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _create_chart(self, df, symbol: str) -> str:
        """Create a professional-looking stock chart"""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for chart generation")
        
        # Prepare data
        df = df.reset_index()
        
        # Check if we have data
        if df.empty:
            raise ValueError("No data available for chart generation")
        
        # Extract data columns
        dates = df['timestamp']
        opens = df['open']
        highs = df['high']
        lows = df['low']
        closes = df['close']
        volumes = df['volume']
        
        # Calculate moving averages only if we have enough data
        closes_array = np.array(closes)
        ma_20 = None
        ma_50 = None
        
        if len(closes_array) >= 20:
            ma_20 = np.convolve(closes_array, np.ones(20)/20, mode='valid')
        
        if len(closes_array) >= 50:
            ma_50 = np.convolve(closes_array, np.ones(50)/50, mode='valid')
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        fig.suptitle(f'{symbol} Stock Chart - {len(df)} Day Analysis', fontsize=16, fontweight='bold')
        
        # Main price chart
        ax1.plot(dates, closes, 'b-', linewidth=2, label='Close Price', alpha=0.8)
        
        # Add moving averages if we have enough data
        if ma_20 is not None and len(ma_20) > 0:
            # Align dates with moving average (skip first 19 days)
            ma_20_dates = dates.iloc[19:19+len(ma_20)]
            ax1.plot(ma_20_dates, ma_20, 'orange', linewidth=1, label='MA 20', alpha=0.7)
        
        if ma_50 is not None and len(ma_50) > 0:
            # Align dates with moving average (skip first 49 days)
            ma_50_dates = dates.iloc[49:49+len(ma_50)]
            ax1.plot(ma_50_dates, ma_50, 'red', linewidth=1, label='MA 50', alpha=0.7)
        
        # Highlight recent trend
        recent_close = closes.iloc[-1]
        previous_close = closes.iloc[-2] if len(closes) > 1 else recent_close
        trend_color = 'green' if recent_close > previous_close else 'red'
        
        # Add price annotations
        ax1.annotate(f'${recent_close:.2f}', 
                    xy=(dates.iloc[-1], recent_close),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=trend_color, alpha=0.7),
                    color='white', fontweight='bold')
        
        # Format price chart
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # Volume chart
        colors = ['green' if close >= open_price else 'red' for close, open_price in zip(closes, opens)]
        ax2.bar(dates, volumes, color=colors, alpha=0.6, width=0.8)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # Add technical analysis annotations
        if len(closes) > 1:
            price_change = recent_close - closes.iloc[0]
            price_change_pct = (price_change / closes.iloc[0]) * 100
            
            # Add performance text
            perf_text = f'{len(df)}-Day Performance: {price_change:+.2f} ({price_change_pct:+.1f}%)'
            fig.text(0.02, 0.02, perf_text, fontsize=10, 
                    color='green' if price_change > 0 else 'red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))
        
        # Add pattern detection hints
        if len(closes_array) >= 10:
            volatility = np.std(closes_array[-10:])
            avg_price = np.mean(closes_array[-30:]) if len(closes_array) >= 30 else np.mean(closes_array)
            
            if volatility > avg_price * 0.02:  # High volatility
                fig.text(0.98, 0.95, '⚠️ High Volatility', ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))
        
        # Save chart
        chart_filename = f'chart_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        chart_path = os.path.join(os.getcwd(), chart_filename)
        
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path
    
    def analyze_symbol(self, symbol: str, days: int = 30) -> Optional[TradingDecision]:
        """Complete analysis pipeline: generate chart -> analyze -> decide"""
        if self.debug:
            print(f"\nStarting complete analysis for {symbol.upper()}")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Step 1: Generate chart
        chart_path = self.generate_chart_from_symbol(symbol, days)
        if not chart_path:
            print_error("Could not generate chart from symbol")
            return None
        
        # Step 2: Analyze chart
        analysis = self.analyze_chart(chart_path)
        if not analysis:
            print_error("Could not analyze generated chart")
            return None
        
        # Step 3: Enhance analysis with symbol context (skip if API unavailable)
        if self.data_client:
            try:
                enhanced_analysis = self._enhance_analysis_with_symbol_data(analysis, symbol, days)
            except Exception as e:
                if self.debug:
                    print_warning(f"Could not enhance with API data: {e}")
                enhanced_analysis = analysis
        else:
            enhanced_analysis = analysis
        
        # Step 4: Make trading decision
        decision = self.make_trading_decision(enhanced_analysis)
        
        # Clean up generated chart file
        try:
            os.remove(chart_path)
            if self.debug:
                print_info(f"Cleaned up chart file: {chart_path}")
        except Exception as e:
            if self.debug:
                print_warning(f"Could not clean up chart file: {e}")
        
        return decision
    
    def _enhance_analysis_with_symbol_data(self, analysis: ChartAnalysis, symbol: str, days: int) -> ChartAnalysis:
        """Enhance OCR analysis with additional context from generated chart"""
        try:
            # Get recent market data for context
            if self.data_client:
                from datetime import datetime, timedelta
                
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days + 5)
                
                # Format as simple YYYY-MM-DD strings (what Alpaca expects)
                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")
                
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol.upper()],
                    timeframe=TimeFrame.Day,
                    start=start_date_str,  # Pre-formatted date string
                    end=end_date_str       # Pre-formatted date string
                )
                
                bars = self.data_client.get_stock_bars(request)
                
                if not bars.df.empty:
                    df = bars.df.reset_index()
                    
                    # Calculate additional metrics
                    recent_close = df['close'].iloc[-1]
                    period_start = df['close'].iloc[0]
                    period_high = df['high'].max()
                    period_low = df['low'].min()
                    avg_volume = df['volume'].mean()
                    recent_volume = df['volume'].iloc[-1]
                    
                    # Performance metrics
                    period_return = ((recent_close - period_start) / period_start) * 100
                    volatility = df['close'].std() / df['close'].mean() * 100
                    
                    # Create enhanced extracted text
                    enhanced_text = f"""
SYMBOL: {symbol.upper()}
CURRENT_PRICE: ${recent_close:.2f}
PERIOD_RETURN: {period_return:+.1f}%
PERIOD_HIGH: ${period_high:.2f}
PERIOD_LOW: ${period_low:.2f}
VOLATILITY: {volatility:.1f}%
VOLUME_RATIO: {recent_volume/avg_volume:.1f}x average
TREND: {'BULLISH' if period_return > 0 else 'BEARISH'}
PERIOD: {days} days
"""
                    
                    # Update analysis
                    analysis.extracted_text = enhanced_text + analysis.extracted_text
                    
                    # Ensure symbol is detected
                    if symbol.upper() not in analysis.detected_symbols:
                        analysis.detected_symbols.append(symbol.upper())
                    
                    # Add current price to detected prices
                    if recent_close not in analysis.detected_prices:
                        analysis.detected_prices.append(float(recent_close))
                    
                    # Add pattern based on performance
                    if period_return > 5:
                        analysis.detected_patterns.append("bullish:strong_uptrend")
                    elif period_return < -5:
                        analysis.detected_patterns.append("bearish:strong_downtrend")
                    
                    if volatility > 30:
                        analysis.detected_patterns.append("momentum:high_volatility")
                    
                    if recent_volume > avg_volume * 1.5:
                        analysis.detected_patterns.append("volume:high_volume")
                    
                    # Increase confidence since we have real data
                    analysis.confidence_score = min(1.0, analysis.confidence_score + 0.3)
        
        except Exception as e:
            if self.debug:
                print_warning(f"Could not enhance analysis: {e}")
        
        return analysis
    
    def make_trading_decision(self, analysis: ChartAnalysis) -> Optional[TradingDecision]:
        """Use LLM to make trading decision based on chart analysis"""
        if not self.llm_client or not analysis:
            return None
        
        try:
            prompt = self._create_analysis_prompt(analysis)
            
            if self.llm_source == "ollama":
                decision = self._get_ollama_decision(prompt)
            else:
                decision = self._get_anthropic_decision(prompt)
            
            if decision:
                self.decision_history.append(decision)
            
            return decision
            
        except Exception as e:
            print_error(f"Error making trading decision: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return None
    
    def _create_analysis_prompt(self, analysis: ChartAnalysis) -> str:
        """Create prompt for LLM analysis"""
        return f"""
You are an expert trading analyst. Analyze this chart data and make a trading recommendation.

CHART DATA:
- Extracted Text: {analysis.extracted_text}
- Detected Symbols: {analysis.detected_symbols}
- Detected Prices: {analysis.detected_prices}
- Detected Patterns: {analysis.detected_patterns}
- Confidence Score: {analysis.confidence_score:.2f}

INSTRUCTIONS:
Based on this information, provide a trading recommendation as JSON:

{{
    "action": "buy|sell|none|watch",
    "symbol": "stock_symbol",
    "quantity": number_of_shares,
    "order_type": "market|limit|stop",
    "price": target_price_or_null,
    "confidence": 0.0_to_1.0,
    "reasoning": "explanation_of_decision",
    "risk_level": "low|medium|high",
    "stop_loss": stop_loss_price_or_null,
    "take_profit": take_profit_price_or_null
}}

RULES:
- Only recommend trades if confidence > 0.6
- If multiple symbols detected, pick the most prominent one
- Consider risk management (stop loss, position sizing)
- Be conservative with position sizes
- Explain your reasoning clearly

Return ONLY the JSON, no additional text.
"""
    
    def _get_anthropic_decision(self, prompt: str) -> Optional[TradingDecision]:
        """Get trading decision from Anthropic"""
        try:
            response = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            return self._parse_decision_response(response_text)
            
        except Exception as e:
            print_error(f"Error getting Anthropic decision: {e}")
            return None
    
    def _get_ollama_decision(self, prompt: str) -> Optional[TradingDecision]:
        """Get trading decision from Ollama"""
        try:
            model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
            
            response = self.llm_client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response['message']['content']
            return self._parse_decision_response(response_text)
            
        except Exception as e:
            print_error(f"Error getting Ollama decision: {e}")
            return None
    
    def _parse_decision_response(self, response_text: str) -> Optional[TradingDecision]:
        """Parse LLM response into TradingDecision"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return TradingDecision(
                    action=data.get('action', 'none'),
                    symbol=data.get('symbol', ''),
                    quantity=data.get('quantity'),
                    order_type=data.get('order_type', 'market'),
                    price=data.get('price'),
                    confidence=float(data.get('confidence', 0.0)),
                    reasoning=data.get('reasoning', ''),
                    risk_level=data.get('risk_level', 'medium'),
                    stop_loss=data.get('stop_loss'),
                    take_profit=data.get('take_profit')
                )
            
        except Exception as e:
            if self.debug:
                print_error(f"Error parsing decision: {e}")
                print_info(f"Response: {response_text[:500]}...")
        
        return None
    
    def execute_decision(self, decision: TradingDecision, confirm: bool = True) -> bool:
        """Execute trading decision"""
        if not decision or decision.action == 'none':
            print("No trading action recommended")
            return False
        
        if decision.action == 'watch':
            print(f"Recommendation: Watch {decision.symbol}")
            print(f"Reasoning: {decision.reasoning}")
            return True
        
        # Display decision details
        print(f"\nTrading Decision Analysis")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Action: {decision.action.upper()}")
        print(f"Symbol: {decision.symbol}")
        print(f"Quantity: {decision.quantity}")
        print(f"Order Type: {decision.order_type}")
        if decision.price:
            print(f"Target Price: ${decision.price:.2f}")
        print(f"Confidence: {decision.confidence:.1%}")
        print(f"Risk Level: {decision.risk_level}")
        if decision.stop_loss:
            print(f"Stop Loss: ${decision.stop_loss:.2f}")
        if decision.take_profit:
            print(f"Take Profit: ${decision.take_profit:.2f}")
        print(f"\nReasoning: {decision.reasoning}")
        
        if confirm:
            response = input(f"\nExecute this trade? [y/N]: ").strip().lower()
            if response != 'y':
                print_error("Trade cancelled")
                return False
        
        # Execute the trade
        return self._submit_order(decision)
    
    def _submit_order(self, decision: TradingDecision) -> bool:
        """Submit order to Alpaca"""
        if not self.trading_client:
            print_error("No trading client configured")
            return False
        
        try:
            side = OrderSide.BUY if decision.action.lower() == 'buy' else OrderSide.SELL
            
            # Create appropriate order type
            if decision.order_type == 'limit' and decision.price:
                order = LimitOrderRequest(
                    symbol=decision.symbol.upper(),
                    qty=str(decision.quantity),
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=str(decision.price)
                )
            else:
                order = MarketOrderRequest(
                    symbol=decision.symbol.upper(),
                    qty=str(decision.quantity),
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            
            result = self.trading_client.submit_order(order)
            print_success(f"Order submitted: {decision.action.upper()} {decision.quantity} {decision.symbol}")
            if self.debug:
                print(f"Order ID: {str(result.id)[:8]}...")
            
            # Log the trade
            self._log_trade(decision, result.id)
            
            return True
            
        except Exception as e:
            print_error(f"Failed to submit order: {e}")
            return False
    
    def _log_trade(self, decision: TradingDecision, order_id):
        """Log trade to file"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'order_id': str(order_id),  # Convert UUID to string
                'action': decision.action,
                'symbol': decision.symbol,
                'quantity': decision.quantity,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'risk_level': decision.risk_level
            }
            
            # Create data directory if it doesn't exist
            data_dir = Path(__file__).parent / 'data'
            data_dir.mkdir(exist_ok=True)
            
            log_file = data_dir / 'chartsniper_trades.json'
            
            # Try to read existing trades, handle corrupted JSON
            trades = []
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        content = f.read().strip()
                        if content:  # Only parse if file has content
                            trades = json.loads(content)
                        else:
                            trades = []
                except json.JSONDecodeError as e:
                    print_warning("Corrupted trade log detected, creating backup...")
                    # Create backup of corrupted file
                    backup_file = log_file.with_suffix('.json.backup')
                    log_file.rename(backup_file)
                    print(f"Corrupted log backed up to {backup_file}")
                    trades = []
                except Exception as e:
                    print_warning(f"Error reading trade log: {e}")
                    trades = []
            
            trades.append(log_entry)
            
            # Write with proper formatting
            with open(log_file, 'w') as f:
                json.dump(trades, f, indent=2, ensure_ascii=False)
            
            if self.debug:
                print(f"Trade logged to {log_file}")
            
        except Exception as e:
            print_warning(f"Could not log trade: {e}")
            # Try to at least log to a simple text file as fallback
            try:
                data_dir = Path(__file__).parent / 'data'
                data_dir.mkdir(exist_ok=True)
                fallback_file = data_dir / 'chartsniper_trades_fallback.txt'
                with open(fallback_file, 'a') as f:
                    f.write(f"{datetime.now().isoformat()} - {decision.action.upper()} {decision.quantity} {decision.symbol} (Order: {str(order_id)[:8]}...)\n")
                if self.debug:
                    print(f"Trade logged to fallback file: {fallback_file}")
            except Exception as fallback_error:
                print_error(f"Complete logging failure: {fallback_error}")
    
    def show_analysis_history(self):
        """Display analysis history"""
        if not self.analysis_history:
            print("No analysis history available")
            return
        
        print(f"\nChart Analysis History ({len(self.analysis_history)} entries)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for i, analysis in enumerate(self.analysis_history[-10:], 1):  # Show last 10
            print(f"{i}. {analysis.timestamp.strftime('%H:%M:%S')} | "
                  f"Symbols: {analysis.detected_symbols} | "
                  f"Confidence: {analysis.confidence_score:.1%}")
    
    def show_decision_history(self):
        """Display decision history"""
        if not self.decision_history:
            print("No decision history available")
            return
        
        print(f"\nTrading Decision History ({len(self.decision_history)} entries)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for i, decision in enumerate(self.decision_history[-10:], 1):  # Show last 10
            action_text = {"buy": "BUY", "sell": "SELL", "watch": "WATCH", "none": "NONE"}.get(decision.action, "UNKNOWN")
            print(f"{i}. {action_text} {decision.symbol} | "
                  f"Qty: {decision.quantity} | "
                  f"Confidence: {decision.confidence:.1%}")
    
    def load_trade_history(self):
        """Load trade history from file"""
        try:
            data_dir = Path(__file__).parent / 'data'
            log_file = data_dir / 'chartsniper_trades.json'
            
            if not log_file.exists():
                return []
            
            with open(log_file, 'r') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
                else:
                    return []
        except Exception as e:
            print_warning(f"Error loading trade history: {e}")
            return []
    
    def show_trade_history(self):
        """Display trade history from file"""
        trades = self.load_trade_history()
        
        if not trades:
            print("No trade history available")
            return
        
        print(f"\nTrade History ({len(trades)} entries)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for i, trade in enumerate(trades[-10:], 1):  # Show last 10
            timestamp = datetime.fromisoformat(trade['timestamp']).strftime('%m/%d %H:%M')
            action_text = {"buy": "BUY", "sell": "SELL", "watch": "WATCH", "none": "NONE"}.get(trade['action'], "UNKNOWN")
            print(f"{i}. {timestamp} | {action_text} {trade['quantity']} {trade['symbol']} | "
                  f"Confidence: {trade['confidence']:.1%} | Risk: {trade['risk_level']}")
            if len(trade.get('reasoning', '')) > 0:
                print(f"   Reasoning: {trade['reasoning'][:100]}{'...' if len(trade['reasoning']) > 100 else ''}")
    
    def show_combined_history(self):
        """Show both analysis and trade history"""
        print_header("CHARTSNIPER ACTIVITY OVERVIEW")
        print("═══════════════════════════════════════════════════════════════════════════════════")
        
        # Show trade history first (most important)
        self.show_trade_history()
        
        # Then show current session analysis and decision history
        if self.analysis_history:
            print(f"\nCurrent Session Analysis ({len(self.analysis_history)} entries)")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            for i, analysis in enumerate(self.analysis_history[-5:], 1):  # Show last 5
                print(f"{i}. {analysis.timestamp.strftime('%H:%M:%S')} | "
                      f"Symbols: {analysis.detected_symbols} | "
                      f"Confidence: {analysis.confidence_score:.1%}")
        
        if self.decision_history:
            print(f"\nCurrent Session Decisions ({len(self.decision_history)} entries)")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            for i, decision in enumerate(self.decision_history[-5:], 1):  # Show last 5
                action_text = {"buy": "BUY", "sell": "SELL", "watch": "WATCH", "none": "NONE"}.get(decision.action, "UNKNOWN")
                print(f"{i}. {action_text} {decision.symbol} | "
                      f"Qty: {decision.quantity} | "
                      f"Confidence: {decision.confidence:.1%}")
    
    def show_analysis_history(self):
        """Display analysis history"""
        if not self.analysis_history:
            print("No analysis history available")
            return
        
        print(f"\nChart Analysis History ({len(self.analysis_history)} entries)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for i, analysis in enumerate(self.analysis_history[-10:], 1):  # Show last 10
            print(f"{i}. {analysis.timestamp.strftime('%H:%M:%S')} | "
                  f"Symbols: {analysis.detected_symbols} | "
                  f"Confidence: {analysis.confidence_score:.1%}")
    
    def show_decision_history(self):
        """Display decision history"""
        if not self.decision_history:
            print("No decision history available")
            return
        
        print(f"\nTrading Decision History ({len(self.decision_history)} entries)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for i, decision in enumerate(self.decision_history[-10:], 1):  # Show last 10
            action_text = {"buy": "BUY", "sell": "SELL", "watch": "WATCH", "none": "NONE"}.get(decision.action, "UNKNOWN")
            print(f"{i}. {action_text} {decision.symbol} | "
                  f"Qty: {decision.quantity} | "
                  f"Confidence: {decision.confidence:.1%}")

# Legacy functions for backward compatibility
def extract_text_from_image(path):
    """Legacy function for backward compatibility"""
    agent = ChartSniperAgent()
    analysis = agent.analyze_chart(path)
    return analysis.extracted_text if analysis else ""

def ask_agent(text):
    """Legacy function for backward compatibility"""
    agent = ChartSniperAgent()
    # Create a minimal analysis object
    analysis = ChartAnalysis(
        extracted_text=text,
        detected_symbols=agent._detect_symbols(text),
        detected_prices=agent._detect_prices(text),
        detected_patterns=agent._detect_patterns(text),
        confidence_score=0.5,
        timestamp=datetime.now()
    )
    decision = agent.make_trading_decision(analysis)
    
    if decision:
        return {
            'action': decision.action,
            'symbol': decision.symbol,
            'quantity': decision.quantity
        }
    return {'action': 'none', 'symbol': '', 'quantity': 0}

def place_order(data):
    """Legacy function for backward compatibility"""
    agent = ChartSniperAgent()
    
    # Convert legacy data format to TradingDecision
    decision = TradingDecision(
        action=data.get('action', 'none'),
        symbol=data.get('symbol', ''),
        quantity=data.get('quantity', 0),
        confidence=0.5,
        reasoning="Legacy order execution"
    )
    
    return agent.execute_decision(decision, confirm=False)

def main():
    """Main CLI interface"""
    try:
        parser = argparse.ArgumentParser(description="ChartSniper - AI-powered chart analysis and trading")
        
        # Create mutually exclusive group for input types
        input_group = parser.add_mutually_exclusive_group()
        input_group.add_argument('image', nargs='?', help='Path to chart image')
        input_group.add_argument('--symbol', '-s', type=str, help='Stock symbol to analyze (e.g., AAPL)')
        
        parser.add_argument('--days', '-d', type=int, default=30, help='Number of days for historical data (default: 30)')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        parser.add_argument('--no-confirm', action='store_true', help='Skip trade confirmation')
        parser.add_argument('--history', action='store_true', help='Show analysis history')
        parser.add_argument('--decisions', action='store_true', help='Show decision history')
        parser.add_argument('--trades', action='store_true', help='Show trade history')
        parser.add_argument('--all-history', action='store_true', help='Show complete activity overview')
        
        args = parser.parse_args()
        
        # Initialize agent
        agent = ChartSniperAgent(debug=args.debug)
        
        # Handle different modes
        if args.history:
            agent.show_analysis_history()
            # Continue to interactive mode
            interactive_mode(agent, args)
            return
        
        if args.decisions:
            agent.show_decision_history()
            # Continue to interactive mode
            interactive_mode(agent, args)
            return
        
        if args.trades:
            agent.show_trade_history()
            # Continue to interactive mode
            interactive_mode(agent, args)
            return
        
        if args.all_history:
            agent.show_combined_history()
            # Continue to interactive mode
            interactive_mode(agent, args)
            return
        
        # Determine analysis mode
        if args.symbol:
            # Symbol analysis mode
            if args.debug:
                print(f"Analyzing symbol: {args.symbol.upper()}")
            
            decision = agent.analyze_symbol(args.symbol, args.days)
            if not decision:
                print_error("Could not complete symbol analysis")
            else:
                # Execute decision
                agent.execute_decision(decision, confirm=not args.no_confirm)
            
            # Continue to interactive mode
            interactive_mode(agent, args)
            
        elif args.image:
            # Image analysis mode
            if args.debug:
                print(f"Analyzing chart: {args.image}")
            
            # Step 1: Analyze chart
            analysis = agent.analyze_chart(args.image)
            if not analysis:
                print_error("Could not analyze chart")
            else:
                # Display analysis results
                print(f"\nChart Analysis Results")
                print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"Symbols: {analysis.detected_symbols}")
                print(f"Prices: {analysis.detected_prices}")
                print(f"Patterns: {len(analysis.detected_patterns)} detected")
                print(f"Confidence: {analysis.confidence_score:.1%}")
                
                if analysis.confidence_score < 0.3:
                    print_warning("Low confidence - chart may not contain clear trading signals")
                
                # Step 2: Make trading decision
                decision = agent.make_trading_decision(analysis)
                
                if not decision:
                    print_error("Could not generate trading decision")
                else:
                    # Step 3: Execute decision
                    agent.execute_decision(decision, confirm=not args.no_confirm)
            
            # Continue to interactive mode
            interactive_mode(agent, args)
            
        else:
            # Interactive mode - continuous loop
            interactive_mode(agent, args)
            
    except KeyboardInterrupt:
        print("\n\nExiting ChartSniper...")
    except Exception as e:
        print_error(f"An error occurred: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()

def interactive_mode(agent, args):
    """Run interactive mode with continuous loop"""
    print_header("ChartSniper Interactive Mode")
    print("Enter commands or choose from the menu. Type 'help' for options.")
    
    # Show menu once at the start
    show_menu()
    
    while True:
        try:
            choice = input("\nsniper> ").strip()
            
            # Handle exit commands
            if choice.lower() in ['/exit', '/quit', 'exit', 'quit']:
                print("Goodbye!")
                break
            
            # Handle help commands
            if choice.lower() in ['/help', 'help', '7']:
                show_menu()
                continue
            
            # Handle menu choices
            if choice == '1':
                handle_image_analysis(agent, args)
            elif choice == '2':
                handle_symbol_analysis(agent, args)
            elif choice == '3':
                agent.show_analysis_history()
            elif choice == '4':
                agent.show_decision_history()
            elif choice == '5':
                agent.show_trade_history()
            elif choice == '6':
                agent.show_combined_history()
            elif choice == '':
                # Empty input, just continue
                continue
            else:
                print_error("Invalid choice. Type 'help' to see available options.")
                
        except KeyboardInterrupt:
            print("\n\nExiting ChartSniper...")
            break
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print_error(f"An error occurred: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

def handle_image_analysis(agent, args):
    """Handle image analysis in interactive mode"""
    try:
        image_path = input("Enter chart image path: ").strip()
        if not image_path:
            print_warning("No image path provided")
            return
        
        # Step 1: Analyze chart
        analysis = agent.analyze_chart(image_path)
        if not analysis:
            print_error("Could not analyze chart")
            return
        
        # Display analysis results
        print(f"\nChart Analysis Results")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Extracted Text: {analysis.extracted_text[:100]}...")
        print(f"Detected Symbols: {analysis.detected_symbols}")
        print(f"Detected Prices: {analysis.detected_prices}")
        print(f"Detected Patterns: {len(analysis.detected_patterns)} patterns")
        print(f"Confidence Score: {analysis.confidence_score:.1%}")
        
        if analysis.confidence_score < 0.3:
            print_warning("Low confidence - chart may not contain clear trading signals")
        
        # Step 2: Make trading decision
        print("\nAnalyzing with AI...")
        decision = agent.make_trading_decision(analysis)
        
        if not decision:
            print_error("Could not generate trading decision")
            return
        
        # Step 3: Execute decision
        agent.execute_decision(decision, confirm=not args.no_confirm)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return

def handle_symbol_analysis(agent, args):
    """Handle symbol analysis in interactive mode"""
    try:
        symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
        if not symbol:
            print_warning("No symbol provided")
            return
        
        days_input = input("Days of history (default 30): ").strip()
        days = int(days_input) if days_input.isdigit() else 30
        
        decision = agent.analyze_symbol(symbol, days)
        if decision:
            agent.execute_decision(decision, confirm=not args.no_confirm)
        else:
            print_error("Could not complete symbol analysis")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return

def show_menu():
    """Show the main menu options"""
    print("\nAvailable commands:")
    print("1. Upload chart image")
    print("2. Analyze stock symbol")
    print("3. Show analysis history")
    print("4. Show decision history")
    print("5. Show trade history")
    print("6. Show complete activity overview")
    print("help - Show this menu")
    print("exit/quit - Exit the application")

def show_help():
    """Show help information"""
    print_header("ChartSniper Help")
    show_menu()
    print("\nFeatures:")
    print("  • AI-powered chart analysis using OCR and LLM")
    print("  • Real-time stock data integration")
    print("  • Automated trading decision making")
    print("  • Trade execution with Alpaca API")
    print("  • Comprehensive logging and history tracking")
    print("  • Support for both paper and live trading")

if __name__ == "__main__":
    main()
