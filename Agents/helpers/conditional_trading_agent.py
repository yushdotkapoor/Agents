"""
Conditional Trading Agent - Uses LLM to parse and execute complex trading conditions
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
import re

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

from dotenv import load_dotenv
load_dotenv()

from Agents.helpers.color_utils import error, success, warning, info

@dataclass
class TradingCondition:
    """Represents a parsed trading condition"""
    condition_type: str  # 'price_threshold', 'portfolio_rebalance', 'top_gainers', 'custom'
    symbol: Optional[str] = None
    action: str = 'buy'  # 'buy', 'sell', 'rebalance'
    quantity: Optional[Decimal] = None
    price_threshold: Optional[Decimal] = None
    operator: str = 'greater_than'  # 'greater_than', 'less_than', 'equal_to'
    target_allocation: Optional[Decimal] = None
    portfolio_value: Optional[Decimal] = None
    screening_criteria: Optional[str] = None
    count: Optional[int] = None
    raw_condition: str = ""
    executable_code: Optional[str] = None

@dataclass
class TradingPlan:
    """Represents an executable trading plan"""
    description: str
    conditions: List[TradingCondition]
    pre_checks: List[str]  # Code to run before execution
    execution_steps: List[str]  # Code to execute the plan
    risk_warnings: List[str]

class ConditionalTradingAgent:
    """Agent that parses and executes conditional trading commands using LLM"""
    
    def __init__(self, trading_client=None, data_client=None, symbol_fetcher=None, debug=False):
        self.trading_client = trading_client
        self.data_client = data_client
        self.symbol_fetcher = symbol_fetcher
        self.debug = debug or os.getenv("CONDITIONAL_TRADING_DEBUG", "").lower() == "true"
        
        # Determine LLM source from environment variable
        self.llm_source = os.getenv("LLM_SOURCE", "anthropic").lower()
        
        # Initialize LLM client based on source
        self.llm_client = None
        if self.llm_source == "ollama":
            self._init_ollama()
        else:
            self._init_anthropic()
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        if ANTHROPIC_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.llm_client = Anthropic(api_key=api_key)
                self.llm_source = "anthropic"
                print(success("Using Anthropic Claude for conditional trading"))
            else:
                print(warning("ANTHROPIC_API_KEY not set - conditional trading will be limited"))
        else:
            print(warning("Anthropic not available - install with: pip install anthropic"))
    
    def _init_ollama(self):
        """Initialize Ollama client"""
        if OLLAMA_AVAILABLE:
            try:
                # Test connection to Ollama
                models = ollama.list()
                if models:
                    self.llm_client = ollama
                    print(success("Using Ollama for conditional trading"))
                else:
                    print(warning("No Ollama models found - falling back to Anthropic"))
                    self._init_anthropic()
            except Exception as e:
                print(error(f"Could not connect to Ollama ({e}) - falling back to Anthropic"))
                self._init_anthropic()
        else:
            print(warning("Ollama not available - install with: pip install ollama"))
            self._init_anthropic()
    
    def parse_conditional_command(self, command: str) -> Optional[TradingPlan]:
        """Parse a natural language conditional trading command using LLM"""
        if not self.llm_client:
            return self._fallback_parse(command)
        
        if self.llm_source == "ollama":
            return self._parse_with_ollama(command)
        else:
            return self._parse_with_anthropic(command)
    
    def _parse_with_anthropic(self, command: str) -> Optional[TradingPlan]:
        """Parse command using Anthropic Claude"""
        system_prompt = self._get_system_prompt()
        user_prompt = f"Parse this trading command: {command}"
        
        try:
            response = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            response_text = response.content[0].text
            return self._extract_json_response(response_text, command)
                
        except Exception as e:
            print(error(f"Error parsing with Anthropic: {e}"))
            return self._fallback_parse(command)
    
    def _parse_with_ollama(self, command: str) -> Optional[TradingPlan]:
        """Parse command using Ollama"""
        system_prompt = self._get_system_prompt()
        user_prompt = f"Parse this trading command: {command}"
        
        try:
            # Get available models and choose one
            model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
            
            response = self.llm_client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response_text = response['message']['content']
            
            if self.debug:
                print(f"Ollama raw response:\n{response_text}\n" + "-"*50)
            
            return self._extract_json_response(response_text, command)
                
        except Exception as e:
            print(error(f"Error parsing with Ollama: {e}"))
            if self.debug:
                import traceback
                traceback.print_exc()
            return self._fallback_parse(command)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for LLM parsing"""
        base_prompt = """You are a financial trading command parser. Your job is to parse natural language trading commands into structured JSON plans.

Parse the user's trading command and return a JSON object with this structure:
{
    "description": "Human readable description of what will be executed",
    "conditions": [
        {
            "condition_type": "price_threshold|portfolio_rebalance|top_gainers|market_screen|custom",
            "symbol": "STOCK_SYMBOL or null",
            "action": "buy|sell|rebalance",
            "quantity": number or null,
            "price_threshold": number or null, 
            "operator": "greater_than|less_than|equal_to",
            "target_allocation": number or null,
            "portfolio_value": number or null,
            "screening_criteria": "gainers|losers|actives or null",
            "count": number or null,
            "raw_condition": "original condition text"
        }
    ],
    "pre_checks": ["list of things to check before execution"],
    "execution_steps": ["step by step execution plan"],
    "risk_warnings": ["potential risks or warnings"]
}

IMPORTANT JSON FORMATTING RULES:
- Use null (not "null" or "None") for missing values
- Numbers should be plain numbers (e.g., 150, 50000, 5) NOT strings
- Do not include $ symbols or commas in numeric values
- Boolean values should be true/false, not strings

Examples:
- "buy aapl if price is greater than $120" → price_threshold condition with price_threshold: 120
- "buy the top 5 gainers divided equally amongst $50,000" → top_gainers condition with count: 5, portfolio_value: 50000
- "maintain equal value for all positions in the portfolio" → portfolio_rebalance condition
- "sell all positions if SPY drops below $400" → price_threshold condition with sell action

Always include practical execution steps and relevant risk warnings."""

        # Add specific formatting instruction for Ollama
        if self.llm_source == "ollama":
            base_prompt += "\n\nIMPORTANT: Return ONLY the JSON object, no additional text or explanations."
        
        return base_prompt
    
    def _extract_json_response(self, response_text: str, original_command: str) -> Optional[TradingPlan]:
        """Extract JSON from LLM response and create trading plan"""
        # Try multiple JSON extraction strategies
        json_candidates = self._find_json_candidates(response_text)
        
        for json_str in json_candidates:
            try:
                plan_data = json.loads(json_str)
                if self.debug:
                    print(f"Debug - Parsed JSON: {json.dumps(plan_data, indent=2)}")
                return self._create_trading_plan(plan_data)
            except json.JSONDecodeError as e:
                if self.debug:
                    print(f"JSON parse failed for candidate: {e}")
                continue
            except Exception as e:
                if self.debug:
                    print(f"Error creating plan from JSON: {e}")
                continue
        
        # If all JSON parsing failed, show debug info and fall back
        print(error(f"Could not parse any valid JSON from LLM response"))
        if self.debug:
            print(f"Raw response: {response_text[:1000]}...")
            print(f"Tried {len(json_candidates)} JSON candidates")
        
        return self._fallback_parse(original_command)
    
    def _find_json_candidates(self, response_text: str) -> List[str]:
        """Find potential JSON objects in the response text"""
        candidates = []
        
        # Strategy 1: Look for markdown code blocks first
        json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        candidates.extend(json_blocks)
        
        # Strategy 2: Find outermost braces
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            candidates.append(response_text[json_start:json_end])
        
        # Strategy 3: Find balanced brace structures
        brace_count = 0
        start_idx = None
        
        for i, char in enumerate(response_text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    candidates.append(response_text[start_idx:i+1])
                    start_idx = None
        
        # Strategy 4: Try to fix common JSON issues for each candidate
        for candidate in list(candidates):
            fixed = self._attempt_json_repair(candidate)
            if fixed and fixed != candidate:
                candidates.append(fixed)
        
        # Strategy 5: Look for JSON-like patterns with regex
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        candidates.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            # Normalize whitespace for comparison
            normalized = re.sub(r'\s+', ' ', candidate.strip())
            if normalized not in seen and len(normalized) > 10:  # Skip very short candidates
                seen.add(normalized)
                unique_candidates.append(candidate)
        
        if self.debug:
            print(f"Found {len(unique_candidates)} JSON candidates")
        
        return unique_candidates
    
    def _attempt_json_repair(self, json_str: str) -> Optional[str]:
        """Attempt to repair common JSON formatting issues"""
        try:
            original = json_str
            
            # Remove markdown code blocks if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1]
            if "```" in json_str:
                json_str = json_str.split("```")[0]
            
            # Remove trailing commas before closing brackets/braces
            repaired = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fix unquoted keys (more sophisticated)
            # Look for word: pattern and quote the word
            repaired = re.sub(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
            
            # Fix single quotes to double quotes
            repaired = repaired.replace("'", '"')
            
            # Remove comments (// or /* */)
            repaired = re.sub(r'//.*?$', '', repaired, flags=re.MULTILINE)
            repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
            
            # Fix common value issues
            repaired = re.sub(r':\s*null\s*([,}])', r': null\1', repaired)
            repaired = re.sub(r':\s*true\s*([,}])', r': true\1', repaired)
            repaired = re.sub(r':\s*false\s*([,}])', r': false\1', repaired)
            
            # Remove extra whitespace and newlines within strings (preserve structure)
            repaired = re.sub(r'\n\s*', ' ', repaired)
            
            # Test if it's valid JSON now
            json.loads(repaired)
            
            if self.debug and repaired != original:
                print(f"JSON repair successful:")
                print(f"   Original: {original[:100]}...")
                print(f"   Repaired: {repaired[:100]}...")
            
            return repaired
            
        except Exception as e:
            if self.debug:
                print(f"JSON repair failed: {e}")
            return None
    
    def _create_trading_plan(self, plan_data: Dict) -> TradingPlan:
        """Create TradingPlan object from parsed data"""
        conditions = []
        for cond_data in plan_data.get("conditions", []):
            try:
                condition = TradingCondition(
                    condition_type=cond_data.get("condition_type", "custom"),
                    symbol=cond_data.get("symbol"),
                    action=cond_data.get("action", "buy"),
                    quantity=self._safe_decimal_convert(cond_data.get("quantity")),
                    price_threshold=self._safe_decimal_convert(cond_data.get("price_threshold")),
                    operator=cond_data.get("operator", "greater_than"),
                    target_allocation=self._safe_decimal_convert(cond_data.get("target_allocation")),
                    portfolio_value=self._safe_decimal_convert(cond_data.get("portfolio_value")),
                    screening_criteria=cond_data.get("screening_criteria"),
                    count=self._safe_int_convert(cond_data.get("count")),
                    raw_condition=cond_data.get("raw_condition", "")
                )
                conditions.append(condition)
            except Exception as e:
                print(warning(f"Skipping invalid condition: {e}"))
                continue
        
        return TradingPlan(
            description=plan_data.get("description", "Conditional trading plan"),
            conditions=conditions,
            pre_checks=plan_data.get("pre_checks", []),
            execution_steps=plan_data.get("execution_steps", []),
            risk_warnings=plan_data.get("risk_warnings", [])
        )
    
    def _safe_decimal_convert(self, value) -> Optional[Decimal]:
        """Safely convert a value to Decimal, handling various input types"""
        if value is None:
            return None
        
        try:
            # Handle string values that might have currency symbols or commas
            if isinstance(value, str):
                # Remove common currency symbols and commas
                cleaned = value.replace('$', '').replace(',', '').replace('%', '').strip()
                if not cleaned or cleaned.lower() in ['null', 'none', '']:
                    return None
                return Decimal(cleaned)
            
            # Handle numeric values
            if isinstance(value, (int, float)):
                if value == 0:
                    return None  # Treat 0 as None for optional fields
                return Decimal(str(value))
            
            # Try direct conversion
            return Decimal(str(value))
            
        except (ValueError, TypeError, InvalidOperation) as e:
            print(error(f"Could not convert '{value}' to Decimal: {e}"))
            return None
    
    def _safe_int_convert(self, value) -> Optional[int]:
        """Safely convert a value to int"""
        if value is None:
            return None
        
        try:
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned or cleaned.lower() in ['null', 'none', '']:
                    return None
            
            return int(float(str(value)))  # float first to handle "5.0" -> 5
            
        except (ValueError, TypeError) as e:
            print(error(f"Could not convert '{value}' to int: {e}"))
            return None
    
    def _fallback_parse(self, command: str) -> Optional[TradingPlan]:
        """Fallback parser for basic conditions when LLM is not available"""
        lower_cmd = command.lower()
        
        # Basic price threshold pattern
        price_pattern = r'buy\s+([a-z]+)\s+if\s+price\s+is\s+(greater than|less than|above|below)\s+\$?(\d+(?:\.\d+)?)'
        match = re.search(price_pattern, lower_cmd)
        
        if match:
            symbol, operator_text, price = match.groups()
            operator = "greater_than" if operator_text in ["greater than", "above"] else "less_than"
            
            condition = TradingCondition(
                condition_type="price_threshold",
                symbol=symbol.upper(),
                action="buy",
                price_threshold=Decimal(price),
                operator=operator,
                raw_condition=command
            )
            
            return TradingPlan(
                description=f"Buy {symbol.upper()} if price is {operator_text} ${price}",
                conditions=[condition],
                pre_checks=["Check current price", "Verify buying power"],
                execution_steps=["Monitor price", "Execute market buy when condition met"],
                risk_warnings=["Market orders execute at current price", "Price may gap through threshold"]
            )
        
        # Top gainers pattern - improved to capture various formats
        gainers_patterns = [
            r'buy\s+(?:the\s+)?top\s+(\d+)\s+gainers?\s+.*?(?:with|amongst|using|for)\s+\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'buy\s+(?:the\s+)?top\s+(\d+)\s+gainers?\s+.*?\$(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'top\s+(\d+)\s+gainers?\s+.*?\$(\d+(?:,\d{3})*(?:\.\d+)?)',
            r'(\d+)\s+gainers?\s+.*?\$(\d+(?:,\d{3})*(?:\.\d+)?)'
        ]
        
        for pattern in gainers_patterns:
            match = re.search(pattern, lower_cmd)
            if match:
                count, total_value = match.groups()
                total_value = total_value.replace(',', '')
                
                condition = TradingCondition(
                    condition_type="top_gainers",
                    action="buy",
                    count=int(count),
                    portfolio_value=Decimal(total_value),
                    screening_criteria="gainers",
                    raw_condition=command
                )
                
                return TradingPlan(
                    description=f"Buy top {count} gainers with ${total_value} total allocation",
                    conditions=[condition],
                    pre_checks=["Fetch top gainers", "Calculate allocation per stock", "Verify buying power"],
                    execution_steps=["Get gainers list", "Calculate quantities", "Submit buy orders"],
                    risk_warnings=["Gainers may reverse quickly", "Equal allocation may not be optimal"]
                )
                break
        
        # Portfolio rebalance pattern
        if "maintain equal" in lower_cmd and "portfolio" in lower_cmd:
            condition = TradingCondition(
                condition_type="portfolio_rebalance",
                action="rebalance",
                raw_condition=command
            )
            
            return TradingPlan(
                description="Rebalance portfolio to maintain equal allocations",
                conditions=[condition],
                pre_checks=["Get current positions", "Calculate target allocations", "Determine required trades"],
                execution_steps=["Calculate rebalancing trades", "Submit adjustment orders"],
                risk_warnings=["Rebalancing incurs transaction costs", "May trigger tax events"]
            )
        
        return None
    
    def execute_plan(self, plan: TradingPlan, confirm: bool = True) -> bool:
        """Execute a trading plan"""
        if not plan:
            print(error("No valid trading plan to execute"))
            return False
        
        print(f"\n{info('Trading Plan:')} {plan.description}")
        
        # Show risk warnings
        if plan.risk_warnings:
            print(f"\n{warning('Risk Warnings:')}")
            for warning in plan.risk_warnings:
                print(f"   • {warning}")
        
        # Show execution steps
        print(f"\n{info('Execution Steps:')}")
        for i, step in enumerate(plan.execution_steps, 1):
            print(f"   {i}. {step}")
        
        if confirm:
            response = input(f"\n{info('Execute this trading plan?')} [y/N]: ").strip().lower()
            if response != 'y':
                print(error("Trading plan cancelled"))
                return False
        
        # Execute each condition
        plan_success = True
        for condition in plan.conditions:
            try:
                if not self._execute_condition(condition):
                    plan_success = False
            except Exception as e:
                print(error(f"Error executing condition: {e}"))
                plan_success = False
        
        return plan_success
    
    def _execute_condition(self, condition: TradingCondition) -> bool:
        """Execute a single trading condition"""
        if condition.condition_type == "price_threshold":
            return self._execute_price_threshold(condition)
        elif condition.condition_type == "top_gainers":
            return self._execute_top_gainers(condition)
        elif condition.condition_type == "portfolio_rebalance":
            return self._execute_portfolio_rebalance(condition)
        else:
            print(error(f"Unknown condition type: {condition.condition_type}"))
            return False
    
    def _execute_price_threshold(self, condition: TradingCondition) -> bool:
        """Execute price threshold condition"""
        if not condition.symbol or not condition.price_threshold:
            print(error("Invalid price threshold condition"))
            return False
        
        try:
            # Get current price
            from alpaca.data.requests import StockLatestQuoteRequest
            request = StockLatestQuoteRequest(symbol_or_symbols=condition.symbol)
            quote = self.data_client.get_stock_latest_quote(request)[condition.symbol]
            current_price = Decimal(str(quote.ask_price if condition.action == 'buy' else quote.bid_price))
            
            print(f"{condition.symbol} current price: ${current_price}")
            print(f"Threshold: ${condition.price_threshold} ({condition.operator})")
            
            # Check condition
            condition_met = False
            if condition.operator == "greater_than" and current_price > condition.price_threshold:
                condition_met = True
            elif condition.operator == "less_than" and current_price < condition.price_threshold:
                condition_met = True
            
            if condition_met:
                print(success(f"Condition met! Executing {condition.action} order..."))
                
                # Determine quantity if not specified
                quantity = condition.quantity
                if not quantity:
                    qty_input = input(f"How many shares of {condition.symbol}? ").strip()
                    quantity = Decimal(qty_input)
                
                # Execute the order
                return self._submit_market_order(condition.action, condition.symbol, quantity)
            else:
                print(info(f"Condition not yet met (${current_price} vs ${condition.price_threshold})"))
                return True  # Condition successfully checked, just not triggered
                
        except Exception as e:
            print(error(f"Error checking price threshold: {e}"))
            return False
    
    def _execute_top_gainers(self, condition: TradingCondition) -> bool:
        """Execute top gainers buying strategy"""
        try:
            # Get top gainers (this would need to be implemented with market data)
            gainers = self._fetch_top_gainers(condition.count or 5)
            if not gainers:
                print(error("Could not fetch top gainers"))
                return False
            
            total_value = condition.portfolio_value or Decimal('1000')
            allocation_per_stock = total_value / len(gainers)
            
            print(f"Top {len(gainers)} gainers:")
            for i, (symbol, price, change_pct) in enumerate(gainers, 1):
                print(f"   {i}. {symbol}: ${price} ({change_pct:+.2f}%)")
            
            print(info(f"Allocating ${allocation_per_stock:.2f} per stock"))
            
            buy_success = True
            for symbol, price, _ in gainers:
                try:
                    quantity = int(allocation_per_stock / Decimal(str(price)))
                    if quantity > 0:
                        print(info(f"Buying {quantity} shares of {symbol}..."))
                        if not self._submit_market_order("buy", symbol, Decimal(quantity)):
                            buy_success = False
                    else:
                        print(warning(f"Skipping {symbol} - allocation too small for 1 share"))
                except Exception as e:
                    print(error(f"Error buying {symbol}: {e}"))
                    buy_success = False
            
            return buy_success
            
        except Exception as e:
            print(error(f"Error executing top gainers strategy: {e}"))
            return False
    
    def _execute_portfolio_rebalance(self, condition: TradingCondition) -> bool:
        """Execute portfolio rebalancing"""
        try:
            # Get current positions
            positions = self.trading_client.get_all_positions()
            if not positions:
                print(error("No positions to rebalance"))
                return False
            
            # Calculate current total value
            total_value = sum(Decimal(str(pos.market_value)) for pos in positions)
            target_value_per_position = total_value / len(positions)
            
            print(f"Portfolio rebalancing:")
            print(f"   Total value: ${total_value:,.2f}")
            print(f"   Target per position: ${target_value_per_position:,.2f}")
            
            rebalance_trades = []
            for pos in positions:
                current_value = Decimal(str(pos.market_value))
                difference = target_value_per_position - current_value
                
                if abs(difference) > Decimal('10'):  # Only rebalance if difference > $10
                    current_price = Decimal(str(pos.avg_entry_price))  # Simplified
                    shares_to_trade = int(abs(difference) / current_price)
                    
                    if shares_to_trade > 0:
                        action = "buy" if difference > 0 else "sell"
                        rebalance_trades.append((pos.symbol, action, shares_to_trade, difference))
            
            if not rebalance_trades:
                print(success("Portfolio already balanced"))
                return True
            
            print(f"\nRebalancing trades:")
            for symbol, action, shares, difference in rebalance_trades:
                print(f"   {action.upper()} {shares} {symbol} (${difference:+.2f})")
            
            # Execute trades
            rebalance_success = True
            for symbol, action, shares, _ in rebalance_trades:
                try:
                    if not self._submit_market_order(action, symbol, Decimal(shares)):
                        rebalance_success = False
                except Exception as e:
                    print(error(f"Error rebalancing {symbol}: {e}"))
                    rebalance_success = False
            
            return rebalance_success
            
        except Exception as e:
            print(error(f"Error executing portfolio rebalance: {e}"))
            return False
    
    def _fetch_top_gainers(self, count: int) -> List[Tuple[str, float, float]]:
        """Fetch top gainers from market data"""
        try:
            # This would integrate with your existing market data fetching
            # For now, return mock data
            import requests
            api_key = os.getenv("FMP_API_KEY")
            if not api_key:
                print(warning("FMP_API_KEY required for market screening"))
                return []
            
            url = f"https://financialmodelingprep.com/api/v3/gainers?apikey={api_key}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            gainers = []
            for item in data[:count]:
                symbol = item.get('ticker', '')
                price = float(item.get('price', 0))
                change_pct = float(item.get('changesPercentage', 0))
                gainers.append((symbol, price, change_pct))
            
            return gainers
            
        except Exception as e:
            print(error(f"Error fetching gainers: {e}"))
            return []
    
    def _submit_market_order(self, action: str, symbol: str, quantity: Decimal) -> bool:
        """Submit a market order"""
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            
            side = OrderSide.BUY if action.lower() == 'buy' else OrderSide.SELL
            
            order = MarketOrderRequest(
                symbol=symbol.upper(),
                qty=str(quantity),
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            result = self.trading_client.submit_order(order)
            print(success(f"Submitted {action.upper()} order: {quantity} {symbol} (id: {str(result.id)[:8]})"))
            return True
            
        except Exception as e:
            print(error(f"Failed to submit {action} order for {symbol}: {e}"))
            return False

# Utility functions
def create_conditional_agent(trading_client, data_client, symbol_fetcher, debug=False):
    """Factory function to create conditional trading agent"""
    return ConditionalTradingAgent(trading_client, data_client, symbol_fetcher, debug=debug)
