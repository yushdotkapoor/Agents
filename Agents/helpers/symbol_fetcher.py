"""
Symbol Fetcher - Uses Alpaca API to fetch and cache stock symbols
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
import re

class SymbolFetcher:
    def __init__(self, api_key, api_secret, paper=True):
        self.trading_client = TradingClient(api_key, api_secret, paper=paper)
        self.stock_client = StockHistoricalDataClient(api_key, api_secret)
        self.crypto_client = CryptoHistoricalDataClient(api_key, api_secret)
        self.symbols_file = Path(__file__).parent.parent / 'data' / 'stock_symbols.json'

    def load_cache(self):
        """Load existing symbol cache"""
        try:
            with open(self.symbols_file, 'r') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            return {
                "company_symbols": {},
                "last_updated": None,
                "cache_duration_hours": 24
            }
    
    def save_cache(self, data):
        """Save symbol cache to JSON file"""
        data["last_updated"] = datetime.now().isoformat()
        with open(self.symbols_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def is_cache_valid(self, data):
        """Check if cache is still valid"""
        if not data.get("last_updated"):
            return False
        
        last_updated = datetime.fromisoformat(data["last_updated"])
        cache_duration = timedelta(hours=data.get("cache_duration_hours", 24))
        
        return datetime.now() - last_updated < cache_duration
    
    def fetch_all_assets(self):
        """Fetch all available assets from Alpaca"""
        print("Fetching all assets from Alpaca...")
        
        # Get all stocks
        stock_request = GetAssetsRequest(
            status=AssetStatus.ACTIVE,
            asset_class=AssetClass.US_EQUITY
        )
        stocks = self.trading_client.get_all_assets(stock_request)
        
        # Get crypto assets
        crypto_request = GetAssetsRequest(
            status=AssetStatus.ACTIVE,
            asset_class=AssetClass.CRYPTO
        )
        cryptos = self.trading_client.get_all_assets(crypto_request)
        
        print(f"Found {len(stocks)} stocks and {len(cryptos)} crypto assets")
        return stocks, cryptos
    
    def clean_company_name(self, name):
        """Clean company name for better matching"""
        if not name:
            return ""
        
        # Convert to lowercase
        name = name.lower()
        
        # Remove common suffixes (more comprehensive list)
        suffixes = [
            r'\s+inc\.?$', r'\s+corp\.?$', r'\s+ltd\.?$', r'\s+llc\.?$',
            r'\s+company$', r'\s+co\.?$', r'\s+plc$', r'\s+sa$',
            r'\s+holdings?$', r'\s+group$', r'\s+technologies?$',
            r'\s+systems?$', r'\s+solutions?$', r'\s+services?$',
            r'\s+international$', r'\s+global$', r'\s+enterprises?$',
            r'\s+industries?$', r'\s+networks?$', r'\s+communications?$',
            r'\s+financial$', r'\s+bancorp$', r'\s+bancorporation$',
            r'\s+trust$', r'\s+capital$', r'\s+investment$',
            r'\s+airlines?$', r'\s+airways?$', r'\s+air\s+lines?$',
            r'\s+motors?$', r'\s+automotive$', r'\s+auto$',
            r'\s+pharmaceuticals?$', r'\s+pharma$', r'\s+healthcare$',
            r'\s+medical$', r'\s+therapeutics?$', r'\s+biotech$',
            r'\s+energy$', r'\s+oil$', r'\s+gas$', r'\s+petroleum$',
            r'\s+mining$', r'\s+metals?$', r'\s+steel$',
            r'\s+real\s+estate$', r'\s+reit$', r'\s+properties$',
            r'\s+entertainment$', r'\s+media$', r'\s+broadcasting$',
            r'\s+telecom$', r'\s+communications?$', r'\s+wireless$',
            r'\s+software$', r'\s+technology$', r'\s+tech$',
            r'\s+semiconductor$', r'\s+electronics$', r'\s+devices$',
            r'\s+retail$', r'\s+stores?$', r'\s+shops?$',
            r'\s+restaurants?$', r'\s+foods?$', r'\s+beverage$',
            r'\s+common\s+stock$', r'\s+class\s+[abc]$', r'\s+series\s+[abc]$'
        ]
        
        for suffix in suffixes:
            name = re.sub(suffix, '', name)
        
        # Remove special characters and extra spaces
        name = re.sub(r'[^\w\s&]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Handle common variations and abbreviations
        replacements = {
            'and': '&',
            'corporation': '',
            'incorporated': '',
            'limited': '',
            'the ': '',
            ' the$': '',
            'international': 'intl',
            'technology': 'tech',
            'technologies': 'tech',
            'systems': 'sys',
            'solutions': 'sol',
            'communications': 'comm',
            'entertainment': 'ent',
            'pharmaceutical': 'pharma',
            'pharmaceuticals': 'pharma',
            'automobile': 'auto',
            'laboratories': 'labs',
            'laboratory': 'lab',
        }
        
        for old, new in replacements.items():
            name = re.sub(r'\b' + re.escape(old) + r'\b', new, name).strip()
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def generate_name_variations(self, name):
        """Generate common variations of a company name including colloquial forms"""
        if not name:
            return []
        
        variations = set()
        clean_name = self.clean_company_name(name)
        variations.add(clean_name)
        
        # Add original name
        variations.add(name.lower())
        
        # Add variations with/without spaces
        variations.add(clean_name.replace(' ', ''))
        
        # Add acronym if multiple words
        words = clean_name.split()
        if len(words) > 1:
            acronym = ''.join(word[0] for word in words if word)
            variations.add(acronym)
        
        # Generate colloquial variations based on common patterns
        colloquial_variations = self._generate_colloquial_variations(clean_name, words)
        variations.update(colloquial_variations)
        
        # Remove empty strings and overly short variations that could cause conflicts
        variations = {v for v in variations if v and len(v) > 2}
        
        return list(variations)
    
    def _generate_colloquial_variations(self, clean_name, words):
        """Generate colloquial and common name variations"""
        variations = set()
        
        # Common company name patterns and their colloquial forms
        colloquial_mappings = {
            # Airlines
            'american airlines': ['american airlines', 'american air', 'aa airlines'],
            'delta air lines': ['delta', 'delta airlines'],
            'united airlines': ['united', 'united air'],
            'southwest airlines': ['southwest', 'southwest air'],
            'jetblue airways': ['jetblue', 'jet blue'],
            
            # Tech companies
            'alphabet': ['google', 'alphabet'],
            'meta platforms': ['facebook', 'meta', 'fb'],
            'tesla': ['tesla motors'],
            'microsoft': ['msft', 'microsoft corp'],
            'apple': ['apple computer', 'apple inc'],
            'amazon': ['amazon.com'],
            'netflix': ['netflix inc'],
            
            # Financial
            'jpmorgan chase': ['jpmorgan', 'jp morgan', 'chase', 'jpm'],
            'bank of america': ['bofa', 'bank america', 'bac'],
            'wells fargo': ['wells', 'wfc'],
            'goldman sachs': ['goldman', 'gs'],
            'morgan stanley': ['morgan stanley', 'ms'],
            'berkshire hathaway': ['berkshire', 'brk'],
            
            # Retail
            'walmart': ['wal-mart', 'wal mart'],
            'target': ['target corp'],
            'home depot': ['homedepot', 'home depot inc'],
            'mcdonalds': ['mcdonald', 'mickey d', 'golden arches'],
            'starbucks': ['sbux', 'starbucks coffee'],
            
            # Automotive
            'general motors': ['gm', 'general motor'],
            'ford motor': ['ford', 'ford motors'],
            'tesla': ['tesla inc', 'tesla motors'],
            
            # Energy
            'exxon mobil': ['exxon', 'exxonmobil', 'mobil'],
            'chevron': ['chevron corp'],
            'conocophillips': ['conoco', 'phillips'],
            
            # Healthcare/Pharma
            'johnson & johnson': ['johnson and johnson', 'jnj', 'j&j'],
            'pfizer': ['pfizer inc'],
            'merck': ['merck & co'],
            
            # Crypto variations
            'bitcoin': ['btc', 'bitcoin cash'],  # Note: bitcoin cash is different, but people often confuse
            'ethereum': ['eth', 'ether'],
            'dogecoin': ['doge', 'dog coin'],
            'litecoin': ['ltc', 'lite coin'],
            'chainlink': ['link'],
            'cardano': ['ada'],
            'polkadot': ['dot'],
            'solana': ['sol'],
            'avalanche': ['avax'],
            'polygon': ['matic'],
            'uniswap': ['uni'],
        }
        
        # Check if the clean name matches any known colloquial patterns
        for canonical_name, colloquials in colloquial_mappings.items():
            if canonical_name in clean_name or any(word in canonical_name for word in words):
                variations.update(colloquials)
        
        # Generate variations based on word patterns
        if len(words) >= 2:
            # For "Company Name Inc/Corp/etc", add just "Company Name"
            if words[-1] in ['inc', 'corp', 'company', 'group', 'holdings']:
                short_name = ' '.join(words[:-1])
                variations.add(short_name)
                variations.add(short_name.replace(' ', ''))
            
            # For "First Second Third", try "First Second" and "First"
            if len(words) >= 3:
                variations.add(' '.join(words[:2]))
                variations.add(words[0])
            elif len(words) == 2:
                variations.add(words[0])
        
        # Handle specific industry patterns
        self._add_industry_specific_variations(clean_name, words, variations)
        
        return variations
    
    def _add_industry_specific_variations(self, clean_name, words, variations):
        """Add industry-specific colloquial variations"""
        
        # Airlines: remove "airlines", "airways", "air"
        if any(word in clean_name for word in ['airlines', 'airways', 'air lines']):
            base_name = clean_name
            for suffix in ['airlines', 'airways', 'air lines', 'air']:
                base_name = base_name.replace(suffix, '').strip()
            if base_name:
                variations.add(base_name)
                variations.add(base_name.replace(' ', ''))
        
        # Banks: add "bank" if not present, remove if present
        if 'bank' in clean_name:
            no_bank = clean_name.replace('bank', '').replace('  ', ' ').strip()
            if no_bank:
                variations.add(no_bank)
        elif any(word in clean_name for word in ['financial', 'trust', 'credit']):
            variations.add(f"{clean_name} bank")
        
        # Tech companies: add common tech suffixes/prefixes
        if any(word in clean_name for word in ['tech', 'systems', 'software', 'digital']):
            base = clean_name.replace('technologies', 'tech').replace('systems', 'sys')
            variations.add(base)
        
        # Automotive: add "motors", "motor", "auto"
        if any(word in clean_name for word in ['motor', 'auto', 'car']):
            base_name = clean_name.replace('motors', 'motor').replace('motor', '').strip()
            if base_name:
                variations.add(base_name)
                variations.add(f"{base_name} motors")
                variations.add(f"{base_name} motor")
        
        # Energy companies
        if any(word in clean_name for word in ['oil', 'gas', 'energy', 'petroleum']):
            # People often refer to energy companies by location or short name
            if 'exxon' in clean_name:
                variations.update(['exxon', 'mobil', 'esso'])
            elif 'chevron' in clean_name:
                variations.update(['chevron', 'texaco'])
            elif 'shell' in clean_name:
                variations.update(['shell', 'royal dutch shell'])
        
        # Retail chains
        if any(word in clean_name for word in ['stores', 'retail', 'mart', 'shop']):
            base_name = clean_name
            for suffix in ['stores', 'retail', 'inc', 'corp']:
                base_name = base_name.replace(suffix, '').strip()
            if base_name:
                variations.add(base_name)
        
        # Crypto-specific patterns
        if 'coin' in clean_name or 'token' in clean_name:
            base_name = clean_name.replace('coin', '').replace('token', '').strip()
            if base_name:
                variations.add(base_name)
                variations.add(f"{base_name}coin")
                variations.add(f"{base_name} coin")
    
    def build_symbol_mapping(self):
        """Build comprehensive symbol mapping from Alpaca data"""
        stocks, cryptos = self.fetch_all_assets()
        
        symbol_mapping = {}
        processed_count = 0
        
        print("Processing stock assets...")
        for asset in stocks:
            try:
                symbol = asset.symbol
                name = asset.name
                
                if name:
                    variations = self.generate_name_variations(name)
                    for variation in variations:
                        if variation not in symbol_mapping:
                            symbol_mapping[variation] = symbol
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} assets...")
                    
            except Exception as e:
                print(f"Error processing stock {asset.symbol}: {e}")
                continue
        
        print("Processing crypto assets...")
        for asset in cryptos:
            try:
                symbol = asset.symbol
                name = asset.name
                
                if name:
                    variations = self.generate_name_variations(name)
                    for variation in variations:
                        if variation not in symbol_mapping:
                            symbol_mapping[variation] = symbol
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} total assets...")
                    
            except Exception as e:
                print(f"Error processing crypto {asset.symbol}: {e}")
                continue
        
        print(f"Generated {len(symbol_mapping)} symbol mappings")
        return symbol_mapping
    
    def update_symbols(self, force_refresh=False):
        """Update symbol mappings from Alpaca API"""
        data = self.load_cache()
        
        if not force_refresh and self.is_cache_valid(data):
            print("Symbol cache is still valid, skipping update")
            return data["company_symbols"]
        
        print("Updating symbol mappings from Alpaca API...")
        
        try:
            symbol_mapping = self.build_symbol_mapping()
            
            data["company_symbols"] = symbol_mapping
            self.save_cache(data)
            
            print(f"Successfully updated {len(symbol_mapping)} symbol mappings")
            return symbol_mapping
            
        except Exception as e:
            print(f"Error updating symbols: {e}")
            print("Using existing cache if available")
            return data.get("company_symbols", {})
    
    def search_symbol(self, company_name):
        """Search for a symbol by company name with enhanced matching"""
        data = self.load_cache()
        symbols = data.get("company_symbols", {})
        
        # First, normalize the input to handle common user patterns
        normalized_input = self._normalize_user_input(company_name)
        
        # Try exact match first
        clean_name = self.clean_company_name(normalized_input)
        if clean_name in symbols:
            return symbols[clean_name]
        
        # Try variations of the input
        variations = self.generate_name_variations(normalized_input)
        for variation in variations:
            if variation in symbols:
                return symbols[variation]
        
        # Try fuzzy matching for common typos and abbreviations
        fuzzy_match = self._fuzzy_search(normalized_input, symbols)
        if fuzzy_match:
            return fuzzy_match
        
        # If not found in cache, try to fetch new data
        print(f"Symbol not found for '{company_name}', trying API search...")
        
        try:
            # Search in all assets
            all_assets = self.trading_client.get_all_assets()
            for asset in all_assets:
                if asset.name and self._is_name_match(normalized_input, asset.name):
                    # Add to cache for future use
                    asset_variations = self.generate_name_variations(asset.name)
                    for variation in asset_variations:
                        if variation not in symbols:
                            symbols[variation] = asset.symbol
                    self.save_cache(data)
                    return asset.symbol
        except Exception as e:
            print(f"Error searching for symbol: {e}")
        
        return None
    
    def _normalize_user_input(self, user_input):
        """Normalize user input to handle common patterns"""
        if not user_input:
            return ""
        
        # Convert to lowercase
        normalized = user_input.lower().strip()
        
        # Handle common user input patterns
        patterns = {
            # Remove common trading-related words
            r'\b(stock|stocks|share|shares|company|corp|corporation)\b': '',
            r'\b(buy|sell|trade|invest in)\b': '',
            
            # Handle possessive forms
            r"'s\b": '',
            
            # Normalize spacing around &
            r'\s*&\s*': ' & ',
            
            # Handle common misspellings and abbreviations
            r'\bgoog\b': 'google',
            r'\bfb\b': 'facebook',
            r'\btsla\b': 'tesla',
            r'\baapl\b': 'apple',
            r'\bmsft\b': 'microsoft',
            r'\bamzn\b': 'amazon',
            r'\bnflx\b': 'netflix',
            r'\bnvda\b': 'nvidia',
            r'\bwal-?mart\b': 'walmart',
            r'\bmcd\b': 'mcdonalds',
            r'\bko\b': 'coca cola',
            r'\bjnj\b': 'johnson & johnson',
            r'\bbofa\b': 'bank of america',
            r'\bjpm\b': 'jpmorgan',
            r'\bgs\b': 'goldman sachs',
        }
        
        for pattern, replacement in patterns.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        # Clean up extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _fuzzy_search(self, query, symbols):
        """Perform fuzzy matching for common variations and typos"""
        query_words = set(query.lower().split())
        
        best_match = None
        best_score = 0
        
        for symbol_name in symbols.keys():
            if len(symbol_name) < 3:  # Skip very short names that could cause false matches
                continue
                
            symbol_words = set(symbol_name.lower().split())
            
            # Calculate word overlap score
            if query_words and symbol_words:
                overlap = len(query_words.intersection(symbol_words))
                score = overlap / len(query_words.union(symbol_words))
                
                # Boost score for exact substring matches
                if query.lower() in symbol_name.lower() or symbol_name.lower() in query.lower():
                    score += 0.3
                
                # Boost score for similar length names
                if abs(len(query) - len(symbol_name)) <= 2:
                    score += 0.1
                
                if score > best_score and score > 0.6:  # Threshold for fuzzy match
                    best_score = score
                    best_match = symbols[symbol_name]
        
        return best_match
    
    def _is_name_match(self, user_input, asset_name):
        """Check if user input matches an asset name"""
        if not user_input or not asset_name:
            return False
        
        user_clean = self.clean_company_name(user_input)
        asset_clean = self.clean_company_name(asset_name)
        
        # Check if any variation of user input matches asset variations
        user_variations = self.generate_name_variations(user_input)
        asset_variations = self.generate_name_variations(asset_name)
        
        return bool(set(user_variations).intersection(set(asset_variations)))

# Utility function for easy usage
def update_symbol_cache(api_key, api_secret, force_refresh=False, paper=True):
    """Convenience function to update symbol cache"""
    fetcher = SymbolFetcher(api_key, api_secret, paper=paper)
    return fetcher.update_symbols(force_refresh=force_refresh)

def search_company_symbol(company_name, api_key=None, api_secret=None, paper=True):
    """Convenience function to search for a company symbol"""
    if api_key and api_secret:
        fetcher = SymbolFetcher(api_key, api_secret, paper=paper)
        return fetcher.search_symbol(company_name)
    else:
        # Just load from cache
        fetcher = SymbolFetcher("", "", paper=paper)
        data = fetcher.load_cache()
        symbols = data.get("company_symbols", {})
        clean_name = fetcher.clean_company_name(company_name)
        return symbols.get(clean_name)

if __name__ == "__main__":
    # Example usage - you'll need to provide your Alpaca credentials
    import os
    
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not api_secret:
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("Or modify the script to include your credentials")
        exit(1)
    
    # Update symbol cache
    update_symbol_cache(api_key, api_secret, force_refresh=True)
    
    # Test search
    test_companies = ["apple", "microsoft", "tesla", "bitcoin"]
    for company in test_companies:
        symbol = search_company_symbol(company)
        print(f"{company} -> {symbol}")
