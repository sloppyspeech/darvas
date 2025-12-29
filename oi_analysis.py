"""
Open Interest Analysis Module
=============================
Fetches OI data from NSE and provides quadrant classification
for derivative market analysis.
"""

import requests
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import json

# NSE Headers for API requests
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.nseindia.com/',
}

# Common F&O stocks
FNO_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT",
    "AXISBANK", "ITC", "BAJFINANCE", "MARUTI", "ASIANPAINT",
    "HCLTECH", "SUNPHARMA", "TATAMOTORS", "WIPRO", "TITAN",
    "ULTRACEMCO", "NTPC", "POWERGRID", "ONGC", "COALINDIA",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "ADANIPORTS", "GRASIM",
    "TECHM", "INDUSINDBK", "BAJAJFINSV", "NESTLEIND", "DRREDDY",
    "CIPLA", "APOLLOHOSP", "DIVISLAB", "BRITANNIA", "EICHERMOT"
]

# Sector mapping
SECTOR_MAP = {
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
    "Oil & Gas": ["RELIANCE", "ONGC", "COALINDIA"],
    "Auto": ["MARUTI", "TATAMOTORS", "EICHERMOT"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "APOLLOHOSP", "DIVISLAB"],
    "FMCG": ["HINDUNILVR", "ITC", "BRITANNIA", "NESTLEIND"],
    "Metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO"],
    "Infra": ["LT", "ADANIPORTS", "ULTRACEMCO", "GRASIM"],
    "Finance": ["BAJFINANCE", "BAJAJFINSV"],
    "Others": ["TITAN", "ASIANPAINT", "NTPC", "POWERGRID"]
}


@dataclass
class OIData:
    """Data class for Open Interest analysis."""
    symbol: str
    spot_price: float
    prev_price: float
    price_change: float
    price_change_pct: float
    current_month_oi: float
    current_month_oi_change: float
    current_month_oi_change_pct: float
    next_month_oi: float = 0.0
    next_month_oi_change: float = 0.0
    next_month_oi_change_pct: float = 0.0
    total_call_oi: float = 0.0
    total_put_oi: float = 0.0
    pcr_ratio: float = 0.0
    quadrant: str = ""
    timestamp: str = ""


def classify_quadrant(price_change: float, oi_change: float) -> str:
    """
    Classify stock into one of four quadrants based on Price and OI movement.
    
    Args:
        price_change: Change in price (positive = up)
        oi_change: Change in open interest (positive = up)
    
    Returns:
        Quadrant name
    """
    if price_change > 0 and oi_change > 0:
        return "Long Buildup"      # Bullish - New longs entering
    elif price_change < 0 and oi_change > 0:
        return "Short Buildup"     # Bearish - New shorts entering
    elif price_change > 0 and oi_change < 0:
        return "Short Covering"    # Rally - Shorts exiting
    elif price_change < 0 and oi_change < 0:
        return "Long Unwinding"    # Bearish Exit - Longs exiting
    else:
        return "Neutral"


def calculate_pcr(put_oi: float, call_oi: float) -> float:
    """
    Calculate Put-Call Ratio.
    
    Args:
        put_oi: Total Put Open Interest
        call_oi: Total Call Open Interest
    
    Returns:
        PCR ratio (Put OI / Call OI)
    """
    if call_oi == 0:
        return 0.0
    return round(put_oi / call_oi, 2)


def get_nse_session() -> requests.Session:
    """Create a session with NSE cookies."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    # Get cookies from NSE homepage first
    try:
        session.get("https://www.nseindia.com/", timeout=10)
    except:
        pass
    return session


def fetch_oi_data_mock(symbol: str) -> Optional[OIData]:
    """
    Generate mock OI data for testing purposes.
    Replace this with actual NSE API calls in production.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
    
    Returns:
        OIData object with mock values
    """
    import random
    
    # Generate realistic mock data
    base_price = random.uniform(500, 3000)
    price_change = random.uniform(-5, 5)
    price_change_pct = random.uniform(-3, 3)
    
    current_oi = random.uniform(1000000, 50000000)
    oi_change = random.uniform(-2000000, 2000000)
    oi_change_pct = (oi_change / current_oi) * 100 if current_oi > 0 else 0
    
    next_oi = random.uniform(500000, 20000000)
    next_oi_change = random.uniform(-500000, 500000)
    next_oi_change_pct = (next_oi_change / next_oi) * 100 if next_oi > 0 else 0
    
    call_oi = random.uniform(500000, 10000000)
    put_oi = random.uniform(500000, 10000000)
    
    quadrant = classify_quadrant(price_change_pct, oi_change_pct)
    pcr = calculate_pcr(put_oi, call_oi)
    
    return OIData(
        symbol=symbol,
        spot_price=round(base_price, 2),
        prev_price=round(base_price - price_change, 2),
        price_change=round(price_change, 2),
        price_change_pct=round(price_change_pct, 2),
        current_month_oi=round(current_oi, 0),
        current_month_oi_change=round(oi_change, 0),
        current_month_oi_change_pct=round(oi_change_pct, 2),
        next_month_oi=round(next_oi, 0),
        next_month_oi_change=round(next_oi_change, 0),
        next_month_oi_change_pct=round(next_oi_change_pct, 2),
        total_call_oi=round(call_oi, 0),
        total_put_oi=round(put_oi, 0),
        pcr_ratio=pcr,
        quadrant=quadrant,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


def fetch_oi_data_live(symbol: str) -> Optional[OIData]:
    """
    Fetch real-time OI data from NSE website.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
    
    Returns:
        OIData object with live values or None if failed
    """
    try:
        print(f"\n{'='*60}")
        print(f"[INFO] Fetching live OI data for {symbol}...")
        
        # Create session with cookies
        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        
        # First hit the main page to get cookies
        print(f"[DEBUG] Getting cookies from NSE homepage...")
        homepage_resp = session.get("https://www.nseindia.com/", timeout=10)
        print(f"[DEBUG] Homepage status: {homepage_resp.status_code}")
        
        # Fetch derivative quote
        url = f"https://www.nseindia.com/api/quote-derivative?symbol={symbol}"
        print(f"[DEBUG] Fetching: {url}")
        response = session.get(url, timeout=15)
        
        print(f"[DEBUG] Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"[ERROR] NSE API returned status {response.status_code} for {symbol}")
            print(f"[DEBUG] Response text: {response.text[:500] if response.text else 'empty'}")
            return None
        
        data = response.json()
        print(f"[DEBUG] Response keys: {list(data.keys()) if data else 'None'}")
        
        # Parse the response
        stock_info = data.get('stocks', [])
        if not stock_info:
            print(f"[ERROR] No 'stocks' data in response for {symbol}")
            print(f"[DEBUG] Full response: {str(data)[:1000]}")
            return None
        
        print(f"[DEBUG] Found {len(stock_info)} stock entries")
        
        # Get underlying value (spot price)
        underlying = data.get('underlyingValue', 0)
        prev_close = data.get('underlyingValue', underlying)  # Fallback
        
        # Aggregate OI from futures contracts
        current_month_oi = 0
        current_month_oi_change = 0
        next_month_oi = 0
        next_month_oi_change = 0
        total_call_oi = 0
        total_put_oi = 0
        
        # Get spot price change from the data
        price_change = 0
        price_change_pct = 0
        
        for stock in stock_info:
            metadata = stock.get('metadata', {})
            market_data = stock.get('marketDeptOrderBook', {}).get('tradeInfo', {})
            
            instrument_type = metadata.get('instrumentType', '')
            expiry_date = metadata.get('expiryDate', '')
            
            # Check if this is current or next month
            # NSE returns data sorted by expiry
            if instrument_type == 'Stock Futures':
                oi = market_data.get('openInterest', 0) or 0
                oi_change = market_data.get('changeinOpenInterest', 0) or 0
                ltp = metadata.get('lastPrice', 0) or 0
                prev_ltp = metadata.get('prevClose', ltp) or ltp
                
                if current_month_oi == 0:
                    # First futures contract (current month)
                    current_month_oi = oi
                    current_month_oi_change = oi_change
                    price_change = ltp - prev_ltp
                    price_change_pct = ((ltp - prev_ltp) / prev_ltp * 100) if prev_ltp > 0 else 0
                    underlying = ltp  # Use futures price if spot not available
                else:
                    # Next month
                    next_month_oi = oi
                    next_month_oi_change = oi_change
                    
            elif instrument_type == 'Stock Options':
                option_type = metadata.get('optionType', '')
                oi = market_data.get('openInterest', 0) or 0
                
                if option_type == 'Call':
                    total_call_oi += oi
                elif option_type == 'Put':
                    total_put_oi += oi
        
        # Calculate percentages
        current_month_oi_change_pct = (current_month_oi_change / current_month_oi * 100) if current_month_oi > 0 else 0
        next_month_oi_change_pct = (next_month_oi_change / next_month_oi * 100) if next_month_oi > 0 else 0
        
        # Calculate PCR and quadrant
        pcr = calculate_pcr(total_put_oi, total_call_oi)
        quadrant = classify_quadrant(price_change_pct, current_month_oi_change_pct)
        
        print(f"[INFO] Successfully fetched data for {symbol}: Price={underlying:.2f}, OI Change={current_month_oi_change_pct:.2f}%, Quadrant={quadrant}")
        
        return OIData(
            symbol=symbol,
            spot_price=round(underlying, 2),
            prev_price=round(underlying - price_change, 2),
            price_change=round(price_change, 2),
            price_change_pct=round(price_change_pct, 2),
            current_month_oi=round(current_month_oi, 0),
            current_month_oi_change=round(current_month_oi_change, 0),
            current_month_oi_change_pct=round(current_month_oi_change_pct, 2),
            next_month_oi=round(next_month_oi, 0),
            next_month_oi_change=round(next_month_oi_change, 0),
            next_month_oi_change_pct=round(next_month_oi_change_pct, 2),
            total_call_oi=round(total_call_oi, 0),
            total_put_oi=round(total_put_oi, 0),
            pcr_ratio=pcr,
            quadrant=quadrant,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except requests.exceptions.Timeout:
        print(f"[ERROR] Timeout fetching OI data for {symbol}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request error for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Error parsing OI data for {symbol}: {e}")
        return None


def fetch_oi_data(symbol: str, use_mock: bool = True) -> Optional[OIData]:
    """
    Fetch OI data for a given stock symbol.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE")
        use_mock: If True, use mock data; else try live NSE API
    
    Returns:
        OIData object or None if failed (for non-F&O stocks)
    """
    if use_mock:
        return fetch_oi_data_mock(symbol)
    
    # Try fetching live data - NO FALLBACK to mock when use_mock is False
    live_data = fetch_oi_data_live(symbol)
    
    if live_data:
        return live_data
    else:
        # Return None for non-F&O stocks - do not fallback to mock data
        print(f"[ERROR] Could not fetch live OI data for {symbol}. This stock may not be in the F&O segment.")
        return None


def fetch_multiple_oi_data(symbols: List[str], use_mock: bool = True) -> List[OIData]:
    """
    Fetch OI data for multiple symbols.
    
    Args:
        symbols: List of stock symbols
        use_mock: If True, use mock data
    
    Returns:
        List of OIData objects
    """
    results = []
    for symbol in symbols:
        data = fetch_oi_data(symbol, use_mock=use_mock)
        if data:
            results.append(data)
    return results


def get_stocks_by_sector(sector: str) -> List[str]:
    """
    Get list of stocks for a given sector.
    
    Args:
        sector: Sector name
    
    Returns:
        List of stock symbols
    """
    if sector == "All":
        return FNO_STOCKS
    return SECTOR_MAP.get(sector, [])


def get_sectors() -> List[str]:
    """Get list of available sectors."""
    return ["All"] + list(SECTOR_MAP.keys())


def get_quadrant_color(quadrant: str) -> str:
    """Get color for each quadrant."""
    colors = {
        "Long Buildup": "#00C853",      # Green - Bullish
        "Short Buildup": "#FF1744",     # Red - Bearish
        "Short Covering": "#2196F3",    # Blue - Rally
        "Long Unwinding": "#FF9800",    # Orange - Bearish Exit
        "Neutral": "#9E9E9E"            # Gray
    }
    return colors.get(quadrant, "#9E9E9E")


def get_quadrant_emoji(quadrant: str) -> str:
    """Get emoji for each quadrant."""
    emojis = {
        "Long Buildup": "🟢",
        "Short Buildup": "🔴",
        "Short Covering": "🔵",
        "Long Unwinding": "🟠",
        "Neutral": "⚪"
    }
    return emojis.get(quadrant, "⚪")


def format_oi_for_llm(data: OIData) -> Dict:
    """
    Format OI data as JSON for LLM input.
    
    Args:
        data: OIData object
    
    Returns:
        Dictionary formatted for LLM prompt
    """
    return {
        "ticker": data.symbol,
        "spot_price": data.spot_price,
        "price_change_pct": f"{data.price_change_pct:+.2f}%",
        "current_month_OI_change": f"{data.current_month_oi_change_pct:+.2f}%",
        "next_month_OI_change": f"{data.next_month_oi_change_pct:+.2f}%",
        "quadrant": data.quadrant,
        "pcr_ratio": data.pcr_ratio
    }


def get_market_sentiment_summary(oi_data_list: List[OIData]) -> Dict:
    """
    Calculate market sentiment summary from OI data.
    
    Args:
        oi_data_list: List of OIData objects
    
    Returns:
        Dictionary with sentiment counts
    """
    summary = {
        "Long Buildup": 0,
        "Short Buildup": 0,
        "Short Covering": 0,
        "Long Unwinding": 0,
        "Neutral": 0,
        "total": len(oi_data_list),
        "bullish_pct": 0.0,
        "bearish_pct": 0.0
    }
    
    for data in oi_data_list:
        quadrant = data.quadrant
        if quadrant in summary:
            summary[quadrant] += 1
    
    if summary["total"] > 0:
        # Long Buildup and Short Covering are bullish signals
        bullish = summary["Long Buildup"] + summary["Short Covering"]
        # Short Buildup and Long Unwinding are bearish signals
        bearish = summary["Short Buildup"] + summary["Long Unwinding"]
        
        summary["bullish_pct"] = round((bullish / summary["total"]) * 100, 1)
        summary["bearish_pct"] = round((bearish / summary["total"]) * 100, 1)
    
    return summary
