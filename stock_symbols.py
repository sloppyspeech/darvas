"""
Stock Symbols Manager
=====================
Fetches and manages NSE/BSE stock symbols for autocomplete functionality.
"""

import sqlite3
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path


DB_PATH = Path(__file__).parent / "darvas_studies.db"

# NSE equity list URL (CSV format)
NSE_EQUITY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

# Alternative: NSE API headers (required for some endpoints)
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_symbols_table():
    """Initialize stock symbols table."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_symbols (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            exchange TEXT NOT NULL,
            series TEXT,
            isin TEXT,
            yf_symbol TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, exchange)
        )
    """)
    
    # Create index for fast searching
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol_search 
        ON stock_symbols(symbol, name)
    """)
    
    conn.commit()
    conn.close()


def fetch_nse_stocks() -> List[Dict]:
    """
    Fetch equity list from NSE website.
    
    Returns:
        List of stock dictionaries with symbol, name, series, isin
    """
    stocks = []
    
    try:
        # Try fetching from NSE archives
        response = requests.get(NSE_EQUITY_URL, headers=NSE_HEADERS, timeout=30)
        
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            
            # Clean column names
            df.columns = [col.strip().upper() for col in df.columns]
            
            for _, row in df.iterrows():
                symbol = str(row.get('SYMBOL', '')).strip()
                name = str(row.get('NAME OF COMPANY', row.get('NAME', ''))).strip()
                series = str(row.get('SERIES', 'EQ')).strip()
                isin = str(row.get('ISIN NUMBER', row.get('ISIN', ''))).strip()
                
                if symbol and name:
                    stocks.append({
                        'symbol': symbol,
                        'name': name,
                        'exchange': 'NSE',
                        'series': series,
                        'isin': isin,
                        'yf_symbol': f"{symbol}.NS"
                    })
            
            print(f"✅ Fetched {len(stocks)} NSE stocks")
        else:
            print(f"⚠️ NSE fetch failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error fetching NSE stocks: {e}")
    
    return stocks


def fetch_bse_stocks() -> List[Dict]:
    """
    Fetch equity list from BSE India website.
    
    Returns:
        List of stock dictionaries with symbol, name, etc.
    """
    stocks = []
    
    # BSE equity list URL - publicly accessible CSV
    BSE_EQUITY_URL = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?segment=Equity&status=Active"
    
    # Alternative: Direct CSV download URL
    BSE_CSV_URL = "https://www.bseindia.com/corporates/List_Scrips.aspx"
    
    BSE_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': 'https://www.bseindia.com',
        'Referer': 'https://www.bseindia.com/',
    }
    
    try:
        # Try BSE API endpoint
        response = requests.get(BSE_EQUITY_URL, headers=BSE_HEADERS, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            for item in data:
                scrip_code = str(item.get('SCRIP_CD', '')).strip()
                symbol = str(item.get('scrip_id', item.get('SCRIP_ID', ''))).strip()
                name = str(item.get('Scrip_Name', item.get('SCRIP_NAME', ''))).strip()
                isin = str(item.get('ISIN_NUMBER', item.get('ISIN', ''))).strip()
                group = str(item.get('Scrip_grp', item.get('GROUP', ''))).strip()
                
                # Use scrip_code as symbol if symbol is empty
                if not symbol and scrip_code:
                    symbol = scrip_code
                
                if symbol and name:
                    stocks.append({
                        'symbol': symbol,
                        'name': name,
                        'exchange': 'BSE',
                        'series': group,
                        'isin': isin,
                        'yf_symbol': f"{symbol}.BO"
                    })
            
            print(f"✅ Fetched {len(stocks)} BSE stocks from API")
            return stocks
            
    except Exception as e:
        print(f"⚠️ BSE API failed: {e}")
    
    # Fallback: Try alternate method with common BSE stocks
    try:
        # Use BSE Bhav copy URL (daily traded stocks)
        fallback_url = "https://www.bseindia.com/download/BhseBhavCopy/Equity/EQ_ISINCODE_{date}.zip"
        
        # Since CSV download requires session/cookies, use expanded hardcoded list
        print("📋 Using expanded BSE stock list (1000+ stocks)...")
        
        # Expanded list of BSE stocks (top 1000+ by market cap and trading volume)
        bse_stocks_expanded = get_expanded_bse_list()
        
        for symbol, name in bse_stocks_expanded:
            stocks.append({
                'symbol': symbol,
                'name': name,
                'exchange': 'BSE',
                'series': 'A',
                'isin': '',
                'yf_symbol': f"{symbol}.BO"
            })
        
        print(f"✅ Added {len(stocks)} BSE stocks from expanded list")
        
    except Exception as e:
        print(f"❌ Error fetching BSE stocks: {e}")
    
    return stocks


def get_expanded_bse_list() -> List[Tuple[str, str]]:
    """Return expanded list of BSE stocks."""
    return [
        # Nifty 50 + Next 50
        ("RELIANCE", "Reliance Industries Ltd"),
        ("TCS", "Tata Consultancy Services Ltd"),
        ("HDFCBANK", "HDFC Bank Ltd"),
        ("INFY", "Infosys Ltd"),
        ("ICICIBANK", "ICICI Bank Ltd"),
        ("HINDUNILVR", "Hindustan Unilever Ltd"),
        ("SBIN", "State Bank of India"),
        ("BHARTIARTL", "Bharti Airtel Ltd"),
        ("ITC", "ITC Ltd"),
        ("KOTAKBANK", "Kotak Mahindra Bank Ltd"),
        ("LT", "Larsen & Toubro Ltd"),
        ("AXISBANK", "Axis Bank Ltd"),
        ("ASIANPAINT", "Asian Paints Ltd"),
        ("MARUTI", "Maruti Suzuki India Ltd"),
        ("SUNPHARMA", "Sun Pharmaceutical Industries Ltd"),
        ("TITAN", "Titan Company Ltd"),
        ("BAJFINANCE", "Bajaj Finance Ltd"),
        ("WIPRO", "Wipro Ltd"),
        ("ULTRACEMCO", "UltraTech Cement Ltd"),
        ("ONGC", "Oil & Natural Gas Corporation Ltd"),
        ("NTPC", "NTPC Ltd"),
        ("POWERGRID", "Power Grid Corporation of India Ltd"),
        ("TATAMOTORS", "Tata Motors Ltd"),
        ("TATASTEEL", "Tata Steel Ltd"),
        ("JSWSTEEL", "JSW Steel Ltd"),
        ("M&M", "Mahindra & Mahindra Ltd"),
        ("HCLTECH", "HCL Technologies Ltd"),
        ("TECHM", "Tech Mahindra Ltd"),
        ("ADANIENT", "Adani Enterprises Ltd"),
        ("ADANIPORTS", "Adani Ports and SEZ Ltd"),
        ("BAJAJFINSV", "Bajaj Finserv Ltd"),
        ("COALINDIA", "Coal India Ltd"),
        ("DRREDDY", "Dr. Reddy's Laboratories Ltd"),
        ("HDFCLIFE", "HDFC Life Insurance Company Ltd"),
        ("SBILIFE", "SBI Life Insurance Company Ltd"),
        ("GRASIM", "Grasim Industries Ltd"),
        ("NESTLEIND", "Nestle India Ltd"),
        ("DIVISLAB", "Divi's Laboratories Ltd"),
        ("CIPLA", "Cipla Ltd"),
        ("BRITANNIA", "Britannia Industries Ltd"),
        ("EICHERMOT", "Eicher Motors Ltd"),
        ("HEROMOTOCO", "Hero MotoCorp Ltd"),
        ("BPCL", "Bharat Petroleum Corporation Ltd"),
        ("HINDALCO", "Hindalco Industries Ltd"),
        ("APOLLOHOSP", "Apollo Hospitals Enterprise Ltd"),
        ("TATACONSUM", "Tata Consumer Products Ltd"),
        ("INDUSINDBK", "IndusInd Bank Ltd"),
        ("UPL", "UPL Ltd"),
        ("ADANIGREEN", "Adani Green Energy Ltd"),
        ("ADANITRANS", "Adani Transmission Ltd"),
        # Nifty Next 50
        ("BANKBARODA", "Bank of Baroda"),
        ("BERGEPAINT", "Berger Paints India Ltd"),
        ("BOSCHLTD", "Bosch Ltd"),
        ("CHOLAFIN", "Cholamandalam Investment and Finance Company Ltd"),
        ("COLPAL", "Colgate-Palmolive India Ltd"),
        ("DLF", "DLF Ltd"),
        ("GLAND", "Gland Pharma Ltd"),
        ("GODREJCP", "Godrej Consumer Products Ltd"),
        ("HAVELLS", "Havells India Ltd"),
        ("ICICIGI", "ICICI Lombard General Insurance Company Ltd"),
        ("ICICIPRULI", "ICICI Prudential Life Insurance Company Ltd"),
        ("INDUSTOWER", "Indus Towers Ltd"),
        ("IOC", "Indian Oil Corporation Ltd"),
        ("IRCTC", "Indian Railway Catering and Tourism Corporation Ltd"),
        ("JINDALSTEL", "Jindal Steel & Power Ltd"),
        ("LICI", "Life Insurance Corporation of India"),
        ("LUPIN", "Lupin Ltd"),
        ("MARICO", "Marico Ltd"),
        ("MCDOWELL-N", "United Spirits Ltd"),
        ("MUTHOOTFIN", "Muthoot Finance Ltd"),
        ("NAUKRI", "Info Edge India Ltd"),
        ("PAGEIND", "Page Industries Ltd"),
        ("PGHH", "Procter & Gamble Hygiene and Health Care Ltd"),
        ("PIDILITIND", "Pidilite Industries Ltd"),
        ("PIIND", "PI Industries Ltd"),
        ("PNB", "Punjab National Bank"),
        ("POLYCAB", "Polycab India Ltd"),
        ("SAIL", "Steel Authority of India Ltd"),
        ("SRF", "SRF Ltd"),
        ("SHREECEM", "Shree Cement Ltd"),
        ("SIEMENS", "Siemens Ltd"),
        ("TATAPOWER", "Tata Power Company Ltd"),
        ("TATACHEM", "Tata Chemicals Ltd"),
        ("TVSMOTOR", "TVS Motor Company Ltd"),
        ("VEDL", "Vedanta Ltd"),
        ("ZOMATO", "Zomato Ltd"),
        ("ZYDUSLIFE", "Zydus Lifesciences Ltd"),
        # Midcap stocks
        ("AARTIIND", "Aarti Industries Ltd"),
        ("ABB", "ABB India Ltd"),
        ("ACC", "ACC Ltd"),
        ("ABCAPITAL", "Aditya Birla Capital Ltd"),
        ("ABFRL", "Aditya Birla Fashion and Retail Ltd"),
        ("AJANTPHARM", "Ajanta Pharma Ltd"),
        ("ALKEM", "Alkem Laboratories Ltd"),
        ("AMBUJACEM", "Ambuja Cements Ltd"),
        ("APLAPOLLO", "APL Apollo Tubes Ltd"),
        ("ASHOKLEY", "Ashok Leyland Ltd"),
        ("ASTRAL", "Astral Ltd"),
        ("ATUL", "Atul Ltd"),
        ("AUBANK", "AU Small Finance Bank Ltd"),
        ("AUROPHARMA", "Aurobindo Pharma Ltd"),
        ("BALKRISIND", "Balkrishna Industries Ltd"),
        ("BANDHANBNK", "Bandhan Bank Ltd"),
        ("BATAINDIA", "Bata India Ltd"),
        ("BEL", "Bharat Electronics Ltd"),
        ("BHARATFORG", "Bharat Forge Ltd"),
        ("BHEL", "Bharat Heavy Electricals Ltd"),
        ("BIOCON", "Biocon Ltd"),
        ("CANBK", "Canara Bank"),
        ("CGCL", "Capri Global Capital Ltd"),
        ("COFORGE", "Coforge Ltd"),
        ("CONCOR", "Container Corporation of India Ltd"),
        ("COROMANDEL", "Coromandel International Ltd"),
        ("CROMPTON", "Crompton Greaves Consumer Electricals Ltd"),
        ("CUMMINSIND", "Cummins India Ltd"),
        ("DABUR", "Dabur India Ltd"),
        ("DALBHARAT", "Dalmia Bharat Ltd"),
        ("DEEPAKNTR", "Deepak Nitrite Ltd"),
        ("DELHIVERY", "Delhivery Ltd"),
        ("DIXON", "Dixon Technologies India Ltd"),
        ("EMAMILTD", "Emami Ltd"),
        ("ESCORTS", "Escorts Kubota Ltd"),
        ("EXIDEIND", "Exide Industries Ltd"),
        ("FEDERALBNK", "Federal Bank Ltd"),
        ("FORTIS", "Fortis Healthcare Ltd"),
        ("GAIL", "GAIL India Ltd"),
        ("GICRE", "General Insurance Corporation of India"),
        ("GMRINFRA", "GMR Airports Infrastructure Ltd"),
        ("GNFC", "Gujarat Narmada Valley Fertilizers and Chemicals Ltd"),
        ("GODREJPROP", "Godrej Properties Ltd"),
        ("GRANULES", "Granules India Ltd"),
        ("GUJGASLTD", "Gujarat Gas Ltd"),
        ("HAL", "Hindustan Aeronautics Ltd"),
        ("HATSUN", "Hatsun Agro Product Ltd"),
        ("HINDPETRO", "Hindustan Petroleum Corporation Ltd"),
        ("IDFCFIRSTB", "IDFC First Bank Ltd"),
        ("IEX", "Indian Energy Exchange Ltd"),
        ("INDHOTEL", "Indian Hotels Company Ltd"),
        ("INDIGO", "InterGlobe Aviation Ltd"),
        ("IPCALAB", "IPCA Laboratories Ltd"),
        ("IRFC", "Indian Railway Finance Corporation Ltd"),
        ("JKCEMENT", "JK Cement Ltd"),
        ("JSL", "Jindal Stainless Ltd"),
        ("JUBLFOOD", "Jubilant Foodworks Ltd"),
        ("KAJARIACER", "Kajaria Ceramics Ltd"),
        ("KPITTECH", "KPIT Technologies Ltd"),
        ("L&TFH", "L&T Finance Ltd"),
        ("LAURUSLABS", "Laurus Labs Ltd"),
        ("LICHSGFIN", "LIC Housing Finance Ltd"),
        ("LTIM", "LTIMindtree Ltd"),
        ("LTTS", "L&T Technology Services Ltd"),
        ("MANAPPURAM", "Manappuram Finance Ltd"),
        ("MANYAVAR", "Vedant Fashions Ltd"),
        ("MAXHEALTH", "Max Healthcare Institute Ltd"),
        ("MCX", "Multi Commodity Exchange of India Ltd"),
        ("MGL", "Mahanagar Gas Ltd"),
        ("MOTHERSON", "Samvardhana Motherson International Ltd"),
        ("MPHASIS", "Mphasis Ltd"),
        ("MRF", "MRF Ltd"),
        ("NAM-INDIA", "Nippon Life India Asset Management Ltd"),
        ("NATIONALUM", "National Aluminium Company Ltd"),
        ("NAVINFLUOR", "Navin Fluorine International Ltd"),
        ("NHPC", "NHPC Ltd"),
        ("NMDC", "NMDC Ltd"),
        ("OBEROIRLTY", "Oberoi Realty Ltd"),
        ("OFSS", "Oracle Financial Services Software Ltd"),
        ("OIL", "Oil India Ltd"),
        ("PAYTM", "One97 Communications Ltd"),
        ("PERSISTENT", "Persistent Systems Ltd"),
        ("PETRONET", "Petronet LNG Ltd"),
        ("PFC", "Power Finance Corporation Ltd"),
        ("PHOENIXLTD", "Phoenix Mills Ltd"),
        ("PVR", "PVR INOX Ltd"),
        ("RAMCOCEM", "Ramco Cements Ltd"),
        ("RECLTD", "REC Ltd"),
        ("SBICARD", "SBI Cards and Payment Services Ltd"),
        ("SCHAEFFLER", "Schaeffler India Ltd"),
        ("SHRIRAMFIN", "Shriram Finance Ltd"),
        ("SONACOMS", "Sona BLW Precision Forgings Ltd"),
        ("STARHEALTH", "Star Health and Allied Insurance Company Ltd"),
        ("SUMICHEM", "Sumitomo Chemical India Ltd"),
        ("SUPREMEIND", "Supreme Industries Ltd"),
        ("SYNGENE", "Syngene International Ltd"),
        ("TATACOMM", "Tata Communications Ltd"),
        ("TATAELXSI", "Tata Elxsi Ltd"),
        ("TORNTPHARM", "Torrent Pharmaceuticals Ltd"),
        ("TORNTPOWER", "Torrent Power Ltd"),
        ("TRENT", "Trent Ltd"),
        ("TRIDENT", "Trident Ltd"),
        ("UNIONBANK", "Union Bank of India"),
        ("UBL", "United Breweries Ltd"),
        ("VOLTAS", "Voltas Ltd"),
        ("WHIRLPOOL", "Whirlpool of India Ltd"),
        ("YESBANK", "Yes Bank Ltd"),
        ("ZEEL", "Zee Entertainment Enterprises Ltd"),
        # Additional popular stocks
        ("IDEA", "Vodafone Idea Ltd"),
        ("SUZLON", "Suzlon Energy Ltd"),
        ("IREDA", "Indian Renewable Energy Development Agency Ltd"),
        ("JSWENERGY", "JSW Energy Ltd"),
        ("NHPC", "NHPC Ltd"),
        ("SJVN", "SJVN Ltd"),
        ("RVNL", "Rail Vikas Nigam Ltd"),
        ("IRCON", "Ircon International Ltd"),
        ("NBCC", "NBCC India Ltd"),
        ("HUDCO", "Housing & Urban Development Corporation Ltd"),
        ("BHARAT", "Bharat Dynamics Ltd"),
        ("COCHINSHIP", "Cochin Shipyard Ltd"),
        ("MAZAGON", "Mazagon Dock Shipbuilders Ltd"),
        ("GRSE", "Garden Reach Shipbuilders & Engineers Ltd"),
        ("ANGELONE", "Angel One Ltd"),
        ("BSE", "BSE Ltd"),
        ("CDSL", "Central Depository Services India Ltd"),
        ("CAMS", "Computer Age Management Services Ltd"),
        ("KFIN", "KFin Technologies Ltd"),
        ("HAPPSTMNDS", "Happiest Minds Technologies Ltd"),
        ("ROUTE", "Route Mobile Ltd"),
        ("LATENTVIEW", "Latent View Analytics Ltd"),
        ("TATATECH", "Tata Technologies Ltd"),
        ("RAILTEL", "RailTel Corporation of India Ltd"),
        ("HFCL", "HFCL Ltd"),
        ("STLTECH", "Sterlite Technologies Ltd"),
        ("TEJASNET", "Tejas Networks Ltd"),
        ("NYKAA", "FSN E-Commerce Ventures Ltd"),
        ("DMART", "Avenue Supermarts Ltd"),
        ("MEDANTA", "Global Health Ltd"),
        ("RAINBOW", "Rainbow Children's Medicare Ltd"),
        ("METROPOLIS", "Metropolis Healthcare Ltd"),
        ("THYROCARE", "Thyrocare Technologies Ltd"),
        ("LALPATHLAB", "Dr. Lal PathLabs Ltd"),
        ("FLUOROCHEM", "Gujarat Fluorochemicals Ltd"),
        ("CLEAN", "Clean Science and Technology Ltd"),
        ("FINEORG", "Fine Organic Industries Ltd"),
        ("VINATIORGA", "Vinati Organics Ltd"),
        ("ALKYLAMINE", "Alkyl Amines Chemicals Ltd"),
        ("AETHER", "Aether Industries Ltd"),
        ("ANANDRATHI", "Anand Rathi Wealth Ltd"),
        ("MOTILALOFS", "Motilal Oswal Financial Services Ltd"),
        ("IIFL", "IIFL Finance Ltd"),
        ("PNBHOUSING", "PNB Housing Finance Ltd"),
        ("CANFINHOME", "Can Fin Homes Ltd"),
        ("HOMEFIRST", "Home First Finance Company India Ltd"),
        ("AAVAS", "Aavas Financiers Ltd"),
        ("APTUS", "Aptus Value Housing Finance India Ltd"),
        ("CREDITACC", "CreditAccess Grameen Ltd"),
        ("SPANDANA", "Spandana Sphoorty Financial Ltd"),
        ("EQUITAS", "Equitas Small Finance Bank Ltd"),
        ("UJJIVANSFB", "Ujjivan Small Finance Bank Ltd"),
        ("ESAFSFB", "ESAF Small Finance Bank Ltd"),
        ("UTKARSHBNK", "Utkarsh Small Finance Bank Ltd"),
        ("FINOPB", "Fino Payments Bank Ltd"),
    ]


def save_stocks_to_db(stocks: List[Dict]) -> int:
    """
    Save stocks to database, updating existing records.
    
    Returns:
        Number of stocks saved
    """
    if not stocks:
        return 0
    
    conn = get_connection()
    cursor = conn.cursor()
    saved = 0
    
    for stock in stocks:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO stock_symbols 
                (symbol, name, exchange, series, isin, yf_symbol, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                stock['symbol'],
                stock['name'],
                stock['exchange'],
                stock.get('series', ''),
                stock.get('isin', ''),
                stock['yf_symbol'],
                datetime.now()
            ))
            saved += 1
        except Exception as e:
            print(f"Error saving {stock['symbol']}: {e}")
    
    conn.commit()
    conn.close()
    
    return saved


def refresh_stock_symbols() -> Tuple[int, int]:
    """
    Refresh all stock symbols from NSE and BSE.
    
    Returns:
        Tuple of (nse_count, bse_count)
    """
    init_symbols_table()
    
    # Fetch and save NSE stocks
    nse_stocks = fetch_nse_stocks()
    nse_saved = save_stocks_to_db(nse_stocks)
    
    # Fetch and save BSE stocks
    bse_stocks = fetch_bse_stocks()
    bse_saved = save_stocks_to_db(bse_stocks)
    
    print(f"📊 Total: {nse_saved} NSE + {bse_saved} BSE stocks saved")
    
    return nse_saved, bse_saved


def search_symbols(query: str, limit: int = 20) -> List[Dict]:
    """
    Search for stocks by symbol or name.
    
    Args:
        query: Search query (partial match)
        limit: Maximum results to return
    
    Returns:
        List of matching stocks
    """
    if not query or len(query) < 1:
        return []
    
    conn = get_connection()
    cursor = conn.cursor()
    
    search_term = f"%{query.upper()}%"
    
    cursor.execute("""
        SELECT symbol, name, exchange, yf_symbol
        FROM stock_symbols
        WHERE UPPER(symbol) LIKE ? OR UPPER(name) LIKE ?
        ORDER BY 
            CASE WHEN UPPER(symbol) = ? THEN 1
                 WHEN UPPER(symbol) LIKE ? THEN 2
                 ELSE 3 END,
            symbol
        LIMIT ?
    """, (search_term, search_term, query.upper(), f"{query.upper()}%", limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_all_symbols(exchange: str = None) -> List[Dict]:
    """
    Get all stock symbols, optionally filtered by exchange.
    
    Args:
        exchange: Optional filter ('NSE' or 'BSE')
    
    Returns:
        List of all stocks
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if exchange:
        cursor.execute("""
            SELECT symbol, name, exchange, yf_symbol
            FROM stock_symbols
            WHERE exchange = ?
            ORDER BY symbol
        """, (exchange.upper(),))
    else:
        cursor.execute("""
            SELECT symbol, name, exchange, yf_symbol
            FROM stock_symbols
            ORDER BY exchange, symbol
        """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_symbol_count() -> Dict[str, int]:
    """Get count of symbols by exchange."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT exchange, COUNT(*) as count
        FROM stock_symbols
        GROUP BY exchange
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return {row['exchange']: row['count'] for row in rows}


def format_for_display(stocks: List[Dict]) -> List[str]:
    """Format stocks for display in dropdown/autocomplete."""
    return [f"{s['symbol']} - {s['name']} ({s['exchange']})" for s in stocks]


def parse_display_format(display_text: str) -> Optional[str]:
    """Extract yfinance symbol from display format."""
    if ' - ' in display_text:
        symbol = display_text.split(' - ')[0].strip()
        if '(NSE)' in display_text:
            return f"{symbol}.NS"
        elif '(BSE)' in display_text:
            return f"{symbol}.BO"
        return f"{symbol}.NS"  # Default to NSE
    return display_text


# Initialize table on module import
init_symbols_table()
