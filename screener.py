"""
Darvas Box Candidate Screener
=============================
Multi-gate funnel to filter Nifty 500 stocks for Darvas Box candidates.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

from nifty500 import get_nse_symbols, get_nifty50_sample


@dataclass
class ScreenerResult:
    """Result for a single stock screening."""
    symbol: str
    current_price: float
    high_52w: float
    low_52w: float
    proximity_pct: float  # % from 52-week high
    strength_ratio: float  # price / 52w low
    sma_200: float
    above_sma: bool
    avg_volume: float
    recent_volume: float
    volume_ratio: float
    volume_spike: bool
    atr_pct: float
    consolidation_range: float
    days_in_consolidation: int
    gates_passed: int
    priority: str
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'current_price': round(self.current_price, 2),
            'high_52w': round(self.high_52w, 2),
            'low_52w': round(self.low_52w, 2),
            'proximity_pct': round(self.proximity_pct, 2),
            'strength_ratio': round(self.strength_ratio, 2),
            'sma_200': round(self.sma_200, 2) if self.sma_200 else None,
            'above_sma': self.above_sma,
            'volume_ratio': round(self.volume_ratio, 2),
            'volume_spike': self.volume_spike,
            'atr_pct': round(self.atr_pct, 2),
            'consolidation_range': round(self.consolidation_range, 2),
            'days_in_consolidation': self.days_in_consolidation,
            'gates_passed': self.gates_passed,
            'priority': self.priority
        }


# Screening thresholds
PROXIMITY_THRESHOLD = 10  # Within 10% of 52-week high
STRENGTH_THRESHOLD = 2.0  # Price doubled from low
VOLUME_SPIKE_THRESHOLD = 1.5  # 1.5x average volume
CONSOLIDATION_RANGE_MAX = 8  # Max 8% range for consolidation
CONSOLIDATION_DAYS_MIN = 10  # Minimum days in consolidation


def fetch_screening_data(symbol: str, retries: int = 2) -> Optional[pd.DataFrame]:
    """
    Fetch 1 year of data for screening with retry logic.
    """
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1y')
            
            if df.empty or len(df) < 200:
                return None
            
            df.columns = [col.lower() for col in df.columns]
            df = df.reset_index()
            df.rename(columns={'Date': 'date', 'index': 'date'}, inplace=True)
            
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            
            return df
            
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5)
            continue
    
    return None


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range as percentage of price."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean().iloc[-1]
    current_price = df['close'].iloc[-1]
    
    return (atr / current_price) * 100


def calculate_consolidation(df: pd.DataFrame, lookback: int = 15) -> Tuple[float, int]:
    """
    Calculate consolidation range and days in consolidation.
    Returns (range_pct, days_in_tight_range)
    """
    recent = df.tail(lookback)
    
    high_range = recent['high'].max()
    low_range = recent['low'].min()
    range_pct = ((high_range - low_range) / low_range) * 100
    
    # Count consecutive days in tight range (5%)
    tight_range_threshold = 0.05
    days_in_consolidation = 0
    
    for i in range(len(df) - 1, max(len(df) - 30, 0), -1):
        recent_high = df['high'].iloc[i-5:i+1].max() if i >= 5 else df['high'].iloc[:i+1].max()
        recent_low = df['low'].iloc[i-5:i+1].min() if i >= 5 else df['low'].iloc[:i+1].min()
        
        if recent_low > 0:
            range_now = (recent_high - recent_low) / recent_low
            if range_now <= tight_range_threshold:
                days_in_consolidation += 1
            else:
                break
    
    return range_pct, days_in_consolidation


def screen_stock(symbol: str) -> Optional[ScreenerResult]:
    """
    Apply multi-gate screening to a single stock.
    """
    df = fetch_screening_data(symbol)
    
    if df is None:
        return None
    
    try:
        current_price = df['close'].iloc[-1]
        high_52w = df['high'].max()
        low_52w = df['low'].min()
        
        # Gate 1: Proximity to 52-week high
        proximity_pct = ((high_52w - current_price) / high_52w) * 100
        gate1_passed = proximity_pct <= PROXIMITY_THRESHOLD
        
        # Gate 2: Strength (price doubled from low)
        strength_ratio = current_price / low_52w if low_52w > 0 else 0
        gate2_passed = strength_ratio >= STRENGTH_THRESHOLD
        
        # Gate 3: Trend (above 200-day SMA)
        sma_200 = df['close'].rolling(window=200).mean().iloc[-1]
        above_sma = current_price > sma_200 if pd.notna(sma_200) else False
        gate3_passed = above_sma
        
        # Gate 4: Interest (volume spike)
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        recent_volume = df['volume'].tail(5).mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
        volume_spike = volume_ratio >= VOLUME_SPIKE_THRESHOLD
        gate4_passed = volume_spike
        
        # Additional checks
        atr_pct = calculate_atr(df)
        consolidation_range, days_in_consolidation = calculate_consolidation(df)
        
        # Count gates passed
        gates = [gate1_passed, gate2_passed, gate3_passed, gate4_passed]
        gates_passed = sum(gates)
        
        # Priority scoring
        if gates_passed == 4 and consolidation_range <= CONSOLIDATION_RANGE_MAX:
            priority = "High"
        elif gates_passed >= 3 and gate1_passed:
            priority = "Medium"
        elif gates_passed >= 2 and gate1_passed:
            priority = "Low"
        else:
            priority = "None"
        
        return ScreenerResult(
            symbol=symbol,
            current_price=current_price,
            high_52w=high_52w,
            low_52w=low_52w,
            proximity_pct=proximity_pct,
            strength_ratio=strength_ratio,
            sma_200=sma_200 if pd.notna(sma_200) else 0,
            above_sma=above_sma,
            avg_volume=avg_volume,
            recent_volume=recent_volume,
            volume_ratio=volume_ratio,
            volume_spike=volume_spike,
            atr_pct=atr_pct,
            consolidation_range=consolidation_range,
            days_in_consolidation=days_in_consolidation,
            gates_passed=gates_passed,
            priority=priority
        )
        
    except Exception as e:
        print(f"Error screening {symbol}: {e}")
        return None


def run_screener(symbols: List[str] = None, 
                 progress_callback=None,
                 filter_priority: str = None) -> Tuple[List[ScreenerResult], Dict]:
    """
    Run the screener on a list of symbols.
    
    Args:
        symbols: List of symbols to screen (defaults to Nifty 500)
        progress_callback: Optional callback(current, total, symbol) for progress
        filter_priority: Optional filter for priority level
    
    Returns:
        Tuple of (list of ScreenerResults, stats dict)
    """
    if symbols is None:
        symbols = get_nse_symbols()
    
    results = []
    stats = {
        'total': len(symbols),
        'processed': 0,
        'failed': 0,
        'candidates': 0,
        'high_priority': 0,
        'medium_priority': 0,
        'low_priority': 0
    }
    
    for i, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(i + 1, len(symbols), symbol)
        
        result = screen_stock(symbol)
        
        if result is None:
            stats['failed'] += 1
        else:
            stats['processed'] += 1
            
            if result.priority != "None":
                stats['candidates'] += 1
                
                if result.priority == "High":
                    stats['high_priority'] += 1
                elif result.priority == "Medium":
                    stats['medium_priority'] += 1
                elif result.priority == "Low":
                    stats['low_priority'] += 1
            
            # Apply priority filter if specified
            if filter_priority is None or result.priority == filter_priority:
                results.append(result)
        
        # Rate limiting to avoid API throttling
        if (i + 1) % 10 == 0:
            time.sleep(0.5)
    
    # Sort by priority and proximity
    priority_order = {"High": 0, "Medium": 1, "Low": 2, "None": 3}
    results.sort(key=lambda x: (priority_order[x.priority], x.proximity_pct))
    
    return results, stats


def get_candidates_only(results: List[ScreenerResult]) -> List[ScreenerResult]:
    """Filter to only return candidates (priority != None)."""
    return [r for r in results if r.priority != "None"]


def results_to_dataframe(results: List[ScreenerResult]) -> pd.DataFrame:
    """Convert results to a Pandas DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])


def get_symbols_for_analysis(results: List[ScreenerResult], 
                             min_priority: str = "Low") -> List[str]:
    """Get list of symbols that meet minimum priority for Darvas analysis."""
    priority_levels = {"High": 3, "Medium": 2, "Low": 1, "None": 0}
    min_level = priority_levels.get(min_priority, 0)
    
    return [r.symbol for r in results if priority_levels.get(r.priority, 0) >= min_level]
