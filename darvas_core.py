"""
Darvas Box Momentum Analyzer - Core Logic Module
=================================================
Contains the core algorithms for Darvas Box detection, signal generation,
and chart creation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


# Configuration Constants
CONFIRMATION_DAYS = 3
VOLUME_MULTIPLIER = 1.5
ENTRY_BUFFER = 0.001
LOOKBACK_PERIOD = '1y'
VOLUME_MA_PERIOD = 20


@dataclass
class DarvasBox:
    """Represents a single Darvas Box."""
    top: float
    bottom: float
    start_date: datetime
    end_date: Optional[datetime] = None
    is_active: bool = True
    breakout_date: Optional[datetime] = None
    breakout_volume_confirmed: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'top': self.top,
            'bottom': self.bottom,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'is_active': self.is_active,
            'breakout_date': self.breakout_date.isoformat() if self.breakout_date else None,
            'breakout_volume_confirmed': self.breakout_volume_confirmed
        }


def fetch_stock_data(symbol: str, period: str = '1y') -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data for a given stock symbol.
    
    Args:
        symbol: Stock ticker (e.g., 'RELIANCE.NS' for NSE, 'RELIANCE.BO' for BSE)
        period: Data period ('1y', '2y', etc.)
    
    Returns:
        DataFrame with OHLCV data or None if fetch fails
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            return None
        
        # Clean column names
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Reset index to make date a column
        df = df.reset_index()
        df.rename(columns={'Date': 'date', 'index': 'date'}, inplace=True)
        
        # Convert timezone-aware datetime to timezone-naive
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None


def detect_darvas_boxes(df: pd.DataFrame, 
                        confirmation_days: int = CONFIRMATION_DAYS,
                        volume_multiplier: float = VOLUME_MULTIPLIER) -> Tuple[List[DarvasBox], pd.DataFrame]:
    """
    Detect Darvas Boxes in the price data.
    
    Algorithm:
    1. Find 52-week highs
    2. If high is not broken for 'confirmation_days', it becomes Box Top
    3. After Box Top is set, find the low
    4. If low is not broken for 'confirmation_days', it becomes Box Bottom
    5. Validate breakouts with volume filter
    """
    df = df.copy()
    boxes = []
    
    # Calculate 52-week high (rolling 252 trading days)
    df['52w_high'] = df['high'].rolling(window=252, min_periods=1).max()
    
    # Calculate 20-day average volume
    df['avg_volume'] = df['volume'].rolling(window=VOLUME_MA_PERIOD).mean()
    
    # State variables
    potential_top = None
    potential_top_date = None
    top_confirmation_count = 0
    
    box_top = None
    box_top_date = None
    
    potential_bottom = None
    potential_bottom_date = None
    bottom_confirmation_count = 0
    
    current_box = None
    
    # Initialize columns for tracking
    df['box_top'] = np.nan
    df['box_bottom'] = np.nan
    df['in_box'] = False
    df['breakout'] = False
    df['volume_confirmed'] = False
    
    for i in range(len(df)):
        row = df.iloc[i]
        current_high = row['high']
        current_low = row['low']
        current_close = row['close']
        current_volume = row['volume']
        current_date = row['date']
        avg_vol = row['avg_volume'] if pd.notna(row['avg_volume']) else current_volume
        
        # Check if we're at or near 52-week high (within 5%)
        is_near_52w_high = current_high >= row['52w_high'] * 0.95
        
        # Phase 1: Looking for new high (potential box top)
        if potential_top is None or current_high > potential_top:
            if is_near_52w_high:
                potential_top = current_high
                potential_top_date = current_date
                top_confirmation_count = 0
                potential_bottom = None
                bottom_confirmation_count = 0
        else:
            if potential_top is not None:
                top_confirmation_count += 1
        
        # Phase 2: Box Top confirmed
        if top_confirmation_count >= confirmation_days and box_top is None:
            box_top = potential_top
            box_top_date = potential_top_date
            potential_bottom = current_low
            potential_bottom_date = current_date
            bottom_confirmation_count = 0
        
        # Phase 3: Looking for box bottom
        if box_top is not None and current_box is None:
            if potential_bottom is None or current_low < potential_bottom:
                potential_bottom = current_low
                potential_bottom_date = current_date
                bottom_confirmation_count = 0
            else:
                bottom_confirmation_count += 1
            
            if bottom_confirmation_count >= confirmation_days:
                current_box = DarvasBox(
                    top=box_top,
                    bottom=potential_bottom,
                    start_date=box_top_date
                )
                boxes.append(current_box)
        
        # Phase 4: Inside a box - check for breakout or breakdown
        if current_box is not None and current_box.is_active:
            df.at[df.index[i], 'box_top'] = current_box.top
            df.at[df.index[i], 'box_bottom'] = current_box.bottom
            df.at[df.index[i], 'in_box'] = True
            
            # Breakout above box top
            if current_close > current_box.top:
                volume_confirmed = current_volume >= avg_vol * volume_multiplier
                current_box.breakout_date = current_date
                current_box.breakout_volume_confirmed = volume_confirmed
                current_box.end_date = current_date
                current_box.is_active = False
                
                df.at[df.index[i], 'breakout'] = True
                df.at[df.index[i], 'volume_confirmed'] = volume_confirmed
                
                potential_top = current_high
                potential_top_date = current_date
                top_confirmation_count = 0
                box_top = None
                potential_bottom = None
                current_box = None
            
            # Breakdown below box bottom
            elif current_close < current_box.bottom:
                current_box.end_date = current_date
                current_box.is_active = False
                
                potential_top = None
                top_confirmation_count = 0
                box_top = None
                potential_bottom = None
                current_box = None
    
    return boxes, df


def generate_signals(df: pd.DataFrame, boxes: List[DarvasBox], symbol: str) -> Dict:
    """Generate trading signals based on Darvas Box analysis."""
    current_price = df['close'].iloc[-1] if len(df) > 0 else None
    
    if not boxes:
        return {
            'symbol': symbol,
            'status': 'No Setup',
            'suggestion': '⚪ SKIP',
            'suggestion_reason': 'Not a Darvas candidate - no box formation',
            'box_top': None,
            'box_bottom': None,
            'entry_price': None,
            'stop_loss': None,
            'current_price': round(current_price, 2) if current_price else None,
            'target_price': None,
            'risk_amount': None,
            'reward_amount': None,
            'risk_percent': None,
            'risk_reward': None,
            'volume_confirmed': False
        }
    
    latest_box = boxes[-1]
    latest_date = df['date'].iloc[-1]
    
    # Determine status and suggestion
    if latest_box.is_active:
        status = 'Inside Box'
        suggestion = '🔵 WATCH'
        suggestion_reason = f'Waiting for breakout above ₹{latest_box.top:.2f}'
    elif latest_box.breakout_date is not None:
        days_since_breakout = (latest_date - latest_box.breakout_date).days
        if days_since_breakout <= 5 and latest_box.breakout_volume_confirmed:
            status = 'Breakout (Volume ✓)'
            suggestion = '🟢 BUY'
            suggestion_reason = 'Strong breakout with volume confirmation'
        elif days_since_breakout <= 5:
            status = 'Breakout (Low Volume)'
            suggestion = '🟡 CAUTION'
            suggestion_reason = 'Breakout without volume - higher risk'
        else:
            status = 'Post-Breakout'
            suggestion = '⚪ SKIP'
            suggestion_reason = 'Breakout too far back - wait for new setup'
    else:
        status = 'Box Closed'
        suggestion = '⚪ SKIP'
        suggestion_reason = 'Box broken down - not a valid setup'
    
    # Calculate entry, stop-loss, and target
    entry_price = latest_box.top * (1 + ENTRY_BUFFER)
    stop_loss = latest_box.bottom
    
    # Risk and reward calculations from ENTRY price (standard Darvas calculation)
    risk_amount = entry_price - stop_loss  # How much you risk from entry
    reward_amount = risk_amount * 2  # Standard 2R target
    target_price = entry_price + reward_amount
    
    # Standard R:R from entry is always 1:2 by design
    entry_risk_reward = "1:2"
    
    # Calculate current position R:R (if already holding or considering buying at current price)
    if current_price and current_price > 0 and stop_loss > 0 and target_price > 0:
        current_risk = current_price - stop_loss  # Risk from current price
        current_reward = target_price - current_price  # Potential reward from current
        
        if current_risk > 0 and current_reward > 0:
            rr_ratio = current_reward / current_risk
            current_risk_reward = f"1:{rr_ratio:.1f}"
        elif current_risk <= 0:
            current_risk_reward = "Below SL"
        else:
            current_risk_reward = "At/Above Target"
    else:
        current_risk_reward = "N/A"
    
    # Use the entry-based R:R as the main display (this is standard for Darvas setups)
    # The format shows: "1:2 (entry)" or current position if price moved
    if current_price and entry_price:
        if abs(current_price - entry_price) / entry_price < 0.02:  # Within 2% of entry
            risk_reward = entry_risk_reward  # Use standard 1:2
        else:
            risk_reward = current_risk_reward  # Use current position R:R
    else:
        risk_reward = entry_risk_reward
    
    risk_percent = (risk_amount / entry_price) * 100 if entry_price > 0 else 0
    
    return {
        'symbol': symbol,
        'status': status,
        'suggestion': suggestion,
        'suggestion_reason': suggestion_reason,
        'box_top': round(latest_box.top, 2),
        'box_bottom': round(latest_box.bottom, 2),
        'entry_price': round(entry_price, 2),
        'stop_loss': round(stop_loss, 2),
        'current_price': round(current_price, 2),
        'target_price': round(target_price, 2),
        'risk_amount': round(risk_amount, 2),
        'reward_amount': round(reward_amount, 2),
        'risk_percent': round(risk_percent, 2),
        'risk_reward': risk_reward,
        'entry_rr': entry_risk_reward,
        'current_rr': current_risk_reward,
        'volume_confirmed': latest_box.breakout_volume_confirmed
    }


def create_chart(df: pd.DataFrame, boxes: List[DarvasBox], symbol: str, 
                 show_last_n_days: int = 120) -> go.Figure:
    """Create an interactive candlestick chart with Darvas Boxes overlay."""
    df_plot = df.tail(show_last_n_days).copy()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol} - Darvas Box Analysis', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df_plot['date'],
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#26a69a' if close >= open else '#ef5350' 
              for close, open in zip(df_plot['close'], df_plot['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df_plot['date'],
            y=df_plot['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Average volume line
    if 'avg_volume' in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot['date'],
                y=df_plot['avg_volume'],
                name='Avg Volume (20d)',
                line=dict(color='orange', width=1, dash='dash')
            ),
            row=2, col=1
        )
    
    # Add Darvas Boxes
    box_colors = ['rgba(66, 133, 244, 0.2)', 'rgba(52, 168, 83, 0.2)', 
                  'rgba(251, 188, 4, 0.2)', 'rgba(234, 67, 53, 0.2)']
    
    plot_start_date = df_plot['date'].min()
    plot_end_date = df_plot['date'].max()
    
    for i, box in enumerate(boxes):
        box_end = box.end_date if box.end_date else plot_end_date
        
        if box.start_date <= plot_end_date and box_end >= plot_start_date:
            color_idx = i % len(box_colors)
            x0 = max(box.start_date, plot_start_date)
            x1 = min(box_end, plot_end_date)
            
            fig.add_shape(
                type="rect",
                x0=x0, y0=box.bottom,
                x1=x1, y1=box.top,
                fillcolor=box_colors[color_idx],
                line=dict(color=box_colors[color_idx].replace('0.2', '0.8'), width=2),
                row=1, col=1
            )
    
    # Mark breakout points
    breakout_df = df_plot[df_plot['breakout'] == True]
    if len(breakout_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=breakout_df['date'],
                y=breakout_df['high'] * 1.02,
                mode='markers+text',
                name='Breakout',
                marker=dict(symbol='triangle-up', size=15, color='lime'),
                text=['🚀' if v else '⚠️' for v in breakout_df['volume_confirmed']],
                textposition='top center'
            ),
            row=1, col=1
        )
    
    fig.update_layout(
        title=dict(text=f'📊 {symbol} - Darvas Box Analysis', font=dict(size=20)),
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def analyze_stock(symbol: str, confirmation_days: int = CONFIRMATION_DAYS,
                  volume_multiplier: float = VOLUME_MULTIPLIER) -> Dict:
    """
    Complete analysis for a single stock.
    
    Returns dictionary with df, boxes, signal, and chart.
    """
    df = fetch_stock_data(symbol, LOOKBACK_PERIOD)
    
    if df is None or len(df) < 50:
        return {
            'success': False,
            'error': 'No data available',
            'signal': {
                'symbol': symbol,
                'status': 'No Data',
                'box_top': None,
                'box_bottom': None,
                'entry_price': None,
                'stop_loss': None,
                'current_price': None,
                'risk_percent': None,
                'risk_reward': None
            }
        }
    
    boxes, enhanced_df = detect_darvas_boxes(df, confirmation_days, volume_multiplier)
    signal = generate_signals(enhanced_df, boxes, symbol)
    chart = create_chart(enhanced_df, boxes, symbol)
    
    return {
        'success': True,
        'df': enhanced_df,
        'boxes': boxes,
        'signal': signal,
        'chart': chart,
        'box_count': len(boxes)
    }
