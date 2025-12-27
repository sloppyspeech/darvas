"""
Intrinsic Value Calculator - "True North"
==========================================
Calculate a stock's fair value using multiple valuation models:
1. Benjamin Graham Formula (growth-oriented)
2. Discounted Cash Flow - DCF (cash-rich companies)
3. Earnings Power Value - EPV (cyclical/no-growth)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# Indian market constants
AAA_BOND_YIELD = 7.5  # AAA Corporate Bond Yield in India (%)
RISK_FREE_RATE = 4.4  # Historical benchmark from Graham's era
BASE_PE_ZERO_GROWTH = 8.5  # P/E assigned for 0% growth company
DISCOUNT_RATE = 11.0  # Cost of capital for Indian market (%)
TERMINAL_GROWTH = 4.5  # Long-term GDP growth rate (%)
TAX_RATE = 25.17  # Indian corporate tax rate (%)

# EPV specific constants
EPV_GROWTH = 3.0  # Conservative perpetual growth for EPV (%)
EPV_DISCOUNT_RATE = 8.5  # Discount rate for EPV (%)
EPV_QUALITY_MULTIPLIER = 1.3  # Quality premium multiplier for exceptional earnings


@dataclass
class ValuationResult:
    """Result of intrinsic value calculation."""
    symbol: str
    current_price: float
    graham_value: Optional[float] = None
    dcf_value: Optional[float] = None
    epv_value: Optional[float] = None
    true_north_value: Optional[float] = None
    valuation_status: str = "N/A"
    upside_percent: Optional[float] = None
    eps: Optional[float] = None
    growth_rate: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'current_price': round(self.current_price, 2) if self.current_price else None,
            'graham_value': round(self.graham_value, 2) if self.graham_value else None,
            'dcf_value': round(self.dcf_value, 2) if self.dcf_value else None,
            'epv_value': round(self.epv_value, 2) if self.epv_value else None,
            'true_north_value': round(self.true_north_value, 2) if self.true_north_value else None,
            'valuation_status': self.valuation_status,
            'upside_percent': round(self.upside_percent, 1) if self.upside_percent else None,
            'eps': round(self.eps, 2) if self.eps else None,
            'growth_rate': round(self.growth_rate, 1) if self.growth_rate else None,
            'error': self.error
        }


def fetch_financial_data(symbol: str) -> Optional[Dict]:
    """
    Fetch financial data required for valuation calculations.
    
    Returns dict with EPS, growth rate, FCF, EBIT, cash, debt, shares outstanding.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get financial statements
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        
        # Current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # EPS (Trailing Twelve Months)
        eps = info.get('trailingEps')
        
        # Growth rate - try to get from info or calculate from historical data
        growth_rate = info.get('earningsGrowth')  # This is a decimal (0.15 = 15%)
        if growth_rate:
            growth_rate = growth_rate * 100  # Convert to percentage
        else:
            # Fallback: use 5-year average or analyst estimate
            growth_rate = info.get('earningsQuarterlyGrowth')
            if growth_rate:
                growth_rate = growth_rate * 100
            else:
                growth_rate = 10  # Default conservative estimate
        
        # Cap growth rate at 15% for conservative estimation
        growth_rate = min(max(growth_rate, 0), 15)
        
        # Shares outstanding
        shares_outstanding = info.get('sharesOutstanding', 0)
        
        # FCF - prefer freeCashflow from info as it's more reliable
        fcf = info.get('freeCashflow')
        
        # If not available, calculate from cashflow statement
        if fcf is None and cashflow is not None and not cashflow.empty:
            try:
                # Try 'Free Cash Flow' row first
                if 'Free Cash Flow' in cashflow.index:
                    fcf = cashflow.loc['Free Cash Flow'].iloc[0]
                else:
                    operating_cf = cashflow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow.index else None
                    capex = cashflow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cashflow.index else 0
                    if operating_cf:
                        fcf = operating_cf - abs(capex) if capex else operating_cf
            except:
                pass
        
        # Calculate FCF per share for validation
        fcf_per_share = None
        if fcf and shares_outstanding and shares_outstanding > 0:
            fcf_per_share = fcf / shares_outstanding
        
        # EBIT (Operating Income)
        ebit = info.get('ebitda')  # Use EBITDA from info as fallback
        if ebit is None and income_stmt is not None and not income_stmt.empty:
            try:
                ebit = income_stmt.loc['Operating Income'].iloc[0] if 'Operating Income' in income_stmt.index else None
                if ebit is None:
                    ebit = income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in income_stmt.index else None
            except:
                pass
        
        # Cash and Debt
        total_cash = info.get('totalCash', 0) or 0
        total_debt = info.get('totalDebt', 0) or 0
        
        # Market cap for validation
        market_cap = info.get('marketCap', 0)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'eps': eps,
            'growth_rate': growth_rate,
            'fcf': fcf,
            'fcf_per_share': fcf_per_share,
            'ebit': ebit,
            'total_cash': total_cash,
            'total_debt': total_debt,
            'shares_outstanding': shares_outstanding,
            'market_cap': market_cap,
            'company_name': info.get('shortName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'enterprise_value': info.get('enterpriseValue', 0)
        }
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


def calculate_benjamin_graham(eps: float, growth_rate: float, 
                               bond_yield: float = AAA_BOND_YIELD) -> Optional[float]:
    """
    Calculate intrinsic value using Revised Benjamin Graham Formula.
    
    Formula: V = (EPS × (8.5 + 2g) × 4.4) / Y
    
    Where:
        - EPS: Trailing 12-month earnings per share
        - g: Expected 5-year growth rate (%)
        - Y: AAA Corporate Bond Yield (%)
        - 8.5: P/E for zero growth company
        - 4.4: Risk-free rate benchmark
    
    Args:
        eps: Earnings Per Share (TTM)
        growth_rate: Expected growth rate in percentage (e.g., 15 for 15%)
        bond_yield: Current AAA bond yield in percentage
    
    Returns:
        Intrinsic value per share or None if calculation fails
    """
    if eps is None or eps <= 0:
        return None
    
    if bond_yield <= 0:
        bond_yield = AAA_BOND_YIELD  # Fallback to default
    
    # Ensure growth rate is reasonable
    growth_rate = max(0, min(growth_rate, 20))  # Cap between 0-20%
    
    # Graham formula
    intrinsic_value = (eps * (BASE_PE_ZERO_GROWTH + 2 * growth_rate) * RISK_FREE_RATE) / bond_yield
    
    return intrinsic_value


def calculate_dcf(fcf: float, shares_outstanding: int, total_cash: float, 
                  total_debt: float, growth_rate: float,
                  discount_rate: float = DISCOUNT_RATE,
                  terminal_growth: float = TERMINAL_GROWTH,
                  projection_years: int = 5,
                  market_cap: float = None,
                  current_price: float = None) -> Optional[float]:
    """
    Calculate intrinsic value using Discounted Cash Flow model.
    Uses FCF Yield based approach for robustness against yfinance unit inconsistencies.
    
    Args:
        fcf: Free Cash Flow (most recent year)
        shares_outstanding: Number of shares outstanding
        total_cash: Total cash and equivalents
        total_debt: Total debt
        growth_rate: Expected FCF growth rate (%)
        discount_rate: Cost of capital (%)
        terminal_growth: Long-term growth rate (%)
        projection_years: Years to project FCF
        market_cap: Market capitalization (for yield calculation)
        current_price: Current stock price (for alternative calculation)
    
    Returns:
        Intrinsic value per share or None if calculation fails
    """
    if fcf is None or fcf <= 0 or shares_outstanding <= 0:
        return None
    
    # Convert percentages to decimals
    g = min(max(growth_rate / 100, 0.02), 0.15)  # Floor at 2%, cap at 15%
    r = discount_rate / 100
    terminal_g = terminal_growth / 100
    
    if r <= terminal_g:
        return None
    
    # Method 1: Use Market Cap for FCF Yield based calculation (more robust)
    if market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap  # FCF Yield
        
        # If FCF yield is unreasonably low (unit mismatch), scale FCF up
        if fcf_yield < 0.005:  # Less than 0.5% yield is suspicious
            # Likely FCF is in smaller units - try scaling by common factors
            fcf_yield = fcf_yield * 100  # Scale up
        
        # Calculate intrinsic FCF yield using perpetuity growth
        # Fair FCF Yield = (r - g) / (1 + g) approximately
        # Or simpler: fair value = FCF * (1+g) / (r - terminal_g)
        
        # Project FCF growth and discount back
        pv_fcf_sum = 0
        current_fcf_factor = 1
        
        for year in range(1, projection_years + 1):
            current_fcf_factor = current_fcf_factor * (1 + g)
            pv_factor = current_fcf_factor / ((1 + r) ** year)
            pv_fcf_sum += pv_factor
        
        # Terminal value factor
        terminal_fcf_factor = current_fcf_factor * (1 + terminal_g)
        terminal_value_factor = terminal_fcf_factor / (r - terminal_g)
        pv_terminal_factor = terminal_value_factor / ((1 + r) ** projection_years)
        
        # Total value factor
        total_factor = pv_fcf_sum + pv_terminal_factor
        
        # Implied fair price from yield
        # If current FCF yield = x%, and we need to achieve yield = y% for fair value
        # fair_price = current_price * (x / y)
        implied_fcf_yield = 1 / total_factor  # Required yield for fair value
        
        # Calculate intrinsic value
        if current_price and implied_fcf_yield > 0:
            # Scale based on FCF growth expectations
            intrinsic_value = current_price * (fcf_yield / implied_fcf_yield)
            
            # Add net cash per share
            net_cash_per_share = (total_cash - total_debt) / shares_outstanding
            intrinsic_value += net_cash_per_share
            
            # Sanity check
            if intrinsic_value > 0 and intrinsic_value < current_price * 5:
                return intrinsic_value
    
    # Method 2: Fallback to traditional per-share calculation
    fcf_per_share = fcf / shares_outstanding
    cash_per_share = total_cash / shares_outstanding
    debt_per_share = total_debt / shares_outstanding
    
    # Check if FCF per share is reasonable (should be positive and meaningful)
    if fcf_per_share <= 0:
        return None
    
    # Project FCF per share
    pv_fcf_sum = 0
    current_fcf = fcf_per_share
    
    for year in range(1, projection_years + 1):
        current_fcf = current_fcf * (1 + g)
        pv = current_fcf / ((1 + r) ** year)
        pv_fcf_sum += pv
    
    # Terminal Value per share
    terminal_fcf = current_fcf * (1 + terminal_g)
    terminal_value = terminal_fcf / (r - terminal_g)
    pv_terminal = terminal_value / ((1 + r) ** projection_years)
    
    intrinsic_value = pv_fcf_sum + pv_terminal + cash_per_share - debt_per_share
    
    return intrinsic_value if intrinsic_value > 0 else None


def calculate_epv(ebit: float, shares_outstanding: int, total_cash: float,
                  total_debt: float, tax_rate: float = TAX_RATE,
                  discount_rate: float = EPV_DISCOUNT_RATE,
                  market_cap: float = None,
                  current_price: float = None,
                  eps: float = None,
                  growth_rate: float = EPV_GROWTH,
                  quality_multiplier: float = EPV_QUALITY_MULTIPLIER) -> Optional[float]:
    """
    Calculate intrinsic value using Earnings Power Value model.
    
    Uses quality-adjusted earnings with perpetual growth.
    Formula: EPV = (EPS × Quality Multiplier) / (r - g)
    
    Args:
        ebit: Earnings Before Interest and Taxes (EBITDA as proxy)
        shares_outstanding: Number of shares outstanding
        total_cash: Total cash
        total_debt: Total debt
        tax_rate: Corporate tax rate (%)
        discount_rate: Discount rate (%) - defaults to 8.5%
        market_cap: Market capitalization
        current_price: Current stock price
        eps: Earnings per share (most reliable from yfinance)
        growth_rate: Perpetual growth rate (%) - defaults to 3%
        quality_multiplier: Quality premium for exceptional earnings - defaults to 1.3x
    
    Returns:
        Intrinsic value per share or None if calculation fails
    """
    if shares_outstanding <= 0 or discount_rate <= 0:
        return None
    
    r = discount_rate / 100  # Discount rate as decimal (0.085)
    g = growth_rate / 100  # Growth rate as decimal (0.03)
    
    # Ensure r > g for valid calculation
    if r <= g:
        r = g + 0.03  # Minimum 3% spread
    
    # Method 1: Use EPS with quality adjustment (most reliable)
    # EPV = (EPS × Quality Multiplier) / (r - g)
    if eps and eps > 0 and current_price and current_price > 0:
        # Adjusted EPS with quality premium
        adjusted_eps = eps * quality_multiplier
        
        # EPV per share
        epv_per_share = adjusted_eps / (r - g)
        
        # Add net cash per share
        net_cash_per_share = (total_cash - total_debt) / shares_outstanding
        intrinsic_value = epv_per_share + net_cash_per_share
        
        # Sanity check
        if intrinsic_value > 0:
            return intrinsic_value
    
    # Method 2: Use Market Cap and EBIT with yield calculation
    if ebit and ebit > 0 and market_cap and market_cap > 0 and current_price and current_price > 0:
        tax_adjustment = 1 - tax_rate / 100
        
        # Calculate EBIT margin to check for unit issues
        ebit_to_market_ratio = (ebit * tax_adjustment) / market_cap
        
        # Expected earnings yield should be roughly = P/E inverse = ~4-8% for most companies
        # If ratio is < 0.01 (1%), there's likely a unit mismatch
        if ebit_to_market_ratio < 0.01:
            # Scale EBIT to match expected earnings yield based on PE
            if current_price > 0:
                # Use PE-implied earnings
                typical_pe = 20  # Conservative PE estimate
                implied_earnings_per_share = current_price / typical_pe
                epv_per_share = implied_earnings_per_share / wacc_decimal
                
                net_cash_per_share = (total_cash - total_debt) / shares_outstanding
                intrinsic_value = epv_per_share + net_cash_per_share
                
                if intrinsic_value > 0:
                    return intrinsic_value
        else:
            # Normal calculation with proper units
            normalized_earnings = ebit * tax_adjustment
            epv = normalized_earnings / wacc_decimal
            equity_epv = epv + total_cash - total_debt
            intrinsic_value = equity_epv / shares_outstanding
            
            if intrinsic_value > 0:
                return intrinsic_value
    
    return None


def calculate_true_north(symbol: str) -> ValuationResult:
    """
    Calculate the "True North" intrinsic value by averaging multiple models.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS')
    
    Returns:
        ValuationResult with all calculated values
    """
    # Fetch financial data
    data = fetch_financial_data(symbol)
    
    if data is None:
        return ValuationResult(
            symbol=symbol,
            current_price=0,
            error="Failed to fetch financial data"
        )
    
    current_price = data['current_price']
    
    if not current_price or current_price <= 0:
        return ValuationResult(
            symbol=symbol,
            current_price=0,
            error="Could not get current price"
        )
    
    # Calculate Benjamin Graham value
    graham_value = None
    if data['eps'] and data['eps'] > 0:
        graham_value = calculate_benjamin_graham(
            eps=data['eps'],
            growth_rate=data['growth_rate']
        )
    
    # Calculate DCF value
    dcf_value = None
    if data['fcf'] and data['shares_outstanding']:
        dcf_value = calculate_dcf(
            fcf=data['fcf'],
            shares_outstanding=data['shares_outstanding'],
            total_cash=data['total_cash'],
            total_debt=data['total_debt'],
            growth_rate=data['growth_rate'],
            market_cap=data.get('market_cap'),
            current_price=current_price
        )
    
    # Calculate EPV value
    epv_value = None
    if data['shares_outstanding']:
        epv_value = calculate_epv(
            ebit=data['ebit'],
            shares_outstanding=data['shares_outstanding'],
            total_cash=data['total_cash'],
            total_debt=data['total_debt'],
            market_cap=data.get('market_cap'),
            current_price=current_price,
            eps=data.get('eps')
        )
    
    # Calculate True North (average of available values)
    valid_values = [v for v in [graham_value, dcf_value, epv_value] if v and v > 0]
    
    if valid_values:
        true_north_value = sum(valid_values) / len(valid_values)
        upside_percent = ((true_north_value - current_price) / current_price) * 100
        
        if upside_percent > 20:
            valuation_status = "🟢 Undervalued"
        elif upside_percent > 0:
            valuation_status = "🟡 Fairly Valued"
        elif upside_percent > -20:
            valuation_status = "🟠 Slightly Overvalued"
        else:
            valuation_status = "🔴 Overvalued"
    else:
        true_north_value = None
        upside_percent = None
        valuation_status = "⚪ Insufficient Data"
    
    return ValuationResult(
        symbol=symbol,
        current_price=current_price,
        graham_value=graham_value,
        dcf_value=dcf_value,
        epv_value=epv_value,
        true_north_value=true_north_value,
        valuation_status=valuation_status,
        upside_percent=upside_percent,
        eps=data['eps'],
        growth_rate=data['growth_rate']
    )
