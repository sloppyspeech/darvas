"""
Stock Target Projection Tool
=============================
Calculate 5-year price targets based on sales growth projections and PE valuations.
Provides sensitivity analysis for margin of safety calculations.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import yfinance as yf


@dataclass
class StockFinancials:
    """Fetched financial data for a stock."""
    symbol: str
    company_name: str = ""
    current_sales: float = 0.0  # Revenue in Crores
    historical_npm: float = 0.0  # Net Profit Margin %
    outstanding_shares: float = 0.0  # Shares in Crores
    current_pe: float = 0.0
    industry_pe: float = 0.0  # For target P/E suggestion
    eps: float = 0.0
    revenue_growth: float = 0.0  # Historical CAGR suggestion
    current_price: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'company_name': self.company_name,
            'current_sales': self.current_sales,
            'historical_npm': self.historical_npm,
            'outstanding_shares': self.outstanding_shares,
            'current_pe': self.current_pe,
            'industry_pe': self.industry_pe,
            'eps': self.eps,
            'revenue_growth': self.revenue_growth,
            'current_price': self.current_price,
            'error': self.error
        }


def fetch_stock_financials(symbol: str) -> StockFinancials:
    """
    Fetch financial data from yfinance for target projection inputs.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE.NS')
        
    Returns:
        StockFinancials with populated data or error message
    """
    result = StockFinancials(symbol=symbol)
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Check if valid stock
        if not info or info.get('regularMarketPrice') is None:
            result.error = f"Could not fetch data for {symbol}. Check the symbol format."
            return result
        
        # Company name
        result.company_name = info.get('shortName', info.get('longName', symbol))
        
        # Current price
        result.current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        
        # Revenue (Total Revenue) - convert to Crores (divide by 10^7)
        total_revenue = info.get('totalRevenue', 0) or 0
        result.current_sales = round(total_revenue / 1e7, 2)  # Convert to Crores
        
        # Net Profit Margin (as percentage)
        profit_margin = info.get('profitMargins', 0) or 0
        result.historical_npm = round(profit_margin * 100, 2)  # Convert to percentage
        
        # Outstanding Shares - convert to Crores (divide by 10^7)
        shares = info.get('sharesOutstanding', 0) or 0
        result.outstanding_shares = round(shares / 1e7, 2)  # Convert to Crores
        
        # Current P/E ratio
        result.current_pe = round(info.get('trailingPE', 0) or 0, 2)
        
        # Industry/Forward P/E as target suggestion
        forward_pe = info.get('forwardPE', 0) or 0
        result.industry_pe = round(forward_pe, 2) if forward_pe > 0 else result.current_pe
        
        # EPS
        result.eps = round(info.get('trailingEps', 0) or 0, 2)
        
        # Revenue Growth (as percentage)
        revenue_growth = info.get('revenueGrowth', 0) or 0
        result.revenue_growth = round(revenue_growth * 100, 2)  # Convert to percentage
        
        # If revenue growth is 0, try earnings growth
        if result.revenue_growth == 0:
            earnings_growth = info.get('earningsGrowth', 0) or info.get('earningsQuarterlyGrowth', 0) or 0
            result.revenue_growth = round(earnings_growth * 100, 2)
        
        # Default to 10% if still 0
        if result.revenue_growth <= 0:
            result.revenue_growth = 10.0
            
    except Exception as e:
        result.error = f"Error fetching data: {str(e)}"
    
    return result


@dataclass
class ProjectionInputs:
    """Input parameters for stock target projection."""
    current_sales: float  # Current TTM revenue in Crores
    projected_cagr: float  # Estimated annual growth (%)
    projection_years: int  # Years into the future
    historical_npm: float  # Net Profit Margin (%)
    outstanding_shares: float  # Total shares in Crores
    current_pe: float  # Current valuation multiple
    target_pe: float  # Industry average or exit multiple
    growth_catalysts: List[str] = field(default_factory=list)  # Growth catalyst list


@dataclass 
class ProjectionResult:
    """Result of stock target projection calculation."""
    # Input echo
    inputs: ProjectionInputs
    
    # Yearly projections
    yearly_sales: List[float] = field(default_factory=list)  # Sales for each year
    yearly_profit: List[float] = field(default_factory=list)  # Net profit for each year
    
    # Final year metrics
    final_sales: float = 0.0
    final_profit: float = 0.0
    final_eps: float = 0.0
    
    # Target prices
    conservative_target: float = 0.0  # Using current P/E
    base_target: float = 0.0  # Using average P/E
    optimistic_target: float = 0.0  # Using target P/E
    
    # Sensitivity analysis results
    sensitivity_low_cagr: float = 0.0  # CAGR - 5%
    sensitivity_low_conservative: float = 0.0
    sensitivity_low_optimistic: float = 0.0
    sensitivity_high_cagr: float = 0.0  # CAGR + 5%
    sensitivity_high_conservative: float = 0.0
    sensitivity_high_optimistic: float = 0.0
    
    # Validation
    is_valid: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'current_sales': self.inputs.current_sales,
            'projected_cagr': self.inputs.projected_cagr,
            'projection_years': self.inputs.projection_years,
            'historical_npm': self.inputs.historical_npm,
            'outstanding_shares': self.inputs.outstanding_shares,
            'current_pe': self.inputs.current_pe,
            'target_pe': self.inputs.target_pe,
            'growth_catalysts': self.inputs.growth_catalysts,
            'yearly_sales': self.yearly_sales,
            'yearly_profit': self.yearly_profit,
            'final_sales': self.final_sales,
            'final_profit': self.final_profit,
            'final_eps': self.final_eps,
            'conservative_target': self.conservative_target,
            'optimistic_target': self.optimistic_target,
            'sensitivity_low_cagr': self.sensitivity_low_cagr,
            'sensitivity_low_conservative': self.sensitivity_low_conservative,
            'sensitivity_low_optimistic': self.sensitivity_low_optimistic,
            'sensitivity_high_cagr': self.sensitivity_high_cagr,
            'sensitivity_high_conservative': self.sensitivity_high_conservative,
            'sensitivity_high_optimistic': self.sensitivity_high_optimistic,
            'is_valid': self.is_valid,
            'error_message': self.error_message
        }


def validate_inputs(inputs: ProjectionInputs) -> tuple[bool, Optional[str]]:
    """
    Validate projection inputs (R1 requirement).
    
    Args:
        inputs: ProjectionInputs dataclass
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if inputs.current_sales <= 0:
        return False, "Current Sales must be positive"
    
    if inputs.outstanding_shares <= 0:
        return False, "Outstanding Shares must be positive"
    
    if inputs.projected_cagr < 0:
        return False, "Projected CAGR cannot be negative"
    
    if inputs.projection_years < 1 or inputs.projection_years > 10:
        return False, "Projection period must be between 1 and 10 years"
    
    if inputs.historical_npm < 0 or inputs.historical_npm > 100:
        return False, "Net Profit Margin must be between 0% and 100%"
    
    if inputs.current_pe <= 0:
        return False, "Current P/E must be positive"
    
    if inputs.target_pe <= 0:
        return False, "Target P/E must be positive"
    
    return True, None


def calculate_projection_core(
    current_sales: float,
    cagr: float,
    years: int,
    npm: float,
    shares: float,
    current_pe: float,
    target_pe: float
) -> tuple[List[float], List[float], float, float, float, float, float]:
    """
    Core calculation logic for projection.
    
    Returns:
        Tuple of (yearly_sales, yearly_profit, final_sales, final_profit, 
                  final_eps, conservative_target, optimistic_target)
    """
    yearly_sales = []
    yearly_profit = []
    
    # Calculate year-by-year projections
    for year in range(1, years + 1):
        # Revenue projection: Future_Sales = Current_Sales × (1 + CAGR)^n
        projected_sales = current_sales * ((1 + cagr / 100) ** year)
        yearly_sales.append(round(projected_sales, 2))
        
        # Net Profit: Projected_Net_Profit = Future_Sales × NPM
        projected_profit = projected_sales * (npm / 100)
        yearly_profit.append(round(projected_profit, 2))
    
    # Final year metrics
    final_sales = yearly_sales[-1] if yearly_sales else 0
    final_profit = yearly_profit[-1] if yearly_profit else 0
    
    # EPS calculation: Projected_EPS = Net_Profit / Outstanding_Shares
    final_eps = final_profit / shares if shares > 0 else 0
    
    # Target Price Valuation
    # Case 1 (Conservative): EPS × Current P/E
    conservative_target = final_eps * current_pe
    
    # Case 2 (Base): EPS × Average P/E
    base_pe = (current_pe + target_pe) / 2
    base_target = final_eps * base_pe
    
    # Case 3 (Optimistic): EPS × Target P/E
    optimistic_target = final_eps * target_pe
    
    return (
        yearly_sales, 
        yearly_profit, 
        round(final_sales, 2), 
        round(final_profit, 2), 
        round(final_eps, 2), 
        round(conservative_target, 2),
        round(base_target, 2),
        round(optimistic_target, 2)
    )


def calculate_projection(inputs: ProjectionInputs) -> ProjectionResult:
    """
    Calculate stock target projection based on inputs.
    
    Args:
        inputs: ProjectionInputs with all required parameters
        
    Returns:
        ProjectionResult with calculated values
    """
    # Validate inputs first
    is_valid, error_msg = validate_inputs(inputs)
    
    if not is_valid:
        result = ProjectionResult(inputs=inputs)
        result.is_valid = False
        result.error_message = error_msg
        return result
    
    # Core calculation
    (yearly_sales, yearly_profit, final_sales, final_profit, 
     final_eps, conservative_target, base_target, optimistic_target) = calculate_projection_core(
        inputs.current_sales,
        inputs.projected_cagr,
        inputs.projection_years,
        inputs.historical_npm,
        inputs.outstanding_shares,
        inputs.current_pe,
        inputs.target_pe
    )
    
    # Sensitivity analysis (R2 requirement): CAGR ± 5%
    low_cagr = max(0, inputs.projected_cagr - 5)  # Don't go below 0
    high_cagr = inputs.projected_cagr + 5
    
    # Low CAGR scenario
    _, _, _, _, low_eps, low_conservative, low_base, low_optimistic = calculate_projection_core(
        inputs.current_sales,
        low_cagr,
        inputs.projection_years,
        inputs.historical_npm,
        inputs.outstanding_shares,
        inputs.current_pe,
        inputs.target_pe
    )
    
    # High CAGR scenario
    _, _, _, _, high_eps, high_conservative, high_base, high_optimistic = calculate_projection_core(
        inputs.current_sales,
        high_cagr,
        inputs.projection_years,
        inputs.historical_npm,
        inputs.outstanding_shares,
        inputs.current_pe,
        inputs.target_pe
    )
    
    return ProjectionResult(
        inputs=inputs,
        yearly_sales=yearly_sales,
        yearly_profit=yearly_profit,
        final_sales=final_sales,
        final_profit=final_profit,
        final_eps=final_eps,
        conservative_target=conservative_target,
        base_target=base_target,
        optimistic_target=optimistic_target,
        sensitivity_low_cagr=low_cagr,
        sensitivity_low_conservative=low_conservative,
        sensitivity_low_optimistic=low_optimistic,
        sensitivity_high_cagr=high_cagr,
        sensitivity_high_conservative=high_conservative,
        sensitivity_high_optimistic=high_optimistic,
        is_valid=True,
        error_message=None
    )
