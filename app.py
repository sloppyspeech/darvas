"""
Darvas Box Momentum Analyzer - Streamlit Web Application V2.0
=============================================================
Interactive web interface for Darvas Box analysis of NSE/BSE stocks.
Includes Nifty 500 Screener for candidate identification.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import time
from streamlit_searchbox import st_searchbox

from darvas_core import (
    analyze_stock, 
    CONFIRMATION_DAYS, 
    VOLUME_MULTIPLIER,
    create_chart,
    fetch_stock_data,
    detect_darvas_boxes,
    generate_signals
)
from database import (
    generate_study_id,
    save_study,
    get_all_studies,
    get_study_results,
    get_study_details,
    delete_study,
    get_symbol_history,
    generate_screener_run_id,
    save_screener_run,
    get_all_screener_runs,
    get_screener_results,
    delete_screener_run
)
from screener import (
    run_screener,
    get_candidates_only,
    results_to_dataframe,
    get_symbols_for_analysis,
    PROXIMITY_THRESHOLD,
    STRENGTH_THRESHOLD,
    VOLUME_SPIKE_THRESHOLD
)
from nifty500 import get_nse_symbols, get_nifty50_sample, get_symbol_count
from stock_symbols import (
    search_symbols, 
    get_all_symbols, 
    refresh_stock_symbols,
    get_symbol_count as get_db_symbol_count,
    parse_display_format
)
from ollama_integration import (
    check_ollama_connection,
    generate_darvas_summary,
    OLLAMA_MODEL
)
from valuation import calculate_true_north, ValuationResult
from target_projection import ProjectionInputs, calculate_projection, fetch_stock_financials, StockFinancials


def search_stock_symbols(query: str) -> list:
    """
    Search function for st_searchbox autocomplete.
    Returns list of tuples (display_text, value).
    """
    if not query or len(query) < 2:
        return []
    
    results = search_symbols(query, limit=10)
    if not results:
        return []
    
    # Return list of tuples (display_text, yf_symbol)
    return [(f"{r['symbol']} - {r.get('company_name', '')[:30]} ({r['exchange']})", r['yf_symbol']) 
            for r in results]


# Page configuration
st.set_page_config(
    page_title="Darvas Box Analyzer V2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }
    .priority-high {
        background-color: #1b5e20;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .priority-medium {
        background-color: #f57c00;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .priority-low {
        background-color: #0d47a1;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .gate-pass {
        color: #4caf50;
        font-weight: bold;
    }
    .gate-fail {
        color: #f44336;
    }
    /* Fix searchbox alignment with buttons */
    [data-testid="stHorizontalBlock"] {
        align-items: flex-end !important;
    }
    /* Ensure searchbox aligns properly */
    [data-testid="stHorizontalBlock"] > div {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stock-market.png", width=80)
        st.title("📊 Darvas Box V2")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🔎 Screener", "🔍 New Analysis", "📚 Study History", "📈 Quick Analyze", "💎 True North", "🎯 Target Projection", "📋 Screener History"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        confirmation_days = st.slider(
            "Confirmation Days",
            min_value=2,
            max_value=5,
            value=CONFIRMATION_DAYS,
            help="Days to confirm box top/bottom"
        )
        
        volume_multiplier = st.slider(
            "Volume Multiplier",
            min_value=1.0,
            max_value=3.0,
            value=VOLUME_MULTIPLIER,
            step=0.1,
            help="Required volume spike for valid breakout"
        )
        
        st.markdown("---")
        st.caption("Built for NSE/BSE Markets 🇮🇳")
        st.caption(f"Nifty 500: {get_symbol_count()} stocks")
    
    # Main content based on navigation
    if page == "🔎 Screener":
        render_screener()
    elif page == "🔍 New Analysis":
        render_new_analysis(confirmation_days, volume_multiplier)
    elif page == "📚 Study History":
        render_study_history()
    elif page == "📋 Screener History":
        render_screener_history()
    elif page == "💎 True North":
        render_true_north()
    elif page == "🎯 Target Projection":
        render_target_projection()
    else:
        render_quick_analyze(confirmation_days, volume_multiplier)


def render_screener():
    """Render the stock screener page with universal exchange options."""
    st.header("🔎 Darvas Candidate Screener")
    
    st.markdown("""
    **Multi-Gate Screening Funnel** to identify high-probability Darvas Box candidates:
    
    | Gate | Criterion | Threshold |
    |------|-----------|-----------|
    | 1️⃣ | Proximity to 52W High | Within 10% |
    | 2️⃣ | Strength (Price Momentum) | 2x from 52W Low |
    | 3️⃣ | Trend (Above 200 SMA) | Price > SMA |
    | 4️⃣ | Interest (Volume Spike) | 1.5x Avg Volume |
    """)
    
    st.markdown("---")
    
    # Get database counts
    db_counts = get_db_symbol_count()
    nse_count = db_counts.get('NSE', 0)
    bse_count = db_counts.get('BSE', 0)
    total_count = nse_count + bse_count
    
    # Screening options
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        exchange_option = st.selectbox(
            "Exchange",
            ["NSE", "BSE", "Both (NSE + BSE)"],
            help=f"NSE: {nse_count} stocks | BSE: {bse_count} stocks"
        )
    
    with col2:
        # Stock count options based on exchange
        if exchange_option == "NSE":
            scan_options = ["Top 50 (Fast)", "Top 200", "Top 500", f"All NSE ({nse_count})"]
            default_counts = [50, 200, 500, nse_count]
        elif exchange_option == "BSE":
            scan_options = ["Top 50 (Fast)", "Top 200", "Top 500", f"All BSE ({bse_count})"]
            default_counts = [50, 200, 500, bse_count]
        else:
            scan_options = ["Top 100 (Fast)", "Top 500", "Top 1000", f"All ({total_count})"]
            default_counts = [100, 500, 1000, total_count]
        
        scan_option = st.selectbox(
            "Scan Size",
            scan_options,
            help="Select number of stocks to screen"
        )
        
        # Get selected count
        selected_idx = scan_options.index(scan_option)
        stock_limit = default_counts[selected_idx]
    
    with col3:
        priority_filter = st.selectbox(
            "Show Priority",
            ["All Candidates", "High Only", "Medium & Above"],
            help="Filter results by priority level"
        )
    
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run", type="primary", width='stretch')
    
    # Show database info
    st.caption(f"📊 Database: {nse_count} NSE + {bse_count} BSE = {total_count} total stocks")
    
    if run_btn:
        # Get symbols based on exchange selection
        if exchange_option == "NSE":
            all_symbols = get_all_symbols("NSE")
        elif exchange_option == "BSE":
            all_symbols = get_all_symbols("BSE")
        else:
            all_symbols = get_all_symbols()  # Both
        
        # Limit to selected count
        symbols_to_scan = [s['yf_symbol'] for s in all_symbols[:stock_limit]]
        
        st.info(f"Screening {len(symbols_to_scan)} {exchange_option} stocks... This may take a few minutes.")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, symbol):
            progress_bar.progress(current / total)
            status_text.text(f"Screening {symbol}... ({current}/{total})")
        
        start_time = time.time()
        
        # Run screener
        results, stats = run_screener(symbols_to_scan, progress_callback=update_progress)
        
        elapsed_time = time.time() - start_time
        status_text.text(f"✅ Screening completed in {elapsed_time:.2f} seconds")
        
        # Get candidates only
        candidates = get_candidates_only(results)
        
        # Save to database
        run_id = generate_screener_run_id()
        results_dicts = [r.to_dict() for r in results]
        
        if save_screener_run(run_id, results_dicts, stats):
            st.success(f"📁 Screener run saved: **{run_id}**")
        
        # Store in session state
        st.session_state['screener_results'] = results
        st.session_state['screener_candidates'] = candidates
        st.session_state['screener_stats'] = stats
        st.session_state['screener_run_id'] = run_id
    
    # Display results if available
    if 'screener_candidates' in st.session_state:
        display_screener_results(
            st.session_state['screener_candidates'],
            st.session_state['screener_stats'],
            priority_filter
        )


def display_screener_results(candidates, stats, priority_filter):
    """Display screener results in heatmap format."""
    st.markdown("---")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Screened", stats['processed'])
    with col2:
        st.metric("Candidates", stats['candidates'])
    with col3:
        st.metric("🟢 High Priority", stats['high_priority'])
    with col4:
        st.metric("🟡 Medium Priority", stats['medium_priority'])
    with col5:
        st.metric("🔵 Low Priority", stats['low_priority'])
    
    if not candidates:
        st.warning("No candidates found matching the screening criteria.")
        return
    
    # Apply priority filter
    filtered = candidates
    if priority_filter == "High Only":
        filtered = [c for c in candidates if c.priority == "High"]
    elif priority_filter == "Medium & Above":
        filtered = [c for c in candidates if c.priority in ["High", "Medium"]]
    
    if not filtered:
        st.info(f"No candidates match the '{priority_filter}' filter.")
        return
    
    st.markdown("### 🔥 Darvas Candidates Heatmap")
    
    # Create DataFrame for display
    df = results_to_dataframe(filtered)
    
    # Reorder and rename columns for heatmap
    display_cols = ['symbol', 'priority', 'proximity_pct', 'strength_ratio', 
                    'above_sma', 'volume_spike', 'gates_passed', 
                    'consolidation_range', 'days_in_consolidation', 'current_price']
    
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    display_df.columns = ['Symbol', 'Priority', '% from 52W High', 'Strength (x)', 
                          'Above SMA', 'Vol Spike', 'Gates', 
                          'Consolidation %', 'Days Consol.', 'Price']
    
    # Style the dataframe
    def style_priority(val):
        if val == 'High':
            return 'background-color: #1b5e20; color: white'
        elif val == 'Medium':
            return 'background-color: #f57c00; color: white'
        elif val == 'Low':
            return 'background-color: #0d47a1; color: white'
        return ''
    
    def style_bool(val):
        if val == True:
            return 'color: #4caf50; font-weight: bold'
        return 'color: #f44336'
    
    styled_df = display_df.style.applymap(style_priority, subset=['Priority'])
    styled_df = styled_df.applymap(style_bool, subset=['Above SMA', 'Vol Spike'])
    styled_df = styled_df.format({
        '% from 52W High': '{:.1f}%',
        'Strength (x)': '{:.2f}x',
        'Consolidation %': '{:.1f}%',
        'Price': '₹{:.2f}'
    })
    
    st.dataframe(styled_df, width='stretch', hide_index=True)
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Results (CSV)",
            csv,
            f"darvas_screener_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    
    with col2:
        # Get symbols for Darvas analysis
        symbols_list = "\n".join([c.symbol for c in filtered])
        st.download_button(
            "📋 Export Symbols for Analysis",
            symbols_list,
            "screener_symbols.txt",
            "text/plain"
        )
    
    # High priority detail cards
    high_priority = [c for c in filtered if c.priority == "High"]
    if high_priority:
        st.markdown("### 🎯 High Priority Candidates")
        
        for candidate in high_priority[:5]:  # Show top 5
            with st.expander(f"🟢 {candidate.symbol} - {candidate.gates_passed}/4 Gates", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Current Price", f"₹{candidate.current_price:.2f}")
                col2.metric("52W High", f"₹{candidate.high_52w:.2f}")
                col3.metric("% from High", f"{candidate.proximity_pct:.1f}%")
                col4.metric("Strength", f"{candidate.strength_ratio:.2f}x")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Above 200 SMA", "✅ Yes" if candidate.above_sma else "❌ No")
                col2.metric("Volume Spike", "✅ Yes" if candidate.volume_spike else "❌ No")
                col3.metric("Consolidation", f"{candidate.consolidation_range:.1f}%")
                col4.metric("Days in Range", candidate.days_in_consolidation)


def render_screener_history():
    """Render the screener history page."""
    st.header("📋 Screener History")
    
    runs = get_all_screener_runs()
    
    if not runs:
        st.info("No screener runs found. Run the screener to create your first scan!")
        return
    
    # Run selector
    run_options = {
        f"{r['run_id']} - {r['candidates_found']} candidates ({r['total_screened']} screened)": r['run_id'] 
        for r in runs
    }
    
    selected_display = st.selectbox("Select Screener Run", list(run_options.keys()))
    selected_run_id = run_options[selected_display]
    
    col1, col2 = st.columns([4, 1])
    
    with col2:
        if st.button("🗑️ Delete Run", type="secondary"):
            if delete_screener_run(selected_run_id):
                st.success("Screener run deleted!")
                st.rerun()
    
    # Get results
    results = get_screener_results(selected_run_id)
    
    # Find the run stats
    run_info = next((r for r in runs if r['run_id'] == selected_run_id), None)
    
    if run_info:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Screened", run_info['total_screened'])
        col2.metric("🟢 High", run_info['high_priority'])
        col3.metric("🟡 Medium", run_info['medium_priority'])
        col4.metric("🔵 Low", run_info['low_priority'])
    
    if results:
        # Filter to candidates only
        candidates = [r for r in results if r.get('priority') != 'None']
        
        if candidates:
            df = pd.DataFrame(candidates)
            display_cols = ['symbol', 'priority', 'proximity_pct', 'strength_ratio', 
                           'above_sma', 'volume_spike', 'gates_passed', 'current_price']
            display_df = df[[c for c in display_cols if c in df.columns]]
            
            st.dataframe(display_df, width='stretch', hide_index=True)
        else:
            st.info("No candidates found in this screener run.")


def render_new_analysis(confirmation_days: int, volume_multiplier: float):
    """Render the new analysis page."""
    st.header("🔍 New Batch Analysis")
    
    # Stock search section
    st.markdown("### 🔎 Search & Add Stocks")
    
    # Stock search on its own row
    selected_stock = st_searchbox(
        search_stock_symbols,
        key="new_analysis_searchbox",
        placeholder="Type to search stocks...",
        clear_on_submit=True,
    )
    # Add selected stock to the list
    if selected_stock:
        current = st.session_state.get('selected_symbols', [])
        if selected_stock not in current:
            current.append(selected_stock)
            st.session_state['selected_symbols'] = current
            st.rerun()
    
    # Exchange filter and refresh button on same row
    col1, col2 = st.columns([3, 1])
    with col1:
        exchange_filter = st.selectbox(
            "Exchange",
            ["All", "NSE", "BSE"],
            key="exchange_filter"
        )
    with col2:
        if st.button("🔄 Refresh DB", help="Re-fetch stock list from NSE/BSE"):
            with st.spinner("Fetching stocks..."):
                nse, bse = refresh_stock_symbols()
                st.success(f"Updated: {nse} NSE, {bse} BSE stocks")
    
    st.markdown("---")
    
    # Stock input area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Build default value from session state or defaults
        if 'selected_symbols' in st.session_state and st.session_state['selected_symbols']:
            default_value = "\n".join(st.session_state['selected_symbols'])
        elif 'screener_candidates' in st.session_state:
            high_symbols = [c.symbol for c in st.session_state['screener_candidates'] 
                           if c.priority == "High"][:10]
            default_value = "\n".join(high_symbols) if high_symbols else "RELIANCE.NS\nTCS.NS\nHDFCBANK.NS"
        else:
            default_value = "RELIANCE.NS\nTCS.NS\nHDFCBANK.NS\nINFY.NS\nICICIBANK.NS"
        
        stocks_input = st.text_area(
            "Selected Stocks (one per line)",
            value=default_value,
            height=150,
            help="Use .NS for NSE stocks, .BO for BSE stocks. Search above to add more."
        )
    
    with col2:
        st.markdown("### Quick Add")
        if st.button("Nifty 50 Sample", width='stretch'):
            st.session_state['selected_symbols'] = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "BHARTIARTL.NS", "SBIN.NS", "WIPRO.NS", "TATAMOTORS.NS", "AXISBANK.NS"
            ]
            st.rerun()
        
        if st.button("Clear All", width='stretch'):
            st.session_state['selected_symbols'] = []
            st.rerun()
        
        if 'screener_candidates' in st.session_state:
            if st.button("From Screener", width='stretch'):
                high_symbols = [c.symbol for c in st.session_state['screener_candidates'] 
                               if c.priority in ["High", "Medium"]]
                if high_symbols:
                    st.session_state['selected_symbols'] = high_symbols
                    st.rerun()
        
        # Show symbol count from DB
        db_counts = get_db_symbol_count()
        total = sum(db_counts.values()) if db_counts else 0
        st.caption(f"📊 {total} stocks in database")
    
    # Study description
    study_desc = st.text_input(
        "Study Description (optional)",
        placeholder="e.g., Weekly momentum scan - Week 52"
    )
    
    # Run analysis button
    if st.button("🚀 Run Analysis", type="primary", width='stretch'):
        stocks = [s.strip() for s in stocks_input.strip().split('\n') if s.strip()]
        
        if not stocks:
            st.error("Please enter at least one stock symbol")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        boxes_data = {}
        all_data = {}
        
        start_time = time.time()
        
        for i, symbol in enumerate(stocks):
            status_text.text(f"Analyzing {symbol}... ({i+1}/{len(stocks)})")
            progress_bar.progress((i + 1) / len(stocks))
            
            result = analyze_stock(symbol, confirmation_days, volume_multiplier)
            results.append(result['signal'])
            
            if result['success']:
                all_data[symbol] = result
                # Store box data for database
                if result.get('boxes'):
                    boxes_data[symbol] = [box.to_dict() for box in result['boxes']]
        
        elapsed_time = time.time() - start_time
        status_text.text(f"✅ Analysis completed in {elapsed_time:.2f} seconds")
        
        # Save to database
        study_id = generate_study_id()
        description = study_desc if study_desc else f"Analysis of {len(stocks)} stocks"
        
        if save_study(study_id, description, results, boxes_data):
            st.success(f"📁 Study saved: **{study_id}**")
        
        # Store in session state for display
        st.session_state['results'] = results
        st.session_state['all_data'] = all_data
        st.session_state['study_id'] = study_id
    
    # Display results if available
    if 'results' in st.session_state and st.session_state['results']:
        display_results(st.session_state['results'], st.session_state.get('all_data', {}))


def display_results(results: list, all_data: dict):
    """Display analysis results."""
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Filter actionable setups
    actionable = df[df['status'].str.contains('Inside Box|Breakout', na=False)]
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Stocks", len(df))
    with col2:
        buy_signals = len(df[df['suggestion'].str.contains('BUY', na=False)]) if 'suggestion' in df.columns else 0
        st.metric("🟢 BUY Signals", buy_signals)
    with col3:
        watch_signals = len(df[df['suggestion'].str.contains('WATCH', na=False)]) if 'suggestion' in df.columns else 0
        st.metric("🔵 WATCH", watch_signals)
    with col4:
        breakouts = len(df[df['status'].str.contains('Breakout', na=False)])
        st.metric("Breakouts", breakouts)
    with col5:
        inside_box = len(df[df['status'].str.contains('Inside Box', na=False)])
        st.metric("Inside Box", inside_box)
    
    # Actionable setups highlight
    if len(actionable) > 0:
        st.markdown("### 🎯 Actionable Setups")
        for _, row in actionable.iterrows():
            suggestion = row.get('suggestion', '⚪ SKIP')
            suggestion_reason = row.get('suggestion_reason', '')
            
            with st.expander(f"{suggestion} {row['symbol']} - {row['status']}", expanded=True):
                # Show suggestion reason
                if suggestion_reason:
                    if 'BUY' in suggestion:
                        st.success(f"**Suggestion:** {suggestion_reason}")
                    elif 'WATCH' in suggestion:
                        st.info(f"**Suggestion:** {suggestion_reason}")
                    elif 'CAUTION' in suggestion:
                        st.warning(f"**Suggestion:** {suggestion_reason}")
                    else:
                        st.write(f"**Suggestion:** {suggestion_reason}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"₹{row['current_price']}")
                col2.metric("Entry Price", f"₹{row['entry_price']}")
                col3.metric("Stop Loss", f"₹{row['stop_loss']}")
                col4.metric("Target (2R)", f"₹{row.get('target_price', 'N/A')}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Risk", f"{row['risk_percent']}%")
                col2.metric("Risk:Reward", row['risk_reward'])
                col3.metric("Box Range", f"₹{row['box_bottom']} - ₹{row['box_top']}")
                col4.metric("Volume OK", "✅" if row.get('volume_confirmed') else "❌")
                
                # Show chart if available
                if row['symbol'] in all_data and all_data[row['symbol']].get('chart'):
                    st.plotly_chart(
                        all_data[row['symbol']]['chart'], 
                        width='stretch',
                        key=f"chart_{row['symbol']}"
                    )
    
    # Full results table
    st.markdown("### 📋 Full Results")
    
    # Format for display with new columns
    display_cols = ['symbol', 'suggestion', 'status', 'current_price', 'entry_price', 
                    'stop_loss', 'target_price', 'risk_percent', 'risk_reward']
    available_cols = [c for c in display_cols if c in df.columns]
    display_df = df[available_cols].copy()
    
    # Rename columns for better readability
    col_rename = {
        'symbol': 'Symbol',
        'suggestion': 'Suggestion', 
        'status': 'Status',
        'current_price': 'Price',
        'entry_price': 'Entry',
        'stop_loss': 'Stop Loss',
        'target_price': 'Target',
        'risk_percent': 'Risk %',
        'risk_reward': 'R:R'
    }
    display_df.columns = [col_rename.get(c, c) for c in display_df.columns]
    
    st.dataframe(display_df, width='stretch', hide_index=True)
    
    # Legend
    st.markdown("""
    **Legend:** 
    🟢 BUY = Enter on breakout | 🔵 WATCH = Wait for breakout | 🟡 CAUTION = Low volume | ⚪ SKIP = Not a setup
    
    **R:R** = Risk:Reward ratio (e.g., 1:2.5 means for every ₹1 risked, potential gain is ₹2.50)
    """)
    
    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Results (CSV)",
        csv,
        f"darvas_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )


def render_study_history():
    """Render the study history page."""
    st.header("📚 Study History")
    
    studies = get_all_studies()
    
    if not studies:
        st.info("No studies found. Run a new analysis to create your first study!")
        return
    
    # Study selector
    study_options = {
        f"{s['study_id']} - {s['description']} ({s['stock_count']} stocks)": s['study_id'] 
        for s in studies
    }
    
    selected_display = st.selectbox("Select Study", list(study_options.keys()))
    selected_study_id = study_options[selected_display]
    
    col1, col2 = st.columns([4, 1])
    
    with col2:
        if st.button("🗑️ Delete Study", type="secondary"):
            if delete_study(selected_study_id):
                st.success("Study deleted!")
                st.rerun()
    
    # Get study details and results
    study_details = get_study_details(selected_study_id)
    results = get_study_results(selected_study_id)
    
    if study_details:
        st.markdown(f"**Date:** {study_details['study_date']}")
        st.markdown(f"**Description:** {study_details['description']}")
    
    if results:
        df = pd.DataFrame(results)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Stocks", len(df))
        with col2:
            actionable = len(df[df['status'].str.contains('Inside Box|Breakout', na=False)])
            st.metric("Actionable", actionable)
        with col3:
            breakouts = len(df[df['status'].str.contains('Breakout', na=False)])
            st.metric("Breakouts", breakouts)
        
        # Results table
        display_cols = ['symbol', 'status', 'current_price', 'box_top', 'box_bottom',
                       'entry_price', 'stop_loss', 'risk_percent', 'risk_reward']
        display_df = df[[c for c in display_cols if c in df.columns]]
        
        st.dataframe(display_df, width='stretch', hide_index=True)


def render_quick_analyze(confirmation_days: int, volume_multiplier: float):
    """Render the quick analyze page for single stock."""
    st.header("📈 Quick Analysis")
    
    # Stock search
    selected = st_searchbox(
        search_stock_symbols,
        key="quick_searchbox",
        placeholder="Type to search stocks (e.g., RELIANCE, TCS)...",
        default=st.session_state.get('quick_symbol', None),
        clear_on_submit=False,
    )
    if selected:
        st.session_state['quick_symbol'] = selected
        symbol = selected
    else:
        symbol = st.session_state.get('quick_symbol', '')
    
    analyze_btn = st.button("🔍 Analyze", type="primary")
    
    # Run analysis and store in session state
    if analyze_btn and symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            result = analyze_stock(symbol.strip(), confirmation_days, volume_multiplier)
        
        if not result['success']:
            st.error(f"❌ {result.get('error', 'Failed to fetch data for')} {symbol}")
            return
        
        # Store in session state for persistence
        st.session_state['quick_result'] = result
        st.session_state['quick_symbol'] = symbol.strip()
        st.session_state['quick_ai_summary'] = None  # Reset AI summary
    
    # Display results from session state
    if 'quick_result' in st.session_state and st.session_state['quick_result']:
        result = st.session_state['quick_result']
        signal = result['signal']
        current_symbol = st.session_state.get('quick_symbol', symbol)
        
        # Display signal summary
        st.markdown("### 📊 Signal Summary")
        
        # Status badge
        status = signal['status']
        if 'Breakout' in status:
            st.success(f"🟢 **{status}**")
        elif 'Inside Box' in status:
            st.info(f"🔵 **{status}**")
        else:
            st.warning(f"⚪ **{status}**")
        
        # Suggestion if available
        if signal.get('suggestion'):
            st.markdown(f"**Suggestion:** {signal.get('suggestion')} - {signal.get('suggestion_reason', '')}")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"₹{signal['current_price']}")
        col2.metric("Box Range", f"₹{signal['box_bottom']} - ₹{signal['box_top']}" if signal['box_top'] else "N/A")
        col3.metric("Entry Price", f"₹{signal['entry_price']}" if signal['entry_price'] else "N/A")
        col4.metric("Stop Loss", f"₹{signal['stop_loss']}" if signal['stop_loss'] else "N/A")
        
        if signal['entry_price']:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Target (2R)", f"₹{signal.get('target_price', signal.get('target_2r', 'N/A'))}")
            col2.metric("Risk", f"{signal['risk_percent']}%")
            col3.metric("Risk:Reward", signal['risk_reward'])
            col4.metric("Volume Confirmed", "✅ Yes" if signal['volume_confirmed'] else "❌ No")
        
        # Chart
        st.markdown("### 📈 Chart")
        st.plotly_chart(result['chart'], width='stretch', key="quick_chart")
        
        # AI Summary Section
        st.markdown("### 🤖 AI Analysis Summary")
        
        if check_ollama_connection():
            col1, col2 = st.columns([1, 4])
            with col1:
                generate_ai_btn = st.button("✨ Generate AI Summary", type="secondary", key="generate_ai_btn")
            
            if generate_ai_btn:
                with st.spinner(f"Generating summary using {OLLAMA_MODEL}..."):
                    ai_summary = generate_darvas_summary(signal, current_symbol)
                st.session_state['quick_ai_summary'] = ai_summary
            
            # Display AI summary from session state
            if st.session_state.get('quick_ai_summary'):
                ai_summary = st.session_state['quick_ai_summary']
                if not ai_summary.startswith("Error"):
                    st.success("**AI Trading Insights:**")
                    st.markdown(ai_summary)
                else:
                    st.warning(ai_summary)
        else:
            st.info("💡 AI Summary available when Ollama is running on port 11435. Start Ollama and refresh.")
        
        # Historical analysis
        st.markdown("### 📜 Historical Analysis")
        history = get_symbol_history(current_symbol)
        
        if history:
            hist_df = pd.DataFrame(history)
            display_cols = ['study_date', 'status', 'current_price', 'entry_price', 'stop_loss']
            hist_display = hist_df[[c for c in display_cols if c in hist_df.columns]]
            st.dataframe(hist_display, width='stretch', hide_index=True)
        else:
            st.info("No historical analysis found for this symbol.")


def render_true_north():
    """Render the True North intrinsic value calculator page."""
    st.header("💎 True North - Intrinsic Value Calculator")
    
    st.markdown("""
    Calculate a stock's fair value using **three valuation models**. The "True North" price 
    is the average of available models, helping identify undervalued opportunities.
    
    | Model | Best For | Method |
    |-------|----------|--------|
    | **Benjamin Graham** | Growth-oriented, mid-caps | EPS × Growth multiplier |
    | **DCF** | Cash-rich, stable large-caps | Discounted future cash flows |
    | **EPV** | Cyclical, no-growth companies | Current earnings power |
    """)
    
    st.markdown("---")
    
    # Stock input
    selected = st_searchbox(
        search_stock_symbols,
        key="valuation_searchbox",
        placeholder="Type to search stocks (e.g., RELIANCE, TCS)...",
        default=st.session_state.get('valuation_symbol', None),
        clear_on_submit=False,
    )
    if selected:
        st.session_state['valuation_symbol'] = selected
        symbol = selected
    else:
        symbol = st.session_state.get('valuation_symbol', '')
    
    calculate_btn = st.button("💎 Calculate Value", type="primary")
    
    # Calculate and store in session state
    if calculate_btn and symbol:
        with st.spinner(f"Calculating intrinsic value for {symbol}..."):
            result = calculate_true_north(symbol.strip())
        
        st.session_state['valuation_result'] = result
        st.session_state['valuation_symbol'] = symbol.strip()
    
    # Display results from session state
    if 'valuation_result' in st.session_state and st.session_state['valuation_result']:
        result = st.session_state['valuation_result']
        
        if result.error:
            st.error(f"❌ {result.error}")
            return
        
        # Main valuation result
        st.markdown("### 🎯 Valuation Result")
        
        # True North value highlight
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            st.metric(
                "Current Price",
                f"₹{result.current_price:.2f}" if result.current_price else "N/A"
            )
        
        with col2:
            st.metric(
                "💎 True North Value",
                f"₹{result.true_north_value:.2f}" if result.true_north_value else "N/A",
                delta=f"{result.upside_percent:.1f}%" if result.upside_percent else None
            )
        
        with col3:
            # Valuation status with color
            status = result.valuation_status
            if "Undervalued" in status:
                st.success(f"**{status}**")
            elif "Fairly" in status:
                st.info(f"**{status}**")
            elif "Slightly" in status:
                st.warning(f"**{status}**")
            elif "Overvalued" in status:
                st.error(f"**{status}**")
            else:
                st.info(f"**{status}**")
        
        # Individual model values
        st.markdown("### 📊 Model Breakdown")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📚 Benjamin Graham**")
            if result.graham_value:
                graham_diff = ((result.graham_value - result.current_price) / result.current_price) * 100
                st.metric(
                    "Graham Value",
                    f"₹{result.graham_value:.2f}",
                    delta=f"{graham_diff:.1f}%"
                )
                st.caption(f"EPS: ₹{result.eps:.2f}" if result.eps else "")
                st.caption(f"Growth: {result.growth_rate:.1f}%" if result.growth_rate else "")
            else:
                st.info("Insufficient data (needs positive EPS)")
        
        with col2:
            st.markdown("**💰 DCF Model**")
            if result.dcf_value:
                dcf_diff = ((result.dcf_value - result.current_price) / result.current_price) * 100
                st.metric(
                    "DCF Value",
                    f"₹{result.dcf_value:.2f}",
                    delta=f"{dcf_diff:.1f}%"
                )
                st.caption("Based on projected cash flows")
            else:
                st.info("Insufficient data (needs FCF)")
        
        with col3:
            st.markdown("**⚡ EPV Model**")
            if result.epv_value:
                epv_diff = ((result.epv_value - result.current_price) / result.current_price) * 100
                st.metric(
                    "EPV Value",
                    f"₹{result.epv_value:.2f}",
                    delta=f"{epv_diff:.1f}%"
                )
                st.caption("No-growth earnings power")
            else:
                st.info("Insufficient data (needs EBIT)")
        
        # Interpretation guide
        st.markdown("---")
        st.markdown("### 📖 How to Interpret")
        
        if result.graham_value and result.epv_value:
            if result.graham_value > result.epv_value * 1.2:
                st.info("📈 **Growth Premium**: Graham value > EPV suggests the market expects significant growth. Verify growth assumptions.")
            elif result.epv_value > result.graham_value * 1.2:
                st.warning("⚠️ **Value Trap Risk**: EPV > Graham may indicate the company is destroying value by growing.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **If True North > Current Price:**
            - Stock may be undervalued
            - Consider for investment with proper due diligence
            - Check qualitative factors (management, moat)
            """)
        
        with col2:
            st.markdown("""
            **If True North < Current Price:**
            - Stock may be overvalued
            - Wait for better entry point
            - Or the market knows something the models don't
            """)
        
        # AI Summary
        st.markdown("### 🤖 AI Valuation Summary")
        
        if check_ollama_connection():
            if st.button("✨ Generate AI Analysis", key="valuation_ai_btn"):
                with st.spinner(f"Generating analysis using {OLLAMA_MODEL}..."):
                    valuation_context = f"""
Stock: {result.symbol}
Current Price: ₹{result.current_price:.2f}
True North (Fair Value): ₹{result.true_north_value:.2f if result.true_north_value else 0}
Valuation Status: {result.valuation_status}
Upside/Downside: {f'{result.upside_percent:.1f}%' if result.upside_percent else 'N/A'}

Graham Value: ₹{result.graham_value:.2f if result.graham_value else 0}
DCF Value: ₹{result.dcf_value:.2f if result.dcf_value else 0}
EPV Value: ₹{result.epv_value:.2f if result.epv_value else 0}

EPS: ₹{result.eps:.2f if result.eps else 0}
Growth Rate: {f'{result.growth_rate:.1f}%' if result.growth_rate else 'N/A'}
"""
                    prompt = f"""Based on this intrinsic value analysis, provide a brief investment recommendation:

{valuation_context}

Give: 1) Overall verdict (BUY/HOLD/AVOID), 2) Key insight, 3) Risk to consider. Keep it under 100 words."""
                    
                    import requests
                    try:
                        response = requests.post(
                            "http://localhost:11435/api/generate",
                            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                            timeout=60
                        )
                        if response.status_code == 200:
                            ai_summary = response.json().get('response', '')
                            st.session_state['valuation_ai_summary'] = ai_summary
                    except:
                        st.session_state['valuation_ai_summary'] = "Error connecting to Ollama"
            
            if st.session_state.get('valuation_ai_summary'):
                st.success(st.session_state['valuation_ai_summary'])
        else:
            st.info("💡 AI Analysis available when Ollama is running on port 11435.")


def render_target_projection():
    """Render the Stock Target Projection Tool page."""
    import plotly.graph_objects as go
    
    st.header("🎯 Stock Target Projection Tool")
    
    st.markdown("""
    Estimate **5-year price targets** based on sales growth projections and P/E valuations.
    This tool provides a standardized framework for estimating future stock prices.
    
    | Step | Description |
    |------|-------------|
    | 1️⃣ | Enter stock symbol to auto-fetch financials |
    | 2️⃣ | Review/adjust fetched data and growth assumptions |
    | 3️⃣ | Define valuation multiples (Current P/E, Target P/E) |
    | 4️⃣ | Review targets with sensitivity analysis |
    """)
    
    st.markdown("---")
    
    # Stock Symbol Input Section
    st.subheader("🔎 Stock Symbol")
    
    selected = st_searchbox(
        search_stock_symbols,
        key="tp_searchbox",
        placeholder="Type to search stocks (e.g., RELIANCE, TCS)...",
        default=st.session_state.get('tp_symbol', None),
        clear_on_submit=False,
    )
    if selected:
        symbol_input = selected
        # Auto-trigger fetch when a new symbol is selected
        if selected != st.session_state.get('tp_symbol', ''):
            st.session_state['tp_pending_fetch'] = selected
    else:
        symbol_input = st.session_state.get('tp_symbol', '')
    
    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        fetch_btn = st.button("📥 Fetch Data", type="primary")
    with col_info:
        if 'tp_company_name' in st.session_state and st.session_state['tp_company_name']:
            st.success(f"✅ {st.session_state['tp_company_name']}")
    
    # Auto-fetch on selection or button click
    should_fetch = fetch_btn or st.session_state.get('tp_pending_fetch')
    if should_fetch and symbol_input:
        symbol = symbol_input.strip().upper() if isinstance(symbol_input, str) else symbol_input
        if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
            symbol = symbol + '.NS'  # Default to NSE
        
        st.session_state['tp_pending_fetch'] = None  # Clear pending fetch
        
        with st.spinner(f"Fetching data for {symbol}..."):
            financials = fetch_stock_financials(symbol)
        
        if financials.error:
            st.error(f"❌ {financials.error}")
        else:
            # Store fetched data in session state
            st.session_state['tp_symbol'] = symbol
            st.session_state['tp_company_name'] = financials.company_name
            st.session_state['tp_sales'] = financials.current_sales
            st.session_state['tp_npm'] = financials.historical_npm
            st.session_state['tp_shares'] = financials.outstanding_shares
            st.session_state['tp_cagr'] = financials.revenue_growth
            st.session_state['tp_current_pe'] = financials.current_pe if financials.current_pe > 0 else 15.0
            st.session_state['tp_target_pe'] = financials.industry_pe if financials.industry_pe > 0 else financials.current_pe
            st.session_state['tp_current_price'] = financials.current_price
            st.session_state['tp_eps'] = financials.eps
            st.success(f"✅ Data fetched for **{financials.company_name}** (₹{financials.current_price:.2f})")
            st.rerun()
    
    st.markdown("---")
    
    # Input form
    st.subheader("📊 Input Parameters")
    
    # Show current price if available
    if 'tp_current_price' in st.session_state and st.session_state['tp_current_price'] > 0:
        st.info(f"📈 Current Market Price: **₹{st.session_state['tp_current_price']:.2f}**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Revenue & Profitability**")
        current_sales = st.number_input(
            "Current Sales (₹ Crores)",
            min_value=0.0,
            value=st.session_state.get('tp_sales', 189.0),
            step=1.0,
            help="Most recent TTM (Trailing Twelve Months) revenue"
        )
        
        historical_npm = st.number_input(
            "Historical NPM (%)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.get('tp_npm', 10.0),
            step=0.5,
            help="Average Net Profit Margin over the last 5 years"
        )
        
        outstanding_shares = st.number_input(
            "Outstanding Shares (Crores)",
            min_value=0.0,
            value=st.session_state.get('tp_shares', 0.8),
            step=0.01,
            format="%.2f",
            help="Total number of shares currently issued"
        )
    
    with col2:
        st.markdown("**Growth Assumptions**")
        projected_cagr = st.number_input(
            "Projected CAGR (%)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.get('tp_cagr', 20.0),
            step=1.0,
            help="Estimated annual growth (Default: 5-year historical average)"
        )
        
        projection_years = st.number_input(
            "Projection Period (Years)",
            min_value=1,
            max_value=10,
            value=st.session_state.get('tp_years', 5),
            step=1,
            help="Years into the future (Default: 5 years)"
        )
    
    with col3:
        st.markdown("**Valuation Multiples**")
        current_pe = st.number_input(
            "Current P/E",
            min_value=0.1,
            value=st.session_state.get('tp_current_pe', 16.0),
            step=0.5,
            help="The stock's current valuation multiple"
        )
        
        target_pe = st.number_input(
            "Target P/E",
            min_value=0.1,
            value=st.session_state.get('tp_target_pe', 42.0),
            step=0.5,
            help="The industry average or exit multiple"
        )
    
    # Growth Catalysts (R3 requirement)
    st.markdown("---")
    st.subheader("📝 Growth Catalysts Checklist")
    st.caption("List the reasons why the projected CAGR is realistic (e.g., Capex, Market Share, New Products)")
    
    growth_catalysts = st.text_area(
        "Growth Catalysts",
        value=st.session_state.get('tp_catalysts', ""),
        height=100,
        placeholder="Example:\n• New capacity expansion (2x by FY26)\n• Market share gain in export markets\n• New product launches in Q3",
        label_visibility="collapsed"
    )
    
    # Calculate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        calculate_btn = st.button("🎯 Calculate Target Prices", type="primary", width='stretch')
    
    # Store inputs in session state
    if calculate_btn:
        st.session_state['tp_sales'] = current_sales
        st.session_state['tp_npm'] = historical_npm
        st.session_state['tp_shares'] = outstanding_shares
        st.session_state['tp_cagr'] = projected_cagr
        st.session_state['tp_years'] = projection_years
        st.session_state['tp_current_pe'] = current_pe
        st.session_state['tp_target_pe'] = target_pe
        st.session_state['tp_catalysts'] = growth_catalysts
        
        # Parse catalysts into list
        catalysts_list = [c.strip().lstrip('•-').strip() 
                        for c in growth_catalysts.split('\n') 
                        if c.strip()]
        
        # Create inputs object
        inputs = ProjectionInputs(
            current_sales=current_sales,
            projected_cagr=projected_cagr,
            projection_years=projection_years,
            historical_npm=historical_npm,
            outstanding_shares=outstanding_shares,
            current_pe=current_pe,
            target_pe=target_pe,
            growth_catalysts=catalysts_list
        )
        
        # Calculate projection
        result = calculate_projection(inputs)
        st.session_state['tp_result'] = result
    
    # Display results
    if 'tp_result' in st.session_state and st.session_state['tp_result']:
        result = st.session_state['tp_result']
        
        if not result.is_valid:
            st.error(f"❌ Validation Error: {result.error_message}")
            return
        
        st.markdown("---")
        st.subheader("📈 Projection Results")
        
        # Final year metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                f"Year {result.inputs.projection_years} Sales",
                f"₹{result.final_sales:,.2f} Cr"
            )
        
        with col2:
            st.metric(
                f"Year {result.inputs.projection_years} Profit",
                f"₹{result.final_profit:,.2f} Cr"
            )
        
        with col3:
            st.metric(
                f"Year {result.inputs.projection_years} EPS",
                f"₹{result.final_eps:,.2f}"
            )
        
        with col4:
            growth_multiple = result.final_sales / result.inputs.current_sales if result.inputs.current_sales > 0 else 0
            st.metric(
                "Growth Multiple",
                f"{growth_multiple:.2f}x"
            )
        
        # Target prices
        st.markdown("### 🎯 Target Prices")
        
        # Get current price for CAGR calculation
        current_price = st.session_state.get('tp_current_price', 0)
        years = result.inputs.projection_years
        
        # Calculate yearly returns (CAGR) if current price is available
        if current_price > 0:
            conservative_cagr = (((result.conservative_target / current_price) ** (1 / years)) - 1) * 100
            optimistic_cagr = (((result.optimistic_target / current_price) ** (1 / years)) - 1) * 100
            conservative_total_return = ((result.conservative_target - current_price) / current_price) * 100
            optimistic_total_return = ((result.optimistic_target - current_price) / current_price) * 100
        else:
            conservative_cagr = None
            optimistic_cagr = None
            conservative_total_return = None
            optimistic_total_return = None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); 
                        padding: 20px; border-radius: 12px; text-align: center; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">📉 Conservative Target</div>
                <div style="font-size: 11px; opacity: 0.7; margin-bottom: 8px;">P/E: {pe}</div>
                <div style="font-size: 36px; font-weight: bold;">₹{target:,.0f}</div>
            </div>
            """.format(pe=result.inputs.current_pe, target=result.conservative_target), unsafe_allow_html=True)
            
            if conservative_cagr:
                st.caption(f"📈 **{conservative_cagr:.1f}% p.a.** ({conservative_total_return:+.0f}% total over {years} yrs)")
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #0d47a1 0%, #1565c0 100%); 
                        padding: 20px; border-radius: 12px; text-align: center; color: white;">
                <div style="font-size: 14px; opacity: 0.9;">📈 Optimistic Target</div>
                <div style="font-size: 11px; opacity: 0.7; margin-bottom: 8px;">P/E: {pe}</div>
                <div style="font-size: 36px; font-weight: bold;">₹{target:,.0f}</div>
            </div>
            """.format(pe=result.inputs.target_pe, target=result.optimistic_target), unsafe_allow_html=True)
            
            if optimistic_cagr:
                st.caption(f"📈 **{optimistic_cagr:.1f}% p.a.** ({optimistic_total_return:+.0f}% total over {years} yrs)")
        
        # Price Trajectory Line Chart
        st.markdown("### 📈 Price Trajectory")
        
        from datetime import datetime
        current_year = datetime.now().year
        
        # Calculate year-by-year EPS and prices for both scenarios
        trajectory_years = list(range(current_year, current_year + years + 2))  # +2 to include endpoint
        conservative_prices = []
        optimistic_prices = []
        
        # Year 0 (current) - use current price if available, else calculate from current EPS
        if current_price > 0:
            conservative_prices.append(current_price)
            optimistic_prices.append(current_price)
        else:
            # Fallback: estimate current price from current EPS and current P/E
            current_eps_estimate = (result.inputs.current_sales * result.inputs.historical_npm / 100) / result.inputs.outstanding_shares
            conservative_prices.append(current_eps_estimate * result.inputs.current_pe)
            optimistic_prices.append(current_eps_estimate * result.inputs.target_pe)
        
        # Calculate prices for each year based on projected EPS growth
        for year_idx in range(1, years + 2):
            if year_idx <= len(result.yearly_profit):
                # Use actual calculated profit/EPS
                year_eps = result.yearly_profit[year_idx - 1] / result.inputs.outstanding_shares
            else:
                # Extend with same growth rate
                year_eps = result.final_eps * ((1 + result.inputs.projected_cagr / 100) ** (year_idx - years))
            
            conservative_prices.append(year_eps * result.inputs.current_pe)
            optimistic_prices.append(year_eps * result.inputs.target_pe)
        
        # FD comparison input and chart type toggle
        col_fd1, col_fd2, col_fd3 = st.columns([2, 2, 1])
        with col_fd1:
            fd_rate = st.number_input(
                "🏦 Bank FD Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=st.session_state.get('tp_fd_rate', 7.0),
                step=0.25,
                help="Enter current FD interest rate for comparison"
            )
            st.session_state['tp_fd_rate'] = fd_rate
        
        with col_fd2:
            chart_type = st.toggle("📊 Bar Chart", value=st.session_state.get('tp_chart_bar', False), 
                                   help="Toggle between Line and Bar chart")
            st.session_state['tp_chart_bar'] = chart_type
        
        # Calculate FD returns (compound interest)
        fd_prices = []
        starting_price = current_price if current_price > 0 else conservative_prices[0]
        for year_idx in range(len(trajectory_years)):
            fd_value = starting_price * ((1 + fd_rate / 100) ** year_idx)
            fd_prices.append(fd_value)
        
        fig_trajectory = go.Figure()
        
        if chart_type:
            # Bar Chart
            fig_trajectory.add_trace(go.Bar(
                x=[str(y) for y in trajectory_years],
                y=conservative_prices,
                name=f'Conservative (P/E: {result.inputs.current_pe})',
                marker_color='#4CAF50',
                text=[f'₹{p:,.0f}' for p in conservative_prices],
                textposition='outside'
            ))
            
            fig_trajectory.add_trace(go.Bar(
                x=[str(y) for y in trajectory_years],
                y=optimistic_prices,
                name=f'Optimistic (P/E: {result.inputs.target_pe})',
                marker_color='#2196F3',
                text=[f'₹{p:,.0f}' for p in optimistic_prices],
                textposition='outside'
            ))
            
            fig_trajectory.add_trace(go.Bar(
                x=[str(y) for y in trajectory_years],
                y=fd_prices,
                name=f'Bank FD ({fd_rate}%)',
                marker_color='#FF9800',
                text=[f'₹{p:,.0f}' for p in fd_prices],
                textposition='outside'
            ))
            
            fig_trajectory.update_layout(barmode='group')
        else:
            # Line Chart
            fig_trajectory.add_trace(go.Scatter(
                x=trajectory_years,
                y=conservative_prices,
                mode='lines+markers',
                name=f'Conservative (P/E: {result.inputs.current_pe})',
                line=dict(color='#4CAF50', width=3),
                marker=dict(size=8)
            ))
            
            fig_trajectory.add_trace(go.Scatter(
                x=trajectory_years,
                y=optimistic_prices,
                mode='lines+markers',
                name=f'Optimistic (P/E: {result.inputs.target_pe})',
                line=dict(color='#2196F3', width=3),
                marker=dict(size=8)
            ))
            
            fig_trajectory.add_trace(go.Scatter(
                x=trajectory_years,
                y=fd_prices,
                mode='lines+markers',
                name=f'Bank FD ({fd_rate}%)',
                line=dict(color='#FF9800', width=2, dash='dot'),
                marker=dict(size=6, symbol='diamond')
            ))
            
            # Add current price reference line if available
            if current_price > 0:
                fig_trajectory.add_hline(
                    y=current_price, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text=f"Current: ₹{current_price:,.0f}",
                    annotation_position="right"
                )
        
        fig_trajectory.update_layout(
            title=f"Projected Stock Price vs FD Returns ({current_year} - {current_year + years + 1})",
            xaxis_title="Year",
            yaxis_title="Value (₹)",
            template="plotly_dark",
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trajectory, width='stretch')
        
        # Sensitivity Analysis (R2 requirement)
        st.markdown("### 📊 Sensitivity Analysis (Margin of Safety)")
        
        show_sensitivity = st.checkbox("Show CAGR ± 5% Sensitivity", value=True)
        
        if show_sensitivity:
            sensitivity_data = {
                "Scenario": ["🔽 Conservative", "📊 Base Case", "🔼 Optimistic"],
                "CAGR": [f"{result.sensitivity_low_cagr:.0f}%", f"{result.inputs.projected_cagr:.0f}%", f"{result.sensitivity_high_cagr:.0f}%"],
                "Conservative Target": [f"₹{result.sensitivity_low_conservative:,.0f}", f"₹{result.conservative_target:,.0f}", f"₹{result.sensitivity_high_conservative:,.0f}"],
                "Optimistic Target": [f"₹{result.sensitivity_low_optimistic:,.0f}", f"₹{result.optimistic_target:,.0f}", f"₹{result.sensitivity_high_optimistic:,.0f}"]
            }
            
            st.table(sensitivity_data)
        
        # Bar Chart Visualization (R4 requirement)
        st.markdown("### 📊 5-Year Sales vs Net Profit Projection")
        
        years = [f"Year {i}" for i in range(1, len(result.yearly_sales) + 1)]
        
        fig = go.Figure()
        
        # Sales bars
        fig.add_trace(go.Bar(
            name='Sales (₹ Cr)',
            x=years,
            y=result.yearly_sales,
            marker_color='#4CAF50',
            text=[f"₹{s:,.0f}" for s in result.yearly_sales],
            textposition='outside'
        ))
        
        # Net Profit bars
        fig.add_trace(go.Bar(
            name='Net Profit (₹ Cr)',
            x=years,
            y=result.yearly_profit,
            marker_color='#2196F3',
            text=[f"₹{p:,.0f}" for p in result.yearly_profit],
            textposition='outside'
        ))
        
        fig.update_layout(
            barmode='group',
            title=f"Projected Sales & Profit at {result.inputs.projected_cagr}% CAGR",
            xaxis_title="Year",
            yaxis_title="Amount (₹ Crores)",
            template="plotly_dark",
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Growth Catalysts Display
        if result.inputs.growth_catalysts:
            st.markdown("### ✅ Growth Catalysts Justification")
            for i, catalyst in enumerate(result.inputs.growth_catalysts, 1):
                st.markdown(f"{i}. {catalyst}")
        
        # Calculation breakdown
        with st.expander("📐 View Calculation Details"):
            st.markdown(f"""
            **Step 1: Revenue Projection**
            ```
            Year {result.inputs.projection_years} Sales = Current Sales × (1 + CAGR)^n
            Year {result.inputs.projection_years} Sales = ₹{result.inputs.current_sales} Cr × (1 + {result.inputs.projected_cagr}%)^{result.inputs.projection_years}
            Year {result.inputs.projection_years} Sales = ₹{result.final_sales:,.2f} Cr
            ```
            
            **Step 2: Net Profit & EPS**
            ```
            Net Profit = Year {result.inputs.projection_years} Sales × NPM
            Net Profit = ₹{result.final_sales:,.2f} Cr × {result.inputs.historical_npm}%
            Net Profit = ₹{result.final_profit:,.2f} Cr
            
            EPS = Net Profit / Outstanding Shares
            EPS = ₹{result.final_profit:,.2f} Cr / {result.inputs.outstanding_shares} Cr shares
            EPS = ₹{result.final_eps:,.2f}
            ```
            
            **Step 3: Target Prices**
            ```
            Conservative Target = EPS × Current P/E
            Conservative Target = ₹{result.final_eps:.2f} × {result.inputs.current_pe}
            Conservative Target = ₹{result.conservative_target:,.0f}
            
            Optimistic Target = EPS × Target P/E
            Optimistic Target = ₹{result.final_eps:.2f} × {result.inputs.target_pe}
            Optimistic Target = ₹{result.optimistic_target:,.0f}
            ```
            """)


if __name__ == "__main__":
    main()
