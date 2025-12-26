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
            ["🔎 Screener", "🔍 New Analysis", "📚 Study History", "📈 Quick Analyze", "📋 Screener History"],
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
        run_btn = st.button("🚀 Run", type="primary", use_container_width=True)
    
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
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
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
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No candidates found in this screener run.")


def render_new_analysis(confirmation_days: int, volume_multiplier: float):
    """Render the new analysis page."""
    st.header("🔍 New Batch Analysis")
    
    # Stock search section
    st.markdown("### 🔎 Search & Add Stocks")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        search_query = st.text_input(
            "Search by symbol or company name",
            placeholder="e.g., RELIANCE or Tata",
            key="stock_search"
        )
    
    with col2:
        exchange_filter = st.selectbox(
            "Exchange",
            ["All", "NSE", "BSE"],
            key="exchange_filter"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh DB", help="Re-fetch stock list from NSE/BSE"):
            with st.spinner("Fetching stocks..."):
                nse, bse = refresh_stock_symbols()
                st.success(f"Updated: {nse} NSE, {bse} BSE stocks")
    
    # Show search results
    if search_query and len(search_query) >= 2:
        results = search_symbols(search_query, limit=15)
        
        if results:
            st.markdown("**Search Results** (click to add):")
            
            # Create columns for results
            cols = st.columns(3)
            for i, stock in enumerate(results):
                col_idx = i % 3
                with cols[col_idx]:
                    display_text = f"{stock['symbol']} ({stock['exchange']})"
                    if st.button(display_text, key=f"add_{stock['yf_symbol']}", use_container_width=True):
                        # Add to text area
                        current = st.session_state.get('selected_symbols', [])
                        if stock['yf_symbol'] not in current:
                            current.append(stock['yf_symbol'])
                            st.session_state['selected_symbols'] = current
                            st.rerun()
            
            st.caption(f"Showing top {len(results)} results for '{search_query}'")
        else:
            st.info(f"No stocks found matching '{search_query}'")
    
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
        if st.button("Nifty 50 Sample", use_container_width=True):
            st.session_state['selected_symbols'] = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "BHARTIARTL.NS", "SBIN.NS", "WIPRO.NS", "TATAMOTORS.NS", "AXISBANK.NS"
            ]
            st.rerun()
        
        if st.button("Clear All", use_container_width=True):
            st.session_state['selected_symbols'] = []
            st.rerun()
        
        if 'screener_candidates' in st.session_state:
            if st.button("From Screener", use_container_width=True):
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
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
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
                        use_container_width=True,
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
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
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
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_quick_analyze(confirmation_days: int, volume_multiplier: float):
    """Render the quick analyze page for single stock."""
    st.header("📈 Quick Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol",
            placeholder="e.g., RELIANCE.NS",
            value=st.session_state.get('quick_symbol', 'RELIANCE.NS'),
            key="quick_symbol_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
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
        st.plotly_chart(result['chart'], use_container_width=True, key="quick_chart")
        
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
            st.dataframe(hist_display, use_container_width=True, hide_index=True)
        else:
            st.info("No historical analysis found for this symbol.")


if __name__ == "__main__":
    main()
