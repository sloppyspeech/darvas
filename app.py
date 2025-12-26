"""
Darvas Box Momentum Analyzer - Streamlit Web Application
=========================================================
Interactive web interface for Darvas Box analysis of NSE/BSE stocks.
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
    get_symbol_history
)


# Page configuration
st.set_page_config(
    page_title="Darvas Box Analyzer",
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
    .status-breakout {
        background-color: #1b5e20;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .status-inside {
        background-color: #0d47a1;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .status-nosetup {
        background-color: #424242;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/stock-market.png", width=80)
        st.title("📊 Darvas Box Analyzer")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🔍 New Analysis", "📚 Study History", "📈 Quick Analyze"],
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
    
    # Main content based on navigation
    if page == "🔍 New Analysis":
        render_new_analysis(confirmation_days, volume_multiplier)
    elif page == "📚 Study History":
        render_study_history()
    else:
        render_quick_analyze(confirmation_days, volume_multiplier)


def render_new_analysis(confirmation_days: int, volume_multiplier: float):
    """Render the new analysis page."""
    st.header("🔍 New Batch Analysis")
    
    # Stock input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stocks_input = st.text_area(
            "Enter Stock Symbols (one per line)",
            value="RELIANCE.NS\nTCS.NS\nHDFCBANK.NS\nINFY.NS\nICICIBANK.NS",
            height=150,
            help="Use .NS for NSE stocks, .BO for BSE stocks"
        )
    
    with col2:
        st.markdown("### Quick Add")
        if st.button("Nifty 50 Sample", use_container_width=True):
            st.session_state['stocks'] = """RELIANCE.NS
TCS.NS
HDFCBANK.NS
INFY.NS
ICICIBANK.NS
BHARTIARTL.NS
SBIN.NS
WIPRO.NS
TATAMOTORS.NS
AXISBANK.NS"""
            st.rerun()
    
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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stocks", len(df))
    with col2:
        st.metric("Actionable Setups", len(actionable))
    with col3:
        breakouts = len(df[df['status'].str.contains('Breakout', na=False)])
        st.metric("Breakouts", breakouts)
    with col4:
        inside_box = len(df[df['status'].str.contains('Inside Box', na=False)])
        st.metric("Inside Box", inside_box)
    
    # Actionable setups highlight
    if len(actionable) > 0:
        st.markdown("### 🎯 Actionable Setups")
        for _, row in actionable.iterrows():
            emoji = '🟢' if 'Breakout' in str(row['status']) else '🔵'
            with st.expander(f"{emoji} {row['symbol']} - {row['status']}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"₹{row['current_price']}")
                col2.metric("Entry Price", f"₹{row['entry_price']}")
                col3.metric("Stop Loss", f"₹{row['stop_loss']}")
                col4.metric("Risk", f"{row['risk_percent']}%")
                
                # Show chart if available
                if row['symbol'] in all_data and all_data[row['symbol']].get('chart'):
                    st.plotly_chart(
                        all_data[row['symbol']]['chart'], 
                        use_container_width=True,
                        key=f"chart_{row['symbol']}"
                    )
    
    # Full results table
    st.markdown("### 📋 Full Results")
    
    # Format for display
    display_df = df[['symbol', 'status', 'current_price', 'box_top', 'box_bottom', 
                     'entry_price', 'stop_loss', 'risk_percent', 'risk_reward']].copy()
    display_df.columns = ['Symbol', 'Status', 'Current Price', 'Box Top', 'Box Bottom',
                          'Entry Price', 'Stop Loss', 'Risk %', 'Risk:Reward']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
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
            value="RELIANCE.NS"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_btn and symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            result = analyze_stock(symbol.strip(), confirmation_days, volume_multiplier)
        
        if not result['success']:
            st.error(f"❌ {result.get('error', 'Failed to fetch data for')} {symbol}")
            return
        
        signal = result['signal']
        
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
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"₹{signal['current_price']}")
        col2.metric("Box Range", f"₹{signal['box_bottom']} - ₹{signal['box_top']}" if signal['box_top'] else "N/A")
        col3.metric("Entry Price", f"₹{signal['entry_price']}" if signal['entry_price'] else "N/A")
        col4.metric("Stop Loss", f"₹{signal['stop_loss']}" if signal['stop_loss'] else "N/A")
        
        if signal['entry_price']:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Target (2R)", f"₹{signal['target_2r']}")
            col2.metric("Risk", f"{signal['risk_percent']}%")
            col3.metric("Risk:Reward", signal['risk_reward'])
            col4.metric("Volume Confirmed", "✅ Yes" if signal['volume_confirmed'] else "❌ No")
        
        # Chart
        st.markdown("### 📈 Chart")
        st.plotly_chart(result['chart'], use_container_width=True)
        
        # Historical analysis
        st.markdown("### 📜 Historical Analysis")
        history = get_symbol_history(symbol.strip())
        
        if history:
            hist_df = pd.DataFrame(history)
            display_cols = ['study_date', 'status', 'current_price', 'entry_price', 'stop_loss']
            hist_display = hist_df[[c for c in display_cols if c in hist_df.columns]]
            st.dataframe(hist_display, use_container_width=True, hide_index=True)
        else:
            st.info("No historical analysis found for this symbol.")


if __name__ == "__main__":
    main()
