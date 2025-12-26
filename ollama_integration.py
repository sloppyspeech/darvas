"""
Ollama LLM Integration
======================
Connects to local Ollama instance to generate human-readable summaries
of Darvas Box analysis results.
"""

import requests
import json
from typing import Dict, Optional


# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11435"
OLLAMA_MODEL = "gpt-oss:120b-cloud"


def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_available_models() -> list:
    """Get list of available models from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
        return []
    except:
        return []


def generate_darvas_summary(signal: Dict, symbol: str) -> Optional[str]:
    """
    Send Darvas analysis to Ollama for human-readable summary.
    
    Args:
        signal: The signal dictionary from Darvas analysis
        symbol: Stock symbol being analyzed
    
    Returns:
        LLM-generated summary or None if failed
    """
    try:
        # Prepare the analysis data as context
        analysis_context = f"""
Stock Symbol: {symbol}

Darvas Box Analysis Results:
- Current Status: {signal.get('status', 'Unknown')}
- Current Price: ₹{signal.get('current_price', 'N/A')}
- Box Top: ₹{signal.get('box_top', 'N/A')}
- Box Bottom: ₹{signal.get('box_bottom', 'N/A')}
- Entry Price: ₹{signal.get('entry_price', 'N/A')}
- Stop Loss: ₹{signal.get('stop_loss', 'N/A')}
- Target Price (2R): ₹{signal.get('target_price', signal.get('target_2r', 'N/A'))}
- Risk Percentage: {signal.get('risk_percent', 'N/A')}%
- Risk:Reward Ratio: {signal.get('risk_reward', 'N/A')}
- Volume Confirmed: {'Yes' if signal.get('volume_confirmed') else 'No'}
- Suggestion: {signal.get('suggestion', 'N/A')}
- Reason: {signal.get('suggestion_reason', 'N/A')}
"""

        # Prompt for the LLM
        prompt = f"""You are a stock trading analyst expert in Darvas Box trading strategy.

Based on the following Darvas Box analysis, provide a clear, actionable summary for a trader:

{analysis_context}

Please provide:
1. A brief assessment (2-3 sentences) of whether this is a good Darvas setup
2. Clear action recommendation (BUY/WAIT/AVOID)
3. Key price levels to watch
4. Risk management advice

Keep your response concise and actionable. Use simple language."""

        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            return f"Error: Ollama returned status {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it's running on port 11435."
    except requests.exceptions.Timeout:
        return "Error: Ollama request timed out. The model may be loading."
    except Exception as e:
        return f"Error: {str(e)}"


def generate_batch_summary(results: list) -> Optional[str]:
    """
    Generate a summary for batch analysis results.
    
    Args:
        results: List of signal dictionaries
    
    Returns:
        LLM-generated summary or None if failed
    """
    try:
        # Count different statuses
        buy_signals = [r for r in results if 'BUY' in r.get('suggestion', '')]
        watch_signals = [r for r in results if 'WATCH' in r.get('suggestion', '')]
        skip_signals = [r for r in results if 'SKIP' in r.get('suggestion', '')]
        
        # Build context
        context = f"""
Batch Darvas Box Analysis Results:
- Total Stocks Analyzed: {len(results)}
- BUY Signals: {len(buy_signals)}
- WATCH Signals: {len(watch_signals)}
- SKIP/No Setup: {len(skip_signals)}

Top BUY opportunities:
"""
        for signal in buy_signals[:5]:
            context += f"- {signal.get('symbol')}: Entry ₹{signal.get('entry_price')}, Risk {signal.get('risk_percent')}%\n"
        
        if watch_signals:
            context += "\nStocks to WATCH:\n"
            for signal in watch_signals[:5]:
                context += f"- {signal.get('symbol')}: Box at ₹{signal.get('box_top')}\n"

        prompt = f"""You are a stock trading analyst expert in Darvas Box strategy.

Summarize this batch analysis in 3-4 sentences, highlighting the best opportunities:

{context}

Be concise and actionable."""

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 200
                }
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        return None
            
    except Exception as e:
        return f"Error: {str(e)}"
