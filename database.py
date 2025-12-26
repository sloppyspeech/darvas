"""
Darvas Box Momentum Analyzer - Database Module
===============================================
SQLite database for storing study results with study_id and date.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


DB_PATH = Path(__file__).parent / "darvas_studies.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Studies table - stores each analysis session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS studies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id TEXT UNIQUE NOT NULL,
            study_date TIMESTAMP NOT NULL,
            description TEXT,
            stock_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Study results table - stores individual stock signals
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS study_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            status TEXT,
            current_price REAL,
            box_top REAL,
            box_bottom REAL,
            entry_price REAL,
            stop_loss REAL,
            target_2r REAL,
            risk_percent REAL,
            risk_reward TEXT,
            volume_confirmed BOOLEAN,
            boxes_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (study_id) REFERENCES studies(study_id)
        )
    """)
    
    # Screener runs table - stores each screening session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screener_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            run_date TIMESTAMP NOT NULL,
            total_screened INTEGER,
            candidates_found INTEGER,
            high_priority INTEGER,
            medium_priority INTEGER,
            low_priority INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Screener results table - stores individual stock screening results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screener_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            current_price REAL,
            high_52w REAL,
            low_52w REAL,
            proximity_pct REAL,
            strength_ratio REAL,
            above_sma BOOLEAN,
            volume_ratio REAL,
            volume_spike BOOLEAN,
            atr_pct REAL,
            consolidation_range REAL,
            days_in_consolidation INTEGER,
            gates_passed INTEGER,
            priority TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES screener_runs(run_id)
        )
    """)
    
    conn.commit()
    conn.close()


def generate_study_id() -> str:
    """Generate a unique study ID based on timestamp."""
    return f"STUDY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_study(study_id: str, description: str, results: List[Dict], 
               boxes_data: Dict[str, List] = None) -> bool:
    """
    Save study results to database.
    
    Args:
        study_id: Unique study identifier
        description: Study description
        results: List of signal dictionaries
        boxes_data: Dictionary mapping symbol to list of box dicts
    
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Insert study record
        cursor.execute("""
            INSERT INTO studies (study_id, study_date, description, stock_count)
            VALUES (?, ?, ?, ?)
        """, (study_id, datetime.now(), description, len(results)))
        
        # Insert result records
        for result in results:
            symbol = result.get('symbol', '')
            boxes_json = None
            
            if boxes_data and symbol in boxes_data:
                boxes_json = json.dumps(boxes_data[symbol])
            
            cursor.execute("""
                INSERT INTO study_results 
                (study_id, symbol, status, current_price, box_top, box_bottom,
                 entry_price, stop_loss, target_2r, risk_percent, risk_reward,
                 volume_confirmed, boxes_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                study_id,
                symbol,
                result.get('status'),
                result.get('current_price'),
                result.get('box_top'),
                result.get('box_bottom'),
                result.get('entry_price'),
                result.get('stop_loss'),
                result.get('target_2r'),
                result.get('risk_percent'),
                result.get('risk_reward'),
                result.get('volume_confirmed', False),
                boxes_json
            ))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error saving study: {e}")
        return False


def get_all_studies() -> List[Dict]:
    """Get list of all studies."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT study_id, study_date, description, stock_count, created_at
        FROM studies
        ORDER BY study_date DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_study_results(study_id: str) -> List[Dict]:
    """Get results for a specific study."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM study_results
        WHERE study_id = ?
        ORDER BY symbol
    """, (study_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        result = dict(row)
        # Parse boxes JSON if present
        if result.get('boxes_json'):
            result['boxes'] = json.loads(result['boxes_json'])
        results.append(result)
    
    return results


def get_study_details(study_id: str) -> Optional[Dict]:
    """Get study metadata."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM studies WHERE study_id = ?
    """, (study_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None


def delete_study(study_id: str) -> bool:
    """Delete a study and its results."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM study_results WHERE study_id = ?", (study_id,))
        cursor.execute("DELETE FROM studies WHERE study_id = ?", (study_id,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting study: {e}")
        return False


def get_symbol_history(symbol: str) -> List[Dict]:
    """Get historical analysis results for a specific symbol across all studies."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT sr.*, s.study_date, s.description
        FROM study_results sr
        JOIN studies s ON sr.study_id = s.study_id
        WHERE sr.symbol = ?
        ORDER BY s.study_date DESC
    """, (symbol,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


# ============================================
# SCREENER FUNCTIONS
# ============================================

def generate_screener_run_id() -> str:
    """Generate a unique screener run ID based on timestamp."""
    return f"SCREEN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_screener_run(run_id: str, results: List[Dict], stats: Dict) -> bool:
    """
    Save screener run and results to database.
    
    Args:
        run_id: Unique run identifier
        results: List of ScreenerResult dictionaries
        stats: Statistics dictionary from screener
    
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Insert screener run record
        cursor.execute("""
            INSERT INTO screener_runs 
            (run_id, run_date, total_screened, candidates_found, 
             high_priority, medium_priority, low_priority)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            datetime.now(),
            stats.get('processed', 0),
            stats.get('candidates', 0),
            stats.get('high_priority', 0),
            stats.get('medium_priority', 0),
            stats.get('low_priority', 0)
        ))
        
        # Insert result records
        for result in results:
            cursor.execute("""
                INSERT INTO screener_results 
                (run_id, symbol, current_price, high_52w, low_52w,
                 proximity_pct, strength_ratio, above_sma, volume_ratio,
                 volume_spike, atr_pct, consolidation_range, 
                 days_in_consolidation, gates_passed, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                result.get('symbol'),
                result.get('current_price'),
                result.get('high_52w'),
                result.get('low_52w'),
                result.get('proximity_pct'),
                result.get('strength_ratio'),
                result.get('above_sma'),
                result.get('volume_ratio'),
                result.get('volume_spike'),
                result.get('atr_pct'),
                result.get('consolidation_range'),
                result.get('days_in_consolidation'),
                result.get('gates_passed'),
                result.get('priority')
            ))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error saving screener run: {e}")
        return False


def get_all_screener_runs() -> List[Dict]:
    """Get list of all screener runs."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT run_id, run_date, total_screened, candidates_found,
               high_priority, medium_priority, low_priority, created_at
        FROM screener_runs
        ORDER BY run_date DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_screener_results(run_id: str, priority_filter: str = None) -> List[Dict]:
    """Get results for a specific screener run."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if priority_filter:
        cursor.execute("""
            SELECT * FROM screener_results
            WHERE run_id = ? AND priority = ?
            ORDER BY proximity_pct ASC
        """, (run_id, priority_filter))
    else:
        cursor.execute("""
            SELECT * FROM screener_results
            WHERE run_id = ?
            ORDER BY 
                CASE priority 
                    WHEN 'High' THEN 1 
                    WHEN 'Medium' THEN 2 
                    WHEN 'Low' THEN 3 
                    ELSE 4 
                END,
                proximity_pct ASC
        """, (run_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def delete_screener_run(run_id: str) -> bool:
    """Delete a screener run and its results."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM screener_results WHERE run_id = ?", (run_id,))
        cursor.execute("DELETE FROM screener_runs WHERE run_id = ?", (run_id,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting screener run: {e}")
        return False


# Initialize database on module import
init_db()
