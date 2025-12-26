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


# Initialize database on module import
init_db()
