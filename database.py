import sqlite3
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from config import DB_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("database")

def get_db_connection() -> sqlite3.Connection:
    """Establishes and returns a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Returns rows as dictionary-like objects
    return conn

def init_db() -> None:
    """Initializes the SQLite database and creates the predictions table if it doesn't exist."""
    logger.info("Initializing database...")
    query = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT NOT NULL,
        image_path TEXT NOT NULL,
        predicted_category TEXT NOT NULL,
        corrected_category TEXT,
        confidence REAL NOT NULL,
        status TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    conn = get_db_connection()
    try:
        conn.execute(query)
        conn.commit()
        logger.info("Database and predictions table initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        conn.close()

def log_prediction(image_name: str, image_path: str, predicted_category: str, confidence: float, status: str) -> int:
    """
    Logs a model prediction.
    
    Args:
        image_name: Name of the uploaded file.
        image_path: Path to the saved local thumbnail.
        predicted_category: Category predicted by the model.
        confidence: Confidence score of the prediction (between 0.0 and 1.0).
        status: Routing status ('Auto-Approved' or 'Manual Review').
        
    Returns:
        The ID of the newly inserted row.
    """
    query = """
    INSERT INTO predictions (image_name, image_path, predicted_category, confidence, status, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, (image_name, image_path, predicted_category, confidence, status, timestamp))
        conn.commit()
        last_id = cursor.lastrowid
        logger.info(f"Successfully logged prediction (ID: {last_id}) for {image_name}.")
        return last_id
    except sqlite3.Error as e:
        logger.error(f"Failed to log prediction for {image_name}: {e}")
        raise
    finally:
        conn.close()

def update_review(prediction_id: int, corrected_category: str) -> bool:
    """
    Updates the prediction record with the manually corrected category, changing the status to 'Reviewed'.
    
    Args:
        prediction_id: The ID of the prediction record to update.
        corrected_category: The corrected category provided by the human reviewer.
        
    Returns:
        True if the update was successful, False otherwise.
    """
    query = """
    UPDATE predictions
    SET corrected_category = ?, status = 'Reviewed'
    WHERE id = ?
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, (corrected_category, prediction_id))
        conn.commit()
        rowcount = cursor.rowcount
        if rowcount > 0:
            logger.info(f"Successfully reviewed and updated prediction ID {prediction_id} to '{corrected_category}'.")
            return True
        else:
            logger.warning(f"Prediction ID {prediction_id} not found for updating review.")
            return False
    except sqlite3.Error as e:
        logger.error(f"Failed to update review for prediction ID {prediction_id}: {e}")
        raise
    finally:
        conn.close()

def get_history(category_filter: Optional[str] = None, status_filter: Optional[str] = None, limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    """
    Retrieves logged prediction records with optional filters, pagination, and total count.
    
    Args:
        category_filter: Filter by predicted (or corrected) category.
        status_filter: Filter by routing status ('Auto-Approved', 'Manual Review', 'Reviewed').
        limit: Max records to fetch.
        offset: Record offset.
        
    Returns:
        A tuple of (list of record dicts, total record count matching filters).
    """
    base_query = "SELECT * FROM predictions"
    count_query = "SELECT COUNT(*) FROM predictions"
    conditions = []
    params = []
    
    if category_filter:
        conditions.append("(predicted_category = ? OR corrected_category = ?)")
        params.extend([category_filter, category_filter])
        
    if status_filter:
        conditions.append("status = ?")
        params.append(status_filter)
        
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)
        base_query += where_clause
        count_query += where_clause
        
    base_query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    query_params = params + [limit, offset]
    
    conn = get_db_connection()
    try:
        # Get total count
        count_cursor = conn.cursor()
        count_cursor.execute(count_query, params)
        total_count = count_cursor.fetchone()[0]
        
        # Get data rows
        cursor = conn.cursor()
        cursor.execute(base_query, query_params)
        rows = cursor.fetchall()
        
        records = [dict(row) for row in rows]
        return records, total_count
    except sqlite3.Error as e:
        logger.error(f"Failed to fetch prediction history: {e}")
        raise
    finally:
        conn.close()

def get_stats() -> Dict[str, Any]:
    """
    Calculates catalog automation dashboard statistics.
    
    Returns:
        A dictionary containing stats summary.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Total count
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0]
        
        if total == 0:
            return {
                "total": 0,
                "auto_approval_rate": 0.0,
                "manual_review_rate": 0.0,
                "reviewed_count": 0,
                "category_distribution": {}
            }
            
        # Status counts
        cursor.execute("SELECT status, COUNT(*) FROM predictions GROUP BY status")
        status_counts = dict(cursor.fetchall())
        
        auto_approved = status_counts.get("Auto-Approved", 0)
        manual_review = status_counts.get("Manual Review", 0)
        reviewed = status_counts.get("Reviewed", 0)
        
        auto_approval_rate = (auto_approved / total) * 100
        manual_review_rate = ((manual_review + reviewed) / total) * 100
        
        # Distribution based on resolved category (corrected_category if present, else predicted_category)
        cursor.execute("""
            SELECT COALESCE(corrected_category, predicted_category) as final_cat, COUNT(*)
            FROM predictions
            GROUP BY final_cat
        """)
        category_distribution = dict(cursor.fetchall())
        
        return {
            "total": total,
            "auto_approval_rate": auto_approval_rate,
            "manual_review_rate": manual_review_rate,
            "reviewed_count": reviewed,
            "category_distribution": category_distribution
        }
    except sqlite3.Error as e:
        logger.error(f"Failed to calculate stats: {e}")
        raise
    finally:
        conn.close()

def reset_db() -> None:
    """Deletes all records from the predictions table."""
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM predictions;")
        conn.commit()
        logger.info("Database reset: cleared all prediction logs.")
    except sqlite3.Error as e:
        logger.error(f"Failed to reset database: {e}")
        raise
    finally:
        conn.close()

