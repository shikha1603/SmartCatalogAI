import unittest
import os
import sqlite3
import shutil
import tempfile
from typing import Generator
import database

class TestDatabase(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary file database path for isolated testing
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        # Override the DB_PATH in database module to point to our test database
        database.DB_PATH = self.db_path
        database.init_db()

    def tearDown(self) -> None:
        # Close file descriptor and remove the temporary file
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_init_db(self) -> None:
        """Tests if the database schema initialized correctly and table exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions';")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], 'predictions')
        conn.close()

    def test_log_prediction(self) -> None:
        """Tests logging a prediction inserts correct details into SQLite."""
        row_id = database.log_prediction(
            image_name="test_item.jpg",
            image_path="outputs/thumbnails/thumb_test.png",
            predicted_category="Fashion",
            confidence=0.92,
            status="Auto-Approved"
        )
        self.assertTrue(row_id > 0)
        
        # Query and verify fields
        records, total = database.get_history()
        self.assertEqual(total, 1)
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["id"], row_id)
        self.assertEqual(rec["image_name"], "test_item.jpg")
        self.assertEqual(rec["image_path"], "outputs/thumbnails/thumb_test.png")
        self.assertEqual(rec["predicted_category"], "Fashion")
        self.assertIsNone(rec["corrected_category"])
        self.assertEqual(rec["confidence"], 0.92)
        self.assertEqual(rec["status"], "Auto-Approved")

    def test_update_review(self) -> None:
        """Tests that human-in-the-loop review updates status and corrected category without overwriting predicted category."""
        row_id = database.log_prediction(
            image_name="shirt.jpg",
            image_path="outputs/thumbnails/thumb_shirt.png",
            predicted_category="Home",
            confidence=0.65,
            status="Manual Review"
        )
        
        # Verify update works
        success = database.update_review(row_id, corrected_category="Fashion")
        self.assertTrue(success)
        
        # Retrieve and inspect fields
        records, _ = database.get_history()
        rec = records[0]
        self.assertEqual(rec["predicted_category"], "Home") # Unchanged
        self.assertEqual(rec["corrected_category"], "Fashion") # Corrected
        self.assertEqual(rec["status"], "Reviewed") # Updated status

    def test_get_stats(self) -> None:
        """Tests that database dashboard summary statistics run proper mathematical aggregations."""
        # Insert 3 predictions
        database.log_prediction("img1.jpg", "path1.jpg", "Fashion", 0.90, "Auto-Approved")
        database.log_prediction("img2.jpg", "path2.jpg", "Electronics", 0.70, "Manual Review")
        row3 = database.log_prediction("img3.jpg", "path3.jpg", "Home", 0.50, "Manual Review")
        
        # Review row 3 (Home -> Beauty)
        database.update_review(row3, "Beauty")
        
        stats = database.get_stats()
        self.assertEqual(stats["total"], 3)
        self.assertAlmostEqual(stats["auto_approval_rate"], (1/3)*100)
        self.assertAlmostEqual(stats["manual_review_rate"], (2/3)*100)
        self.assertEqual(stats["reviewed_count"], 1)
        
        # Distribution: count resolved category (corrected if present, else predicted)
        # Resolved classes: Fashion (predicted), Electronics (predicted), Beauty (corrected)
        dist = stats["category_distribution"]
        self.assertEqual(dist.get("Fashion"), 1)
        self.assertEqual(dist.get("Electronics"), 1)
        self.assertEqual(dist.get("Beauty"), 1)
        self.assertEqual(dist.get("Home", 0), 0)

if __name__ == "__main__":
    unittest.main()
