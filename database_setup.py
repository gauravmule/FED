import pymysql
from pymysql.cursors import DictCursor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "admin@123",
    "database": "face_emotion_detection",
    "cursorclass": DictCursor,
    "autocommit": True,
    "charset": "utf8mb4"
}

def get_db_connection():
    try:
        # Verify/Create database
        temp_conn = pymysql.connect(
            host="localhost",
            user="root",
            password="admin@123",
            autocommit=True
        )
        with temp_conn.cursor() as cursor:
            cursor.execute("CREATE DATABASE IF NOT EXISTS face_emotion_detection")
        temp_conn.close()
        
        # Connect to the database
        conn = pymysql.connect(**db_config)
        logger.info("Database connection successful")
        return conn
    except pymysql.MySQLError as e:
        logger.error(f"Database connection failed: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to initialize database")
        return
    
    try:
        with conn.cursor() as cursor:
            # Drop existing tables (for development only)
            cursor.execute("DROP TABLE IF EXISTS emotion_logs")
            cursor.execute("DROP TABLE IF EXISTS sessions")
            cursor.execute("DROP TABLE IF EXISTS users")
            
            # Create tables
            cursor.execute('''
                CREATE TABLE users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(128) NOT NULL,
                    total_sessions INT DEFAULT 0,
                    total_faces_detected INT DEFAULT 0,
                    most_common_emotion VARCHAR(50)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE sessions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    total_faces INT DEFAULT 0,
                    most_common_emotion VARCHAR(50),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE emotion_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id INT,
                    emotion VARCHAR(50),
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            ''')
            
            conn.commit()
            logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()