import mysql.connector
from mysql.connector import Error
import os

# --- DB Configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',        # Replace with your MySQL username
    'password': 'nidhi@sql',  # Replace with your MySQL password
    'database': 'plant_disease_db'
}

# --- Create connection ---
def create_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            print("✅ Database connected successfully.")
            return connection
    except Error as e:
        print(f"\n❌ Error while connecting to MySQL: {e}")
    return None


# --- Create necessary tables ---
def create_tables():
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100) NOT NULL UNIQUE,
                    password VARCHAR(100) NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100),
                    image_path VARCHAR(255),
                    ml_result VARCHAR(100),
                    cnn_result VARCHAR(100),
                    qml_result VARCHAR(100),
                    qdl_result VARCHAR(100),
                    final_result VARCHAR(100)
                )
            """)
            connection.commit()
            print("✅ Tables created or already exist.")
        except Error as e:
            print(f"\n❌ Failed to create tables: {e}")
        finally:
            cursor.close()
            connection.close()
    else:
        print("❌ Could not create tables because database connection failed.")

# --- Insert new user ---
def insert_user(username, password):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        try:
            print(f"Inserting user: {username}")
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            connection.commit()
            print(f"✅ User {username} inserted successfully.")
        except Error as e:
            print(f"❌ Failed to insert user: {e}")
            raise
        finally:
            cursor.close()
            connection.close()
    else:
        print("❌ Failed to insert user: no database connection.")


# --- Validate user login ---
def validate_user(username, password):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
            return cursor.fetchone() is not None
        except Error as e:
            print(f"❌ Failed to validate user: {e}")
            return False
        finally:
            cursor.close()
            connection.close()
    else:
        print("❌ Failed to validate user: no database connection.")
        return False

# --- Save prediction result ---
def save_prediction(username, image_path, ml_result, cnn_result, qml_result, qdl_result, final_result):
    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        try:
            cursor.execute("""
                INSERT INTO predictions (username, image_path, ml_result, cnn_result, qml_result, qdl_result, final_result)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (username, image_path, ml_result, cnn_result, qml_result, qdl_result, final_result))
            connection.commit()
        except Error as e:
            print(f"❌ Failed to save prediction: {e}")
        finally:
            cursor.close()
            connection.close()
    else:
        print("❌ Failed to save prediction: no database connection.")
