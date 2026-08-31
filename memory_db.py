from datetime import datetime, timedelta
import sqlite3

DB_NAME = "chat_memory.db"


def init_db():
  """Initializes the SQLite database table for chat history."""
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
  conn.commit()
  conn.close()


def save_message(role: str, content: str):
  """Saves a user or assistant message with an exact timestamp."""
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute(
      "INSERT INTO chat_history (role, content, timestamp) VALUES (?, ?, ?)",
      (role, content, datetime.now()),
  )
  conn.commit()
  conn.close()


def clean_old_messages():
  """Deletes any chat logs older than 6 hours."""
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  purge_threshold = datetime.now() - timedelta(hours=6)
  cursor.execute(
      "DELETE FROM chat_history WHERE timestamp < ?", (purge_threshold,)
  )
  conn.commit()
  conn.close()


def get_recent_conversation_history():
  """Cleans old data and returns messages from the last 6 hours."""
  clean_old_messages()  # Automatically purge old records first

  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  time_threshold = datetime.now() - timedelta(hours=6)

  cursor.execute(
      "SELECT role, content FROM chat_history WHERE timestamp >= ? ORDER BY"
      " timestamp ASC",
      (time_threshold,),
  )
  rows = cursor.fetchall()
  conn.close()

  # Format into the message list structure expected by LLMs
  messages = []
  for role, content in rows:
    messages.append({"role": role, "content": content})

  return messages


# Initialize the database table immediately when this file is run/imported
if __name__ == "__main__":
  init_db()
  print("Chat memory database initialized successfully!")

def log_conversation(user_text: str, ai_text: str):
  """Saves both the user prompt and AI response in one simple step."""
  save_message("user", user_text)
  save_message("assistant", ai_text)