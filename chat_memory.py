from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
import sqlite3
import json
from typing import List
from pydantic.v1 import BaseModel

class SQLiteChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect("chat_history.db") as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history
                (session_id TEXT PRIMARY KEY, messages TEXT)
            """)
    
    @property
    def messages(self) -> List[BaseMessage]:
        with sqlite3.connect("chat_history.db") as conn:
            cursor = conn.execute(
                "SELECT messages FROM chat_history WHERE session_id = ?",
                (self.session_id,)
            )
            row = cursor.fetchone()
            if row:
                messages_dict = json.loads(row[0])
                return messages_from_dict(messages_dict)
        return []
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        current_messages = self.messages
        current_messages.extend(messages)
        messages_dict = messages_to_dict(current_messages)
        
        with sqlite3.connect("chat_history.db") as conn:
            conn.execute(
                """INSERT OR REPLACE INTO chat_history (session_id, messages)
                   VALUES (?, ?)""",
                (self.session_id, json.dumps(messages_dict))
            )
    
    def clear(self) -> None:
        with sqlite3.connect("chat_history.db") as conn:
            conn.execute(
                "DELETE FROM chat_history WHERE session_id = ?",
                (self.session_id,)
            )

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLiteChatMessageHistory(session_id=session_id) 