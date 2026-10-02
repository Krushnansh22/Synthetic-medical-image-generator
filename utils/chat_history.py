from contextlib import contextmanager
from pathlib import Path
import sqlite3
from uuid import uuid4


DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "medsynth_history.sqlite3"
)


@contextmanager
def _connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New conversation',
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%d %H:%M:%f', 'now')
                ),
                updated_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%d %H:%M:%f', 'now')
                )
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                image BLOB,
                image_mime TEXT,
                created_at TEXT NOT NULL DEFAULT (
                    strftime('%Y-%m-%d %H:%M:%f', 'now')
                ),
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS messages_conversation_order
                ON messages(conversation_id, id);
            CREATE INDEX IF NOT EXISTS conversations_recent
                ON conversations(updated_at DESC);
            """
        )


def create_conversation():
    conversation_id = uuid4().hex
    with _connection() as connection:
        connection.execute(
            "INSERT INTO conversations (id) VALUES (?)",
            (conversation_id,),
        )
    return conversation_id


def list_conversations():
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_messages(conversation_id):
    if not conversation_id:
        return []

    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id, role, content, image, image_mime, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def add_message(
    conversation_id,
    role,
    content,
    image=None,
    image_mime=None,
):
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")

    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO messages (
                conversation_id, role, content, image, image_mime
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, image, image_mime),
        )

        if role == "user":
            normalized_content = " ".join(content.split())
            title = normalized_content[:64]
            if len(normalized_content) > 64:
                title = f"{title.rstrip()}..."
            connection.execute(
                """
                UPDATE conversations
                SET title = CASE
                    WHEN title = 'New conversation' THEN ?
                    ELSE title
                END,
                updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
                WHERE id = ?
                """,
                (title or "New conversation", conversation_id),
            )
        else:
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
                WHERE id = ?
                """,
                (conversation_id,),
            )


def delete_conversation(conversation_id):
    if not conversation_id:
        return False

    with _connection() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
    return cursor.rowcount > 0


initialize_database()