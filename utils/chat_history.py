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
        _migrate(connection)


# Generation settings are stored per conversation so that switching between
# threads in the sidebar restores the exact setup that conversation was made
# with, and so a reload never silently drops a hand-tuned configuration.
SETTINGS_DEFAULTS: dict[str, object] = {
    "image_type": "Dermoscopy",
    "clinical": 1,
    "steps": 30,
    "guidance": 6.5,
    "seed": -1,
    "negative_prompt": "",
}

_SETTINGS_COLUMNS = {
    "image_type": "TEXT NOT NULL DEFAULT 'Dermoscopy'",
    "clinical": "INTEGER NOT NULL DEFAULT 1",
    "steps": "INTEGER NOT NULL DEFAULT 30",
    "guidance": "REAL NOT NULL DEFAULT 6.5",
    "seed": "INTEGER NOT NULL DEFAULT -1",
    "negative_prompt": "TEXT NOT NULL DEFAULT ''",
}


def _migrate(connection):
    """Add columns that older databases do not have yet."""
    existing = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(conversations)")
    }
    for name, definition in _SETTINGS_COLUMNS.items():
        if name not in existing:
            connection.execute(
                f"ALTER TABLE conversations ADD COLUMN {name} {definition}"
            )


def create_conversation(settings=None):
    conversation_id = uuid4().hex
    with _connection() as connection:
        connection.execute(
            "INSERT INTO conversations (id) VALUES (?)",
            (conversation_id,),
        )
    if settings:
        save_settings(conversation_id, **settings)
    return conversation_id


def list_conversations(query=""):
    """All conversations, most recently active first, with activity counts.

    The counts let the sidebar show at a glance which threads actually produced
    images without opening them. `query` narrows the list by conversation title
    or by anything said inside it, so old prompts stay findable.
    """
    needle = f"%{(query or '').strip()}%"
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COALESCE(m.message_count, 0) AS message_count,
                   COALESCE(m.image_count, 0)  AS image_count
            FROM conversations AS c
            LEFT JOIN (
                SELECT conversation_id,
                       COUNT(*)                AS message_count,
                       SUM(image IS NOT NULL)  AS image_count
                FROM messages
                GROUP BY conversation_id
            ) AS m ON m.conversation_id = c.id
            WHERE ? = '%'
               OR c.title LIKE ?
               OR EXISTS (
                    SELECT 1 FROM messages AS hit
                    WHERE hit.conversation_id = c.id
                      AND hit.content LIKE ?
               )
            ORDER BY c.updated_at DESC, c.id DESC
            """,
            (needle, needle, needle),
        ).fetchall()
    return [dict(row) for row in rows]


def get_settings(conversation_id):
    """The generation settings stored on a conversation, with defaults filled in."""
    settings = dict(SETTINGS_DEFAULTS)
    if not conversation_id:
        return settings

    with _connection() as connection:
        row = connection.execute(
            f"SELECT {', '.join(_SETTINGS_COLUMNS)} "
            "FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

    if row is None:
        return settings

    stored = dict(row)
    stored["clinical"] = bool(stored.get("clinical", 1))
    settings.update(stored)
    return settings


def save_settings(conversation_id, **fields):
    """Persist a subset of the generation settings on one conversation."""
    if not conversation_id:
        return get_settings(conversation_id)

    updates = {
        key: int(bool(value)) if key == "clinical" else value
        for key, value in fields.items()
        if key in _SETTINGS_COLUMNS
    }
    if not updates:
        return get_settings(conversation_id)

    assignments = ", ".join(f"{key} = ?" for key in updates)
    with _connection() as connection:
        connection.execute(
            f"UPDATE conversations SET {assignments} WHERE id = ?",
            (*updates.values(), conversation_id),
        )
    return get_settings(conversation_id)


def reset_settings(conversation_id):
    """Restore the shipped defaults for one conversation."""
    return save_settings(conversation_id, **SETTINGS_DEFAULTS)


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


def rename_conversation(conversation_id, title):
    """Give a conversation a human-readable name of its own."""
    title = " ".join((title or "").split())
    if not conversation_id or not title:
        return False

    with _connection() as connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
            WHERE id = ?
            """,
            (title[:64], conversation_id),
        )
    return cursor.rowcount > 0


def clear_messages(conversation_id):
    """Drop every message and image in a conversation but keep the thread.

    Used by "Clear chat": the conversation stays in the sidebar, the title
    resets so the next prompt names it again, and generation settings survive.
    """
    if not conversation_id:
        return 0

    with _connection() as connection:
        cursor = connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            "UPDATE conversations "
            "SET title = 'New conversation', "
            "    updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now') "
            "WHERE id = ?",
            (conversation_id,),
        )
    return cursor.rowcount


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