from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import html as html_mod
import re
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
        # Created after the migration so an older database gains the column
        # before anything references it in an index.
        connection.execute(
            "CREATE INDEX IF NOT EXISTS conversations_recent "
            "ON conversations(pinned DESC, updated_at DESC)"
        )


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

# Kept out of SETTINGS_COLUMNS: pinning is a per-conversation flag, not a
# generation setting, so it must not be touched by save_settings/reset_settings.
_PINNED_COLUMN = "pinned INTEGER NOT NULL DEFAULT 0"


def _migrate(connection):
    """Add columns that older databases do not have yet."""
    existing = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(conversations)")
    }
    for name, definition in {
        **_SETTINGS_COLUMNS,
        "pinned": _PINNED_COLUMN,
    }.items():
        if name not in existing:
            connection.execute(
                f"ALTER TABLE conversations ADD COLUMN {name} {definition}"
            )


def create_conversation(settings=None, title=None):
    conversation_id = uuid4().hex
    with _connection() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conversation_id, " ".join((title or "New conversation").split())[:64]),
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
                   COALESCE(c.pinned, 0)      AS pinned,
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
            ORDER BY COALESCE(c.pinned, 0) DESC, c.updated_at DESC, c.id DESC
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


def get_conversation(conversation_id):
    """One conversation's row, or None when the id is unknown."""
    if not conversation_id:
        return None

    with _connection() as connection:
        row = connection.execute(
            "SELECT id, title, created_at, updated_at, "
            "       COALESCE(pinned, 0) AS pinned "
            "FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else None


def set_pinned(conversation_id, pinned=True):
    """Pin a conversation so it stays at the top of the sidebar.

    Toggles when `pinned` is None, which is how the sidebar's pin button
    behaves: one control, two states, no separate unpin path.
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return None
    next_value = not conversation["pinned"] if pinned is None else bool(pinned)
    with _connection() as connection:
        connection.execute(
            "UPDATE conversations SET pinned = ? WHERE id = ?",
            (int(next_value), conversation_id),
        )
    return next_value


def duplicate_conversation(conversation_id):
    """Copy a thread, its settings and every message into a new one.

    Used to branch off an experiment: the copy keeps the exact prompts, images
    and generation settings so the two threads can diverge from one shared
    starting point without touching each other's history.
    """
    source = get_conversation(conversation_id)
    if source is None:
        return None

    new_id = create_conversation(
        get_settings(conversation_id),
        title=f"{source['title']} (copy)",
    )

    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content, image,
                                  image_mime, created_at)
            SELECT ?, role, content, image, image_mime, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id
            """,
            (new_id, conversation_id),
        )
        connection.execute(
            "UPDATE conversations "
            "SET updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now') "
            "WHERE id = ?",
            (new_id,),
        )
    return new_id


EXPORT_ROOT = Path(__file__).resolve().parents[1] / "results" / "exports"


def export_conversation(conversation_id):
    """Write a conversation to results/exports as Markdown plus PNGs.

    Images are pulled straight from the BLOB column, so an export is a
    self-contained folder that survives deleting the thread. Returns the path
    to the Markdown file.
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        return None

    messages = get_messages(conversation_id)
    settings = get_settings(conversation_id)

    safe_name = "".join(
        character if character.isalnum() or character in " -_" else "-"
        for character in conversation["title"]
    ).strip() or "conversation"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = EXPORT_ROOT / f"{safe_name}-{stamp}"
    folder.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# {conversation['title']}",
        "",
        f"- Category: {settings['image_type']}",
        f"- Clinical enhancement: {'on' if settings['clinical'] else 'off'}",
        f"- Steps {settings['steps']} / CFG {settings['guidance']:g}",
        f"- Seed {settings['seed'] if settings['seed'] >= 0 else 'random'}",
        f"- Created {conversation['created_at']}",
        f"- Exported {stamp}",
        "",
        "> Synthetic output. Not for diagnostic use.",
        "",
    ]

    image_number = 0
    for message in messages:
        lines.append("## You" if message["role"] == "user" else "## Medsynth")
        lines.append("")
        caption = _export_text(message["content"])
        if caption:
            lines.extend([caption, ""])
        if message["image"]:
            image_number += 1
            name = f"image-{image_number:02d}.png"
            (folder / name).write_bytes(message["image"])
            lines.extend([f"![{name}]({name})", ""])

    export_path = folder / f"{safe_name}.md"
    export_path.write_text("\n".join(lines), encoding="utf-8")
    return export_path


def _export_text(content):
    """Flatten the stored caption markup back into plain Markdown."""
    text = re.sub(r"<br\s*/?>", "\n", str(content or ""))
    text = re.sub(r"<[^>]+>", "", text)
    return html_mod.unescape(text).replace(" · ", ", ").strip()


def prune_empty_conversations():
    """Drop untouched empty threads, keeping the oldest one as a fallback.

    A user who clicks "New conversation" a few times should not be left with a
    sidebar full of empty duplicates; the most recent one always survives so
    there is always somewhere to type.
    """
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT c.id
            FROM conversations AS c
            LEFT JOIN messages AS m ON m.conversation_id = c.id
            WHERE m.id IS NULL AND COALESCE(c.pinned, 0) = 0
            ORDER BY c.updated_at DESC, c.id DESC
            """
        ).fetchall()

        survivors = [row["id"] for row in rows[:1]]
        stale = [row["id"] for row in rows[1:]]
        for conversation_id in stale:
            connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
    return stale


initialize_database()