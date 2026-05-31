import datetime
import uuid
import uuid6
from datetime import datetime, timezone
import os

def generate_conversation_id(useralias, topic) -> str:
    """
    Generates a new unique conversation ID.

    Returns:
        str: A unique conversation ID.
    """
    readable_env = os.environ.get('READABLE_CONVERSATION_ID', 'no').lower()
    readable = readable_env in {'yes', 'true', '1', 'on'}
    if readable:
        conv_id = f"{useralias}{str(uuid.uuid4()).split('-')[0]}{_sanitize_message(topic.text)}"
    else:
        conv_id = str(uuid.uuid4()).replace('-', '_')
    return conv_id

def _sanitize_message(message: str) -> str:
    """Sanitize the message by removing special characters or trimming."""
    sanitized = ''.join(e if e.isalnum() else '' for e in message).strip(':')
    return sanitized[:20]

def generate_uuid7_id() -> str:
    """
    Generates a UUID version 7 (UUIDv7) string.

    UUIDv7 is a time-ordered UUID format suitable for distributed systems requiring unique and sortable identifiers.

    Returns:
        str: A string representation of a UUIDv7.
    """
    return str(uuid6.uuid7())

def uuidv7_to_datetime(uuid_str: str) -> datetime:
    """
    Convert a UUIDv7 string to a datetime (UTC).
    """
    u = uuid6.UUID(uuid_str)
    # The first 48 bits are milliseconds since Unix epoch
    unix_ms = (u.int >> 80) & ((1 << 48) - 1)
    return datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc)