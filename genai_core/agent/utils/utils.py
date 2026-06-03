

import uuid


def generate_uuid7_id() -> str:
    """
    Generates a UUID version 7 (UUIDv7) string.

    UUIDv7 is a time-ordered UUID format suitable for distributed systems requiring unique and sortable identifiers.

    Returns:
        str: A string representation of a UUIDv7.
    """
    return str(uuid.uuid7())