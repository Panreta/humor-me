"""Client-to-server websocket messages, validated before anything else sees them (backend/README.md lists both ways)."""
import json

from app.log import KEY_LEVELS
from sim.env import AGES, ENERGIES, VENUES


class ProtocolError(ValueError):
    pass


def _int(msg: dict, field: str) -> int:
    value = msg.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProtocolError(f"{field} must be a non-negative integer")
    return value


def parse(frame: dict) -> dict:
    """The ASGI websocket frame in, a checked message out. Raises ProtocolError; the connection stays open."""
    text = frame.get("text")
    if text is None:
        raise ProtocolError("expected a text frame")
    try:
        msg = json.loads(text)
    except ValueError:
        raise ProtocolError("not valid JSON") from None
    if not isinstance(msg, dict):
        raise ProtocolError("message must be a JSON object")
    kind = msg.get("type")
    if kind == "played":
        return {"type": kind}
    if kind == "reaction":
        key = msg.get("key")
        if not isinstance(key, str) or key not in KEY_LEVELS:
            raise ProtocolError(f"key must be one of {tuple(KEY_LEVELS)}")
        return {"type": kind, "bit": _int(msg, "bit"), "joke": _int(msg, "joke"), "key": key}
    if kind == "start":
        context = msg.get("context")
        if not isinstance(context, dict):
            raise ProtocolError("context must be an object")
        for field, allowed in (("venue", VENUES), ("energy", ENERGIES), ("age", AGES)):
            if context.get(field) not in allowed:
                raise ProtocolError(f"context.{field} must be one of {allowed}")
        if msg.get("room") not in ("sim", "live"):
            raise ProtocolError("room must be 'sim' or 'live'")
        operator = msg.get("operator")
        if not isinstance(operator, str) or not operator.strip():
            raise ProtocolError("operator must be a non-empty string")
        return {"type": kind, "context": {f: context[f] for f in ("venue", "energy", "age")}, "room": msg["room"],
                "operator": operator}
    raise ProtocolError(f"unknown message type {kind!r}")
