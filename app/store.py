import json
import uuid

from app.crypto import decrypt, encrypt
from app.database import connect, get_config, row, rows, set_config


def seed_codex_from_auth():
    from app.codex import load_auth_tokens
    tokens = load_auth_tokens()
    if not tokens:
        return None
    with connect() as conn:
        existing = conn.execute("SELECT id FROM providers WHERE provider_type = 'codex' LIMIT 1").fetchone()
        if existing:
            return existing["id"]
    provider = create_provider("Codex (ChatGPT)", "codex", None, json.dumps(tokens))
    create_model(provider["id"], "gpt-5.5", "gpt-5.5", enabled=True, is_default=True)
    return provider["id"]


def list_providers():
    seed_codex_from_auth()
    with connect() as conn:
        return rows(conn.execute(
            "SELECT id, name, provider_type, base_url, enabled, created_at, updated_at FROM providers ORDER BY created_at"
        ).fetchall())


def get_provider(provider_id: str):
    with connect() as conn:
        return row(conn.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone())


def create_provider(name: str, provider_type: str, base_url: str | None, api_key: str):
    if provider_type == "codex":
        from app.codex import normalize_tokens
        api_key = json.dumps(normalize_tokens(api_key))
        base_url = None
    provider_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            "INSERT INTO providers (id, name, provider_type, base_url, api_key_encrypted) VALUES (?, ?, ?, ?, ?)",
            (provider_id, name, provider_type, base_url, encrypt(api_key)),
        )
    return get_provider_public(provider_id)


def get_provider_public(provider_id: str):
    with connect() as conn:
        return row(conn.execute(
            "SELECT id, name, provider_type, base_url, enabled, created_at, updated_at FROM providers WHERE id = ?",
            (provider_id,),
        ).fetchone())


def update_provider(provider_id: str, **kwargs):
    if kwargs.get("provider_type") == "codex":
        kwargs["base_url"] = None
    if "api_key" in kwargs:
        provider_type = kwargs.get("provider_type") or (get_provider(provider_id) or {}).get("provider_type")
        if provider_type == "codex":
            from app.codex import normalize_tokens
            kwargs["api_key"] = json.dumps(normalize_tokens(kwargs["api_key"]))
        kwargs["api_key_encrypted"] = encrypt(kwargs.pop("api_key"))
    if not kwargs:
        return get_provider_public(provider_id)
    fields = [f"{k} = ?" for k in kwargs] + ["updated_at = CURRENT_TIMESTAMP"]
    values = list(kwargs.values()) + [provider_id]
    with connect() as conn:
        conn.execute(f"UPDATE providers SET {', '.join(fields)} WHERE id = ?", values)
    return get_provider_public(provider_id)


def delete_provider(provider_id: str):
    with connect() as conn:
        cur = conn.execute("DELETE FROM providers WHERE id = ?", (provider_id,))
        return cur.rowcount > 0


def list_models(provider_id: str | None = None):
    sql = """
        SELECT m.*, p.name AS provider_name, p.provider_type
        FROM models m JOIN providers p ON m.provider_id = p.id
    """
    args = []
    if provider_id:
        sql += " WHERE m.provider_id = ?"
        args.append(provider_id)
    sql += " ORDER BY p.name, m.display_name"
    with connect() as conn:
        return rows(conn.execute(sql, args).fetchall())


def get_model(id: str):
    with connect() as conn:
        return row(conn.execute(
            """
            SELECT m.*, p.name AS provider_name, p.provider_type
            FROM models m JOIN providers p ON m.provider_id = p.id
            WHERE m.id = ?
            """,
            (id,),
        ).fetchone())


def create_model(provider_id: str, display_name: str, model_id: str, enabled: bool = True, is_default: bool = False):
    id = str(uuid.uuid4())
    with connect() as conn:
        if is_default:
            conn.execute("UPDATE models SET is_default = 0")
        conn.execute(
            "INSERT INTO models (id, provider_id, display_name, model_id, enabled, is_default) VALUES (?, ?, ?, ?, ?, ?)",
            (id, provider_id, display_name, model_id, int(enabled), int(is_default)),
        )
    return get_model(id)


def update_model(id: str, **kwargs):
    if kwargs.get("is_default"):
        with connect() as conn:
            conn.execute("UPDATE models SET is_default = 0")
    if not kwargs:
        return get_model(id)
    fields = [f"{k} = ?" for k in kwargs] + ["updated_at = CURRENT_TIMESTAMP"]
    values = [int(v) if isinstance(v, bool) else v for v in kwargs.values()] + [id]
    with connect() as conn:
        conn.execute(f"UPDATE models SET {', '.join(fields)} WHERE id = ?", values)
    return get_model(id)


def delete_model(id: str):
    with connect() as conn:
        cur = conn.execute("DELETE FROM models WHERE id = ?", (id,))
        return cur.rowcount > 0


def get_model_with_key(id: str):
    with connect() as conn:
        item = row(conn.execute(
            """
            SELECT m.model_id, p.provider_type, p.base_url, p.api_key_encrypted, p.id AS provider_id
            FROM models m JOIN providers p ON m.provider_id = p.id
            WHERE m.id = ? AND m.enabled = 1 AND p.enabled = 1
            """,
            (id,),
        ).fetchone())
    if not item:
        return None
    item["api_key"] = decrypt(item.pop("api_key_encrypted"))
    return item


def get_chain():
    return get_config("llm_chain", {"chain": []})["chain"]


def set_chain(chain: list[dict]):
    set_config("llm_chain", {"chain": chain})
