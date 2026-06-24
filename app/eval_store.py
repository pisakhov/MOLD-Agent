import json
import uuid
from typing import Any

from app.database import connect, row, rows


def newid():
    return str(uuid.uuid4())


def create_sample(system_prompt: str, user_message: str, source: str = "case"):
    id = newid()
    with connect() as conn:
        conn.execute(
            "INSERT INTO eval_samples (id, system_prompt, user_message, source) VALUES (?, ?, ?, ?)",
            (id, system_prompt, user_message, source),
        )
    return id


def sample_count():
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) AS count FROM eval_samples").fetchone()["count"]


def random_samples(limit: int = 5):
    with connect() as conn:
        return rows(conn.execute(
            "SELECT system_prompt, user_message FROM eval_samples ORDER BY RANDOM() LIMIT ?",
            (limit,),
        ).fetchall())


def create_batch(mold_name: str, count: int = 5):
    id = newid()
    with connect() as conn:
        conn.execute(
            "INSERT INTO eval_batches (id, mold_name, target_count, status) VALUES (?, ?, ?, 'running')",
            (id, mold_name, count),
        )
    return id


def finish_batch(batch_id: str, status: str = "done", error: str | None = None):
    with connect() as conn:
        conn.execute(
            "UPDATE eval_batches SET status = ?, error = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, error, batch_id),
        )


def fail_stale_batches(hours: int = 2):
    with connect() as conn:
        conn.execute(
            """
            UPDATE eval_batches
            SET status = 'failed', error = 'Generation stopped before finishing.', finished_at = CURRENT_TIMESTAMP
            WHERE status = 'running' AND created_at < datetime('now', ?)
            """,
            (f"-{hours} hours",),
        )


def active_batch():
    fail_stale_batches()
    with connect() as conn:
        return row(conn.execute(
            "SELECT * FROM eval_batches WHERE status = 'running' ORDER BY created_at DESC LIMIT 1"
        ).fetchone())


def get_batch(batch_id: str):
    with connect() as conn:
        return row(conn.execute("SELECT * FROM eval_batches WHERE id = ?", (batch_id,)).fetchone())


def batch_progress(batch_id: str):
    with connect() as conn:
        batch = row(conn.execute("SELECT * FROM eval_batches WHERE id = ?", (batch_id,)).fetchone())
        if not batch:
            return None
        counts = row(conn.execute(
            """
            SELECT
                COUNT(*) AS created_cases,
                COALESCE(SUM(CASE WHEN human_winner IS NULL THEN 1 ELSE 0 END), 0) AS pending_cases,
                COALESCE(SUM(CASE WHEN human_winner IS NOT NULL THEN 1 ELSE 0 END), 0) AS voted_cases
            FROM eval_cases
            WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone())
    batch.update(counts or {})
    batch["progress"] = (batch.get("created_cases") or 0) / batch["target_count"] if batch.get("target_count") else 0
    return batch


def clear_eval_data():
    with connect() as conn:
        conn.execute("DELETE FROM eval_cases")
        conn.execute("DELETE FROM eval_batches")
        conn.execute("DELETE FROM eval_samples")


def clear_eval_votes():
    with connect() as conn:
        conn.execute(
            """
            UPDATE eval_cases
            SET human_winner = NULL, criteria_votes_json = NULL, voted_at = NULL
            WHERE human_winner IS NOT NULL OR criteria_votes_json IS NOT NULL OR voted_at IS NOT NULL
            """
        )


def create_case(data: dict[str, Any]):
    id = newid()
    payload = {
        "id": id,
        "batch_id": data["batch_id"],
        "seed_samples_json": json.dumps(data.get("seed_samples", [])),
        "system_prompt": data["system_prompt"],
        "user_message": data["user_message"],
        "generation_notes": data.get("generation_notes", ""),
        "mold_name": data["mold_name"],
        "option_a_kind": data["option_a_kind"],
        "option_b_kind": data["option_b_kind"],
        "option_a_answer": data["option_a_answer"],
        "option_b_answer": data["option_b_answer"],
        "simple_answer": data["simple_answer"],
        "mold_answer": data["mold_answer"],
        "mold_state_json": json.dumps(data.get("mold_state", {})),
        "mold_trace_json": json.dumps(data.get("mold_trace", [])),
        "rubric_json": json.dumps(data.get("rubric", [])),
        "judge_suggestion": data.get("judge_suggestion"),
        "judge_confidence": data.get("judge_confidence"),
        "judge_rationale": data.get("judge_rationale", ""),
    }
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO eval_cases (
                id, batch_id, seed_samples_json, system_prompt, user_message, generation_notes, mold_name,
                option_a_kind, option_b_kind, option_a_answer, option_b_answer, simple_answer, mold_answer,
                mold_state_json, mold_trace_json, rubric_json, judge_suggestion, judge_confidence, judge_rationale
            ) VALUES (
                :id, :batch_id, :seed_samples_json, :system_prompt, :user_message, :generation_notes, :mold_name,
                :option_a_kind, :option_b_kind, :option_a_answer, :option_b_answer, :simple_answer, :mold_answer,
                :mold_state_json, :mold_trace_json, :rubric_json, :judge_suggestion, :judge_confidence, :judge_rationale
            )
            """,
            payload,
        )
    return get_case(id)


def get_case(id: str):
    with connect() as conn:
        return row(conn.execute("SELECT * FROM eval_cases WHERE id = ?", (id,)).fetchone())


def delete_case(id: str):
    with connect() as conn:
        conn.execute("DELETE FROM eval_cases WHERE id = ?", (id,))


def list_cases(pending: bool | None = None, limit: int = 50):
    where = ""
    if pending is True:
        where = "WHERE human_winner IS NULL"
    elif pending is False:
        where = "WHERE human_winner IS NOT NULL"
    with connect() as conn:
        return rows(conn.execute(
            f"SELECT * FROM eval_cases {where} ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall())


def next_case():
    with connect() as conn:
        return row(conn.execute(
            "SELECT * FROM eval_cases WHERE human_winner IS NULL ORDER BY created_at ASC LIMIT 1"
        ).fetchone())


def vote_case(case_id: str, winner: str, criteria_votes: dict[str, Any]):
    with connect() as conn:
        conn.execute(
            """
            UPDATE eval_cases
            SET human_winner = ?, criteria_votes_json = ?, voted_at = CURRENT_TIMESTAMP
            WHERE id = ? AND human_winner IS NULL
            """,
            (winner, json.dumps(criteria_votes), case_id),
        )
    return get_case(case_id)


def voted_examples(limit: int = 24):
    with connect() as conn:
        return rows(conn.execute(
            """
            SELECT system_prompt, user_message, option_a_answer, option_b_answer, rubric_json,
                   human_winner, criteria_votes_json
            FROM eval_cases
            WHERE human_winner IS NOT NULL
            ORDER BY voted_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall())


def stats():
    with connect() as conn:
        summary = row(conn.execute(
            """
            SELECT
                COUNT(*) AS total_cases,
                COALESCE(SUM(CASE WHEN human_winner IS NULL THEN 1 ELSE 0 END), 0) AS pending_cases,
                COALESCE(SUM(CASE WHEN human_winner IS NOT NULL THEN 1 ELSE 0 END), 0) AS voted_cases,
                COALESCE(SUM(CASE WHEN human_winner IS NOT NULL AND judge_suggestion = human_winner THEN 1 ELSE 0 END), 0) AS judge_correct
            FROM eval_cases
            """
        ).fetchone())
        per_mold = rows(conn.execute(
            """
            SELECT mold_name,
                   COUNT(*) AS voted,
                   SUM(CASE
                         WHEN human_winner = 'A' AND option_a_kind = 'mold' THEN 1
                         WHEN human_winner = 'B' AND option_b_kind = 'mold' THEN 1
                         ELSE 0
                       END) AS mold_wins,
                   SUM(CASE WHEN judge_suggestion = human_winner THEN 1 ELSE 0 END) AS judge_correct
            FROM eval_cases
            WHERE human_winner IS NOT NULL
            GROUP BY mold_name
            ORDER BY voted DESC
            """
        ).fetchall())
        recent = rows(conn.execute(
            """
            SELECT id, created_at, voted_at, mold_name, user_message, human_winner, judge_suggestion,
                   option_a_kind, option_b_kind
            FROM eval_cases
            WHERE human_winner IS NOT NULL
            ORDER BY voted_at DESC
            LIMIT 20
            """
        ).fetchall())
    return {"summary": summary, "per_mold": per_mold, "recent": recent}
