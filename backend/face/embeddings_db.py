"""SQLite store for face embeddings and person assignments."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


class FaceEmbeddingsDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
                    embedding BLOB NOT NULL,
                    person_id TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id)
                )
                """
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_faces_file_hash ON faces(file_hash)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)")
            conn.commit()
        finally:
            conn.close()

    def create_new_person(self) -> str:
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT COALESCE(MAX(CAST(SUBSTR(person_id,2) AS INTEGER)),0) FROM persons")
            new_id = f"p{c.fetchone()[0] + 1}"
            c.execute("INSERT INTO persons (person_id) VALUES (?)", (new_id,))
            conn.commit()
            return new_id
        finally:
            conn.close()

    def store_face(
        self,
        file_hash: str,
        face_index: int,
        bbox: tuple[int, int, int, int],
        embedding: np.ndarray,
        person_id: str | None = None,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO faces (file_hash, face_index, bbox_x1, bbox_y1, bbox_x2,
                                   bbox_y2, embedding, person_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_hash,
                    face_index,
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                    embedding.astype(np.float32).tobytes(),
                    person_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_faces_for_hash(self, file_hash: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM faces WHERE file_hash = ?", (file_hash,))
            conn.commit()
        finally:
            conn.close()

    def load_all_faces(self, *, only_unassigned: bool = False) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            if only_unassigned:
                c.execute("SELECT id, file_hash, embedding, person_id FROM faces WHERE person_id IS NULL")
            else:
                c.execute("SELECT id, file_hash, embedding, person_id FROM faces")
            rows = c.fetchall()
        finally:
            conn.close()
        return [
            {
                "id": r[0],
                "file_hash": r[1],
                "emb": np.frombuffer(r[2], dtype=np.float32),
                "person_id": r[3],
            }
            for r in rows
        ]

    def assign_face_to_person(self, face_id: int, person_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("UPDATE faces SET person_id=? WHERE id=?", (person_id, face_id))
            conn.commit()
        finally:
            conn.close()

    def reset_person_assignments(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM persons")
            conn.execute("UPDATE faces SET person_id = NULL")
            conn.commit()
        finally:
            conn.close()

    def delete_orphan_hashes(self, missing_hashes: set[str]) -> int:
        if not missing_hashes:
            return 0
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            for h in missing_hashes:
                c.execute("DELETE FROM faces WHERE file_hash = ?", (h,))
            c.execute(
                """
                DELETE FROM persons
                WHERE person_id NOT IN (
                    SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()
        return len(missing_hashes)

    def get_file_person_tags(self, person_prefix: str = "person:") -> dict[str, set[str]]:
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT file_hash, person_id FROM faces WHERE person_id IS NOT NULL")
            file_tags: dict[str, set[str]] = {}
            for fhash, pid in c.fetchall():
                file_tags.setdefault(fhash, set()).add(f"{person_prefix}{pid}")
            return file_tags
        finally:
            conn.close()

    def stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM faces")
            faces = int(c.fetchone()[0])
            c.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NULL")
            unassigned = int(c.fetchone()[0])
            c.execute("SELECT COUNT(*) FROM persons")
            persons = int(c.fetchone()[0])
            c.execute("SELECT COUNT(DISTINCT file_hash) FROM faces")
            files = int(c.fetchone()[0])
        finally:
            conn.close()
        return {
            "faces": faces,
            "unassigned_faces": unassigned,
            "persons": persons,
            "files_with_faces": files,
            "db_path": str(self.db_path),
        }

    def distinct_file_hashes(self) -> list[str]:
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT DISTINCT file_hash FROM faces")
            return [row[0] for row in c.fetchall()]
        finally:
            conn.close()

    def file_hashes_in_db(self) -> set[str]:
        """All file hashes with stored embeddings (one query per detect batch)."""
        return set(self.distinct_file_hashes())

    def file_has_embeddings(self, file_hash: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        try:
            c = conn.cursor()
            c.execute("SELECT 1 FROM faces WHERE file_hash = ? LIMIT 1", (file_hash,))
            return c.fetchone() is not None
        finally:
            conn.close()
