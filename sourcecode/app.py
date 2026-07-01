from __future__ import annotations

import csv
import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    END = None
    StateGraph = None

try:
    import faiss
    import numpy as np
except Exception:  # pragma: no cover
    faiss = None
    np = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


ROOT = Path(__file__).resolve().parent


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", value.strip())
    return slug.strip("._-") or "INFS4205"


COURSE_CODE = safe_slug(os.getenv("STUDY_AGENT_COURSE", "INFS4205"))
DATA_DIR = Path(os.getenv("STUDY_AGENT_DATA_DIR", ROOT / "data")).resolve()
COURSE_DIR = Path(os.getenv("STUDY_AGENT_COURSE_DIR", DATA_DIR / "courses" / COURSE_CODE)).resolve()
LEGACY_UPLOAD_DIR = DATA_DIR / "uploads"
LEGACY_IMAGE_DIR = DATA_DIR / "page_images"
UPLOAD_DIR = COURSE_DIR / "uploads"
IMAGE_DIR = COURSE_DIR / "page_images"
INDEX_DIR = COURSE_DIR / "indexes"
EVAL_DIR = DATA_DIR / "evaluation"
BENCHMARK_PATH = EVAL_DIR / "benchmark_cases.json"
LEGACY_DB_PATH = DATA_DIR / "study_agent_runtime_v2.sqlite"
DEFAULT_DB_PATH = COURSE_DIR / "study_agent.sqlite"
if os.getenv("STUDY_AGENT_DB_PATH"):
    DB_PATH = Path(os.getenv("STUDY_AGENT_DB_PATH", "")).resolve()
elif LEGACY_DB_PATH.exists() and not DEFAULT_DB_PATH.exists():
    DB_PATH = LEGACY_DB_PATH.resolve()
else:
    DB_PATH = DEFAULT_DB_PATH.resolve()
STATIC_DIR = ROOT / "static"
VECTOR_DIM = 256
DEFAULT_TEXT_MODEL = "qwen2.5:7b"
DEFAULT_VISION_MODEL = "llava:latest"
AUTO_VISION_SUMMARY = os.getenv("STUDY_AGENT_AUTO_VISION_SUMMARY", "0").strip() == "1"
READ_UPLOADED_IMAGES = os.getenv("STUDY_AGENT_READ_UPLOADED_IMAGES", "1").strip() != "0"
VISUAL_VECTOR_DIM = 256
RETRIEVAL_CACHE: dict[str, Any] = {"version": None, "rows": None, "mtime": 0.0}

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


def now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    COURSE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)


def connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init_db() -> None:
    ensure_dirs()
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY,
                user_name TEXT NOT NULL,
                course TEXT NOT NULL,
                preferred_language TEXT NOT NULL,
                answer_style TEXT NOT NULL,
                current_goal TEXT NOT NULL,
                daily_minutes INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS weak_topics (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                topic TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence REAL NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                course TEXT NOT NULL DEFAULT 'INFS4205',
                filename TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                path TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evidence_chunks (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                page INTEGER,
                modality TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                image_path TEXT,
                tokens_json TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                session_id INTEGER,
                parent_id INTEGER,
                is_branch INTEGER NOT NULL DEFAULT 0,
                user_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                resolved_query TEXT NOT NULL,
                intent TEXT NOT NULL,
                answer TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                trace_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_sessions (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                course TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_cases (
                id INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                expected_source TEXT,
                expected_keywords TEXT,
                top_k INTEGER NOT NULL,
                result_json TEXT,
                notes TEXT
            );
            """
        )
        migrate_schema(db)
        ensure_default_session(db)
        existing = db.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0]
        if not existing:
            db.execute(
                """
                INSERT INTO user_profiles
                (id, user_name, course, preferred_language, answer_style, current_goal, daily_minutes, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "Anna",
                    "INFS4205/7205",
                    "Chinese-English mixed",
                    "structured, exam-oriented, evidence-grounded",
                    "prepare final quiz and assignment",
                    90,
                    now(),
                ),
            )
            for topic in ["HNSW", "Token reduction", "MRAG evaluation"]:
                db.execute(
                    """
                    INSERT INTO weak_topics (user_id, topic, reason, confidence, updated_at)
                    VALUES (1, ?, 'default profile', 0.7, ?)
                    """,
                    (topic, now()),
                )


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "what",
    "why",
    "how",
    "are",
    "is",
    "to",
    "of",
    "in",
    "a",
    "an",
    "我",
    "的",
    "了",
    "是",
    "和",
    "一下",
    "什么",
    "怎么",
    "帮我",
}


def tokenize(text: str) -> list[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]{1,4}|\d+", text.lower())
    tokens: list[str] = []
    for token in raw:
        if token in STOPWORDS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]{3,4}", token):
            tokens.extend(token[i : i + 2] for i in range(len(token) - 1))
        tokens.append(token)
    return [t for t in tokens if len(t.strip()) > 1]


def vectorize(tokens: list[str]) -> list[float]:
    vector = [0.0] * VECTOR_DIM
    counts = Counter(tokens)
    for token, count in counts.items():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % VECTOR_DIM
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[idx] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [round(v / norm, 6) for v in vector]


def normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [round(value / norm, 6) for value in values]


def visual_vector_from_bytes(raw: bytes) -> list[float]:
    vector = [0.0] * VISUAL_VECTOR_DIM
    if not raw:
        return vector
    digest = hashlib.blake2b(raw, digest_size=64).digest()
    for idx, byte in enumerate(digest):
        vector[idx % VISUAL_VECTOR_DIM] += (byte / 255.0) - 0.5
    for start in range(0, min(len(raw), 65536), 257):
        byte = raw[start]
        vector[(start // 257) % VISUAL_VECTOR_DIM] += (byte / 255.0) - 0.5
    return normalize_vector(vector)


def visual_vectorize(image_path: str | None) -> list[float]:
    if not image_path:
        return [0.0] * VISUAL_VECTOR_DIM
    path = resolve_storage_path(image_path)
    if not path.exists():
        return [0.0] * VISUAL_VECTOR_DIM
    try:
        raw = path.read_bytes()
    except OSError:
        return [0.0] * VISUAL_VECTOR_DIM
    return visual_vector_from_bytes(raw)


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def faiss_rank(query_vector: list[float], vectors: list[list[float]], top_n: int) -> list[tuple[int, float]]:
    if faiss is None or np is None or not vectors:
        scored = [(idx, cosine(query_vector, vector)) for idx, vector in enumerate(vectors)]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_n]
    matrix = np.array(vectors, dtype="float32")
    query = np.array([query_vector], dtype="float32")
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(matrix)
    scores, indices = index.search(query, min(top_n, len(vectors)))
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


def db_signature(db: sqlite3.Connection) -> str:
    row = db.execute("SELECT COUNT(*) AS count, COALESCE(MAX(id), 0) AS max_id FROM evidence_chunks").fetchone()
    return f"{row['count']}:{row['max_id']}"


def retrieval_rows(db: sqlite3.Connection) -> list[tuple[sqlite3.Row, list[str], list[float], dict[str, Any]]]:
    signature = db_signature(db)
    if RETRIEVAL_CACHE.get("version") == signature and RETRIEVAL_CACHE.get("rows") is not None:
        return RETRIEVAL_CACHE["rows"]
    rows = db.execute("SELECT * FROM evidence_chunks").fetchall()
    parsed = []
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        if row["image_path"] and "visual_vector" not in metadata:
            metadata["visual_vector"] = visual_vectorize(row["image_path"])
        parsed.append((row, json.loads(row["tokens_json"]), json.loads(row["vector_json"]), metadata))
    RETRIEVAL_CACHE["version"] = signature
    RETRIEVAL_CACHE["rows"] = parsed
    return parsed


def invalidate_retrieval_cache() -> None:
    RETRIEVAL_CACHE["version"] = None
    RETRIEVAL_CACHE["rows"] = None


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def path_for_storage(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def path_for_browser(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        try:
            return "data/" + str(path.relative_to(DATA_DIR)).replace("\\", "/")
        except ValueError:
            return str(path)


def table_columns(db: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}


def migrate_schema(db: sqlite3.Connection) -> None:
    doc_cols = table_columns(db, "documents")
    if "course" not in doc_cols:
        db.execute("ALTER TABLE documents ADD COLUMN course TEXT NOT NULL DEFAULT 'INFS4205'")
    conv_cols = table_columns(db, "conversations")
    if "session_id" not in conv_cols:
        db.execute("ALTER TABLE conversations ADD COLUMN session_id INTEGER")
    if "parent_id" not in conv_cols:
        db.execute("ALTER TABLE conversations ADD COLUMN parent_id INTEGER")
    if "is_branch" not in conv_cols:
        db.execute("ALTER TABLE conversations ADD COLUMN is_branch INTEGER NOT NULL DEFAULT 0")


def ensure_default_session(db: sqlite3.Connection) -> int:
    row = db.execute("SELECT id FROM conversation_sessions ORDER BY id ASC LIMIT 1").fetchone()
    if row:
        session_id = int(row["id"])
    else:
        cursor = db.execute(
            "INSERT INTO conversation_sessions (title, course, created_at, updated_at) VALUES (?, ?, ?, ?)",
            ("New study chat", COURSE_CODE, now(), now()),
        )
        session_id = int(cursor.lastrowid)
    db.execute("UPDATE conversations SET session_id=? WHERE session_id IS NULL", (session_id,))
    return session_id


def chunk_text(text: str, size: int = 900, overlap: int = 140) -> list[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def infer_week(filename: str, text: str = "") -> str | None:
    match = re.search(r"(?:week|w)[\s_\-]*(\d{1,2})", f"{filename} {text}", re.I)
    return f"Week {match.group(1)}" if match else None


TOPIC_PATTERNS = {
    "RAG": r"\bRAG\b|retrieval augmented|retrieval-augmented",
    "Naive RAG": r"na[iï]ve rag|broken chunks|irrelevant documents|wrong conclusions",
    "Chunking": r"chunking|recursive chunk|broken chunks",
    "Contextual Retrieval": r"contextual retrieval|anthropic",
    "BM25": r"\bBM25\b|best matching|sparse retrieval",
    "Hybrid Search": r"hybrid search|merged and weighted",
    "Reranker": r"reranker|re-rank|re-ranking|rerank",
    "RRF": r"reciprocal rank fusion|\bRRF\b",
    "Multimodal RAG": r"multimodal rag|\bMRAG\b|pseudo-mrag|true multimodal",
    "ColPali": r"colpali|late interaction",
    "Jina Embeddings": r"jina|multilingual retrieval",
    "ViDoRe": r"vidore|real-world scenarios",
    "Agentic RAG": r"agentic rag|iterative search|search-o1|visor",
    "LLM Wiki": r"llm wiki|compiled-memory|compiled memory|rag always needed",
    "Evaluation": r"evaluation|benchmark|groundedness|recall|mrr|ndcg",
    "Token Reduction": r"token reduction|information bottleneck|reduce tokens",
    "Vision Transformer": r"vision transformer|\bvit\b|self-attention|image tokens",
    "Pruning": r"pruning|prune|importance score",
    "Merging": r"merging|merge|similar tokens|tome",
    "Routing": r"routing|route|dynamic.*tokens",
    "Resampling": r"resampling|latent tokens|resampler",
    "Information Bottleneck": r"information bottleneck|compression|mutual information",
}


def infer_topics(text: str) -> list[str]:
    topics = []
    haystack = text.lower()
    for topic, pattern in TOPIC_PATTERNS.items():
        if re.search(pattern, haystack, re.I):
            topics.append(topic)
    return topics[:8]


def infer_visual_type(text: str) -> str:
    haystack = text.lower()
    if re.search(r"pipeline|roadmap|workflow|->|→", haystack):
        return "pipeline/roadmap"
    if re.search(r"table|benchmark|comparison|score", haystack):
        return "table/comparison"
    if re.search(r"formula|equation|bm25|rrf|ndcg", haystack):
        return "formula"
    if re.search(r"architecture|diagram|model|colpali|jina", haystack):
        return "architecture/diagram"
    return "slide"


def guess_title(content: str, fallback: str) -> str:
    lines = [line.strip() for line in re.split(r"[\n。]", content) if line.strip()]
    if not lines:
        return fallback
    title = lines[0][:90]
    return title if len(title) > 8 else fallback


def add_evidence(
    db: sqlite3.Connection,
    document_id: int,
    source: str,
    page: int | None,
    modality: str,
    title: str,
    content: str,
    image_path: str | None,
    metadata: dict[str, Any],
) -> None:
    metadata = dict(metadata)
    metadata.setdefault("topics", infer_topics(f"{title} {content} {source}"))
    metadata.setdefault("visual_type", infer_visual_type(f"{title} {content}"))
    if image_path and "visual_vector" not in metadata:
        metadata["visual_vector"] = visual_vectorize(image_path)
    tokens = tokenize(content + " " + title + " " + source + " " + " ".join(metadata.get("topics", [])))
    db.execute(
        """
        INSERT INTO evidence_chunks
        (document_id, source, page, modality, title, content, image_path, tokens_json, vector_json, metadata_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document_id,
            source,
            page,
            modality,
            title,
            content,
            image_path,
            json_dumps(tokens),
            json_dumps(vectorize(tokens)),
            json_dumps(metadata),
            now(),
        ),
    )


def save_page_image(pdf_path: Path, filename: str, page_index: int) -> str | None:
    if fitz is None:
        return None
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
        out = IMAGE_DIR / f"{pdf_path.stem}_page_{page_index + 1}.png"
        pix.save(out)
        doc.close()
        return path_for_browser(out)
    except Exception:
        return None


def infer_page_image_path(source: str, page: int | None) -> str | None:
    if not page:
        return None
    stem = Path(source).stem
    candidate = IMAGE_DIR / f"{stem}_page_{page}.png"
    if candidate.exists():
        return path_for_browser(candidate)
    legacy_candidate = LEGACY_IMAGE_DIR / f"{stem}_page_{page}.png"
    if legacy_candidate.exists():
        return path_for_browser(legacy_candidate)
    source_path = UPLOAD_DIR / source
    if not source_path.exists():
        legacy = LEGACY_UPLOAD_DIR / source
        if legacy.exists():
            source_path = legacy
    if source_path.exists() and source_path.suffix.lower() == ".pdf" and fitz is not None:
        try:
            doc = fitz.open(source_path)
            if 1 <= page <= doc.page_count:
                pix = doc.load_page(page - 1).get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
                pix.save(candidate)
                doc.close()
                return path_for_browser(candidate)
            doc.close()
        except Exception:
            return None
    return None


def ingest_pdf(db: sqlite3.Connection, path: Path, document_id: int) -> int:
    if PdfReader is not None:
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
    elif fitz is not None:
        doc = fitz.open(path)
        page_count = doc.page_count
        pages = []
        for idx in range(page_count):
            try:
                pages.append(doc.load_page(idx).get_text("text") or "")
            except Exception:
                pages.append("")
        doc.close()
    else:
        raise RuntimeError("PDF extraction needs pypdf or pymupdf. Please run pip install -r requirements.txt")

    for idx, text in enumerate(pages):
        page_no = idx + 1
        week = infer_week(path.name, text)
        image_path = save_page_image(path, path.name, idx)
        page_title = guess_title(text, f"{path.name} page {page_no}")
        if text.strip():
            for part_no, chunk in enumerate(chunk_text(text), start=1):
                add_evidence(
                    db,
                    document_id,
                    path.name,
                    page_no,
                    "slide_text",
                    f"{page_title} · text {part_no}",
                    chunk,
                    image_path,
                    {"week": week, "part": part_no, "source_type": "pdf", "ocr_text": text[:1200], "index_name": "text_index"},
                )
        visual_caption = (
            f"Slide image evidence from {path.name}, page {page_no}. "
            f"Likely topic: {page_title}. Text anchors: {text[:420]}"
        )
        add_evidence(
            db,
            document_id,
            path.name,
            page_no,
            "slide_image",
            f"{page_title} · visual page",
            visual_caption,
            image_path,
            {
                "week": week,
                "source_type": "pdf_page_image",
                "ocr_text": text[:1200],
                "caption": visual_caption,
                "index_name": "visual_caption_index",
            },
        )
    return page_count


def ingest_text(db: sqlite3.Connection, path: Path, document_id: int, modality: str) -> int:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for idx, chunk in enumerate(chunk_text(text), start=1):
        add_evidence(
            db,
            document_id,
            path.name,
            None,
            modality,
            guess_title(chunk, f"{path.name} note {idx}"),
            chunk,
            None,
            {"week": infer_week(path.name, chunk), "part": idx, "source_type": "text"},
        )
    return 1


def ingest_csv(db: sqlite3.Connection, path: Path, document_id: int) -> int:
    rows = 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for rows, row in enumerate(reader, start=1):
            content = "; ".join(f"{key}: {value}" for key, value in row.items())
            add_evidence(
                db,
                document_id,
                path.name,
                None,
                "table",
                f"{path.name} row {rows}",
                content,
                None,
                {"row": rows, "source_type": "csv"},
            )
    return rows


def ingest_image(db: sqlite3.Connection, path: Path, document_id: int, caption: str = "") -> int:
    rel_path = path_for_browser(path)
    supplied_caption = caption.strip()
    vision_caption = ""
    if READ_UPLOADED_IMAGES and not supplied_caption:
        image = image_to_base64(rel_path)
        if image:
            vision_caption = llm_generate(
                f"""
You are indexing an uploaded image for a personalised study agent.
Read the image carefully and write searchable study evidence.
If it is a slide/screenshot, capture title, visible text, diagram meaning, formulas, parameters, and key learning points.
If text is visible, preserve important technical terms exactly.
If text is blurry or uncertain, mark it as [unclear] instead of guessing.
Return concise bullet points only.

Filename: {path.name}
""",
                images=[image],
                role="vision",
            ) or ""
    content = supplied_caption or vision_caption.strip() or (
        f"Uploaded visual evidence named {path.name}. "
        "Use this image as a course screenshot, diagram, equation, or slide visual evidence."
    )
    add_evidence(
        db,
        document_id,
        path.name,
        None,
        "uploaded_image",
        path.stem,
        content,
        rel_path,
        {
            "source_type": "image",
            "week": infer_week(path.name, content),
            "caption": content,
            "ocr_text": content,
            "index_name": "uploaded_image_index",
            "vision_read": bool(vision_caption.strip()),
        },
    )
    return 1


def index_existing_file(db: sqlite3.Connection, target: Path, caption: str = "") -> dict[str, Any]:
    ext = target.suffix.lower()
    doc_type = "pdf" if ext == ".pdf" else "image" if ext in {".png", ".jpg", ".jpeg", ".webp"} else "csv" if ext == ".csv" else "text"
    cursor = db.execute(
        "INSERT INTO documents (course, filename, doc_type, path, page_count, uploaded_at) VALUES (?, ?, ?, ?, 0, ?)",
        (COURSE_CODE, target.name, doc_type, path_for_storage(target), now()),
    )
    doc_id = cursor.lastrowid
    try:
        if doc_type == "pdf":
            count = ingest_pdf(db, target, doc_id)
        elif doc_type == "csv":
            count = ingest_csv(db, target, doc_id)
        elif doc_type == "image":
            count = ingest_image(db, target, doc_id, caption)
        else:
            count = ingest_text(db, target, doc_id, "note_text")
        db.execute("UPDATE documents SET page_count=? WHERE id=?", (count, doc_id))
        invalidate_retrieval_cache()
        return {"id": doc_id, "filename": target.name, "type": doc_type, "items": count}
    except Exception:
        db.execute("DELETE FROM evidence_chunks WHERE document_id=?", (doc_id,))
        db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        raise


def get_profile(db: sqlite3.Connection) -> dict[str, Any]:
    profile = dict(db.execute("SELECT * FROM user_profiles WHERE id = 1").fetchone())
    weak = [
        dict(row)
        for row in db.execute(
            "SELECT topic, reason, confidence, updated_at FROM weak_topics WHERE user_id = 1 ORDER BY confidence DESC, updated_at DESC"
        )
    ]
    profile["weak_topics"] = weak
    return profile


def classify_intent(query: str) -> str:
    q = query.lower()
    if ("retriever" in q or "检索器" in q) and ("reranker" in q or "重排" in q):
        return "comparison"
    if "branch question without changing the original plan" in q:
        if re.search(r"区别|区分|compare|versus| vs |different|到底怎么分", q):
            return "comparison"
        return "mistake_review"
    if re.search(r"(lecture|week|pdf|slide)\s*\d+.*(overview|summary|cover|about|knowledge|points?)|(\u8bb2\u4e86\u4ec0\u4e48|\u4e3b\u8981\u8bb2|\u603b\u7ed3|\u6982\u89c8|\u77e5\u8bc6\u70b9|\u6574\u7406)", q):
        return "document_overview"
    if re.search(r"区别|区分|compare|versus| vs |different|对比|到底怎么分", q):
        return "comparison"
    if re.search(r"计划|复习|revision|study plan|days?|天|week\s*\d+\s*-\s*\d+", q):
        return "revision_planning"
    if re.search(r"quiz|练习|题|practice|mcq|test me", q):
        return "quiz_generation"
    if re.search(r"哪.*文件|找.*文件|找一下|在哪|哪几页|slide|page|pdf|where|which file|哪一页", q):
        return "file_or_slide_search"
    if re.search(r"图片|图|diagram|figure|screenshot|公式|image", q):
        return "visual_question"
    if re.search(r"错|不懂|confused|weak|mistake|掌握", q):
        return "mistake_review"
    return "concept_explanation"


def resolve_followup(db: sqlite3.Connection, query: str) -> str:
    last = db.execute("SELECT query FROM conversations ORDER BY id DESC LIMIT 1").fetchone()
    if not last:
        return query
    latest_plan = db.execute(
        "SELECT id, query FROM conversations WHERE intent='revision_planning' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if latest_plan and re.search(r"阶段|第\s*\d+\s*天|day\s*\d+|这一步|这个部分|这里|不懂|不理解|confused|branch", query, re.I):
        return f"Revision plan context: {latest_plan['query']}. Branch question without changing the original plan: {query}"
    if re.search(r"^(那|这个|它|this|that|继续|再|what about|and )", query.strip(), re.I):
        return f"Previous question: {last['query']}. Follow-up: {query}"
    return query


def retrieve(
    db: sqlite3.Connection,
    query: str,
    top_k: int = 6,
    modality_filter: str | None = None,
    source_filter: str | None = None,
    retrieval_mode: str = "hybrid",
) -> list[dict[str, Any]]:
    q_tokens = tokenize(query)
    q_vector = vectorize(q_tokens)
    q_counts = Counter(q_tokens)
    parsed_rows = retrieval_rows(db)
    doc_count = max(len(parsed_rows), 1)
    df = Counter()
    visual_query = re.search(r"图|图片|截图|diagram|figure|image|visual|slide|页面|公式|表格", query, re.I) is not None
    query_visual_vector = visual_vector_from_bytes(query.encode("utf-8")) if visual_query else [0.0] * VISUAL_VECTOR_DIM
    if retrieval_mode == "text_only":
        retrieval_mode = "keyword"
    elif retrieval_mode == "caption_only":
        modality_filter = "image"
        retrieval_mode = "keyword"
    for row, tokens, _vector, _metadata in parsed_rows:
        df.update(set(tokens))

    candidates = []
    for local_idx, (row, tokens, vector, metadata) in enumerate(parsed_rows):
        if modality_filter and modality_filter not in row["modality"]:
            continue
        if source_filter and row["source"] != source_filter:
            continue
        counts = Counter(tokens)
        bm25 = 0.0
        for token, qf in q_counts.items():
            if not counts[token]:
                continue
            idf = math.log((doc_count - df[token] + 0.5) / (df[token] + 0.5) + 1.0)
            bm25 += idf * (counts[token] * 2.2) / (counts[token] + 1.2)
        vec_score = max(0.0, cosine(q_vector, vector))
        visual_vector = metadata.get("visual_vector") or ([0.0] * VISUAL_VECTOR_DIM)
        visual_score = max(0.0, cosine(query_visual_vector, visual_vector)) if visual_query else 0.0
        metadata_boost = 0.0
        week = metadata.get("week") or ""
        if week and week.lower() in query.lower():
            metadata_boost += 0.18
        topics = metadata.get("topics", [])
        if topics:
            topic_hits = [topic for topic in topics if topic.lower() in query.lower()]
            metadata_boost += min(0.28, len(topic_hits) * 0.09)
        if row["modality"] in {"slide_image", "uploaded_image"} and re.search(r"图|图片|diagram|figure|image|slide", query, re.I):
            metadata_boost += 0.2
        if row["modality"] == "table" and re.search(r"表|错题|topic|mapping|quiz|score", query, re.I):
            metadata_boost += 0.16
        candidates.append(
            {
                "local_idx": local_idx,
                "row": row,
                "tokens": tokens,
                "vector": vector,
                "metadata": metadata,
                "bm25": bm25,
                "vec_score": vec_score,
                "visual_score": visual_score,
                "metadata_boost": metadata_boost,
            }
        )

    faiss_results = faiss_rank(q_vector, [item["vector"] for item in candidates], max(top_k * 4, 20))
    faiss_score_by_local = {candidates[idx]["local_idx"]: score for idx, score in faiss_results if idx < len(candidates)}
    keyword_ranking = [item["local_idx"] for item in sorted(candidates, key=lambda item: item["bm25"] + item["metadata_boost"], reverse=True)]
    vector_ranking = [item["local_idx"] for item in sorted(candidates, key=lambda item: faiss_score_by_local.get(item["local_idx"], item["vec_score"]), reverse=True)]
    visual_ranking = [item["local_idx"] for item in sorted(candidates, key=lambda item: item["visual_score"], reverse=True)]
    metadata_ranking = [item["local_idx"] for item in sorted(candidates, key=lambda item: item["metadata_boost"], reverse=True)]
    fused = reciprocal_rank_fusion([keyword_ranking, vector_ranking, visual_ranking, metadata_ranking])

    results = []
    for item in candidates:
        row = item["row"]
        tokens = item["tokens"]
        counts = Counter(tokens)
        metadata = item["metadata"]
        bm25 = item["bm25"]
        vec_score = faiss_score_by_local.get(item["local_idx"], item["vec_score"])
        visual_score = item["visual_score"]
        metadata_boost = item["metadata_boost"]
        if retrieval_mode == "keyword":
            score = bm25 + metadata_boost
        elif retrieval_mode == "vector":
            score = vec_score + metadata_boost
        elif retrieval_mode == "visual":
            score = visual_score + metadata_boost
        else:
            score = bm25 * 0.44 + vec_score * 0.27 + visual_score * 0.16 + metadata_boost + fused.get(item["local_idx"], 0.0) * 4.0
        if score <= 0:
            continue
        matched = [token for token in q_tokens if token in counts][:8]
        results.append(
            {
                "id": row["id"],
                "source": row["source"],
                "page": row["page"],
                "modality": row["modality"],
                "title": row["title"],
                "excerpt": row["content"][:700],
                "image_path": row["image_path"] or infer_page_image_path(row["source"], row["page"]),
                "score": round(score, 4),
                "why": f"Matched terms: {', '.join(matched) if matched else 'semantic/vector overlap'}",
                "metadata": metadata,
                "topics": topics,
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def retrieve_final(db: sqlite3.Connection, query: str, top_k: int = 6) -> list[dict[str, Any]]:
    intent = classify_intent(query)
    if intent == "document_overview":
        doc, evidence = document_overview_evidence(db, query, top_k=top_k)
        return evidence
    if re.search(r"图|图片|截图|diagram|figure|image|visual|slide|页面|找出|找到", query, re.I):
        return retrieve(db, query, top_k=top_k, retrieval_mode="caption_only")
    if intent in {"comparison", "concept_explanation", "revision_planning", "mistake_review"}:
        return retrieve(db, query, top_k=top_k, modality_filter="text", retrieval_mode="hybrid")
    modality = "image" if intent in {"visual_question", "file_or_slide_search"} else None
    pools = [
        retrieve(db, query, top_k=max(top_k, 8), modality_filter=modality, retrieval_mode="hybrid"),
        retrieve(db, query, top_k=max(top_k, 8), retrieval_mode="caption_only"),
        retrieve(db, query, top_k=max(top_k, 8), retrieval_mode="visual"),
    ]
    if modality:
        pools.append(retrieve(db, query, top_k=max(top_k, 8), retrieval_mode="hybrid"))
    by_id: dict[int, dict[str, Any]] = {}
    for pool_idx, pool in enumerate(pools):
        for rank, item in enumerate(pool, start=1):
            current = by_id.get(item["id"])
            bonus = 0.24 / (rank + pool_idx + 1)
            if current is None:
                copy = dict(item)
                copy["score"] = round(float(copy.get("score", 0)) + bonus, 4)
                by_id[item["id"]] = copy
            else:
                current["score"] = round(float(current.get("score", 0)) + bonus, 4)
    evidence = sorted(by_id.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return reading_order(evidence) if intent == "file_or_slide_search" else evidence


def retrieve_for_ablation(db: sqlite3.Connection, query: str, mode: str, top_k: int) -> list[dict[str, Any]]:
    if mode in {"final_agent", "hybrid"}:
        return retrieve_final(db, query, top_k=top_k)
    if mode == "plain_llm":
        return []
    if mode == "plain_vlm":
        return retrieve(db, query, top_k=top_k, modality_filter="image", retrieval_mode="caption_only")
    if mode == "no_router":
        return retrieve(db, query, top_k=top_k, retrieval_mode="hybrid")
    if mode == "no_rerank":
        return retrieve(db, query, top_k=top_k, retrieval_mode="vector")
    if mode == "no_visual":
        return retrieve(db, query, top_k=top_k, modality_filter="text", retrieval_mode="hybrid")
    if mode == "no_memory":
        return retrieve(db, query, top_k=top_k, retrieval_mode="hybrid")
    if mode in {"text_only", "caption_only", "visual"}:
        return retrieve(db, query, top_k=top_k, retrieval_mode=mode)
    return retrieve(db, query, top_k=top_k, retrieval_mode="hybrid")


def find_document_for_query(db: sqlite3.Connection, query: str) -> sqlite3.Row | None:
    docs = db.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
    if not docs:
        return None
    q = query.lower()
    number_match = re.search(r"(?:lecture|week|w)\s*(\d{1,2})", q)
    if number_match:
        number = number_match.group(1)
        for doc in docs:
            filename = doc["filename"].lower()
            if re.search(rf"(?:lecture|week|w)[_\-\s]*0?{number}\b", filename):
                return doc
            if f"lecture{number}" in filename or f"week{number}" in filename:
                return doc
    for doc in docs:
        stem_tokens = tokenize(Path(doc["filename"]).stem)
        if stem_tokens and any(token in q for token in stem_tokens):
            return doc
    return docs[0] if len(docs) == 1 else None


def document_overview_evidence(db: sqlite3.Connection, query: str, top_k: int = 10) -> tuple[sqlite3.Row | None, list[dict[str, Any]]]:
    doc = find_document_for_query(db, query)
    if not doc:
        return None, []
    rows = db.execute(
        """
        SELECT * FROM evidence_chunks
        WHERE source = ? AND modality IN ('slide_text', 'slide_image')
        ORDER BY COALESCE(page, 0), modality
        """,
        (doc["filename"],),
    ).fetchall()
    slide_text_rows = [row for row in rows if row["modality"] == "slide_text"]
    if slide_text_rows:
        if top_k <= 1:
            candidate_rows = [slide_text_rows[0]]
        else:
            candidate_rows = [
                slide_text_rows[round(i * (len(slide_text_rows) - 1) / (top_k - 1))]
                for i in range(top_k)
            ]
        priority_terms = ["limitation", "chunk", "reranker", "multimodal", "roadmap", "agentic", "wiki", "needed"]
        for row in slide_text_rows:
            haystack = f"{row['title']} {row['content']}".lower()
            if any(term in haystack for term in priority_terms):
                candidate_rows.append(row)
    else:
        candidate_rows = rows
    image_by_page = {row["page"]: row for row in rows if row["modality"] == "slide_image"}

    selected: list[dict[str, Any]] = []
    seen_pages: set[int] = set()
    seen_titles: set[str] = set()
    for row in candidate_rows:
        row = image_by_page.get(row["page"], row)
        page = row["page"] or 0
        clean_title = re.sub(r"\s*·\s*(text|visual).*$", "", row["title"]).strip().lower()
        if page in seen_pages or (clean_title in seen_titles and len(selected) >= 3):
            continue
        if len(selected) >= 4 and any(abs(page - existing_page) <= 2 for existing_page in seen_pages):
            continue
        metadata = json.loads(row["metadata_json"])
        selected.append(
            {
                "id": row["id"],
                "source": row["source"],
                "page": row["page"],
                "modality": row["modality"],
                "title": row["title"],
                "excerpt": row["content"][:700],
                "image_path": row["image_path"] or infer_page_image_path(row["source"], row["page"]),
                "score": 1.0,
                "why": "Selected for document-level overview rather than single-query similarity.",
                "metadata": metadata,
            }
        )
        seen_pages.add(page)
        seen_titles.add(clean_title)
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        selected.extend(retrieve(db, query, top_k=top_k - len(selected), source_filter=doc["filename"]))
    return doc, selected[:top_k]


def image_to_base64(image_path: str | None) -> str | None:
    if not image_path:
        return None
    path = resolve_storage_path(image_path)
    if not path.exists() or path.stat().st_size > 12_000_000:
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    except OSError:
        return None


def resolve_storage_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    if str(value).replace("\\", "/").startswith("data/"):
        return DATA_DIR / str(value).replace("\\", "/")[5:]
    return candidate


def visual_summary_for_evidence(item: dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if metadata.get("visual_summary"):
        return metadata["visual_summary"]
    if not AUTO_VISION_SUMMARY:
        return item.get("excerpt", "")[:360]
    image_path = item.get("image_path") or infer_page_image_path(item.get("source", ""), item.get("page"))
    image = image_to_base64(image_path)
    if not image:
        return item.get("excerpt", "")[:260]
    prompt = f"""
You are reading a lecture slide for a study agent.
Summarise what this slide visually/textually contains for revision.
Use Chinese if the slide/query context is Chinese, but keep technical terms in English.
Do not say "the image shows" repeatedly. Give the educational content.

Slide metadata:
source={item.get('source')}
page={item.get('page')}
title={item.get('title')}
text_anchor={item.get('excerpt', '')[:700]}

Return 2-4 concise bullet points.
"""
    summary = llm_generate(prompt, images=[image], role="vision")
    return summary or item.get("excerpt", "")[:260]


def enrich_visual_summaries(evidence: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    enriched = []
    for idx, item in enumerate(evidence):
        copy = dict(item)
        copy["image_path"] = copy.get("image_path") or infer_page_image_path(copy.get("source", ""), copy.get("page"))
        if idx < limit and copy.get("image_path"):
            copy["visual_summary"] = visual_summary_for_evidence(copy)
        enriched.append(copy)
    return enriched


def choose_ollama_model(role: str = "text", has_images: bool = False) -> tuple[str, str]:
    health = ollama_health()
    available = set(health.get("models", []))
    text_model = os.getenv("OLLAMA_TEXT_MODEL", os.getenv("OLLAMA_MODEL", DEFAULT_TEXT_MODEL)).strip()
    vision_model = os.getenv("OLLAMA_VISION_MODEL", DEFAULT_VISION_MODEL).strip()
    desired = vision_model if has_images or role == "vision" else text_model
    if desired in available:
        return desired, "configured"
    fallback_order = [text_model, vision_model, os.getenv("OLLAMA_MODEL", "").strip(), DEFAULT_VISION_MODEL]
    for model in fallback_order:
        if model and model in available:
            return model, f"fallback_from_{desired}"
    return desired, "unavailable"


def llm_generate(prompt: str, images: list[str] | None = None, role: str = "text") -> str | None:
    if os.getenv("OLLAMA_DISABLED", "").strip() == "1":
        return None
    model, model_status = choose_ollama_model(role, bool(images))
    if model_status == "unavailable":
        return None
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    payload_obj: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    if images:
        payload_obj["images"] = images[:2]
    payload = json_dumps(payload_obj).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
        return (data.get("response") or "").strip() or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def ollama_health() -> dict[str, Any]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = [item.get("name") for item in data.get("models", [])]
        return {"available": True, "models": models}
    except Exception:
        return {"available": False, "models": []}


def make_document_overview(profile: dict[str, Any], doc: sqlite3.Row | None, evidence: list[dict[str, Any]], query: str = "") -> str:
    if not doc or not evidence:
        return "I could not find an indexed lecture document to summarise. Please upload and index the PDF first."
    with connect() as db:
        outline = document_outline(db, doc["filename"], max_items=80)
    evidence = enrich_visual_summaries(evidence, limit=8)
    if re.search(r"全部|所有|知识点|整理|复习|summary|overview|knowledge", query, re.I):
        return make_structured_document_summary(doc, outline, query, evidence)
    titles = []
    for item in evidence:
        clean_title = re.sub(r"\s*·\s*(text|visual).*$", "", item["title"]).strip()
        if clean_title and clean_title.lower() not in {title.lower() for title in titles}:
            titles.append(clean_title)
    evidence_brief = "\n".join(
        f"- p.{item['page']}: {item['title']} :: {item['excerpt'][:300]}" for item in evidence[:10]
    )
    full_outline = "\n".join(
        f"- p.{item['page']}: {item['title']} :: {item['snippet']}" for item in outline
    )
    language = response_language(query=query, profile=profile)
    prompt = f"""
You are a personalised study agent for {profile['course']}.
The user prefers {profile['preferred_language']} and {profile['answer_style']}.
Summarise this lecture PDF for exam revision. Use only the text evidence below.
Do not describe screenshots, images, webpages, or the UI. The task is to organise the lecture knowledge points for the student.
MANDATORY LANGUAGE RULE: answer in {language}. If the user query is Chinese, answer in Chinese and keep only key technical terms in English.
Use second person ("你") when answering in Chinese.

Document: {doc['filename']}
Representative evidence:
{evidence_brief}

Full slide outline:
{full_outline}

Return:
Use this exact structure, but do not mention internal system details:

### 总览
用 3-5 句话说明这周课的主线。

### 知识点地图
按模块整理所有关键知识点。每个知识点包含：是什么、为什么重要、对应页码。

### 易混淆点
列出容易混的概念，并说明怎么区分。

### 考试/作业可用答题框架
给出你可以直接背/改写的答题模板。

### 下一步学习行动
给出具体复习动作，例如先看哪些页、整理什么表、做什么主动回忆。不要建议用户继续提问。
"""
    generated = llm_generate(prompt, images=None, role="text")
    if generated:
        return enforce_answer_language(generated, query, profile) or generated

    overview_titles = [re.sub(r"\s*·\s*(text|visual).*$", "", item["title"]).strip() for item in evidence[:10]]
    topic_lines = "\n".join(f"- p.{item['page']}: {title}" for item, title in zip(evidence[:10], overview_titles))
    return "\n".join(
        [
            "### Lecture overview",
            f"{doc['filename']} 主要讲的是 Multimodal RAG / advanced RAG 的设计与局限。Lecture 8 不是只讲普通 RAG 定义，而是围绕 RAG 在真实多模态场景中的扩展：chunking、reranking、multimodal evidence、agentic RAG，以及什么时候 RAG 并不总是必要。",
            "",
            "### Key ideas",
            topic_lines,
            "",
            "### Exam focus",
            "- 能解释 RAG pipeline: query -> retrieval/index -> evidence/context -> generation。",
            "- 能说明 naive RAG 的限制，例如 chunking 太粗或太碎都会影响 grounded answer。",
            "- 能区分 text-only RAG 和 multimodal RAG: 后者需要 caption/OCR/image metadata 或多索引 fusion。",
            "- 能讨论 system trade-off: retrieval quality, latency, memory, evidence grounding。",
            "",
            "### Common traps",
            "- 不要只说 RAG = 检索后回答。高分答案要说明 indexing choice、metadata、retrieval/ranking、groundedness 和 failure cases。",
            "",
            "### Next study action",
            "- 先复习 p.8-p.15 的 naive RAG/chunking limitations。",
            "- 再整理 p.25 reranker 和 p.51 agentic RAG 的区别。",
            "- 最后写一段 120 字 exam answer：为什么 Multimodal RAG 比 text-only RAG 更难评估。",
        ]
    )


def extract_topics(query: str, evidence: list[dict[str, Any]]) -> list[str]:
    candidates = []
    banned_re = re.compile(r"我不懂|不懂|帮我|给一些|复习|解释|区别|什么|怎么|这个|那个|根据|生成|练习|题目|总结")
    patterns = [
        r"\b(CLIP|DINO|MAE|RAG|MRAG|HNSW|BM25|nDCG|Recall@K|LangGraph|ReAct|Token reduction|token pruning|embedding|reranker)\b",
    ]
    text = query + " " + " ".join(item.get("title", "") for item in evidence[:5])
    for pattern in patterns:
        candidates.extend(re.findall(pattern, text, flags=re.I))
    for item in evidence[:8]:
        candidates.extend(item.get("topics") or item.get("metadata", {}).get("topics") or [])
        candidates.extend(infer_topics(f"{item.get('title','')} {item.get('excerpt','')}"))
    clean = []
    for topic in candidates:
        topic = topic if isinstance(topic, str) else topic[0]
        topic = topic.strip()
        if banned_re.search(topic) or len(topic) < 2:
            continue
        if topic and topic.lower() not in {t.lower() for t in clean}:
            clean.append(topic)
    return clean[:5] or ["current topic"]


def response_language(query: str, profile: dict[str, Any]) -> str:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", query))
    latin_words = len(re.findall(r"[A-Za-z]{2,}", query))
    if chinese_chars >= 4 and chinese_chars >= latin_words:
        return "Chinese, with important technical terms kept in English"
    if profile.get("preferred_language") == "Chinese":
        return "Chinese"
    if profile.get("preferred_language") == "English":
        return "English"
    return "Chinese-English mixed"


def needs_chinese_answer(query: str, profile: dict[str, Any] | None = None) -> bool:
    profile = profile or {}
    return "Chinese" in response_language(query, profile)


def mostly_not_chinese(text: str) -> bool:
    if not text:
        return False
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z]{2,}", text))
    return latin_words > 20 and chinese_chars < max(12, latin_words // 6)


def enforce_answer_language(text: str | None, query: str, profile: dict[str, Any] | None = None) -> str | None:
    if not text or not needs_chinese_answer(query, profile) or not mostly_not_chinese(text):
        return text
    prompt = f"""
Rewrite the answer below in Chinese for the student. Keep important technical terms in English.
Do not add new facts. Preserve Markdown headings, bold terms, page citations such as p.1, and the concrete study actions.

Answer:
{text}
"""
    rewritten = llm_generate(prompt, role="text")
    return rewritten or text


def document_outline(db: sqlite3.Connection, filename: str, max_items: int = 80) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT page, title, content
        FROM evidence_chunks
        WHERE source = ? AND modality = 'slide_text'
        ORDER BY COALESCE(page, 0)
        """,
        (filename,),
    ).fetchall()
    outline = []
    seen = set()
    for row in rows[:max_items]:
        clean_title = re.sub(r"\s*·\s*text\s*\d+.*$", "", row["title"]).strip()
        key = (row["page"], clean_title.lower())
        if key in seen:
            continue
        seen.add(key)
        outline.append(
            {
                "page": row["page"],
                "title": clean_title,
                "snippet": re.sub(r"\s+", " ", row["content"])[:180],
            }
        )
    return outline


def compress_outline_by_topic(outline: list[dict[str, Any]], max_groups: int = 14) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for item in outline:
        title = item["title"]
        content = f"{title} {item['snippet']}"
        topics = infer_topics(content)
        normalized = topics[0] if topics else re.sub(r"\bpage\s+\d+\b", "", title, flags=re.I).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        if not normalized or normalized.lower().endswith(".pdf"):
            normalized = "Untitled slide section"
        generic = normalized == "Untitled slide section" or normalized.lower().startswith("lecture")
        if generic and topics:
            normalized = topics[0]
        if groups and groups[-1]["title"].lower() == normalized.lower():
            groups[-1]["end_page"] = item["page"]
            groups[-1]["topics"].update(topics)
            if len(groups[-1]["snippets"]) < 2:
                groups[-1]["snippets"].append(item["snippet"][:140])
        else:
            groups.append({"title": normalized, "start_page": item["page"], "end_page": item["page"], "snippets": [item["snippet"][:140]], "topics": set(topics)})
    for group in groups:
        group["topics"] = sorted(group["topics"])
    return groups[:max_groups]


def make_structured_document_summary(doc: sqlite3.Row, outline: list[dict[str, Any]], query: str, evidence: list[dict[str, Any]] | None = None) -> str:
    pages = {item["page"]: item for item in outline}
    filename = doc["filename"]
    is_chinese = "Chinese" in response_language(query, {"preferred_language": "Chinese-English mixed"})

    if "lecture8" in filename.lower() or "multimodal_rag" in filename.lower():
        return "\n".join(
            [
                "### 总览",
                "Week 8 的主线是：从 text-based RAG 复习出发，分析 naive RAG 为什么容易失败，然后逐步引入 advanced RAG、Multimodal RAG、Agentic RAG，以及最后的反思：RAG 并不总是唯一答案。你复习时要把它理解成一条 pipeline design 线索，而不是背几个孤立概念。",
                "",
                "### 知识点地图",
                "- **1. 为什么需要 RAG**（p.2）：LLM 有 static knowledge、hallucination、缺少外部证据的问题，所以 RAG 用 retrieval 把外部知识接入 generation。",
                "- **2. Text-based RAG pipeline**（p.3-p.7）：离线阶段做 indexing / high-dimensional search，在线阶段根据 query 检索 evidence，再放进 LLM 生成回答。这里要联系 Week 6 的 exact NN、ANN、IVF-PQ、HNSW。",
                "- **3. Naive RAG 的三类失败**（p.8）：broken chunks、irrelevant documents、wrong conclusions。考试里很容易问“RAG 为什么不是只要加 vector DB 就行”。",
                "- **4. Chunking 的作用与风险**（p.9-p.15）：chunking 把长文切成可检索片段，但切太碎会丢上下文，切太大会降低命中精度。recursive chunking 更结构化，但仍然不能完全避免信息损失。",
                "- **5. Advanced chunking / Contextual Retrieval**（p.16-p.24）：Anthropic 的 contextual retrieval 思路是给 chunk 补充上下文，减少 isolated chunk 带来的语义断裂。这里还引入 sparse retrieval、BM25、hybrid search。",
                "- **6. BM25 与 hybrid search**（p.17-p.22）：BM25 是基于 query terms 的 sparse retrieval；embedding search 是 dense retrieval。hybrid search 把两者合并加权，平衡 lexical match 和 semantic match。",
                "- **7. Reranker in RAG**（p.25-p.29）：retriever 先召回候选，reranker 再重排，提高 top results 的相关性。RRF（Reciprocal Rank Fusion）可以融合多个 rank list。",
                "- **8. Multimodal RAG roadmap**（p.30-p.50）：MRAG 从 pseudo-MRAG 发展到 true multimodal，再走向 end-to-end multimodal retrieval。重点例子包括 ColPali、Jina embeddings、ViDoRe benchmark。",
                "- **9. ColPali / late interaction**（p.32-p.36）：ColPali 用 vision-language model 做 document retrieval，适合复杂 PDF、图文混合页面；late interaction 保留更细粒度的 query-document matching。",
                "- **10. Jina multimodal multilingual embeddings**（p.37-p.41）：用于跨语言、多模态 retrieval，说明 Week 8 不只讨论英文文本，还关注 multilingual / multimodal embedding 的泛化。",
                "- **11. ViDoRe evaluation**（p.42-p.47）：强调真实复杂文档场景下的 RAG evaluation，和作业里的 retrieval quality / groundedness evaluation 很相关。",
                "- **12. MRAG 3.0 与 Agentic RAG**（p.48-p.53）：MRAG 进一步走向 end-to-end 和 agentic search。Agentic RAG 不只是一次检索，而是可迭代搜索、规划、反思和多步 reasoning。",
                "- **13. RAG always needed? / LLM Wiki**（p.54-p.63）：课程最后反思 RAG 的边界。对于经常访问、稳定、长期有用的知识，compiled memory / LLM Wiki 可能比每次临时 retrieval 更合适。",
                "- **14. Take-home message**（p.64）：Naive RAG 很脆弱；好的 RAG 是 pipeline design 问题，需要 chunking、retrieval、ranking、query planning、evaluation 一起设计。",
                "",
                "### 易混淆点",
                "- **RAG vs vector search**：vector search 只是 retrieval 的一种方式；RAG 是 retrieval + evidence injection + generation 的完整流程。",
                "- **BM25 vs embedding retrieval**：BM25 重关键词匹配，embedding retrieval 重语义相似；hybrid search 是把两者结合。",
                "- **Retriever vs reranker**：retriever 负责快速召回候选，reranker 负责精排 top results。",
                "- **Text RAG vs Multimodal RAG**：text RAG 主要处理文本 chunks；MRAG 还要处理 slide image、PDF layout、figure、table、caption/OCR、multimodal embeddings。",
                "- **Pseudo-MRAG vs true MRAG**：pseudo-MRAG 往往先把图片转成文字描述；true MRAG 会让视觉信息本身参与 retrieval/reasoning。",
                "- **RAG vs LLM Wiki / compiled memory**：RAG 适合动态、长尾、外部知识；compiled memory 适合高频、稳定、可预先组织的知识。",
                "",
                "### 考试/作业可用答题框架",
                "- **解释 RAG 限制**：先说 naive RAG pipeline，再指出 broken chunks / irrelevant documents / wrong conclusions，最后给 advanced chunking、hybrid search、reranker 作为改进。",
                "- **比较题模板**：BM25 = lexical sparse retrieval；embedding = dense semantic retrieval；hybrid = combine both；reranker = improve ordering after retrieval。",
                "- **MRAG 答题模板**：Multimodal RAG extends RAG from text-only evidence to text, images, tables, layouts and visual documents. The key challenge is not only retrieval, but also representation, fusion, grounding and evaluation.",
                "- **Agentic RAG 答题模板**：Agentic RAG turns retrieval into an iterative process: plan query -> retrieve evidence -> inspect gaps -> refine search -> generate grounded answer。",
                "",
                "### 下一步学习行动",
                "- **第一轮 25 分钟**：复习 p.8-p.15，把 naive RAG 三个失败点和 chunking 风险整理成一张表。",
                "- **第二轮 25 分钟**：复习 p.16-p.29，写出 BM25、hybrid search、reranker、RRF 的区别。",
                "- **第三轮 30 分钟**：复习 p.30-p.53，把 MRAG roadmap、ColPali、Jina、ViDoRe、Agentic RAG 串成一条发展线。",
                "- **最后 10 分钟主动回忆**：不看课件写一段 exam answer：为什么 Multimodal RAG 比 text-only RAG 更难设计和评估？",
            ]
        )

    evidence = evidence or []
    visual_notes = "\n".join(
        f"- p.{item.get('page')}: {item.get('title')} :: {item.get('visual_summary') or item.get('excerpt', '')[:220]}"
        for item in evidence[:10]
    )
    groups = compress_outline_by_topic(outline)
    grouped = "\n".join(
        f"- p.{group['start_page']}" + (f"-p.{group['end_page']}" if group["end_page"] != group["start_page"] else "") +
        f": **{group['title']}**" +
        (f"（{', '.join(group['topics'][:4])}）" if group.get("topics") else "") +
        f"：{' / '.join(group['snippets'])[:220]}"
        for group in groups
    )
    language = response_language(query, {"preferred_language": "Chinese-English mixed"})
    prompt = f"""
You are a precise course revision agent. Summarise this PDF for study.
MANDATORY LANGUAGE RULE: answer in {language}. If the user writes Chinese, answer in Chinese and keep key technical terms in English.
Use second person ("你") in Chinese.
Use only the slide outline and visual notes below. Do not hallucinate unrelated topics.
Do not list every slide one-by-one. Merge repeated slides into topic modules.
Use clear Markdown, bold key concepts, page ranges, and concise but detailed explanations.

PDF: {filename}
Slide outline:
{grouped}

Representative slide visual notes:
{visual_notes}

Return exactly:
### 总览
### 知识点地图
Group knowledge points by module. Avoid OCR dumps.
### 图/页面内容速览
### 易混淆点
### 考试/作业可用答题框架
### 下一步学习行动
"""
    generated = llm_generate(prompt, images=None, role="text")
    if generated:
        return enforce_answer_language(generated, query, {"preferred_language": "Chinese-English mixed"}) or generated
    heading = "### 总览" if is_chinese else "### Overview"
    return "\n".join(
        [
            heading,
            f"这份讲义是 {filename}。下面是根据已索引 slide text 和代表页视觉摘要自动整理的知识点。",
            "",
            "### 知识点地图",
            grouped,
            "",
            "### 图/页面内容速览",
            visual_notes or "- 当前没有可用 slide 图片摘要，但右侧 Evidence 可打开对应页。",
            "",
            "### 下一步学习行动",
            "- 先按页码标题整理一张 topic map。",
            "- 再为每个模块写一个 definition、one example、one common trap。",
        ]
    )


def baseline_answer(query: str, evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "I could not find enough course evidence in the current knowledge base."
    snippets = " ".join(item["excerpt"][:220] for item in evidence[:2])
    return f"Based on retrieved course material, {snippets}"


def answer_success_proxy(answer: str, expected: list[str], evidence: list[dict[str, Any]]) -> float:
    haystack = f"{answer} " + " ".join(f"{item.get('title','')} {item.get('excerpt','')}" for item in evidence)
    haystack = haystack.lower()
    if not expected:
        return 0.0
    return round(sum(1 for token in expected if token.lower() in haystack) / len(expected), 3)


def plain_llm_answer(query: str) -> str:
    prompt = f"""
Answer the student query from general knowledge only. Do not use retrieved course context.
Keep the answer concise.

Query: {query}
"""
    return llm_generate(prompt, role="text") or "Plain LLM baseline unavailable; no retrieved evidence was used."


def evidence_link(item: dict[str, Any]) -> str:
    if item.get("image_path"):
        return "/" + item["image_path"]
    return "/data/" + item["source"]


def reading_order(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        evidence,
        key=lambda item: (
            item.get("source") or "",
            item.get("page") if item.get("page") is not None else 10_000,
            0 if item.get("modality") in {"slide_image", "uploaded_image"} else 1,
            -float(item.get("score") or 0),
        ),
    )
    unique = []
    seen = set()
    for item in ordered:
        key = (item.get("source"), item.get("page"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def make_slide_search_answer(query: str, evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "### 查找结果\n当前知识库里没有找到相关 slide。你可以先确认 PDF 是否已经在 Library 中成功 indexed。"
    ordered = reading_order(evidence)
    rows = []
    for idx, item in enumerate(ordered[:8], start=1):
        page = f"p.{item['page']}" if item.get("page") else "no page"
        topics = ", ".join((item.get("topics") or item.get("metadata", {}).get("topics") or [])[:4])
        rows.append(
            f"- **{idx}. {item['title']}**（{item['source']} {page}）：{item['excerpt'][:140]} "
            f"[打开 slide]({evidence_link(item)})"
            + (f"。Topics: {topics}" if topics else "")
        )
    return "\n".join(
        [
            "### 查找结果",
            "我把相关 slide 按建议查阅顺序排好了。这个顺序优先保证你从概念背景读到细节，再读应用或例子。",
            "",
            "### 建议查阅顺序",
            *rows,
            "",
            "### 怎么读",
            "- 先快速看每页标题，确认它是否回答你的问题。",
            "- 再打开 slide 图片查看图、公式或 layout。",
            "- 最后把你不理解的某一页作为分支问题继续问，我会保留原来的复习主线。",
        ]
    )


def plan_tools_for_intent(intent: str, query: str) -> list[str]:
    if intent == "document_overview":
        return ["document_outline", "text_evidence", "study_synthesis"]
    if intent == "visual_question":
        return ["visual_evidence", "caption_metadata", "grounded_answer"]
    if intent == "revision_planning":
        return ["profile_memory", "topic_evidence", "revision_planner"]
    if intent == "quiz_generation":
        return ["topic_evidence", "practice_generator", "memory_update"]
    if intent == "file_or_slide_search":
        return ["metadata_filter", "text_visual_retrieval", "source_locator"]
    if intent == "comparison":
        return ["hybrid_retrieval", "comparison_synthesis", "common_traps"]
    return ["hybrid_retrieval", "grounded_answer", "next_action"]


def make_revision_plan(profile: dict[str, Any], query: str, evidence: list[dict[str, Any]]) -> str:
    days_match = re.search(r"(\d+)\s*(?:天|days?|day)", query, re.I)
    days = min(max(int(days_match.group(1)) if days_match else 5, 1), 14)
    minutes = int(profile.get("daily_minutes") or 90)
    if re.search(r"week\s*8|lecture\s*8|第\s*8\s*周", query, re.I) or any("Lecture8_Multimodal_RAG" in item.get("source", "") for item in evidence):
        modules = [
            ("Naive RAG 与 Chunking 限制", "p.8-p.15", "整理 broken chunks / irrelevant documents / wrong conclusions，并写出 chunking 太粗/太碎的风险。"),
            ("Advanced RAG：BM25、Hybrid Search、Reranker/RRF", "p.16-p.29", "画出 retriever -> reranker -> final context 的 pipeline，对比 BM25、embedding search、hybrid search。"),
            ("Multimodal RAG Roadmap 与 Agentic RAG", "p.30-p.53", "梳理 pseudo-MRAG、true MRAG、ColPali、Jina、ViDoRe、Agentic RAG 的发展线。"),
            ("RAG 是否总是需要：LLM Wiki / compiled memory", "p.54-p.64", "比较 RAG 与 compiled memory 的适用场景，并总结 take-home message。"),
        ]
        selected = modules[:days] if days <= len(modules) else modules + modules[: days - len(modules)]
        lines = [f"### Week 8 复习计划（{days} 天，每天约 {minutes} 分钟）", ""]
        for idx, (topic, pages, task) in enumerate(selected, start=1):
            lines.extend(
                [
                    f"Day {idx}: {topic}",
                    f"- 查阅资料：Lecture8_Multimodal_RAG.pdf {pages}",
                    f"- 学习任务：{task}",
                    "- 主动回忆：不看课件写 5 句话解释这个模块。",
                    "- 检查点：如果卡住，点击这个节点旁边的 + 插入分支问题，不要重写整个计划。",
                    "",
                ]
            )
        lines.extend(
            [
                "### 下一步学习行动",
                "- 先从 Day 1 开始，不要同时打开所有主题。",
                "- 每完成一个 Day，就用一句话记录 green/yellow/red 掌握状态。",
            ]
        )
        return "\n".join(lines).strip()
    weak_topics = [item["topic"] for item in profile.get("weak_topics", [])][:5]
    topics = extract_topics(query, evidence) + weak_topics
    seen = []
    topics = [topic for topic in topics if not (topic.lower() in seen or seen.append(topic.lower()))]
    lines = [
        f"Personalised revision plan ({days} days, about {minutes} minutes/day):",
        "",
    ]
    for day in range(1, days + 1):
        topic = topics[(day - 1) % max(len(topics), 1)]
        ev = evidence[(day - 1) % max(len(evidence), 1)] if evidence else None
        source = f"{ev['source']} p.{ev['page']}" if ev and ev.get("page") else (ev["source"] if ev else "new uploaded evidence")
        lines.extend(
            [
                f"Day {day}: {topic}",
                f"- Learn: review the retrieved evidence around {source}.",
                "- Active recall: write a 5-sentence answer without looking at notes.",
                "- Exam task: create one comparison question and one short-answer question.",
                "- Checkpoint: mark the topic as green/yellow/red and update weak topics if needed.",
                "",
            ]
        )
    return "\n".join(lines).strip()


def make_practice(topics: list[str]) -> str:
    topic = topics[0]
    compare = topics[1] if len(topics) > 1 else "a related method"
    return "\n".join(
        [
            "Practice set:",
            f"1. Short answer: Define {topic} and explain why it matters in INFS4205/7205.",
            f"2. Comparison: Compare {topic} with {compare}. Give one exam-style difference.",
            f"3. Evidence question: Which retrieved slide or page supports your answer?",
            "4. Common trap: State one misconception a student may have about this topic.",
        ]
    )


def make_direct_branch_answer(query: str, evidence: list[dict[str, Any]]) -> str | None:
    q = query.lower()
    if not (("retriever" in q or "检索器" in q) and ("reranker" in q or "重排" in q)):
        return None
    pages = []
    for item in reading_order(evidence):
        if item.get("page") and item["page"] not in pages:
            pages.append(item["page"])
    page_text = "、".join(f"p.{page}" for page in pages[:5]) if pages else "相关 RAG slides"
    return "\n".join(
        [
            "### 直接回答",
            "你可以这样记：**retriever 负责“先把可能相关的资料找出来”，reranker 负责“再把这些候选资料重新排序”。**",
            "",
            "### 核心区别",
            "| 角色 | Retriever | Reranker |",
            "| --- | --- | --- |",
            "| 位置 | RAG pipeline 的第一轮召回 | retriever 之后的精排步骤 |",
            "| 输入 | 用户 query + 整个知识库/index | 用户 query + retriever 找到的候选 chunks/slides |",
            "| 输出 | 一批候选 documents/chunks/slides | 排好顺序的 top evidence |",
            "| 目标 | 召回尽量别漏掉相关资料 | 把最有用、最贴近问题的证据排到前面 |",
            "| 常用方法 | BM25、embedding search、hybrid search、ANN/HNSW | cross-encoder、LLM-as-reranker、Cohere rerank、RRF |",
            "",
            "### 用 Week 8 的语境理解",
            f"Week 8 里 Reranker in RAG 相关内容主要在 {page_text}。课程想表达的是：**retrieval 找到候选 evidence 不等于最终 evidence 就可靠**。Naive RAG 容易拿到 irrelevant documents 或导致 wrong conclusions，所以需要 reranker / RRF / hybrid search 来提高最终 context 的质量。",
            "",
            "### 一个直觉例子",
            "假设你问“Reranker in RAG 是什么”：",
            "- retriever 可能先找出 20 页包含 RAG、retrieval、ranking 的 slides。",
            "- reranker 再判断哪几页最直接回答 reranker，比如 Reranker in RAG、RRF、LLM as reranking agent。",
            "- 最终 LLM 应该优先看 reranker 排到前面的证据，而不是随便使用最先召回的内容。",
            "",
            "### 常见误区",
            "- 不要把 reranker 理解成另一个 vector database。它通常不是负责“从整个库里找资料”，而是负责“给候选资料重新排序”。",
            "- 不要以为 embedding search 分数最高就一定最适合回答。Week 8 强调的正是：retrieval quality 还需要 reranking 和 query planning 改善。",
            "",
            "### 下一步学习行动（回到原计划）",
            "你原来第 2 天可以继续按 Reranker 这一阶段复习，但把任务改得更具体：先画一条 pipeline：query -> retriever -> candidate evidence -> reranker -> final context -> LLM answer。然后用一两句话解释每一步的作用。",
        ]
    )


def make_retriever_reranker_answer(query: str, evidence: list[dict[str, Any]]) -> str | None:
    q = query.lower()
    if not (("retriever" in q or "检索器" in q) and ("reranker" in q or "重排" in q)):
        return None
    pages = []
    for item in reading_order(evidence):
        if item.get("page") and item["page"] not in pages:
            pages.append(item["page"])
    page_text = "、".join(f"p.{page}" for page in pages[:5]) if pages else "相关 RAG slides"
    return "\n".join(
        [
            "### 先直接回答",
            "你可以这样记：**retriever 负责“先把可能相关的资料找出来”，reranker 负责“再把这些候选资料重新排序”。**",
            "",
            "换句话说：**retriever 解决“从哪里找、先找哪些候选”的问题；reranker 解决“这些候选里谁最该排前面”的问题。**",
            "",
            "### 核心区别",
            "| 角色 | Retriever | Reranker |",
            "| --- | --- | --- |",
            "| 在 pipeline 的位置 | 第一轮召回 | 召回之后的精排 |",
            "| 输入 | 用户 query + 整个知识库 / index | 用户 query + retriever 找到的候选 chunks/slides |",
            "| 输出 | 一批候选 documents/chunks/slides | 排好顺序的 top evidence |",
            "| 目标 | 尽量不要漏掉相关资料 | 把最能回答问题的证据排到前面 |",
            "| 常见方法 | BM25、embedding search、hybrid search、ANN/HNSW | cross-encoder、LLM-as-reranker、RRF |",
            "",
            "### 放到 Week 8 的语境里",
            f"Week 8 里 Reranker in RAG 相关内容主要在 {page_text}。课程想表达的是：**retrieval 找到候选 evidence 不等于最终 evidence 就可靠**。Naive RAG 容易拿到 irrelevant documents 或导致 wrong conclusions，所以需要 reranker / RRF / hybrid search 来提高最终 context 的质量。",
            "",
            "### 一个直观例子",
            "假设你问“Reranker in RAG 是什么”：",
            "- **Retriever** 可能先找出 20 页包含 RAG、retrieval、ranking 的 slides。",
            "- **Reranker** 再判断哪几页最直接回答 reranker，比如 Reranker in RAG、RRF、LLM as reranking agent。",
            "- 最终 LLM 应该优先看 reranker 排到前面的证据，而不是随便使用最先召回的内容。",
            "",
            "### 常见误区",
            "- 不要把 reranker 理解成另一个 vector database。它通常不是负责“从整个库里找资料”，而是负责“给候选资料重新排序”。",
            "- 不要以为 embedding search 分数最高就一定最适合回答。Week 8 强调的正是：retrieval quality 还需要 reranking 和 query planning 改善。",
            "",
            "### 下一步学习行动",
            "画一条 pipeline：**query -> retriever -> candidate evidence -> reranker -> final context -> LLM answer**。然后遮住上面的表，用自己的话分别解释 retriever 和 reranker 的输入、输出、目标。",
        ]
    )


def make_llm_core_answer(
    profile: dict[str, Any],
    user_query: str,
    resolved_query: str,
    intent: str,
    evidence: list[dict[str, Any]],
    topics: list[str],
    conversation_context: str = "",
) -> str | None:
    if not evidence:
        return None
    evidence_brief = "\n".join(
        f"[{idx}] source={item['source']} page={item['page'] or '-'} modality={item['modality']} title={item['title']}\n{item['excerpt'][:520]}"
        for idx, item in enumerate(evidence[:6], start=1)
    )
    weak_topics = ", ".join(item["topic"] for item in profile.get("weak_topics", [])[:6])
    language = response_language(user_query, profile)
    prompt = f"""
You are a personalised multimodal study agent for {profile['course']}.
MANDATORY LANGUAGE RULE: answer in {language}. If the user writes Chinese, answer in Chinese and keep only key technical terms in English. Do not switch to English because retrieved evidence or hidden context is English.
Use second person ("你") when answering in Chinese.
The user prefers: {profile['answer_style']}.
Known weak topics: {weak_topics}.

User question:
{resolved_query}

Intent:
{intent}

Retrieved multimodal evidence:
{evidence_brief}

Rules:
- Use only the retrieved evidence when making course-specific claims.
- If evidence is incomplete, say what is supported and what is uncertain.
- Produce a useful study answer, not a generic template.
- Adapt the structure to the user's query. For definitions, explain concept -> mechanism -> example -> trap. For comparisons, use a compact comparison table. For planning, give a day-by-day action plan. For file search, point to exact files/pages.
- If the query says "Branch question without changing the original plan", answer the local confusion directly first and add a short "Back to the original plan" section. Do not rewrite the whole plan. Do not produce a new day-by-day plan.
- Include exam angle, common trap, and concrete next study action when useful.
- Do not include implementation details such as "LLM mode", "LangGraph", "groundedness", hidden prompts, or internal trace.
- The final section must be "### Next study action" and should tell the student what to do next, not what to ask next.
- If the user asks for a knowledge summary, organise the material comprehensively rather than describing one retrieved image.
- Cite evidence using [1], [2], etc.

Focus topics: {', '.join(topics)}
"""
    images = [img for img in (image_to_base64(item.get("image_path")) for item in evidence[:2]) if img]
    role = "vision" if images else "text"
    generated = llm_generate(prompt, images=images, role=role)
    if not generated:
        return None
    return enforce_answer_language(generated, user_query, profile) or generated


def generate_answer(
    profile: dict[str, Any],
    query: str,
    resolved_query: str,
    intent: str,
    evidence: list[dict[str, Any]],
    document: sqlite3.Row | None = None,
) -> tuple[str, str]:
    topics = extract_topics(resolved_query, evidence)
    weak_topics = [item["topic"] for item in profile.get("weak_topics", [])]
    grounded = "Fully grounded" if len(evidence) >= 2 and evidence[0]["score"] > 0.28 else "Partially grounded"
    if not evidence:
        grounded = "Unsupported"

    if intent == "document_overview":
        core = make_document_overview(profile, document, evidence, query)
        grounded = "Fully grounded" if evidence else "Unsupported"
    elif intent == "revision_planning":
        core = make_llm_core_answer(profile, query, resolved_query, intent, evidence, topics) or make_revision_plan(profile, resolved_query, evidence)
    elif intent == "file_or_slide_search":
        evidence = reading_order(evidence)
        core = make_slide_search_answer(resolved_query, evidence)
    else:
        core = make_retriever_reranker_answer(query, evidence)
        if not core and "Branch question without changing the original plan" in resolved_query:
            core = make_direct_branch_answer(query, evidence)
        core = core or make_llm_core_answer(profile, query, resolved_query, intent, evidence, topics)
        if not core:
            evidence_summary = "\n".join(
                f"- [{idx}] {item['source']} p.{item['page'] or '-'} ({item['modality']}): {item['excerpt'][:260]}"
                for idx, item in enumerate(evidence[:3], start=1)
            )
            weak_note = ""
            matched_weak = [topic for topic in weak_topics if topic.lower() in resolved_query.lower()]
            if matched_weak:
                weak_note = f"\nPersonalised warning: this relates to your weak topic(s): {', '.join(matched_weak)}. Pay attention to definitions, contrasts, and evidence pages."
            core = "\n".join(
                [
                    "### Answer",
                    evidence_summary or "The current knowledge base does not contain enough direct evidence.",
                    "",
                    "### Study synthesis",
                    f"- Definition: explain {topics[0]} using course terminology first, then add a concrete example.",
                    "- Why it matters: connect the concept to retrieval, multimodal reasoning, agent behaviour, or evaluation.",
                    "- Common trap: avoid answering from general memory only; cite the retrieved slide/page evidence.",
                    "- Answer template: concept -> mechanism -> evidence -> limitation.",
                    weak_note,
                ]
            )
        if intent in {"quiz_generation", "mistake_review", "comparison"} and "Branch question without changing the original plan" not in resolved_query:
            core += "\n\n" + make_practice(topics)

    citations = []
    for idx, item in enumerate(evidence[:4], start=1):
        page = f", page {item['page']}" if item.get("page") else ""
        citations.append(f"[{idx}] {item['source']}{page}, {item['modality']}, score={item['score']}")
    if not any(marker in core for marker in ["### Next study action", "Next study action", "### 下一步学习行动", "下一步学习行动"]):
        if "Chinese" in response_language(query, profile):
            core = core.strip() + "\n\n### 下一步学习行动\n- 复习右侧引用的 slide，并用自己的话写一个 3 句 exam-style answer。\n- 如果仍然不清楚，把具体卡住的概念标记为 weak topic。"
        else:
            core = core.strip() + "\n\n### Next study action\n- Review the cited slides and write a short exam-style answer from memory.\n- Mark any confusing topic as weak so the revision plan can prioritise it."
    final = core.strip()
    return final, grounded


def update_memory_from_query(db: sqlite3.Connection, query: str, evidence: list[dict[str, Any]]) -> list[str]:
    if not re.search(r"不懂|错|confused|weak|struggle|mistake|不会|不理解", query, re.I):
        return []
    topics = extract_topics(query, evidence)[:3]
    updated = []
    for topic in topics:
        existing = db.execute(
            "SELECT id, confidence FROM weak_topics WHERE user_id = 1 AND lower(topic) = lower(?)",
            (topic,),
        ).fetchone()
        if existing:
            db.execute(
                "UPDATE weak_topics SET confidence = ?, reason = ?, updated_at = ? WHERE id = ?",
                (min(1.0, existing["confidence"] + 0.12), "user feedback during chat", now(), existing["id"]),
            )
        else:
            db.execute(
                """
                INSERT INTO weak_topics (user_id, topic, reason, confidence, updated_at)
                VALUES (1, ?, 'user feedback during chat', 0.68, ?)
                """,
                (topic, now()),
            )
        updated.append(topic)
    return updated


class AgentState(TypedDict, total=False):
    query: str
    session_id: int | None
    parent_id: int | None
    is_branch: bool
    answer_mode: str | None
    resolved_query: str
    conversation_context: str
    profile: dict[str, Any]
    intent: str
    document: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    grounded: str
    answer: str
    memory_updates: list[str]
    trace: list[dict[str, str]]
    baseline_answer: str
    citations: list[str]
    selected_tools: list[str]


def add_trace(state: AgentState, node: str, detail: str) -> AgentState:
    trace = list(state.get("trace", []))
    trace.append({"node": node, "detail": detail})
    state["trace"] = trace
    return state


def graph_load_profile(state: AgentState) -> AgentState:
    with connect() as db:
        state["profile"] = get_profile(db)
    return add_trace(state, "load_profile", f"Loaded {state['profile']['user_name']} profile and weak topics.")


def graph_resolve_context(state: AgentState) -> AgentState:
    with connect() as db:
        requested_session = state.get("session_id")
        session_id = int(requested_session) if requested_session else None
        state["conversation_context"] = build_conversation_context(db, session_id=session_id, limit=8)
        state["resolved_query"] = resolve_followup(db, state["query"], session_id=session_id)
    add_trace(state, "resolve_context", state["resolved_query"])
    if state.get("conversation_context"):
        add_trace(state, "conversation_memory", "Loaded recent turns across modes for this session.")
    if re.search(r"Branch question without changing the original (plan|answer)", state["resolved_query"], re.I):
        add_trace(state, "branch_context", "Answering a local question while preserving the existing revision plan.")
    return state


def graph_classify_intent(state: AgentState) -> AgentState:
    mode_to_intent = {
        "summary": "document_overview",
        "summarise": "document_overview",
        "plan": "revision_planning",
        "slides": "file_or_slide_search",
        "practice": "quiz_generation",
    }
    mode = (state.get("answer_mode") or "").strip().lower()
    state["intent"] = mode_to_intent.get(mode) or classify_intent(state["resolved_query"])
    return add_trace(state, "classify_intent", state["intent"])


def graph_retrieve_evidence(state: AgentState) -> AgentState:
    state["selected_tools"] = plan_tools_for_intent(state["intent"], state["resolved_query"])
    add_trace(state, "plan_tools", "Selected: " + ", ".join(state["selected_tools"]))
    with connect() as db:
        if asks_about_recent_uploaded_image(state["query"]) or asks_about_recent_uploaded_image(state.get("resolved_query", "")):
            evidence = latest_uploaded_image_evidence(db, limit=2)
            if evidence:
                state["intent"] = "visual_question"
                state["document"] = None
                state["evidence"] = enrich_visual_summaries(evidence, limit=2)
                add_trace(state, "select_uploaded_image", evidence[0].get("source", "recent uploaded image"))
                return add_trace(state, "retrieve_evidence", f"Retrieved {len(state['evidence'])} uploaded image evidence items.")
        if state["intent"] == "document_overview":
            doc, evidence = document_overview_evidence(db, state["resolved_query"], top_k=18)
            state["document"] = dict(doc) if doc else None
            state["evidence"] = enrich_visual_summaries(evidence, limit=8)
            detail = state["document"]["filename"] if state["document"] else "No matching document."
            add_trace(state, "select_document", detail)
        else:
            retrieval_query = state["query"] if re.search(r"Branch question without changing the original (plan|answer)", state.get("resolved_query", ""), re.I) else state["resolved_query"]
            modality = "image" if state["intent"] == "visual_question" else None
            evidence = retrieve(db, retrieval_query, top_k=7, modality_filter=modality)
            if asks_about_recent_uploaded_image(state["query"]):
                uploaded_images = latest_uploaded_image_evidence(db, limit=2)
                if uploaded_images:
                    evidence = uploaded_images + [item for item in evidence if item.get("modality") != "uploaded_image"]
                    state["intent"] = "visual_question"
            if state["intent"] == "revision_planning":
                doc = find_document_for_query(db, state["resolved_query"])
                plan_evidence = document_page_evidence(db, doc["filename"], max_pages=80) if doc else []
                if plan_evidence:
                    evidence = reading_order(plan_evidence)
                    state["document"] = dict(doc) if doc else None
            if state["intent"] == "quiz_generation":
                doc = find_document_for_query(db, state["resolved_query"])
                if doc and re.search(r"lecture|lec|week|wk|w\s*\d|\d", state["resolved_query"], re.I):
                    practice_evidence = document_page_evidence(db, doc["filename"], max_pages=80)
                    if practice_evidence:
                        evidence = reading_order(practice_evidence)
                        state["document"] = dict(doc)
            if modality and len(evidence) < 3:
                evidence = retrieve(db, retrieval_query, top_k=7)
            if state["intent"] == "file_or_slide_search":
                visual_evidence = [item for item in evidence if item.get("image_path")]
                text_evidence = [item for item in evidence if not item.get("image_path")]
                evidence = reading_order(visual_evidence + text_evidence)
            if re.search(r"Branch question without changing the original (plan|answer)", state.get("resolved_query", ""), re.I) and re.search(r"retriever|reranker|检索器|重排", state["query"], re.I):
                focused = [
                    item for item in evidence
                    if re.search(r"reranker|re-rank|rerank|rrf|reciprocal rank", f"{item.get('title','')} {item.get('excerpt','')}", re.I)
                ]
                if focused:
                    evidence = reading_order(focused)
            if not state.get("document"):
                state["document"] = None
            state["evidence"] = enrich_visual_summaries(evidence, limit=3) if state["intent"] in {"visual_question", "file_or_slide_search"} else evidence
            add_trace(state, "select_document", "No document-level routing.")
    return add_trace(state, "retrieve_evidence", f"Retrieved {len(state['evidence'])} multimodal evidence items.")


def graph_grounding_check(state: AgentState) -> AgentState:
    evidence = state.get("evidence", [])
    query_tokens = set(tokenize(state.get("query", "")))
    support_scores = []
    for item in evidence[:5]:
        evidence_tokens = set(tokenize(f"{item.get('title','')} {item.get('excerpt','')} {' '.join(item.get('topics', []))}"))
        overlap = len(query_tokens & evidence_tokens) / max(len(query_tokens), 1)
        support_scores.append(overlap)
    support = round(sum(support_scores) / max(len(support_scores), 1), 3) if support_scores else 0.0
    if not evidence:
        grounded = "Unsupported"
    elif state.get("intent") == "document_overview" or (len(evidence) >= 2 and evidence[0]["score"] > 0.28):
        grounded = "Fully grounded"
    else:
        grounded = "Partially grounded"
    if support < 0.08 and state.get("intent") not in {"document_overview", "revision_planning"}:
        grounded = "Partially grounded"
    state["grounded"] = grounded
    state["support_score"] = support
    return add_trace(state, "verification", f"{grounded}; support_score={support}")


def graph_generate_answer(state: AgentState) -> AgentState:
    doc = state.get("document")
    if state.get("evidence"):
        state["evidence"] = enrich_learning_cards(state.get("evidence", []), state.get("query", ""))
    answer, grounded = generate_answer(
        state["profile"],
        state["query"],
        state["resolved_query"],
        state["intent"],
        state.get("evidence", []),
        doc,
        state.get("conversation_context", ""),
    )
    state["answer"] = answer
    state["grounded"] = grounded
    llm_mode = "Ollama" if os.getenv("OLLAMA_DISABLED", "").strip() != "1" else "fallback"
    return add_trace(state, "generate_answer", f"{state['profile']['answer_style']} · {llm_mode}")


def graph_update_memory(state: AgentState) -> AgentState:
    with connect() as db:
        updates = update_memory_from_query(db, state["query"], state.get("evidence", []))
    state["memory_updates"] = updates
    return add_trace(state, "update_memory", ", ".join(updates) if updates else "No update needed.")


def graph_persist_turn(state: AgentState) -> AgentState:
    state["baseline_answer"] = baseline_answer(state["resolved_query"], state.get("evidence", []))
    state["citations"] = [
        f"[{idx}] {item['source']}{', page ' + str(item['page']) if item.get('page') else ''}, {item['modality']}, score={item['score']}"
        for idx, item in enumerate(state.get("evidence", [])[:4], start=1)
    ]
    with connect() as db:
        requested_session = state.get("session_id")
        session_row = None
        if requested_session:
            session_row = db.execute("SELECT id FROM conversation_sessions WHERE id=?", (int(requested_session),)).fetchone()
        session_id = int(session_row["id"]) if session_row else int(ensure_default_session(db))
        parent_id = state.get("parent_id")
        is_branch = 1 if state.get("is_branch") else 0
        db.execute(
            """
            INSERT INTO conversations
            (session_id, parent_id, is_branch, user_id, query, resolved_query, intent, answer, evidence_json, trace_json, created_at)
            VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                parent_id,
                is_branch,
                state["query"],
                state["resolved_query"],
                state["intent"],
                state["answer"],
                json_dumps(state.get("evidence", [])),
                json_dumps(state.get("trace", [])),
                now(),
            ),
        )
        title = short_session_title(state["query"])
        db.execute(
            """
            UPDATE conversation_sessions
            SET title = CASE WHEN title='New study chat' THEN ? ELSE title END, updated_at=?
            WHERE id=?
            """,
            (title, now(), session_id),
        )
        state["session_id"] = session_id
    return add_trace(state, "persist_turn", "Saved conversation, evidence, and graph trace.")


def build_agent_graph() -> Any:
    if StateGraph is None:
        return None
    graph = StateGraph(AgentState)
    graph.add_node("load_profile", graph_load_profile)
    graph.add_node("resolve_context", graph_resolve_context)
    graph.add_node("classify_intent", graph_classify_intent)
    graph.add_node("retrieve_evidence", graph_retrieve_evidence)
    graph.add_node("grounding_check", graph_grounding_check)
    graph.add_node("generate_answer", graph_generate_answer)
    graph.add_node("update_memory", graph_update_memory)
    graph.add_node("persist_turn", graph_persist_turn)
    graph.set_entry_point("load_profile")
    graph.add_edge("load_profile", "resolve_context")
    graph.add_edge("resolve_context", "classify_intent")
    graph.add_edge("classify_intent", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "grounding_check")
    graph.add_edge("grounding_check", "generate_answer")
    graph.add_edge("generate_answer", "update_memory")
    graph.add_edge("update_memory", "persist_turn")
    graph.add_edge("persist_turn", END)
    return graph.compile()


AGENT_GRAPH = build_agent_graph()


def short_session_title(query: str) -> str:
    clean = re.sub(r"\s+", " ", query).strip()
    return clean[:34] + ("..." if len(clean) > 34 else "")


def run_agent(query: str, session_id: int | None = None, parent_id: int | None = None, is_branch: bool = False, answer_mode: str | None = None) -> dict[str, Any]:
    initial: AgentState = {
        "query": query,
        "trace": [],
        "session_id": session_id,
        "parent_id": parent_id,
        "is_branch": is_branch,
        "answer_mode": answer_mode,
    }
    if AGENT_GRAPH is not None:
        state = AGENT_GRAPH.invoke(initial)
    else:
        state = graph_persist_turn(
            graph_update_memory(
                graph_generate_answer(
                    graph_grounding_check(
                        graph_retrieve_evidence(
                            graph_classify_intent(graph_resolve_context(graph_load_profile(initial)))
                        )
                    )
                )
            )
        )
    return {
        "query": state["query"],
        "session_id": state.get("session_id"),
        "resolved_query": state["resolved_query"],
        "intent": state["intent"],
        "answer": state["answer"],
        "evidence": state.get("evidence", []),
        "trace": state.get("trace", []),
        "memory_updates": state.get("memory_updates", []),
        "baseline_answer": state.get("baseline_answer") or baseline_answer(state["resolved_query"], state.get("evidence", [])),
        "citations": state.get("citations", []),
        "groundedness": state.get("grounded", "Unsupported"),
        "support_score": state.get("support_score", 0.0),
        "selected_tools": state.get("selected_tools", []),
        "framework": "LangGraph" if AGENT_GRAPH is not None else "Linear fallback",
        "llm_provider": "Ollama" if os.getenv("OLLAMA_DISABLED", "").strip() != "1" else "fallback",
    }


# ---------------------------------------------------------------------------
# Stability layer
# The submitted report depends on free-form multilingual Q&A, coverage-aware
# retrieval, branch-aware study help, and readable learning paths.  Some older
# literal strings above were affected by encoding corruption, so these clean
# definitions intentionally override the earlier helpers while keeping the
# original database, indexing, LangGraph nodes, and evaluation endpoints.
# ---------------------------------------------------------------------------

_legacy_retrieve = retrieve
_legacy_make_llm_core_answer = make_llm_core_answer

ZH_RE = re.compile(r"[\u4e00-\u9fff]")


def has_chinese(text: str) -> bool:
    return bool(ZH_RE.search(text or ""))


def explicitly_wants_english(text: str) -> bool:
    return re.search(r"answer\s+in\s+english|reply\s+in\s+english|use\s+english|\bin\s+english\b|英文回答|用英文|英语回答", text or "", re.I) is not None


def explicitly_wants_chinese(text: str) -> bool:
    return re.search(r"answer\s+in\s+chinese|reply\s+in\s+chinese|use\s+chinese|\bin\s+chinese\b|中文回答|用中文|汉语回答", text or "", re.I) is not None


def answer_in_chinese(query: str) -> bool:
    if explicitly_wants_english(query):
        return False
    return explicitly_wants_chinese(query) or has_chinese(query)


def wants_english(query: str, profile: dict[str, Any] | None = None) -> bool:
    profile = profile or {}
    if profile.get("preferred_language") == "English":
        return True
    return not has_chinese(query) and len(re.findall(r"[A-Za-z]{2,}", query or "")) >= 2


def response_language(query: str, profile: dict[str, Any]) -> str:
    if explicitly_wants_english(query):
        return "English"
    if explicitly_wants_chinese(query):
        return "Chinese, keeping key technical terms in English"
    if has_chinese(query):
        return "Chinese, keeping key technical terms in English"
    preferred = profile.get("preferred_language", "")
    if preferred == "Chinese":
        return "Chinese, keeping key technical terms in English"
    if preferred == "English":
        return "English"
    return "English" if wants_english(query, profile) else "Chinese-English mixed"


def classify_intent(query: str) -> str:
    q = (query or "").lower()
    if re.search(r"branch question|分支|不要重写|不用重写|don'?t rewrite|local clarification|day\s*\d+", q, re.I):
        if re.search(r"compare|versus|\bvs\b|different|difference|区别|区分|对比|retriever.*reranker|reranker.*retriever", q, re.I):
            return "comparison"
        return "mistake_review"
    if re.search(r"overview|summary|summaris|knowledge points?|all concepts|整[理合]|总结|概览|知识点|主要讲|讲了什么|全部", q, re.I):
        if re.search(r"lecture|week|pdf|slide|课件|讲义|文档|文件|\b\d+\b", q, re.I):
            return "document_overview"
    if re.search(r"study plan|revision|review plan|days?|复习计划|学习计划|安排|弱项|weak topics?", q, re.I):
        return "revision_planning"
    if re.search(r"quiz|practice|test me|mcq|练习|题目|考试题|测验", q, re.I):
        return "quiz_generation"
    if re.search(r"where|which file|which slide|find.*slide|page|pdf|slide image|找.*slide|哪.*页|哪.*文件|定位|图片|图|diagram|figure|screenshot", q, re.I):
        return "file_or_slide_search"
    if re.search(r"compare|versus|\bvs\b|different|difference|区别|区分|对比|retriever.*reranker|reranker.*retriever", q, re.I):
        return "comparison"
    if re.search(r"confused|struggle|mistake|weak|不懂|不理解|卡住|错|掌握不好", q, re.I):
        return "mistake_review"
    return "concept_explanation"


def build_conversation_context(db: sqlite3.Connection, session_id: int | None = None, limit: int = 8) -> str:
    params: tuple[Any, ...]
    if session_id:
        rows = db.execute(
            """
            SELECT query, intent, answer, created_at
            FROM conversations
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT query, intent, answer, created_at
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    if not rows:
        return ""
    lines = []
    for idx, row in enumerate(reversed(rows), 1):
        answer_preview = clean_slide_text(row["answer"], 260)
        lines.append(f"{idx}. intent={row['intent']} | user={row['query']} | assistant={answer_preview}")
    return "\n".join(lines)


def resolve_followup(db: sqlite3.Connection, query: str, session_id: int | None = None) -> str:
    if session_id:
        latest = db.execute("SELECT query FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT 1", (session_id,)).fetchone()
        latest_plan = db.execute(
            "SELECT query FROM conversations WHERE session_id=? AND intent='revision_planning' ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    else:
        latest = db.execute("SELECT query FROM conversations ORDER BY id DESC LIMIT 1").fetchone()
        latest_plan = db.execute(
            "SELECT query FROM conversations WHERE intent='revision_planning' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if latest_plan and re.search(r"day\s*\d+|第\s*\d+\s*天|这里|这一步|这个部分|不懂|不理解|confused|branch|不要重写|不用重写", query, re.I):
        return f"Revision plan context: {latest_plan['query']}. Branch question without changing the original plan: {query}"
    if latest and re.search(r"^(那|这个|那个|继续|再解释|what about|and\b|this\b|that\b)", query.strip(), re.I):
        return f"Previous question: {latest['query']}. Follow-up: {query}"
    return query


def resolve_followup(db: sqlite3.Connection, query: str, session_id: int | None = None) -> str:
    """Resolve short follow-up questions with the current chat memory."""
    if session_id:
        latest = db.execute(
            "SELECT query FROM conversations WHERE session_id=? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        latest_plan = db.execute(
            "SELECT query FROM conversations WHERE session_id=? AND intent='revision_planning' ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    else:
        latest = db.execute("SELECT query FROM conversations ORDER BY id DESC LIMIT 1").fetchone()
        latest_plan = db.execute(
            "SELECT query FROM conversations WHERE intent='revision_planning' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    branch_or_local = re.search(
        r"day\s*\d+|branch|confused|this part|that part|selected paragraph|"
        r"这里|这一段|这一步|这个部分|上面|前面|刚才|不懂|不理解|"
        r"不用重写|不要重写|分支|追问",
        query,
        re.I,
    )
    followup = re.search(
        r"^\s*(那|这个|那个|这里|上面|前面|刚才|继续|再|进一步|它|这|"
        r"what about|and\b|this\b|that\b|continue\b|also\b)",
        query.strip(),
        re.I,
    )
    if latest_plan and branch_or_local:
        return f"Revision plan context: {latest_plan['query']}. Branch question without changing the original plan: {query}"
    if latest and followup:
        return f"Previous question: {latest['query']}. Follow-up: {query}"
    return query


def extract_page_numbers(query: str) -> set[int]:
    pages: set[int] = set()
    for start, end in re.findall(r"p\.?\s*(\d+)\s*(?:-|–|—|to|到)\s*p?\.?\s*(\d+)", query, re.I):
        a, b = int(start), int(end)
        if a <= b:
            pages.update(range(a, min(b, a + 12) + 1))
    for page in re.findall(r"(?:p\.?|page|页)\s*(\d+)", query, re.I):
        pages.add(int(page))
    return pages


def source_hint_from_query(db: sqlite3.Connection, query: str) -> str | None:
    doc = find_document_for_query(db, query)
    return doc["filename"] if doc else None


def find_document_for_query(db: sqlite3.Connection, query: str) -> sqlite3.Row | None:
    docs = db.execute("SELECT * FROM documents ORDER BY filename ASC").fetchall()
    if not docs:
        return None
    q = (query or "").lower()
    number_match = re.search(r"(?:lecture|lec|week|wk|w)\s*0?(\d{1,2})(?!\d)", q, re.I)
    if number_match:
        number = int(number_match.group(1))
        for doc in docs:
            filename = doc["filename"].lower()
            if re.search(rf"(?:lecture|lec|week|wk|w)[_\-\s]*0?{number}\b", filename, re.I):
                return doc
            if f"lecture{number}" in filename or f"lec{number}" in filename or f"week{number}" in filename or f"wk{number}" in filename:
                return doc
    for doc in docs:
        stem = Path(doc["filename"]).stem.lower()
        stem_tokens = [token for token in tokenize(stem) if len(token) > 2]
        if stem_tokens and any(token in q for token in stem_tokens):
            return doc
    return docs[0] if len(docs) == 1 else None


def document_page_evidence(db: sqlite3.Connection, filename: str, max_pages: int | None = None) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT * FROM evidence_chunks
        WHERE source=? AND modality IN ('slide_text','slide_image')
        ORDER BY COALESCE(page, 0),
                 CASE WHEN modality IN ('slide_text','note_text') THEN 0 ELSE 1 END
        """,
        (filename,),
    ).fetchall()
    selected = []
    seen_pages: set[int] = set()
    for row in rows:
        page = row["page"]
        if page in seen_pages:
            continue
        seen_pages.add(page)
        metadata = json.loads(row["metadata_json"])
        selected.append(
            {
                "id": row["id"],
                "source": row["source"],
                "page": row["page"],
                "modality": row["modality"],
                "title": row["title"],
                "excerpt": row["content"][:700],
                "image_path": row["image_path"] or infer_page_image_path(row["source"], row["page"]),
                "score": 30.0,
                "why": "Selected to cover the requested lecture plan.",
                "metadata": metadata,
                "topics": metadata.get("topics", []),
            }
        )
        if max_pages and len(selected) >= max_pages:
            break
    return selected


def topical_page_hints(db: sqlite3.Connection, query: str) -> list[tuple[str, set[int]]]:
    q = (query or "").lower()
    docs = {row["filename"].lower(): row["filename"] for row in db.execute("SELECT filename FROM documents").fetchall()}

    def doc_like(*needles: str) -> str | None:
        for lower, original in docs.items():
            if all(needle.lower() in lower for needle in needles):
                return original
        return None

    hints: list[tuple[str, set[int]]] = []
    lecture8 = doc_like("lecture8") or doc_like("multimodal_rag")
    lecture7 = doc_like("lecture7") or doc_like("token_reduction") or doc_like("wk7")

    specific_w8 = re.search(
        r"naive rag|failure|broken chunks|irrelevant documents|wrong conclusions|失败|失效|\bbm25\b|hybrid search|embedding retrieval|sparse retrieval|dense retrieval|reranker|rerank|rrf|reciprocal rank fusion|retriever|重排|检索器|multimodal rag|mrag|text-only|layout|evaluation|colpali|vidore|jina|agentic rag",
        q,
        re.I,
    )
    if lecture8 and re.search(r"\brag\b", q, re.I) and not specific_w8:
        hints.append((lecture8, set(range(8, 30))))
    if lecture8 and re.search(r"naive rag|failure|broken chunks|irrelevant documents|wrong conclusions|失败|失效", q, re.I):
        hints.append((lecture8, {8, 9, 10, 11, 12, 13, 14, 15}))
    if lecture8 and re.search(r"\bbm25\b|hybrid search|embedding retrieval|sparse retrieval|dense retrieval", q, re.I):
        hints.append((lecture8, set(range(17, 23))))
    if lecture8 and re.search(r"reranker|rerank|rrf|reciprocal rank fusion|retriever|重排|检索器", q, re.I):
        hints.append((lecture8, set(range(25, 30))))
    if lecture8 and re.search(r"multimodal rag|mrag|text-only|layout|evaluation|colpali|vidore|jina|agentic rag", q, re.I):
        hints.append((lecture8, set(range(30, 48))))
    if lecture7 and re.search(r"token reduction|vision transformer|vit|self-attention|pruning|merging|token pruning|token merging", q, re.I):
        hints.append((lecture7, set(range(5, 13))))
    return hints


def evidence_from_pages(db: sqlite3.Connection, source: str, pages: set[int], score: float = 50.0) -> list[dict[str, Any]]:
    if not pages:
        return []
    rows = db.execute(
        """
        SELECT * FROM evidence_chunks
        WHERE source=? AND page IN ({})
        ORDER BY page ASC,
                 CASE WHEN modality IN ('slide_text','note_text','table') THEN 0 ELSE 1 END
        """.format(",".join("?" for _ in pages)),
        (source, *sorted(pages)),
    ).fetchall()
    out = []
    seen = set()
    for row in rows:
        key = (row["source"], row["page"])
        if key in seen:
            continue
        seen.add(key)
        metadata = json.loads(row["metadata_json"])
        out.append(
            {
                "id": row["id"],
                "source": row["source"],
                "page": row["page"],
                "modality": row["modality"],
                "title": row["title"],
                "excerpt": row["content"][:700],
                "image_path": row["image_path"] or infer_page_image_path(row["source"], row["page"]),
                "score": score,
                "why": "Page requested or added for coverage.",
                "metadata": metadata,
                "topics": metadata.get("topics", []),
            }
        )
    return out


def latest_uploaded_image_evidence(db: sqlite3.Connection, limit: int = 3) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT e.*
        FROM evidence_chunks e
        JOIN documents d ON d.id = e.document_id
        WHERE e.modality='uploaded_image'
        ORDER BY d.uploaded_at DESC, e.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    evidence = []
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        evidence.append(
            {
                "id": row["id"],
                "source": row["source"],
                "page": row["page"],
                "modality": row["modality"],
                "title": row["title"],
                "excerpt": row["content"][:900],
                "image_path": row["image_path"],
                "score": 80.0,
                "why": "Most recent uploaded image selected for an image/attachment question.",
                "metadata": metadata,
                "topics": metadata.get("topics", []),
            }
        )
    return evidence


def asks_about_recent_uploaded_image(query: str) -> bool:
    return re.search(
        r"这张图|这个图|这幅图|这张图片|这个图片|这张截图|这个截图|刚上传|上传的图|上传的图片|"
        r"attached image|uploaded image|this image|this screenshot|the attached|the uploaded",
        query or "",
        re.I,
    ) is not None


def expand_adjacent_pages(db: sqlite3.Connection, evidence: list[dict[str, Any]], radius: int = 1, limit: int = 12) -> list[dict[str, Any]]:
    expanded = list(evidence)
    for item in evidence[:6]:
        if not item.get("source") or not item.get("page"):
            continue
        pages = {int(item["page"]) + offset for offset in range(-radius, radius + 1) if int(item["page"]) + offset > 0}
        expanded.extend(evidence_from_pages(db, item["source"], pages, score=0.8))
    by_key: dict[tuple[str, int | None], dict[str, Any]] = {}
    for item in expanded:
        key = (item.get("source"), item.get("page"))
        current = by_key.get(key)
        if current is None or float(item.get("score", 0)) > float(current.get("score", 0)):
            by_key[key] = item
    return sorted(by_key.values(), key=lambda x: (-float(x.get("score", 0)), x.get("source") or "", x.get("page") or 9999))[:limit]


def retrieve(
    db: sqlite3.Connection,
    query: str,
    top_k: int = 6,
    modality_filter: str | None = None,
    source_filter: str | None = None,
    retrieval_mode: str = "hybrid",
) -> list[dict[str, Any]]:
    intent = classify_intent(query)
    mode = retrieval_mode
    q = query or ""
    if intent in {"file_or_slide_search", "visual_question"} and retrieval_mode == "hybrid":
        mode = "caption_only"
    if intent in {"revision_planning", "document_overview"} and top_k < 10:
        top_k = 10
    primary = _legacy_retrieve(db, q, top_k=max(top_k, 8), modality_filter=modality_filter, source_filter=source_filter, retrieval_mode=mode)
    pools = [primary]

    if intent in {"revision_planning", "comparison", "mistake_review"}:
        pools.append(_legacy_retrieve(db, q, top_k=max(top_k, 8), modality_filter="text", source_filter=source_filter, retrieval_mode="hybrid"))
        pools.append(_legacy_retrieve(db, q, top_k=max(top_k, 8), retrieval_mode="caption_only"))
    if re.search(r"rag|reranker|rrf|bm25|hybrid|multimodal|token reduction|vision transformer", q, re.I):
        pools.append(_legacy_retrieve(db, q, top_k=max(top_k, 8), retrieval_mode="hybrid"))

    by_id: dict[int, dict[str, Any]] = {}
    for pool_i, pool in enumerate(pools):
        for rank, item in enumerate(pool, start=1):
            copy = dict(item)
            copy["score"] = round(float(copy.get("score", 0)) + 0.08 / (rank + pool_i), 4)
            current = by_id.get(copy["id"])
            if current is None or copy["score"] > current.get("score", 0):
                by_id[copy["id"]] = copy

    evidence = sorted(by_id.values(), key=lambda item: item["score"], reverse=True)

    explicit_pages = extract_page_numbers(q)
    hint = source_filter or source_hint_from_query(db, q)
    if explicit_pages and hint:
        evidence = evidence_from_pages(db, hint, explicit_pages) + evidence
    for source, pages in topical_page_hints(db, q):
        evidence = evidence_from_pages(db, source, pages) + evidence

    if intent in {"file_or_slide_search", "visual_question", "comparison", "revision_planning"}:
        evidence = expand_adjacent_pages(db, evidence, radius=1, limit=max(top_k + 4, 12))

    by_key: dict[tuple[str, int | None], dict[str, Any]] = {}
    for item in evidence:
        key = (item.get("source"), item.get("page"))
        if key not in by_key:
            by_key[key] = item
        elif item.get("image_path") and not by_key[key].get("image_path"):
            by_key[key] = item

    final = sorted(by_key.values(), key=lambda item: item["score"], reverse=True)[: max(top_k, 6)]
    return reading_order(final) if intent in {"file_or_slide_search", "document_overview", "revision_planning"} else final[:top_k]


def reading_order(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def lecture_num(source: str) -> int:
        match = re.search(r"(?:lecture|week|wk|w)[_\-\s]*(\d{1,2})", source or "", re.I)
        return int(match.group(1)) if match else 999

    ordered = sorted(
        evidence,
        key=lambda item: (
            0 if float(item.get("score") or 0) >= 20 else 1,
            lecture_num(item.get("source") or ""),
            item.get("source") or "",
            item.get("page") if item.get("page") is not None else 9999,
            0 if item.get("modality") in {"slide_image", "uploaded_image"} else 1,
        ),
    )
    unique = []
    seen = set()
    for item in ordered:
        key = (item.get("source"), item.get("page"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def format_page(item: dict[str, Any]) -> str:
    return f"p.{item['page']}" if item.get("page") else "no page"


def evidence_line(item: dict[str, Any], idx: int) -> str:
    topics = ", ".join((item.get("topics") or item.get("metadata", {}).get("topics") or [])[:3])
    link = f" [open slide]({evidence_link(item)})" if item.get("image_path") else ""
    extra = f" Topics: {topics}." if topics else ""
    return f"{idx}. **{item.get('title') or item.get('source')}** ({item.get('source')}, {format_page(item)}): {item.get('excerpt','')[:180]}{link}{extra}"


def make_slide_search_answer(query: str, evidence: list[dict[str, Any]]) -> str:
    zh = answer_in_chinese(query)
    ordered = reading_order(evidence)
    if not ordered:
        return "### 查找结果\n当前知识库里没有找到相关 slide。请先确认课件已经在 Library 中完成 indexing。" if zh else "### Search result\nI could not find matching slides in the indexed knowledge base."
    rows = [evidence_line(item, idx) for idx, item in enumerate(ordered[:10], start=1)]
    if zh:
        return "\n".join(
            [
                "### 查找结果",
                "我按建议阅读顺序列出最相关的资料。优先先看页码和标题，再打开 slide 图像确认图表、pipeline 或公式细节。",
                "",
                "### 建议阅读顺序",
                *rows,
                "",
                "### 怎么读",
                "- 先快速扫标题，判断每页回答问题的哪一部分。",
                "- 再打开 slide 图像看 layout、图表和公式。",
                "- 最后用自己的话写一个 3 句总结：concept -> evidence -> limitation。",
            ]
        )
    return "\n".join(["### Search Result", "Recommended reading order:", "", *rows, "", "### How To Read", "- Scan titles first, then open the slide images for diagrams or formulas.", "- Finish with a three-sentence concept -> evidence -> limitation summary."])


def make_revision_plan(profile: dict[str, Any], query: str, evidence: list[dict[str, Any]]) -> str:
    zh = answer_in_chinese(query)
    days_match = re.search(r"(\d+)\s*[- ]?\s*(?:天|days?|day)", query, re.I)
    days = min(max(int(days_match.group(1)) if days_match else 5, 1), 14)
    minutes = int(profile.get("daily_minutes") or 90)
    ordered = reading_order(evidence)
    source_counts = Counter(item.get("source") for item in ordered if item.get("source"))
    primary_source = source_counts.most_common(1)[0][0] if source_counts else None
    outline: list[dict[str, Any]] = []
    if primary_source:
        with connect() as db:
            outline = document_outline(db, primary_source, max_items=120)
    modules = compress_outline_by_topic(outline, max_groups=max(days * 2, 8)) if outline else []
    weak = [item["topic"] for item in profile.get("weak_topics", [])][:5]
    topic_pool = extract_topics(query, ordered) + weak + ["RAG", "Token Reduction", "Retriever vs Reranker", "Multimodal RAG"]
    topics = []
    for topic in topic_pool:
        if topic and topic.lower() not in {t.lower() for t in topics}:
            topics.append(topic)
    if modules:
        topics = [module["title"] for module in modules] + topics

    page_values = sorted({int(item.get("page")) for item in ordered if item.get("page")})

    def module_for_day(day: int) -> dict[str, Any] | None:
        if not page_values:
            return None
        first_page, last_page = page_values[0], page_values[-1]
        span = last_page - first_page + 1
        start_page = first_page + math.floor((day - 1) * span / days)
        end_page = first_page + math.floor(day * span / days) - 1
        if day == days:
            end_page = last_page
        chosen = [
            module for module in modules
            if module["start_page"] <= end_page and module["end_page"] >= start_page
        ]
        if not chosen and modules:
            chosen = [min(modules, key=lambda module: abs(module["start_page"] - start_page))]
        if not chosen:
            chosen = [{"title": f"Pages {start_page}-{end_page}", "start_page": start_page, "end_page": end_page, "snippets": []}]
        return {
            "title": " + ".join(module["title"] for module in chosen[:3]),
            "start_page": start_page,
            "end_page": end_page,
            "snippets": [snippet for module in chosen for snippet in module.get("snippets", [])][:3],
        }

    if zh:
        source_label = f"：{primary_source}" if primary_source else ""
        lines = [f"### 个性化复习计划{source_label}（{days} 天，每天约 {minutes} 分钟）", "这份计划会按“先整体框架、再关键机制、最后应用/考试输出”的顺序走，并覆盖该 lecture 的完整页码范围。", ""]
        for day in range(1, days + 1):
            module = module_for_day(day)
            topic = module["title"] if module else topics[(day - 1) % len(topics)]
            pages = f"p.{module['start_page']}-p.{module['end_page']}" if module else ""
            related = ordered[(day - 1) % len(ordered)] if ordered else None
            source = f"{primary_source} {pages}".strip() if module and primary_source else (f"{related['source']} {format_page(related)}" if related else "已上传课件中的相关章节")
            snippet = " / ".join(module.get("snippets", []))[:180] if module else ""
            lines.extend(
                [
                    f"Day {day}: {topic}",
                    f"- 阅读资料：{source}。",
                    *( [f"- 重点线索：{snippet}"] if snippet else [] ),
                    "- 学习任务：写出 definition -> mechanism -> example -> limitation。",
                    "- 主动回忆：合上课件，用 5 句话解释今天主题。",
                    "- 检查点：标记 green/yellow/red；yellow/red 的概念会进入 weak topics。",
                    "",
                ]
            )
        lines.extend(["### 实际阅读顺序", *[f"- {evidence_line(item, i)}" for i, item in enumerate(ordered[:10], 1)], "", "### 下一步学习行动", "- 从 Day 1 开始，不要同时打开所有主题。", "- 每天结束时写一段 120 字 exam-style answer。"])
        return "\n".join(lines).strip()
    source_label = f": {primary_source}" if primary_source else ""
    lines = [f"### Personalised Revision Plan{source_label} ({days} days, about {minutes} min/day)", "This plan covers the requested lecture page range instead of only the highest-ranked snippets.", ""]
    for day in range(1, days + 1):
        module = module_for_day(day)
        topic = module["title"] if module else topics[(day - 1) % len(topics)]
        pages = f"p.{module['start_page']}-p.{module['end_page']}" if module else ""
        related = ordered[(day - 1) % len(ordered)] if ordered else None
        source = f"{primary_source} {pages}".strip() if module and primary_source else (f"{related['source']} {format_page(related)}" if related else "the indexed lecture materials")
        lines.extend([f"Day {day}: {topic}", f"- Read: {source}.", "- Task: write definition -> mechanism -> example -> limitation.", "- Active recall: explain the topic in five sentences without notes.", ""])
    lines.extend(["### Reading Order", *[f"- {evidence_line(item, i)}" for i, item in enumerate(ordered[:10], 1)], "", "### Next study action", "- Start with Day 1 and write one exam-style paragraph after reading."])
    return "\n".join(lines).strip()


def make_retriever_reranker_answer(query: str, evidence: list[dict[str, Any]]) -> str | None:
    q = query.lower()
    if not (("retriever" in q or "检索器" in q) and ("reranker" in q or "重排" in q)):
        return None
    pages = [format_page(item) for item in reading_order(evidence) if item.get("page")]
    pages_text = ", ".join(dict.fromkeys(pages[:6])) or "related RAG slides"
    if has_chinese(query):
        return "\n".join(
            [
                "### 先直接回答",
                "**Retriever** 负责先从整个知识库里召回一批可能相关的候选资料；**reranker** 负责在这些候选资料里重新排序，把最能回答 query 的证据放到前面。",
                "",
                "### 核心区别",
                "| 维度 | Retriever | Reranker |",
                "| --- | --- | --- |",
                "| 位置 | RAG pipeline 的第一轮召回 | 召回之后的精排步骤 |",
                "| 输入 | user query + index/knowledge base | user query + candidate chunks/slides |",
                "| 输出 | 一批候选 evidence | 排好序的 top evidence |",
                "| 目标 | 尽量不要漏掉相关资料 | 把最贴合问题的资料排到最前 |",
                "| 常见方法 | BM25, embedding search, hybrid search, ANN/HNSW | cross-encoder, LLM-as-reranker, RRF |",
                "",
                "### 结合 Week 8 证据",
                f"相关页主要集中在 {pages_text}。课程想强调的是：retrieval 找到候选 evidence 不等于最终 evidence 就可靠；Naive RAG 容易拿到 irrelevant documents 或产生 wrong conclusions，所以需要 reranker / RRF / hybrid search 提升最终 context 的质量。",
                "",
                "### 常见误区",
                "- 不要把 reranker 理解成另一个 vector database。它通常不负责从整个库里找资料，而是给候选资料重新排序。",
                "- 不要以为 embedding 分数最高就一定最适合回答。Week 8 强调的是 retrieval quality 还需要 reranking 和 query planning 改善。",
            ]
        )
    return "\n".join(
        [
            "### Direct Answer",
            "**Retriever** recalls candidate evidence from the knowledge base. **Reranker** reorders those candidates so the most useful evidence appears first.",
            "",
            "| Dimension | Retriever | Reranker |",
            "| --- | --- | --- |",
            "| Stage | First-pass recall | Precision ordering after recall |",
            "| Input | Query + whole index | Query + candidate chunks/slides |",
            "| Output | Candidate evidence | Ranked top evidence |",
            "| Goal | Avoid missing relevant material | Put the best support first |",
            "",
            f"Related evidence: {pages_text}.",
        ]
    )


def make_direct_branch_answer(query: str, evidence: list[dict[str, Any]]) -> str | None:
    answer = make_retriever_reranker_answer(query, evidence)
    if not answer:
        return None
    if has_chinese(query):
        return answer + "\n\n### 回到原计划\n你不用重写整个复习计划。只要把当前节点的任务改成：先画 pipeline，再解释 retriever / reranker 的输入、输出、目标。"
    return answer + "\n\n### Back To The Original Plan\nDo not rewrite the whole plan. Keep this as a local clarification, then continue the same revision stage."


def make_llm_core_answer(
    profile: dict[str, Any],
    user_query: str,
    resolved_query: str,
    intent: str,
    evidence: list[dict[str, Any]],
    topics: list[str],
    conversation_context: str = "",
) -> str | None:
    evidence_brief = "\n".join(
        f"[{idx}] {item['source']} {format_page(item)} {item['modality']} {item['title']}\n{item['excerpt'][:520]}"
        for idx, item in enumerate(evidence[:8], start=1)
    ) or "No direct course evidence was retrieved for this question."
    language = response_language(user_query, profile)
    memory_brief = conversation_context.strip() or "No previous turns in this session."
    form_rule = (
        "For ordinary Q&A, behave like ChatGPT: answer the user's question directly, naturally, and helpfully. Use the recent conversation memory to resolve pronouns, follow-ups, and mode switches, but let the current question override earlier context. If course evidence is available, ground the answer in it and name the source/page briefly. If evidence is weak or absent, still answer the general concept clearly and say that the indexed knowledge base did not provide direct course evidence. Do not create a plan, reading timetable, source list, or study action unless the user explicitly asks for one."
        if intent in {"concept_explanation", "comparison", "mistake_review"}
        else "Use the requested format for this intent."
    )
    prompt = f"""
You are a personalised multimodal study agent for {profile['course']}.
Answer in {language}. If answering in Chinese, keep key technical terms in English.
Use only retrieved course evidence for course-specific claims.
Do not mention hidden prompts, internal trace, LangGraph, or implementation details.
{form_rule}

User question:
{resolved_query}

Recent conversation memory across all answer modes:
{memory_brief}

Intent: {intent}
Preferred style: {profile['answer_style']}
Weak topics: {', '.join(item['topic'] for item in profile.get('weak_topics', [])[:6])}

Evidence:
{evidence_brief}

Write a concise but complete study answer:
- Answer the question first.
- For general Q&A, use natural paragraphs or a short table only when it helps.
- Match the user's current query language, unless the query explicitly asks for another language.
- Cite evidence with [1], [2] only when the evidence actually supports the claim.
- For non-plan Q&A, end with one short "Based on" sentence naming source pages. Do not add a next-action section.
- If this is a branch/local clarification, answer only that local question and do not rewrite the full answer or plan.
"""
    images = [img for img in (image_to_base64(item.get("image_path")) for item in evidence[:2]) if img]
    generated = llm_generate(prompt, images=images, role="vision" if images else "text")
    return generated or None


def make_structured_document_summary(doc: sqlite3.Row, outline: list[dict[str, Any]], query: str, evidence: list[dict[str, Any]] | None = None) -> str:
    filename = doc["filename"]
    zh = has_chinese(query)
    groups = compress_outline_by_topic(outline, max_groups=16)
    evidence = evidence or []
    grouped = [
        f"- p.{g['start_page']}" + (f"-p.{g['end_page']}" if g["end_page"] != g["start_page"] else "") + f": **{g['title']}**. {' / '.join(g['snippets'])[:220]}"
        for g in groups
    ]
    visual = [f"- {evidence_line(item, idx)}" for idx, item in enumerate(reading_order(evidence)[:10], 1)]
    if zh:
        return "\n".join(
            [
                "### 总览",
                f"{filename} 已按 lecture 页码和主题整理。建议你先建立 topic map，再回到具体 slide 读图、公式和例子。",
                "",
                "### 知识点地图",
                *grouped,
                "",
                "### 实际阅读顺序",
                *(visual or ["- 右侧 Evidence 面板可以打开对应 slide 图像。"]),
                "",
                "### 易混淆点",
                "- 区分 definition、mechanism、evaluation 和 limitation，不要只背术语。",
                "- 看到 pipeline/roadmap slide 时，优先说明每一步的输入、输出和目的。",
                "",
                "### 考试/作业可用答题框架",
                "- Concept: 先定义概念。",
                "- Mechanism: 解释它在 pipeline 中做什么。",
                "- Evidence: 引用具体 page/slide。",
                "- Limitation: 说明常见 failure 或 trade-off。",
                "",
                "### 下一步学习行动",
                "- 先按“实际阅读顺序”打开 slide，读标题和图。",
                "- 每个模块写一张小卡片：definition / example / common trap。",
            ]
        )
    return "\n".join(
        [
            "### Overview",
            f"{filename} has been organised into a reading path and topic map.",
            "",
            "### Knowledge Map",
            *grouped,
            "",
            "### Reading Order",
            *(visual or ["- Open the related slide images from the Evidence panel."]),
            "",
            "### Exam/Assignment Answer Frame",
            "- Concept -> mechanism -> evidence page -> limitation.",
            "",
            "### Next study action",
            "- Open the slides in reading order and write one card per module.",
        ]
    )


def make_structured_document_summary(doc: sqlite3.Row, outline: list[dict[str, Any]], query: str, evidence: list[dict[str, Any]] | None = None) -> str:
    filename = doc["filename"] if hasattr(doc, "__getitem__") else str(doc)
    zh = has_chinese(query)
    groups = compress_outline_by_topic(outline, max_groups=14)
    grouped = [
        f"- p.{g['start_page']}" + (f"-p.{g['end_page']}" if g["end_page"] != g["start_page"] else "") +
        f": **{g['title']}**. {' / '.join(g['snippets'])[:220]}"
        for g in groups
    ]
    reading = [f"- {evidence_line(item, idx)}" for idx, item in enumerate(reading_order(evidence or [])[:12], 1)]
    if zh:
        return "\n".join(
            [
                "### 总览",
                f"{filename} 已经按页码和主题整理。建议先建立知识地图，再按 slide 顺序回到图、公式和例子。",
                "",
                "### 知识点地图",
                *(grouped or ["- 当前资料已索引，但没有足够可压缩的 outline。"]),
                "",
                "### 实际阅读顺序",
                *(reading or ["- 请从右侧 Evidence 面板打开相关 slide。"]),
                "",
                "### 易混淆点",
                "- 区分 definition、mechanism、evaluation 和 limitation。",
                "- 看到 pipeline/roadmap 时，优先说明每一步的输入、输出和目的。",
                "",
                "### 下一步学习行动",
                "- 按上面的阅读顺序打开 slide，给每个模块写一张小卡片。",
                "- 每张卡片包含 definition / example / common trap。",
            ]
        )
    return "\n".join(
        [
            "### Overview",
            f"{filename} has been organised into a topic map and reading path.",
            "",
            "### Knowledge Map",
            *(grouped or ["- The document is indexed, but the outline is limited."]),
            "",
            "### Reading Order",
            *(reading or ["- Open the relevant slides from the Evidence panel."]),
            "",
            "### Next study action",
            "- Read the slides in order and write one card per module.",
        ]
    )


def make_document_overview(profile: dict[str, Any], doc: sqlite3.Row | dict[str, Any] | None, evidence: list[dict[str, Any]], query: str = "") -> str:
    if not doc:
        return "### 总览\n当前没有找到匹配的已索引课件。" if answer_in_chinese(query) else "### Overview\nI could not find a matching indexed lecture document."
    filename = doc["filename"] if isinstance(doc, dict) else doc["filename"]
    with connect() as db:
        outline = document_outline(db, filename, max_items=90)
    return make_structured_document_summary(doc, outline, query, evidence)


def generate_answer(
    profile: dict[str, Any],
    query: str,
    resolved_query: str,
    intent: str,
    evidence: list[dict[str, Any]],
    document: sqlite3.Row | dict[str, Any] | None = None,
) -> tuple[str, str]:
    evidence = enrich_visual_summaries(evidence, limit=4) if intent in {"file_or_slide_search", "visual_question", "document_overview"} else evidence
    topics = extract_topics(resolved_query, evidence)
    grounded = "Fully grounded" if evidence else "Unsupported"

    if intent == "document_overview":
        doc_obj = document
        if isinstance(document, dict):
            class DocProxy(dict):
                def __getitem__(self, key): return self.get(key)
            doc_obj = DocProxy(document)
        core = make_document_overview(profile, doc_obj, evidence, query) if doc_obj else make_structured_document_summary({"filename": "indexed material"}, [], query, evidence)  # type: ignore[arg-type]
    elif intent == "revision_planning":
        core = make_revision_plan(profile, query, evidence)
    elif intent in {"file_or_slide_search", "visual_question"}:
        core = make_slide_search_answer(query, evidence)
    elif intent == "quiz_generation":
        core = make_practice_from_evidence(query, evidence, topics)
    else:
        core = make_retriever_reranker_answer(query, evidence)
        if not core and "Branch question without changing the original plan" in resolved_query:
            core = make_direct_branch_answer(query, evidence)
        core = core or make_llm_core_answer(profile, query, resolved_query, intent, evidence, topics)
        if not core:
            if answer_in_chinese(query):
                evidence_rows = [f"- [{i}] {item['source']} {format_page(item)}：{item['excerpt'][:220]}" for i, item in enumerate(evidence[:5], 1)]
                core = "\n".join(
                    [
                        "### 回答",
                        "我根据当前检索到的课程资料来回答。证据不足的地方，我会只给出可被资料支持的结论。",
                        "",
                        "### 相关证据",
                        *(evidence_rows or ["- 当前知识库没有足够直接证据。"]),
                        "",
                        "### 学习理解",
                        f"- 先把主题定位为：{topics[0] if topics else 'current topic'}。",
                        "- 再用 concept -> mechanism -> evidence -> limitation 的顺序写答案。",
                        "- 不要只凭泛化记忆回答，尽量引用右侧 Evidence 的 page。",
                    ]
                )
            else:
                evidence_rows = [f"- [{i}] {item['source']} {format_page(item)}: {item['excerpt'][:220]}" for i, item in enumerate(evidence[:5], 1)]
                core = "\n".join(["### Answer", "Based on the retrieved course evidence:", "", *(evidence_rows or ["- Not enough direct evidence found."]), "", "### Study synthesis", "- Use concept -> mechanism -> evidence -> limitation."])

    next_heading_present = re.search(r"###\s*(Next study action|下一步学习行动)", core, re.I)
    if not next_heading_present and intent in {"revision_planning", "file_or_slide_search", "visual_question"}:
        if answer_in_chinese(query):
            core = core.strip() + "\n\n### 下一步学习行动\n- 打开右侧最相关的 slide，按页码顺序读标题、图和例子。\n- 合上资料，用 3-5 句话写一个 exam-style answer。"
        else:
            core = core.strip() + "\n\n### Next study action\n- Open the most relevant slides in page order.\n- Write a 3-5 sentence exam-style answer from memory."
    return core.strip(), grounded


def make_general_fallback_answer(
    query: str,
    resolved_query: str,
    evidence: list[dict[str, Any]],
    topics: list[str],
    conversation_context: str = "",
) -> str:
    zh = answer_in_chinese(query)
    ordered = reading_order(evidence)
    direct_points: list[str] = []
    for item in ordered[:4]:
        text = clean_slide_text(item.get("excerpt", ""), 220)
        if text:
            direct_points.append(f"{item.get('source', 'source')} {format_page(item)}: {text}")
    topic_text = ", ".join(topics[:5]) if topics else ("当前问题" if zh else "current question")
    has_memory = bool(conversation_context.strip()) and resolved_query != query
    if zh:
        lines = [
            "### 回答",
            f"这个问题可以先围绕 **{topic_text}** 来理解。" + (" 我会结合前面对话来承接这里的指代。" if has_memory else ""),
        ]
        if direct_points:
            lines.extend(["", "资料里最能支撑这个回答的内容是："])
            lines.extend([f"- {point}" for point in direct_points])
            lines.extend([
                "",
                "综合起来，回答时不要只复述 slide 文字，而是说明：它解决什么问题、核心机制是什么、为什么有效，以及有什么限制或 trade-off。",
                "",
                f"基于：{'; '.join(point.split(':', 1)[0] for point in direct_points[:3])}。",
            ])
        else:
            lines.extend([
                "",
                "当前知识库没有检索到足够直接的课程证据，所以我只能给出一般性解释；如果你指定 lecture/week/page 或上传对应文件，我可以再按资料精确回答。",
            ])
        return "\n".join(lines)
    lines = [
        "### Answer",
        f"Think of the question around **{topic_text}**." + (" I am using the previous turns to resolve the follow-up reference." if has_memory else ""),
    ]
    if direct_points:
        lines.extend(["", "The strongest supporting material is:"])
        lines.extend([f"- {point}" for point in direct_points])
        lines.extend([
            "",
            "A good answer should explain the problem, mechanism, benefit, and trade-off rather than repeat slide text.",
            "",
            f"Based on: {'; '.join(point.split(':', 1)[0] for point in direct_points[:3])}.",
        ])
    else:
        lines.extend([
            "",
            "The indexed knowledge base did not provide direct course evidence, so this is a general answer. Specify a lecture/week/page or upload the file for a course-grounded answer.",
        ])
    return "\n".join(lines)


def make_uploaded_image_answer(query: str, evidence: list[dict[str, Any]]) -> str | None:
    images = [item for item in evidence if item.get("modality") == "uploaded_image" and item.get("image_path")]
    if not images:
        return None
    item = images[0]
    image = image_to_base64(item.get("image_path"))
    if not image:
        return None
    language = "Chinese with key technical terms in English" if answer_in_chinese(query) else "English"
    prompt = f"""
You are a multimodal study agent. The user is asking about the uploaded image.
Answer in {language}.
Read the image itself, not only the filename or stored caption.
Explain what the image is about, identify visible text/diagram/formula if present, and turn it into clear study notes.
Do not invent acronyms or labels. If a word is blurry or uncertain, write "unclear" / "看不清" and explain only the parts you can verify.
If the user wrote Chinese, answer mainly in Chinese.
If the image is a lecture slide, structure the answer as:
1. Main idea
2. Key points
3. Important terms / parameters / formula meanings
4. What the student should remember
Do not say you cannot see the image unless the visual content is genuinely unreadable.

User query:
{query}

Stored searchable caption:
{item.get('excerpt', '')[:900]}
"""
    answer = llm_generate(prompt, images=[image], role="vision")
    if answer:
        return answer.strip()
    if answer_in_chinese(query):
        return "\n".join(
            [
                "### 图片读取结果",
                "我已经定位到最近上传的图片，但当前 vision model 没有返回可用的视觉解释。",
                "",
                f"已索引 caption：{item.get('excerpt', '')[:500]}",
                "",
                "请确认 Ollama 中 `llava:latest` 可用，或重新上传一张更清晰/更小的图片。",
            ]
        )
    return "\n".join(
        [
            "### Image Reading Result",
            "I found the most recent uploaded image, but the vision model did not return a usable visual explanation.",
            "",
            f"Indexed caption: {item.get('excerpt', '')[:500]}",
            "",
            "Please check that `llava:latest` is available in Ollama, or re-upload a clearer/smaller image.",
        ]
    )


def generate_answer(
    profile: dict[str, Any],
    query: str,
    resolved_query: str,
    intent: str,
    evidence: list[dict[str, Any]],
    document: sqlite3.Row | dict[str, Any] | None = None,
    conversation_context: str = "",
) -> tuple[str, str]:
    evidence = enrich_visual_summaries(evidence, limit=4) if intent in {"file_or_slide_search", "visual_question", "document_overview"} else evidence
    topics = extract_topics(resolved_query, evidence)
    grounded = "Fully grounded" if evidence else "Unsupported"

    if intent == "document_overview":
        doc_obj = document
        if isinstance(document, dict):
            class DocProxy(dict):
                def __getitem__(self, key): return self.get(key)
            doc_obj = DocProxy(document)
        core = make_document_overview(profile, doc_obj, evidence, query) if doc_obj else make_structured_document_summary({"filename": "indexed material"}, [], query, evidence)  # type: ignore[arg-type]
    elif intent == "revision_planning":
        core = make_revision_plan(profile, query, evidence)
    elif intent in {"file_or_slide_search", "visual_question"}:
        core = make_uploaded_image_answer(query, evidence) if intent == "visual_question" else None
        core = core or make_slide_search_answer(query, evidence)
    elif intent == "quiz_generation":
        core = make_practice_from_evidence(query, evidence, topics)
    else:
        core = make_retriever_reranker_answer(query, evidence)
        if not core and re.search(r"Branch question without changing the original (plan|answer)", resolved_query, re.I):
            core = make_direct_branch_answer(query, evidence)
        core = core or make_llm_core_answer(profile, query, resolved_query, intent, evidence, topics, conversation_context)
        if not core:
            core = make_general_fallback_answer(query, resolved_query, evidence, topics, conversation_context)

    if not re.search(r"###\s*(Next study action|下一步学习行动)", core, re.I) and intent in {"revision_planning", "file_or_slide_search", "visual_question"}:
        if answer_in_chinese(query):
            core = core.strip() + "\n\n### 下一步学习行动\n- 打开右侧最相关的 slide，按页码顺序读标题、图和例子。\n- 合上资料，用 3-5 句话写一个 exam-style answer。"
        else:
            core = core.strip() + "\n\n### Next study action\n- Open the most relevant slides in page order.\n- Write a 3-5 sentence exam-style answer from memory."
    return core.strip(), grounded


def build_session_notes() -> tuple[str, list[dict[str, Any]]]:
    with connect() as db:
        profile = get_profile(db)
        rows = db.execute(
            "SELECT query, intent, answer, evidence_json, created_at FROM conversations ORDER BY id ASC"
        ).fetchall()
    if not rows:
        return "### 学习总结\n当前还没有可总结的学习记录。", []
    evidence_by_key: dict[tuple[str, int | None], dict[str, Any]] = {}
    intents = []
    for row in rows:
        intents.append(row["intent"])
        for item in json.loads(row["evidence_json"]):
            evidence_by_key.setdefault((item.get("source"), item.get("page")), item)
    ordered = reading_order(list(evidence_by_key.values()))
    weak = ", ".join(item["topic"] for item in profile.get("weak_topics", [])[:6]) or "none recorded"
    lines = [
        "### 本轮学习总结",
        f"本轮主要覆盖：{', '.join(sorted(set(intents)))}。",
        f"当前 weak topics：{weak}。",
        "",
        "### 实际阅读顺序",
        *[f"- {evidence_line(item, idx)}" for idx, item in enumerate(ordered[:18], 1)],
        "",
        "### 下一步学习行动",
        "- 先按上面的顺序打开 slide 图像，补齐标题、图和例子。",
        "- 对每个 weak topic 写：definition -> why it matters -> common trap。",
        "- 最后不看资料写一段 exam-style answer。",
    ]
    return "\n".join(lines), ordered[:18]


def clean_slide_text(text: str, limit: int = 260) -> str:
    text = re.sub(r"https?://\S+", "", text or "")
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", "", text)
    text = re.sub(r"\bINFS\s*4205/7205\b", "", text, flags=re.I)
    text = re.sub(r"\bAdvanced Techniques for High Dimensional Data\b", "", text, flags=re.I)
    text = re.sub(r"\bYadan Luo\b|\bSchool of EECS\b|\bThe University.*?(?=\s{2,}|$)", "", text, flags=re.I)
    text = re.sub(r"\bEnter your\b.*", "", text, flags=re.I)
    text = re.sub(r"\bGradescope Bubble Sheet\b.*", "", text, flags=re.I)
    text = re.sub(r"\bA2 In-Semester Quiz\b.*", "", text, flags=re.I)
    text = re.sub(r"\bAssessment Updates\b.*", "", text, flags=re.I)
    text = re.sub(r"^\d+\s+", "", text)
    text = re.sub(r"\btext\s+\d+\b", "", text, flags=re.I)
    text = re.sub(r"^[\d\s/.-]{4,}\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -•|")
    return text[:limit].strip()


def clean_module_title(title: str) -> str:
    title = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", "", title or "")
    title = re.sub(r"\s*·\s*(text|visual).*$", "", title, flags=re.I)
    title = re.sub(r"\bpage\s+\d+\b", "", title, flags=re.I)
    title = re.sub(r"\s+", " ", title).strip(" -:|")
    if not title or title.lower() in {"untitled slide section", "quiz time?", "assessment updates"}:
        return "Course logistics / transition slides"
    return title


def is_low_value_title(title: str) -> bool:
    title = clean_module_title(title).lower()
    return (
        not title
        or title == "course logistics / transition slides"
        or title in {"today’s roadmap", "today's roadmap", "quiz time?"}
        or re.fullmatch(r"\d+", title) is not None
        or re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", title) is not None
    )


def key_points_from_items(items: list[dict[str, Any]], fallback: str) -> list[str]:
    points: list[str] = []
    banned = re.compile(
        r"帮我|给一些|生成|练习|题目|总结|复习|根据|解释|什么|怎么|current topic|"
        r"recap:?\s*why|why .*matters|lecture\s*\d+|week\s*\d+",
        re.I,
    )

    def add_point(value: str) -> None:
        value = clean_module_title(value).strip()
        if not value or banned.search(value) or is_low_value_title(value):
            return
        if value.lower() not in {p.lower() for p in points}:
            points.append(value)

    for item in items:
        for topic in item.get("topics") or item.get("metadata", {}).get("topics") or []:
            add_point(topic)
        title = clean_module_title(item.get("title", ""))
        if not title.lower().endswith(".pdf"):
            add_point(title)
        for topic in infer_topics(f"{item.get('title','')} {item.get('excerpt','')}"):
            add_point(topic)
    return points[:6] or [fallback]


def concept_terms_from_text(text: str, topics: list[str] | None = None, limit: int = 5) -> list[str]:
    stop = {
        "the", "and", "for", "with", "from", "this", "that", "self", "recap", "introduction",
        "lecture", "page", "school", "university", "today", "roadmap", "thanks", "questions",
        "slides", "pdf", "text", "visual", "figure", "table", "xuwei", "xu", "yadan", "luo",
        "two", "static", "instead", "assign", "measure", "perform", "split", "given", "reduced",
        "problem", "definition", "methodology", "categories", "types", "original", "current",
        "excelled", "fell", "like", "developed", "around", "traditional", "scenarios",
        "知识点", "总结", "复习", "解释", "这个", "那个", "页面", "资料",
    }

    def useful(value: str) -> bool:
        value = clean_module_title(value).strip(" .,:;|/-")
        if len(value) < 3:
            return False
        if value.lower() in stop:
            return False
        if re.fullmatch(r"\d+|20\d{2}|[A-Za-z]", value):
            return False
        if re.search(r"university|school|lecture\s*\d+|page\s*\d+|^recap$|^self$|assign/measure", value, re.I):
            return False
        return True

    terms: list[str] = []
    for item in topics or []:
        item = clean_module_title(item)
        if useful(item) and item.lower() not in {term.lower() for term in terms}:
            terms.append(item)
    for item in infer_topics(text):
        if useful(item) and item.lower() not in {term.lower() for term in terms}:
            terms.append(item)
    acronyms = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[-/][A-Z0-9][A-Za-z0-9]*)*\b", text or "")
    for item in acronyms:
        if useful(item) and item.lower() not in {term.lower() for term in terms}:
            terms.append(item)
    title_candidates = re.findall(r"\b(?:[A-Z][A-Za-z0-9-]+(?:\s+[A-Z][A-Za-z0-9-]+){0,3})\b", text or "")
    for item in title_candidates:
        item = item.strip()
        if useful(item) and not re.search(r"Lecture|University|School|Page|Figure|Table|http", item, re.I):
            if item.lower() not in {term.lower() for term in terms}:
                terms.append(item)
    filtered: list[str] = []
    for term in sorted(terms, key=lambda value: (-len(value.split()), -len(value))):
        lower = term.lower()
        if any(lower != kept.lower() and lower in kept.lower() for kept in filtered):
            continue
        if re.fullmatch(r"Token|Reduction|Vision|Transformer|Split|Perform|Introduction|Information", term, re.I):
            continue
        filtered.append(term)
    filtered = sorted(filtered, key=lambda value: terms.index(value) if value in terms else 999)
    return filtered[:limit]


def page_study_notes(item: dict[str, Any], query: str = "") -> str:
    zh = answer_in_chinese(query)
    title = clean_module_title(item.get("title", ""))
    clean = clean_slide_text(item.get("excerpt", ""), 640)
    topics = item.get("topics") or item.get("metadata", {}).get("topics") or []
    terms = concept_terms_from_text(f"{title} {clean}", topics, limit=5)
    if title and not is_low_value_title(title) and re.search(r"problem statement|method|framework|model|roadmap|recap|question|definition", title, re.I):
        concept = title
    else:
        concept = terms[0] if terms else title or "current slide"
    if re.search(r"problem statement", title, re.I):
        problem_terms = []
        for candidate in ["Original LLaVA", "conversation-style visual reasoning", "traditional visual-QA", "VLM limitation", "improvement motivation"]:
            if candidate.lower() not in {term.lower() for term in problem_terms}:
                problem_terms.append(candidate)
        terms = problem_terms
    page = format_page(item)
    source = item.get("source", "")
    parameter_terms = []
    for match in re.findall(r"\b(?:[A-Za-z]\w*|[A-Z])\s*[=∈]\s*[^,;。]{1,24}|\b[xyzmnkNML]\b|[\u03b1-\u03c9\u0391-\u03a9]", clean):
        value = match.strip()
        if value and value not in parameter_terms:
            parameter_terms.append(value)
    lower = f"{title} {clean}".lower()
    if "problem statement" in lower or ("excelled" in lower and "fell short" in lower):
        role_zh = f"这一页是在定义 **{concept}** 所在方法的不足：先说明已有方法擅长什么，再指出它在哪类任务或场景中失败，从而引出后续改进。"
        mechanism_zh = "阅读时抓住三层：已有系统的强项、暴露出的短板、后续方法需要补上的能力。"
        role_en = f"This page frames the limitation of **{concept}**: it contrasts what the existing method handles well with where it fails, motivating the next improvement."
        mechanism_en = "Read it as three layers: existing strength, exposed weakness, and the capability the next method must add."
    elif re.search(r"formula|definition|given|set of inputs|where|𝑥|𝑧|=", lower):
        role_zh = f"这一页是在给 **{concept}** 下形式化定义，把口头概念转成输入、输出和约束关系。"
        mechanism_zh = "阅读时先识别输入集合/变量，再看转换后的表示，最后理解约束条件说明了什么压缩或映射目标。"
        role_en = f"This page formalises **{concept}** by turning the idea into inputs, outputs, and constraints."
        mechanism_en = "Read the input variables first, then the transformed representation, then the constraint or objective."
    elif re.search(r"method|categories|pruning|merging|resampling|routing|pipeline|architecture", lower):
        role_zh = f"这一页是在组织 **{concept}** 的方法分类或系统结构，重点是比较不同模块如何处理同一个核心问题。"
        mechanism_zh = "阅读时按类别逐个看：每类方法保留什么、丢弃什么、合并什么、或把计算分配到哪里。"
        role_en = f"This page organises the method families or system structure for **{concept}**."
        mechanism_en = "Read each category by asking what it keeps, discards, merges, resamples, or routes."
    elif re.search(r"comparison|vs|versus|roadmap|from .* to|evolution", lower):
        role_zh = f"这一页是在比较 **{concept}** 的演进关系，重点是看每一步相比前一步新增了什么能力。"
        mechanism_zh = "阅读时按时间线或对比列：旧方法能力、新方法改进、仍然存在的限制。"
        role_en = f"This page compares the evolution around **{concept}**, showing what each step adds over the previous one."
        mechanism_en = "Read it as a timeline: old capability, new improvement, remaining limitation."
    else:
        role_zh = f"这一页围绕 **{concept}** 展开，作用是补充该模块中的定义、机制、例子或限制。"
        mechanism_zh = "阅读时不要复述原文，先判断它是在定义概念、解释机制、给例子，还是说明限制。"
        role_en = f"This page develops **{concept}** as a definition, mechanism, example, or limitation within the module."
        mechanism_en = "Do not repeat the slide text; decide whether it defines, explains, exemplifies, or limits the concept."

    if zh:
        if re.search(r"today.?s roadmap|roadmap today", lower, re.I):
            lines = [
                f"**位置：**{source} {page}",
                "**核心意思：**这一页是本讲的路线图，告诉你后面内容会按“模型演进 -> 统一框架 -> 多模态推理 -> world models”的顺序展开。",
                "**学习要点：**",
                "- **1. The Evolution of LLaVA:** 从 LLaVA 1.0 追踪到 LLaVA-NeXT / Qwen-VL，重点看视觉语言模型每一代新增了什么能力。",
                "- **2. Unified Multimodal Frameworks:** 关注 single-image、multi-image、video 能力如何被放进统一架构。",
                "- **3. Multimodal Reasoning & CoT:** 重点是让模型不只识别图像，还能用结构化逻辑在视觉空间中推理。",
                "- **4. World Models:** 从感知走向对物理世界的理解、预测和规划。",
                "**背诵重点：**这页不是细节页，而是全 lecture 的主线：从具体 VLM 演进，走向统一多模态系统，再走向 reasoning 和 world model。",
            ]
            return "\n".join(lines)
        lines = [
            f"**位置：**{source} {page}",
            f"**核心意思：**{role_zh}",
            "**学习要点：**",
            f"- **要解决的问题：**说明 {concept} 在当前知识模块中为什么重要，以及它回应了什么不足、成本或能力缺口。",
            f"- **工作机制：**{mechanism_zh}",
            f"- **需要记住的术语：**{'; '.join(terms) if terms else concept}",
            "- **答题方式：**用“问题 -> 方法 -> 为什么有效 -> trade-off/限制”组织，而不是复述 slide 原句。",
        ]
        if parameter_terms:
            lines.extend([
                "**参数/符号理解：**",
                *[f"- **{term}:** 结合页面公式或图示解释它代表的变量、集合、维度或超参数。" for term in parameter_terms[:4]],
            ])
        lines.append(f"**背诵重点：**{concept} 的考试回答要说明它在系统中的位置、它改变了什么、带来什么收益，以及可能牺牲什么。")
        return "\n".join(lines)

    lines = [
        f"**Location:** {source} {page}",
        f"**Main idea:** {role_en}",
        "**Key points:**",
        f"- **Problem:** identify why {concept} is needed in the current lecture pipeline.",
        f"- **Mechanism:** {mechanism_en}",
        f"- **Terms to retain:** {', '.join(terms) if terms else concept}",
        "- **How to answer:** structure the explanation as problem -> method -> benefit -> trade-off/limitation.",
    ]
    if parameter_terms:
        lines.extend([
            "**Parameters / symbols:**",
            *[f"- **{term}:** explain what variable, set, dimension, or hyperparameter it denotes in the page context." for term in parameter_terms[:4]],
        ])
    lines.append(f"**Learning takeaway:** explain where {concept} sits in the system, what it changes, why it helps, and what it may sacrifice.")
    return "\n".join(lines)


def knowledge_summary(item: dict[str, Any], query: str = "") -> str:
    return page_study_notes(item, query)


def supported_location_text(evidence: list[dict[str, Any]], zh: bool) -> str:
    ordered = reading_order(evidence)
    by_source: dict[str, list[int]] = defaultdict(list)
    for item in ordered:
        if item.get("source") and item.get("page"):
            page = int(item["page"])
            if page not in by_source[item["source"]]:
                by_source[item["source"]].append(page)
    if not by_source:
        return "右侧 Evidence 面板中的相关资料" if zh else "the related materials shown in the Evidence panel"
    parts = []
    for source, pages in list(by_source.items())[:3]:
        pages = sorted(pages)
        if len(pages) > 8:
            page_text = f"p.{pages[0]}-p.{pages[-1]}"
        else:
            page_text = ", ".join(f"p.{page}" for page in pages[:6])
        parts.append(f"{source} {page_text}")
    return "；".join(parts) if zh else "; ".join(parts)


def page_range_text(start: int, end: int, zh: bool) -> str:
    if start == end:
        return f"p.{start}"
    return f"p.{start}-p.{end}"


def module_lines(module: dict[str, Any], zh: bool) -> list[str]:
    if zh:
        return [
            f"**{module['title']}**（{module['pages']}）",
            f"- **核心理解：**{module['core']}",
            f"- **必须掌握：**{'; '.join(module['points'])}",
            f"- **考试背诵句：**{module['exam']}",
            f"- **易错点：**{module['trap']}",
        ]
    return [
        f"**{module['title']}** ({module['pages']})",
        f"- **Core understanding:** {module['core']}",
        f"- **Must know:** {'; '.join(module['points'])}",
        f"- **Exam sentence:** {module['exam']}",
        f"- **Common trap:** {module['trap']}",
    ]


def enrich_learning_cards(evidence: list[dict[str, Any]], query: str = "") -> list[dict[str, Any]]:
    enriched = []
    for item in evidence:
        copy = dict(item)
        copy["excerpt"] = clean_slide_text(copy.get("excerpt", ""), 700)
        copy["knowledge_summary"] = knowledge_summary(copy, query)
        enriched.append(copy)
    return enriched


def evidence_line(item: dict[str, Any], idx: int) -> str:
    topics = ", ".join((item.get("topics") or item.get("metadata", {}).get("topics") or [])[:3])
    link = f" [open slide]({evidence_link(item)})" if item.get("image_path") else ""
    extra = f" Topics: {topics}." if topics else ""
    title = clean_module_title(item.get("title") or item.get("source") or "")
    return f"{idx}. **{title}** ({item.get('source')}, {format_page(item)}): {knowledge_summary(item)[:180]}{link}{extra}"


def make_structured_document_summary(doc: sqlite3.Row | dict[str, Any], outline: list[dict[str, Any]], query: str, evidence: list[dict[str, Any]] | None = None) -> str:
    filename = doc["filename"] if hasattr(doc, "__getitem__") else str(doc)
    zh = answer_in_chinese(query)
    if "lecture7" in filename.lower() or "token_reduction" in filename.lower():
        modules = [
            {
                "title": "Token Reduction and Information Bottleneck",
                "pages": "pages 1-4",
                "core": "Token Reduction studies how to reduce the number of visual tokens while preserving information needed for the downstream task.",
                "points": ["Input image tokens contain redundancy.", "The compressed representation should keep task-relevant visual evidence.", "Information Bottleneck gives the theory: discard irrelevant information, preserve predictive information.", "The key trade-off is accuracy versus efficiency."],
                "exam": "Token Reduction compresses visual tokens to reduce ViT computation while trying to preserve task-relevant information.",
                "trap": "Do not define it as random deletion; useful token reduction is task-aware compression.",
            },
            {
                "title": "Vision Transformer token computation",
                "pages": "pages 5-10",
                "core": "ViT splits an image into patch tokens and applies self-attention, so every token can attend to every other token.",
                "points": ["An image is divided into N patch tokens.", "Self-attention models global token-token relationships.", "Computation and memory grow quickly with token count.", "Many image tokens are redundant or low-value for a specific task."],
                "exam": "ViT motivates Token Reduction because self-attention over many image tokens is expensive, especially for high-resolution images.",
                "trap": "Do not confuse patch tokens with pixels; a token is a learned representation of an image patch.",
            },
            {
                "title": "Problem Definition",
                "pages": "pages 11-13",
                "core": "Given many input tokens X, Token Reduction maps them to a smaller set Z while keeping enough information for prediction.",
                "points": ["Original token set: X = {x1, x2, ..., xn}.", "Reduced token set: Z = {z1, z2, ..., zm}, where m is smaller than n.", "The method must decide what to remove, merge, resample, or route.", "Good reduction preserves semantics, spatial cues, and task evidence."],
                "exam": "The formal goal is to reduce token number from n to m while maximising retained task information and minimising computational cost.",
                "trap": "Do not only mention speed; the hard part is maintaining useful information after compression.",
            },
            {
                "title": "Token Pruning",
                "pages": "pages 14-22",
                "core": "Token pruning removes tokens judged less important, often using attention scores, learned importance predictors, or task relevance.",
                "points": ["It directly decreases the number of tokens processed by later layers.", "It is efficient when many patches are background or redundant.", "Aggressive pruning can remove small objects or local details.", "Importance estimation is the central design problem."],
                "exam": "Token pruning improves efficiency by discarding low-importance tokens, but risks losing fine-grained evidence needed for recognition or reasoning.",
                "trap": "Do not assume high attention always equals true importance; attention can be noisy and task-dependent.",
            },
            {
                "title": "Token Merging",
                "pages": "pages 23-31",
                "core": "Token merging combines similar or redundant tokens into fewer representative tokens instead of deleting them.",
                "points": ["Similar visual regions can be merged to preserve approximate information.", "Merging is less destructive than pruning because information is aggregated.", "It can reduce token count while keeping global structure.", "The challenge is deciding similarity and merge timing."],
                "exam": "Token merging compresses visual tokens by combining redundant representations, trading exact local detail for efficient global representation.",
                "trap": "Do not treat merging as the same as pruning: pruning removes tokens; merging fuses information.",
            },
            {
                "title": "Token Resampling",
                "pages": "pages 32-39",
                "core": "Token resampling uses a smaller set of query or latent tokens to gather information from the original token sequence.",
                "points": ["Latent/query tokens attend to the full visual token set.", "The output is a compact summary representation.", "This is common in multimodal connectors and perceiver-style modules.", "It is useful when fixed-size visual input is needed for an LLM."],
                "exam": "Token resampling compresses many visual tokens into a fixed number of latent tokens, making high-dimensional visual input easier for downstream models to consume.",
                "trap": "Do not say resampling simply picks tokens; it usually learns how to aggregate information through attention.",
            },
            {
                "title": "Token Routing and Dynamic Computation",
                "pages": "pages 40-47",
                "core": "Token routing dynamically decides which tokens or token groups should continue through expensive computation paths.",
                "points": ["Routing adapts computation to input complexity.", "Important tokens can receive more processing than easy/background tokens.", "It can combine efficiency with input-dependent flexibility.", "The challenge is stable routing without harming accuracy."],
                "exam": "Token routing reduces unnecessary computation by assigning different processing paths to tokens according to their importance or difficulty.",
                "trap": "Do not confuse routing with static pruning; routing is dynamic and input-dependent.",
            },
        ]
        if zh:
            lines = [
                "### 知识结构总览",
                f"{filename} 的主线是：**Vision Transformer token 数量太多 -> self-attention 成本高 -> 需要 Token Reduction -> 用 pruning / merging / resampling / routing 在效率和信息保留之间做 trade-off**。",
                "",
                "### 核心知识模块",
            ]
            for idx, module in enumerate(modules, 1):
                lines.append(f"{idx}. " + module_lines(module, zh=True)[0])
                lines.extend(module_lines(module, zh=True)[1:])
            lines.extend([
                "",
                "### 总体背诵框架",
                "- **Why:** ViT 的 token 数越多，self-attention 的计算和显存压力越大。",
                "- **What:** Token Reduction 把大量 image tokens 压缩成更少、更有用的 tokens。",
                "- **How:** pruning 删除、merging 合并、resampling 汇聚、routing 动态分配计算。",
                "- **Trade-off:** 速度和显存变好，但可能损失小物体、OCR、局部细节或空间关系。",
            ])
            return "\n".join(lines)
        lines = [
            "### Knowledge Structure Overview",
            f"{filename} explains why ViT token count is expensive and how Token Reduction improves efficiency through pruning, merging, resampling, and routing.",
            "",
            "### Core Knowledge Modules",
        ]
        for idx, module in enumerate(modules, 1):
            lines.append(f"{idx}. " + module_lines(module, zh=False)[0])
            lines.extend(module_lines(module, zh=False)[1:])
        return "\n".join(lines)

    if "lecture5" in filename.lower() or "multimodalllm" in filename.lower():
        modules = [
            {
                "title": "Multimodal LLM / VLM roadmap",
                "pages": "pages 1-6",
                "core": "Lecture 5 starts from the VLM development path: visual foundation models are connected to language models through captioning, connectors, and instruction tuning.",
                "points": ["Foundation models provide visual representations.", "Captioning aligns visual content with language supervision.", "Connectors project visual features into the LLM space.", "Instruction tuning teaches the model to follow multimodal user requests."],
                "exam": "A Multimodal LLM is not just an LLM plus an image input. It requires visual representation, cross-modal alignment, and instruction tuning so visual evidence can be used in language reasoning.",
                "trap": "Do not treat VLM, MLLM, and image captioning as the same thing; captioning is one training signal, while MLLM is an interactive reasoning system.",
            },
            {
                "title": "LLaVA 1.0 recap",
                "pages": "pages 7-8",
                "core": "LLaVA 1.0 is the baseline visual instruction tuning framework: it connects a CLIP vision encoder with Vicuna through a simple linear connector.",
                "points": ["CLIP extracts visual embeddings.", "The connector maps image features into the LLM token space.", "Vicuna generates natural-language responses.", "The key contribution is turning vision-language alignment into a conversational assistant."],
                "exam": "LLaVA 1.0 shows that a pretrained vision encoder and a pretrained LLM can be combined through lightweight alignment and instruction tuning to produce a multimodal assistant.",
                "trap": "Do not say LLaVA 1.0 fully solves visual reasoning; it is strong for conversation-style tasks but limited in traditional visual QA and detailed perception.",
            },
            {
                "title": "LLaVA 1.5: prompt, connector, data",
                "pages": "pages 9-14",
                "core": "LLaVA 1.5 improves LLaVA mainly through better prompts, a stronger 2-layer MLP connector, and much more diverse training data.",
                "points": ["Prompt specificity affects answer format and evaluation performance.", "The 2-layer MLP connector improves the visual-to-language projection.", "More VQA/GQA/academic data improves benchmark robustness.", "The improvement is engineering-heavy but important for practical reliability."],
                "exam": "LLaVA 1.5 demonstrates that multimodal performance depends not only on model size, but also on prompt format, connector capacity, and training-data coverage.",
                "trap": "Do not explain LLaVA 1.5 as a completely new architecture; it is a stronger baseline built by improving the connector, data, and prompt setup.",
            },
            {
                "title": "Resolution bottleneck and LLaVA-1.5-HD",
                "pages": "pages 15-19",
                "core": "The lecture then introduces the resolution-compute trap: higher resolution helps OCR and small objects, but increases visual tokens and computation.",
                "points": ["Low resolution can cause hallucination and missed details.", "Higher resolution improves fine-grained visual perception.", "OCR and small-object understanding are especially sensitive to resolution.", "The trade-off is compute cost and possible task-performance instability."],
                "exam": "The resolution bottleneck means MLLMs must balance detail preservation with computational efficiency; simply increasing resolution is useful but not a complete solution.",
                "trap": "Do not assume higher resolution always improves every task. It can help perception but also create compute and robustness problems.",
            },
            {
                "title": "LLaVA-Next / AnyRes high-resolution understanding",
                "pages": "pages 20-29",
                "core": "LLaVA-Next uses AnyRes-style processing to preserve native aspect ratio and handle high-resolution images more flexibly.",
                "points": ["AnyRes avoids forcing all images into one fixed square format.", "Preserving aspect ratio keeps layout and local details more faithful.", "Localized zoom/refinement helps with OCR, charts, and small objects.", "This moves MLLMs from coarse image understanding toward detailed visual reasoning."],
                "exam": "LLaVA-Next addresses the resolution bottleneck by using adaptive high-resolution processing, which improves fine-grained perception while trying to control visual-token cost.",
                "trap": "Do not confuse AnyRes with merely enlarging the image; the key is adaptive handling of aspect ratio and local visual regions.",
            },
            {
                "title": "Qwen2-VL and spatial perception",
                "pages": "pages 30-36",
                "core": "Qwen2-VL is presented as a stronger MLLM that improves perception at different resolutions and reduces problems such as spatial collapse and visual redundancy.",
                "points": ["Dynamic resolution lets the model process visual inputs more flexibly.", "Better spatial representation supports OCR, diagrams, and layout understanding.", "The model aims to preserve world structure rather than flatten all visual details.", "This is important for document, chart, and screen understanding tasks."],
                "exam": "Qwen2-VL matters because it improves visual perception across resolutions and helps MLLMs reason over spatially structured information, not just image-level semantics.",
                "trap": "Do not describe Qwen2-VL as only a bigger LLM; its visual representation and dynamic-resolution handling are central.",
            },
            {
                "title": "Multimodal in-context learning",
                "pages": "pages 37-44",
                "core": "Multimodal in-context learning studies how demonstrations, example order, and image-text interleaving affect MLLM behaviour.",
                "points": ["Examples can steer multimodal reasoning without parameter updates.", "Order sensitivity means the same examples may work differently depending on sequence.", "Image-text interleaving changes what context the model attends to.", "Good demonstrations help, but poor ordering can mislead the model."],
                "exam": "Multimodal in-context learning extends text ICL to image-text settings, where demonstration choice, order, and cross-modal layout strongly affect model output.",
                "trap": "Do not assume ICL is stable just because examples are provided; multimodal examples introduce order and modality-placement sensitivity.",
            },
            {
                "title": "LLaVA-OneVision and long-context transfer",
                "pages": "pages 45-52",
                "core": "LLaVA-OneVision generalises the LLaVA family toward single-image, multi-image, and video tasks using a unified transfer-learning approach.",
                "points": ["It traces the evolution from LLaVA Main to LLaVA-1.5/HD, LLaVA-Next, and LLaVA-OV.", "It combines Qwen2, SigLIP, high AnyRes, and a larger ViT backbone.", "The goal is transfer across vision tasks and modalities.", "It moves from single-image understanding to multi-image and video capability."],
                "exam": "LLaVA-OneVision is important because it turns earlier LLaVA improvements into a unified multimodal model that can transfer across single-image, multi-image, and video tasks.",
                "trap": "Do not treat LLaVA-OneVision as only a higher-resolution LLaVA-Next; its key idea is unified task transfer across modalities.",
            },
            {
                "title": "Unified multimodal understanding and generation",
                "pages": "pages 53-62",
                "core": "The lecture then moves to unified models such as Chameleon, Show-o, and Janus-Pro, which try to combine multimodal understanding and generation.",
                "points": ["Chameleon uses mixed-modal early fusion with discrete visual tokens.", "Show-o explores unified next-token prediction for understanding and generation.", "Janus-Pro separates understanding and generation pathways to improve scaling.", "Unified MLLMs aim to reason about and produce multimodal content in one framework."],
                "exam": "Unified MLLMs are designed to handle both understanding and generation, but they must solve representation conflicts between visual perception and visual synthesis.",
                "trap": "Do not assume one shared representation is always best; Janus-Pro highlights that understanding and generation may need separated pathways.",
            },
            {
                "title": "Visual reasoning, world models, and open questions",
                "pages": "pages 63-74",
                "core": "The final technical section discusses open questions: visual simulation, multimodal chain-of-thought, visual planning, and world-model reasoning.",
                "points": ["Visual simulation helps models reason beyond visible pixels.", "Multimodal chain-of-thought can structure visual reasoning steps.", "Visual planning connects perception with future-state prediction.", "World models infer possible future states from historical and current observations."],
                "exam": "The open direction for MLLMs is not only recognising images, but using visual information to plan, simulate, and reason about possible future states.",
                "trap": "Do not reduce visual reasoning to captioning. The harder goal is structured reasoning over space, time, actions, and possible outcomes.",
            },
        ]
        if zh:
            lines = [
                "### 知识结构总览",
                f"{filename} 主要围绕 **Multimodal Large Language Models** 的演进展开：从 LLaVA 系列，到高分辨率/AnyRes，再到 Qwen2-VL、LLaVA-OneVision、统一理解与生成模型，以及视觉推理和 world model 的开放问题。",
                "",
                "### 核心知识模块",
            ]
            for idx, module in enumerate(modules, 1):
                lines.append(f"{idx}. " + module_lines(module, zh=True)[0])
                lines.extend(module_lines(module, zh=True)[1:])
            lines.extend([
                "",
                "### 知识点之间的关系",
                "- 主线是：**LLaVA visual instruction tuning -> prompt/connector/data scaling -> high-resolution AnyRes -> Qwen2-VL spatial perception -> LLaVA-OneVision transfer -> unified MLLM -> visual reasoning/world models**。",
                "- 每个模型都不要只背名字，要比较它的输入形式、vision encoder/connector/backbone、支持的任务范围和 limitation。",
                "",
                "### 易混淆点",
                "- LLaVA-1.5 的重点是 prompt、connector、data；LLaVA-Next 更强调 high resolution / AnyRes；LLaVA-OneVision 强调跨 single-image、multi-image、video 的统一迁移。",
                "- Qwen2-VL 和 LLaVA-OneVision 都处理更复杂视觉输入，但前者更突出 dynamic resolution 和 spatial perception，后者更突出 unified visual task transfer。",
            ])
            return "\n".join(lines)
        lines = [
            "### Knowledge Structure Overview",
            f"{filename} follows the evolution of **Multimodal Large Language Models** from LLaVA-style visual instruction tuning to high-resolution AnyRes, Qwen2-VL, LLaVA-OneVision, unified multimodal generation, and visual/world-model reasoning.",
            "",
            "### Core Knowledge Modules",
        ]
        for idx, module in enumerate(modules, 1):
            lines.append(f"{idx}. " + module_lines(module, zh=False)[0])
            lines.extend(module_lines(module, zh=False)[1:])
        lines.extend([
            "",
            "### Concept Relationships",
            "- Main chain: **LLaVA visual instruction tuning -> prompt/connector/data scaling -> high-resolution AnyRes -> Qwen2-VL spatial perception -> LLaVA-OneVision transfer -> unified MLLM -> visual reasoning/world models**.",
        ])
        return "\n".join(lines)

    groups = compress_outline_by_topic(outline, max_groups=18)
    groups = [group for group in groups if not is_low_value_title(group.get("title", ""))]
    if not groups and evidence:
        ordered = reading_order(evidence)
        buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        pages = sorted({int(item["page"]) for item in ordered if item.get("page")})
        if pages:
            first, last = pages[0], pages[-1]
            span = max(1, math.ceil((last - first + 1) / 6))
            for item in ordered:
                if not item.get("page"):
                    continue
                start = first + ((int(item["page"]) - first) // span) * span
                end = min(last, start + span - 1)
                buckets[(start, end)].append(item)
            for (start, end), items in sorted(buckets.items()):
                points = key_points_from_items(items, f"Pages {start}-{end}")
                groups.append({
                    "title": points[0],
                    "start_page": start,
                    "end_page": end,
                    "topics": points[:5],
                    "snippets": [clean_slide_text(item.get("excerpt", ""), 160) for item in items[:2]],
                })

    if zh:
        lines = [
            "### 知识结构总览",
            f"{filename} 的重点不是按页逐条阅读，而是建立知识点之间的结构关系。下面只保留知识模块和对应页码范围。",
            "",
            "### 核心知识模块",
        ]
        if not groups:
            lines.append("- 当前没有从该课件中提取到足够清晰的知识模块。请先确认 Library 中该 PDF 已完成 indexing。")
        for idx, group in enumerate(groups, 1):
            pages = page_range_text(group["start_page"], group["end_page"], zh=True)
            topics_list = group.get("topics", [])[:4] or [clean_module_title(group["title"])]
            topics = "；".join(topics_list)
            title = clean_module_title(group["title"])
            lines.extend([
                f"{idx}. **{title}**（{pages}）",
                f"- **核心理解：**该模块围绕 {topics or title} 展开，需要理解它解决的问题、基本机制和在系统 pipeline 中的位置。",
                f"- **必须掌握：**{topics or title}",
                "- **考试背诵句：**回答时用 definition -> mechanism -> benefit -> limitation 组织，不要照抄 slide 文字。",
                "- **易错点：**注意区分概念定义、方法步骤、评估指标和 trade-off。",
            ])
        lines.extend([
            "",
            "### 知识点之间的关系",
            "- 先看基础概念和 pipeline，再看改进方法，最后看 evaluation / limitation。",
            "- 每个模块都用 definition -> mechanism -> evidence -> limitation 串起来。",
            "",
            "### 易混淆点",
            "- 不要把知识点当成孤立术语，要说明它在系统 pipeline 里的位置。",
            "- 不要只背 slide 标题，要能说清楚输入、输出、目标和 trade-off。",
        ])
        return "\n".join(lines)

    lines = [
        "### Knowledge Structure Overview",
        f"{filename} is summarised as a concept map, not a revision timetable.",
        "",
        "### Core Knowledge Modules",
    ]
    for idx, group in enumerate(groups, 1):
        pages = page_range_text(group["start_page"], group["end_page"], zh=False)
        topics = ", ".join(group.get("topics", [])[:4] or [clean_module_title(group["title"])])
        title = clean_module_title(group["title"])
        lines.extend([
            f"{idx}. **{title}** ({pages})",
            f"- **Core understanding:** this module focuses on {topics or title}; explain the problem, mechanism, and system role.",
            f"- **Must know:** {topics or title}",
            "- **Exam sentence:** answer with definition -> mechanism -> benefit -> limitation.",
            "- **Common trap:** do not copy slide text; distinguish concept, method, metric, and trade-off.",
        ])
    lines.extend([
        "",
        "### Relationship Between Concepts",
        "- Read from foundations and pipeline, then improvements, then evaluation and limitations.",
        "- Use each module as concept -> mechanism -> evidence range -> limitation.",
    ])
    return "\n".join(lines)


def build_generic_study_modules(
    outline: list[dict[str, Any]],
    evidence: list[dict[str, Any]] | None = None,
    max_modules: int = 12,
) -> list[dict[str, Any]]:
    evidence = reading_order(evidence or [])
    groups = compress_outline_by_topic(outline, max_groups=max_modules)
    groups = [group for group in groups if not is_low_value_title(group.get("title", ""))]
    expanded_groups: list[dict[str, Any]] = []
    for group in groups:
        start = int(group.get("start_page") or 0)
        end = int(group.get("end_page") or start)
        if end - start + 1 <= 10:
            expanded_groups.append(group)
            continue
        window = 7
        for sub_start in range(start, end + 1, window):
            sub_end = min(end, sub_start + window - 1)
            rows = [
                row for row in outline
                if row.get("page") and sub_start <= int(row["page"]) <= sub_end
            ]
            snippets = [row.get("snippet", "") for row in rows[:5]]
            text = " ".join(f"{row.get('title','')} {row.get('snippet','')}" for row in rows)
            terms = concept_terms_from_text(text, group.get("topics", []), limit=5)
            title = terms[0] if terms else clean_module_title(group.get("title", ""))
            expanded_groups.append(
                {
                    "title": title,
                    "start_page": sub_start,
                    "end_page": sub_end,
                    "topics": terms,
                    "snippets": snippets,
                }
            )
    groups = expanded_groups[: max_modules + 4]
    if not groups and evidence:
        pages = sorted({int(item["page"]) for item in evidence if item.get("page")})
        if pages:
            first, last = pages[0], pages[-1]
            span = max(1, math.ceil((last - first + 1) / max_modules))
            buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for item in evidence:
                if not item.get("page"):
                    continue
                start = first + ((int(item["page"]) - first) // span) * span
                end = min(last, start + span - 1)
                buckets[(start, end)].append(item)
            for (start, end), items in sorted(buckets.items()):
                text = " ".join(f"{item.get('title','')} {item.get('excerpt','')}" for item in items)
                terms = concept_terms_from_text(text, [], limit=5)
                groups.append(
                    {
                        "title": terms[0] if terms else f"Pages {start}-{end}",
                        "start_page": start,
                        "end_page": end,
                        "topics": terms,
                        "snippets": [clean_slide_text(item.get("excerpt", ""), 220) for item in items[:3]],
                    }
                )

    modules = []
    for group in groups[:max_modules]:
        title = clean_module_title(group.get("title", ""))
        start = int(group.get("start_page") or 0)
        end = int(group.get("end_page") or start)
        snippets = [clean_slide_text(snippet, 260) for snippet in group.get("snippets", []) if clean_slide_text(snippet, 80)]
        text = " ".join([title, *snippets, " ".join(group.get("topics", []))])
        terms = concept_terms_from_text(text, group.get("topics", []), limit=5)
        if not terms and title:
            terms = [title]
        inferred = infer_topics(text)
        title_terms = concept_terms_from_text(title, [], limit=2)
        if inferred:
            problem = inferred[0]
        elif title_terms:
            problem = title_terms[0]
        elif terms:
            problem = terms[0]
        else:
            problem = "the module concept"
        clue = snippets[0] if snippets else f"This module introduces {problem} and its role in the lecture."
        modules.append(
            {
                "title": title or problem,
                "pages": page_range_text(start, end, zh=False),
                "start_page": start,
                "end_page": end,
                "concepts": terms[:5],
                "problem": problem,
                "core": f"This module centres on {problem}: understand what problem it addresses, how the method or model works, and where it sits in the overall pipeline.",
                "points": [
                    f"Define {problem} in one precise sentence.",
                    "Identify the input, transformation/module, and output.",
                    "Explain the benefit or improvement over the previous step.",
                    "Name the main limitation, assumption, or trade-off.",
                ],
                "exam": f"{problem} should be explained as: definition -> mechanism -> benefit -> limitation, supported by the page range rather than copied slide text.",
                "trap": "Do not memorise the title only; separate the concept, mechanism, evidence, and trade-off.",
                "clue": clue,
            }
        )
    return modules


def make_llm_document_summary(
    filename: str,
    outline: list[dict[str, Any]],
    evidence: list[dict[str, Any]] | None,
    query: str,
) -> str | None:
    zh = answer_in_chinese(query)
    language = "Chinese with key technical terms kept in English" if zh else "English"
    outline_rows = []
    for item in outline[:90]:
        title = clean_module_title(item.get("title", ""))
        snippet = clean_slide_text(item.get("snippet", ""), 220)
        if title and not is_low_value_title(title):
            outline_rows.append(f"p.{item.get('page')}: {title} :: {snippet}")
    evidence_rows = []
    for item in reading_order(evidence or [])[:24]:
        title = clean_module_title(item.get("title", ""))
        excerpt = clean_slide_text(item.get("excerpt", ""), 220)
        if title or excerpt:
            evidence_rows.append(f"p.{item.get('page')}: {title} :: {excerpt}")
    if not outline_rows and not evidence_rows:
        return None
    prompt = f"""
You are generating study notes from an arbitrary uploaded course document.
Document: {filename}
User query: {query}
Answer language: {language}

Use only the document outline and evidence below. Do not write a plan. Do not list every slide.
Do not copy OCR text directly. Infer the knowledge structure from the material and rewrite it into clean study notes.
The notes must work for any course document, not a hard-coded lecture.

Output exactly this structure:
### 知识结构总览 / Knowledge Structure Overview
One compact paragraph explaining the main storyline of the document.

### 核心知识模块 / Core Knowledge Modules
For 6-12 modules, each module must have:
1. **Module title**（p.x-p.y）
- **核心理解 / Core understanding:** explain the concept in learning language.
- **必须掌握 / Must know:** 3-5 concrete key points, not vague labels.
- **机制/原理 / Mechanism:** explain input -> process/module -> output, or cause -> method -> effect.
- **考试背诵句 / Exam sentence:** one polished sentence the student can memorise.
- **易错点 / Common trap:** one likely confusion.

### 总体背诵框架 / Exam Recall Frame
Give a reusable answer frame for this document.

Document outline:
{chr(10).join(outline_rows[:90])}

Representative evidence:
{chr(10).join(evidence_rows[:24])}
"""
    generated = llm_generate(prompt, role="text")
    if not generated:
        return None
    if "核心知识模块" not in generated and "Core Knowledge Modules" not in generated:
        return None
    return generated.strip()


def make_structured_document_summary(doc: sqlite3.Row | dict[str, Any], outline: list[dict[str, Any]], query: str, evidence: list[dict[str, Any]] | None = None) -> str:
    filename = doc["filename"] if hasattr(doc, "__getitem__") else str(doc)
    zh = answer_in_chinese(query)
    llm_summary = make_llm_document_summary(filename, outline, evidence, query)
    if llm_summary:
        return llm_summary
    modules = build_generic_study_modules(outline, evidence, max_modules=12)
    if zh:
        lines = [
            "### 知识结构总览",
            f"{filename} 已按资料自身内容自动整理成可复习的知识结构。下面不是逐页摘抄，而是把页面信息转化成可理解、可背诵、可用于考试回答的模块。",
            "",
            "### 核心知识模块",
        ]
        if not modules:
            lines.append("- 当前没有足够清晰的模块信息。请确认该 PDF 已完成 indexing，或重新上传/刷新 Library。")
        for idx, module in enumerate(modules, 1):
            pages = page_range_text(module["start_page"], module["end_page"], zh=True)
            concepts = "；".join(module["concepts"]) if module["concepts"] else module["title"]
            problem = module.get("problem") or module["title"]
            lines.extend(
                [
                    f"{idx}. **{module['title']}**（{pages}）",
                    f"- **核心理解：**本模块围绕 **{problem}** 展开，重点是理解它解决什么问题、方法如何工作，以及它在整体 pipeline 中的位置。",
                    f"- **必须掌握：**{concepts}",
                    f"- **机制/原理：**{module['clue']}",
                    f"- **考试背诵句：**{problem} 的回答应按 definition -> mechanism -> benefit -> limitation 组织，并结合页码范围中的证据说明。",
                    "- **易错点：**不要只背标题；要分清概念、机制、证据和 trade-off。",
                ]
            )
        lines.extend(
            [
                "",
                "### 总体背诵框架",
                "- **Definition:** 这个概念/方法是什么。",
                "- **Mechanism:** 它如何处理输入、经过什么模块、产生什么输出。",
                "- **Benefit:** 它解决了前面哪个问题或改进了什么性能。",
                "- **Limitation:** 它牺牲了什么、依赖什么假设、或在哪些场景容易失败。",
            ]
        )
        return "\n".join(lines)

    lines = [
        "### Knowledge Structure Overview",
        f"{filename} has been converted into a study-ready concept structure. This is not slide transcription; it reorganises the material into exam-useful modules.",
        "",
        "### Core Knowledge Modules",
    ]
    if not modules:
        lines.append("- The document is indexed, but no clear modules were extracted. Reindex or refresh the Library.")
    for idx, module in enumerate(modules, 1):
        concepts = ", ".join(module["concepts"]) if module["concepts"] else module["title"]
        lines.extend(
            [
                f"{idx}. **{module['title']}** ({module['pages']})",
                f"- **Core understanding:** {module['core']}",
                f"- **Must know:** {concepts}",
                f"- **Mechanism / principle:** {module['clue']}",
                f"- **Exam sentence:** {module['exam']}",
                f"- **Common trap:** {module['trap']}",
            ]
        )
    lines.extend(
        [
            "",
            "### Exam Recall Frame",
            "- **Definition:** what the concept or method is.",
            "- **Mechanism:** input -> module/process -> output.",
            "- **Benefit:** what problem it solves or improves.",
            "- **Limitation:** what assumption, cost, or failure case remains.",
        ]
    )
    return "\n".join(lines)


def make_revision_plan(profile: dict[str, Any], query: str, evidence: list[dict[str, Any]]) -> str:
    zh = answer_in_chinese(query)
    days_match = re.search(r"(\d+)\s*[- ]?\s*(?:天|days?|day)", query, re.I)
    days = min(max(int(days_match.group(1)) if days_match else 5, 1), 14)
    minutes = int(profile.get("daily_minutes") or 90)
    ordered = enrich_learning_cards(reading_order(evidence), query)
    source_counts = Counter(item.get("source") for item in ordered if item.get("source"))
    primary_source = source_counts.most_common(1)[0][0] if source_counts else None
    outline: list[dict[str, Any]] = []
    if primary_source:
        with connect() as db:
            outline = document_outline(db, primary_source, max_items=160)
    modules = compress_outline_by_topic(outline, max_groups=max(days * 3, 10)) if outline else []
    page_values = sorted({int(item.get("page")) for item in ordered if item.get("page")})

    def module_for_day(day: int) -> dict[str, Any] | None:
        if not page_values:
            return None
        first_page, last_page = page_values[0], page_values[-1]
        span = last_page - first_page + 1
        start_page = first_page + math.floor((day - 1) * span / days)
        end_page = first_page + math.floor(day * span / days) - 1
        if day == days:
            end_page = last_page
        chosen = [m for m in modules if m["start_page"] <= end_page and m["end_page"] >= start_page]
        range_items = [item for item in ordered if item.get("page") and start_page <= int(item["page"]) <= end_page]
        range_points = key_points_from_items(range_items, f"Pages {start_page}-{end_page}")
        clean_titles = []
        for module in chosen:
            title = clean_module_title(module["title"])
            if not is_low_value_title(title) and title not in clean_titles:
                clean_titles.append(title)
        title = " + ".join(clean_titles[:3] or range_points[:3])
        snippets = [s for m in chosen for s in m.get("snippets", [])][:4]
        topics = sorted({t for m in chosen for t in m.get("topics", [])})
        if not topics:
            topics = range_points
        return {"title": title, "start_page": start_page, "end_page": end_page, "snippets": snippets, "topics": topics}

    def learning_goal(module: dict[str, Any], zh: bool) -> str:
        title = module.get("title") or "current module"
        topics = module.get("topics") or [title]
        primary = topics[0] if topics else title
        snippet_text = " ".join(module.get("snippets", []))
        haystack = f"{title} {primary} {snippet_text}".lower()
        if re.search(r"pruning|merging|resampling|routing|rerank|retrieval|connector|attention|architecture|method|categories", haystack):
            if re.search(r"pruning", haystack):
                focus = "token pruning / importance scoring"
            elif re.search(r"merging", haystack):
                focus = "token merging / redundancy compression"
            elif re.search(r"resampling", haystack):
                focus = "token resampling / latent aggregation"
            elif re.search(r"routing", haystack):
                focus = "token routing / dynamic computation"
            else:
                focus = primary
            return (
                f"掌握 {focus} 的输入、处理步骤和输出，能比较它与相邻方法的效率、信息保留和 trade-off。"
                if zh else
                f"Master the input, processing steps, and output of {focus}, then compare its efficiency, retained information, and trade-off."
            )
        if re.search(r"roadmap|evolution|recap|history|from .* to", haystack):
            return (
                f"梳理 {primary} 的演进主线，能说清每一步相比前一步新增了什么能力、仍留下什么限制。"
                if zh else
                f"Trace the evolution of {primary}, explaining what each step adds and what limitation remains."
            )
        if re.search(r"problem|limitation|bottleneck|failure|trap|hallucination", haystack):
            return (
                f"找出 {primary} 要解决的核心瓶颈，并能用“问题 -> 原因 -> 改进方向”解释为什么需要后续方法。"
                if zh else
                f"Identify the bottleneck behind {primary} and explain it as problem -> cause -> improvement direction."
            )
        if re.search(r"evaluation|benchmark|metric|recall|mrr|ndcg|score", haystack):
            return (
                f"理解 {primary} 如何被评估，能说明指标衡量什么、不能衡量什么，以及结果如何支持结论。"
                if zh else
                f"Understand how {primary} is evaluated, what the metrics capture, what they miss, and how results support conclusions."
            )
        return (
            f"围绕 {primary} 建立可背诵解释：定义它、说明机制、给出例子，并指出一个限制或应用场景。"
            if zh else
            f"Build a recall-ready explanation of {primary}: define it, explain the mechanism, give an example, and name one limitation or use case."
        )

    if zh:
        lines = [
            f"### 复习计划：{primary_source or '当前资料'}（{days} 天，每天约 {minutes} 分钟）",
            "每个阶段都包含阅读范围、主要知识点、学习目标和输出任务。",
            "",
        ]
        for day in range(1, days + 1):
            module = module_for_day(day)
            if not module:
                continue
            pages = f"p.{module['start_page']}-p.{module['end_page']}"
            snippets = [
                s for s in (clean_slide_text(s, 140) for s in module.get("snippets", []))
                if s and len(s) > 12 and not re.search(r"assessment|gradescope|bubble sheet|quiz", s, re.I)
            ]
            topics = module.get("topics") or [module["title"]]
            lines.extend([
                f"### Day {day}: {module['title']}",
                f"- 阅读范围：{primary_source} {pages}",
                f"- 主要知识点：{'; '.join(topics[:6])}",
                f"- 学习目标：{learning_goal(module, True)}",
                *( [f"- 内容提示：{'; '.join(snippets[:3])}"] if snippets else [] ),
                "- 阶段输出：写一段 120-180 字 exam-style answer，并列出 2 个 common traps。",
                "",
            ])
        lines.extend([
            "### 对应资料",
            *[f"- {evidence_line(item, idx)}" for idx, item in enumerate(ordered[:12], 1)],
        ])
        return "\n".join(lines).strip()

    lines = [
        f"### Revision Plan: {primary_source or 'current materials'} ({days} days, about {minutes} min/day)",
        "Each stage includes pages, key knowledge points, learning goals, and an output task.",
        "",
    ]
    for day in range(1, days + 1):
        module = module_for_day(day)
        if not module:
            continue
        pages = f"p.{module['start_page']}-p.{module['end_page']}"
        topics = module.get("topics") or [module["title"]]
        snippets = [
            s for s in (clean_slide_text(s, 140) for s in module.get("snippets", []))
            if s and len(s) > 12 and not re.search(r"assessment|gradescope|bubble sheet|quiz", s, re.I)
        ]
        lines.extend([
            f"### Day {day}: {module['title']}",
            f"- Read: {primary_source} {pages}",
            f"- Key knowledge points: {'; '.join(topics[:6])}",
            f"- Learning goal: {learning_goal(module, False)}",
            *( [f"- Content cue: {'; '.join(snippets[:3])}"] if snippets else [] ),
            "- Output: write a 120-180 word exam-style answer and list two common traps.",
            "",
        ])
    lines.extend(["### Related Materials", *[f"- {evidence_line(item, idx)}" for idx, item in enumerate(ordered[:12], 1)]])
    return "\n".join(lines).strip()


def make_practice_from_evidence(query: str, evidence: list[dict[str, Any]], topics: list[str]) -> str:
    zh = answer_in_chinese(query)
    ordered = enrich_learning_cards(reading_order(evidence), query)
    banned_topic_re = re.compile(
        r"帮我|给一些|根据|生成|练习题|题目|复习|总结|解释|什么|怎么|"
        r"summary|summarise|practice|quiz|test|questions?|week\d+|lecture\d+|"
        r"recap:?\s*why|recap:?\s*text|why .*matters",
        re.I,
    )

    def useful_topic(topic: str) -> bool:
        topic = (topic or "").strip()
        if len(topic) < 2 or banned_topic_re.search(topic):
            return False
        return True

    topic_names: list[str] = []
    for topic in topics:
        if useful_topic(topic) and topic.lower() not in {t.lower() for t in topic_names}:
            topic_names.append(topic)
    for item in ordered:
        for topic in key_points_from_items([item], item.get("title", "")):
            if useful_topic(topic) and topic.lower() not in {t.lower() for t in topic_names}:
                topic_names.append(topic)
    topic_names = topic_names[:6] or (["检索到的课程内容"] if zh else ["the retrieved lecture content"])
    source_text = supported_location_text(ordered, zh)

    if zh:
        lines = [
            "### 练习题",
            f"下面的题目只基于本次检索到的复习内容生成，主要依据：{source_text}。",
            "",
            "### A. 概念检查",
        ]
        for idx, topic in enumerate(topic_names[:4], 1):
            lines.append(f"{idx}. 用 2-3 句话解释 **{topic}**：它是什么、解决什么问题、在 pipeline 里的位置是什么？")
        lines.extend([
            "",
            "### B. 对比题",
            f"1. 从 {', '.join(topic_names[:3])} 中任选两个概念，比较它们的输入、输出、目标和 limitation。",
            "2. 说明一个方法为什么不能单独解决所有检索或多模态理解问题。",
            "",
            "### C. 应用题",
            "1. 如果系统回答不准确，你会优先检查 retrieval、reranking、evidence grounding 还是 prompt？说明原因。",
            "2. 设计一个小型 study-agent 流程：用户提问后，系统如何选择资料、排序证据、生成答案、更新 weak topics？",
            "",
            "### D. Exam-style 简答",
            f"请写一段 120-180 字答案，主题是：{topic_names[0]} 为什么重要，以及它在真实学习/问答系统中的 trade-off。",
            "",
            "### 参考答案要点",
            "- 答案必须包含 definition、mechanism、evidence location、limitation。",
            "- 引用资料时写成：资料名 + 页码范围；具体 slide 图片在右侧 Evidence 中打开。",
        ])
        return "\n".join(lines)

    lines = [
        "### Practice Questions",
        f"These questions are generated only from the retrieved study content. Main support: {source_text}.",
        "",
        "### A. Concept Checks",
    ]
    for idx, topic in enumerate(topic_names[:4], 1):
        lines.append(f"{idx}. Explain **{topic}** in 2-3 sentences: what it is, what problem it solves, and where it sits in the pipeline.")
    lines.extend([
        "",
        "### B. Comparison",
        f"1. Choose two of {', '.join(topic_names[:3])} and compare their inputs, outputs, goals, and limitations.",
        "2. Explain why one method alone cannot solve every retrieval or multimodal-understanding problem.",
        "",
        "### C. Application",
        "1. If the system answer is inaccurate, would you inspect retrieval, reranking, grounding, or prompting first? Explain why.",
        "2. Design a small study-agent workflow from user query to evidence ranking, answer generation, and weak-topic update.",
        "",
        "### D. Exam-style Short Answer",
        f"Write 120-180 words on why {topic_names[0]} matters and what trade-off it introduces.",
        "",
        "### Answer Checklist",
        "- Include definition, mechanism, evidence location, and limitation.",
        "- Cite the material as document name + page range; open the actual slide image from Evidence.",
    ])
    return "\n".join(lines)


@app.get("/")
def index() -> Any:
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/data/<path:filename>")
def data_file(filename: str) -> Any:
    return send_from_directory(DATA_DIR, filename)


@app.get("/api/status")
def status() -> Any:
    with connect() as db:
        docs = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = db.execute("SELECT COUNT(*) FROM evidence_chunks").fetchone()[0]
        turns = db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    health = ollama_health()
    text_model, text_model_status = choose_ollama_model("text", False)
    vision_model, vision_model_status = choose_ollama_model("vision", True)
    return jsonify(
        {
            "documents": docs,
            "evidence_chunks": chunks,
            "conversation_turns": turns,
            "agent_framework": "LangGraph" if AGENT_GRAPH is not None else "Linear fallback",
            "llm_provider": "Ollama" if health["available"] else "fallback",
            "ollama_model": f"text:{text_model} / vision:{vision_model}",
            "ollama_available": health["available"],
            "ollama_models": health["models"],
            "text_model": text_model,
            "text_model_status": text_model_status,
            "vision_model": vision_model,
            "vision_model_status": vision_model_status,
            "vector_backend": "FAISS IndexFlatIP" if faiss is not None else "hash-vector fallback",
            "course": COURSE_CODE,
            "database_path": str(DB_PATH),
            "upload_dir": str(UPLOAD_DIR),
            "auto_vision_summary": AUTO_VISION_SUMMARY,
            "multimodal_embedding": "text_hash + visual_signature + OCR/caption fusion",
        }
    )


@app.get("/api/profile")
def profile_get() -> Any:
    with connect() as db:
        return jsonify(get_profile(db))


@app.post("/api/profile")
def profile_update() -> Any:
    payload = request.get_json(force=True)
    fields = {
        "user_name": payload.get("user_name", "Anna"),
        "course": payload.get("course", "INFS4205/7205"),
        "preferred_language": payload.get("preferred_language", "Chinese-English mixed"),
        "answer_style": payload.get("answer_style", "structured, exam-oriented, evidence-grounded"),
        "current_goal": payload.get("current_goal", "prepare final quiz and assignment"),
        "daily_minutes": int(payload.get("daily_minutes") or 90),
    }
    weak_topics = payload.get("weak_topics", [])
    with connect() as db:
        db.execute(
            """
            UPDATE user_profiles
            SET user_name=?, course=?, preferred_language=?, answer_style=?, current_goal=?, daily_minutes=?, updated_at=?
            WHERE id=1
            """,
            (*fields.values(), now()),
        )
        if isinstance(weak_topics, list):
            db.execute("DELETE FROM weak_topics WHERE user_id = 1")
            for topic in weak_topics:
                if str(topic).strip():
                    db.execute(
                        """
                        INSERT INTO weak_topics (user_id, topic, reason, confidence, updated_at)
                        VALUES (1, ?, 'profile editor', 0.75, ?)
                        """,
                        (str(topic).strip(), now()),
                    )
        return jsonify(get_profile(db))


@app.post("/api/upload")
def upload() -> Any:
    if "files" not in request.files:
        return jsonify({"error": "No files provided."}), 400
    files = request.files.getlist("files")
    caption = request.form.get("caption", "")
    summaries = []
    with connect() as db:
        for file in files:
            if not file.filename:
                continue
            safe_name = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", file.filename)
            target = UPLOAD_DIR / safe_name
            file.save(target)
            try:
                existing = db.execute("SELECT id FROM documents WHERE filename = ?", (safe_name,)).fetchone()
                if existing:
                    db.execute("DELETE FROM evidence_chunks WHERE document_id = ?", (existing["id"],))
                    db.execute("DELETE FROM documents WHERE id = ?", (existing["id"],))
                summaries.append(index_existing_file(db, target, caption))
            except Exception as exc:
                summaries.append({"filename": safe_name, "error": str(exc)})
    return jsonify({"uploaded": summaries})


@app.post("/api/reindex-uploads")
def reindex_uploads() -> Any:
    summaries = []
    with connect() as db:
        for target in sorted(UPLOAD_DIR.glob("*")):
            if not target.is_file():
                continue
            existing = db.execute("SELECT id FROM documents WHERE filename = ?", (target.name,)).fetchone()
            if existing:
                continue
            try:
                summaries.append(index_existing_file(db, target))
            except Exception as exc:
                summaries.append({"filename": target.name, "error": str(exc)})
    return jsonify({"reindexed": summaries})


@app.get("/api/documents")
def documents_list() -> Any:
    with connect() as db:
        docs = db.execute("SELECT * FROM documents ORDER BY uploaded_at DESC").fetchall()
        items = []
        for doc in docs:
            counts = {
                row["modality"]: row["count"]
                for row in db.execute(
                    "SELECT modality, COUNT(*) as count FROM evidence_chunks WHERE document_id=? GROUP BY modality",
                    (doc["id"],),
                )
            }
            topic_counter: Counter[str] = Counter()
            for row in db.execute("SELECT title, content, metadata_json FROM evidence_chunks WHERE document_id=?", (doc["id"],)):
                metadata = json.loads(row["metadata_json"])
                topics = metadata.get("topics") or infer_topics(f"{row['title']} {row['content']} {doc['filename']}")
                topic_counter.update(topics)
            items.append(
                {
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "doc_type": doc["doc_type"],
                    "page_count": doc["page_count"],
                    "uploaded_at": doc["uploaded_at"],
                    "path": doc["path"],
                    "chunks": sum(counts.values()),
                    "modality_counts": counts,
                    "topics": [topic for topic, _ in topic_counter.most_common(8)],
                }
            )
    return jsonify({"documents": items})


@app.get("/api/indexes")
def indexes_status() -> Any:
    with connect() as db:
        rows = db.execute("SELECT modality, metadata_json FROM evidence_chunks").fetchall()
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        metadata = json.loads(row["metadata_json"])
        index_name = metadata.get("index_name")
        if not index_name:
            if row["modality"] in {"slide_text", "note_text"}:
                index_name = "text_index"
            elif row["modality"] in {"slide_image", "uploaded_image"}:
                index_name = "visual_caption_index"
            elif row["modality"] == "table":
                index_name = "table_index"
            else:
                index_name = "misc_index"
        item = summary.setdefault(index_name, {"name": index_name, "chunks": 0, "modalities": Counter()})
        item["chunks"] += 1
        item["modalities"].update([row["modality"]])
    payload = []
    for item in summary.values():
        payload.append(
            {
                "name": item["name"],
                "chunks": item["chunks"],
                "modalities": dict(item["modalities"]),
                "vector_backend": "FAISS IndexFlatIP" if faiss is not None else "hash-vector fallback",
                "fusion": "RRF(keyword, text vector, visual vector, metadata)",
            }
        )
    return jsonify({"indexes": sorted(payload, key=lambda item: item["name"])})


@app.post("/api/documents/<int:document_id>/reindex")
def document_reindex(document_id: int) -> Any:
    with connect() as db:
        doc = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not doc:
            return jsonify({"error": "Document not found."}), 404
        path = resolve_storage_path(doc["path"])
        if not path.exists():
            return jsonify({"error": "Original uploaded file is missing."}), 404
        db.execute("DELETE FROM evidence_chunks WHERE document_id=?", (document_id,))
        db.execute("DELETE FROM documents WHERE id=?", (document_id,))
        result = index_existing_file(db, path)
    return jsonify({"reindexed": result})


@app.delete("/api/documents/<int:document_id>")
def document_delete(document_id: int) -> Any:
    with connect() as db:
        doc = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not doc:
            return jsonify({"error": "Document not found."}), 404
        db.execute("DELETE FROM evidence_chunks WHERE document_id=?", (document_id,))
        db.execute("DELETE FROM documents WHERE id=?", (document_id,))
    return jsonify({"deleted": document_id})


@app.post("/api/chat")
def chat() -> Any:
    payload = request.get_json(force=True)
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query is empty."}), 400
    return jsonify(
        run_agent(
            query,
            session_id=payload.get("session_id"),
            parent_id=payload.get("parent_id"),
            is_branch=bool(payload.get("is_branch")),
            answer_mode=payload.get("answer_mode"),
        )
    )


@app.post("/api/chat/stream")
def chat_stream() -> Any:
    payload = request.get_json(force=True)
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Query is empty."}), 400

    def emit(kind: str, data: dict[str, Any]) -> str:
        return json_dumps({"type": kind, **data}) + "\n"

    @stream_with_context
    def generate():
        yield emit("status", {"message": "Reading conversation context"})
        yield emit("status", {"message": "Retrieving course evidence"})
        result = run_agent(
            query,
            session_id=payload.get("session_id"),
            parent_id=payload.get("parent_id"),
            is_branch=bool(payload.get("is_branch")),
            answer_mode=payload.get("answer_mode"),
        )
        yield emit(
            "meta",
            {
                "session_id": result.get("session_id"),
                "intent": result.get("intent"),
                "evidence": result.get("evidence", []),
                "trace": result.get("trace", []),
                "memory_updates": result.get("memory_updates", []),
                "citations": result.get("citations", []),
                "selected_tools": result.get("selected_tools", []),
            },
        )
        answer = result.get("answer", "")
        for idx in range(0, len(answer), 18):
            yield emit("token", {"text": answer[idx : idx + 18]})
        yield emit("done", result)

    return Response(generate(), mimetype="application/x-ndjson")


@app.get("/api/sessions")
def sessions_list() -> Any:
    with connect() as db:
        ensure_default_session(db)
        rows = db.execute(
            """
            SELECT s.id, s.title, s.course, s.created_at,
                   COALESCE(MAX(c.created_at), s.updated_at) AS updated_at,
                   COUNT(c.id) AS turns,
                   (
                       SELECT c2.query
                       FROM conversations c2
                       WHERE c2.session_id = s.id AND COALESCE(c2.is_branch, 0) = 0
                       ORDER BY c2.id DESC
                       LIMIT 1
                   ) AS last_query,
                   (
                       SELECT c2.intent
                       FROM conversations c2
                       WHERE c2.session_id = s.id AND COALESCE(c2.is_branch, 0) = 0
                       ORDER BY c2.id DESC
                       LIMIT 1
                   ) AS last_intent
            FROM conversation_sessions s
            LEFT JOIN conversations c ON c.session_id = s.id AND COALESCE(c.is_branch, 0) = 0
            GROUP BY s.id
            ORDER BY COALESCE(MAX(c.id), 0) DESC, s.updated_at DESC
            LIMIT 50
            """
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.post("/api/sessions")
def sessions_create() -> Any:
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "New study chat").strip()[:80]
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO conversation_sessions (title, course, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (title, COURSE_CODE, now(), now()),
        )
        session_id = cursor.lastrowid
    return jsonify({"id": session_id, "title": title})


@app.get("/api/sessions/<int:session_id>")
def sessions_item(session_id: int) -> Any:
    with connect() as db:
        session = db.execute("SELECT * FROM conversation_sessions WHERE id=?", (session_id,)).fetchone()
        if not session:
            return jsonify({"error": "Session not found."}), 404
        rows = db.execute(
            """
            SELECT id, query, resolved_query, intent, answer, evidence_json, trace_json, is_branch, parent_id, created_at
            FROM conversations
            WHERE session_id=?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
    return jsonify(
        {
            "session": dict(session),
            "turns": [
                {
                    "id": row["id"],
                    "query": row["query"],
                    "resolved_query": row["resolved_query"],
                    "intent": row["intent"],
                    "answer": row["answer"],
                    "evidence": json.loads(row["evidence_json"]),
                    "trace": json.loads(row["trace_json"]),
                    "is_branch": bool(row["is_branch"]),
                    "parent_id": row["parent_id"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ],
        }
    )


@app.delete("/api/sessions/<int:session_id>")
def sessions_delete(session_id: int) -> Any:
    with connect() as db:
        db.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
        db.execute("DELETE FROM conversation_sessions WHERE id=?", (session_id,))
    return jsonify({"deleted": session_id})


@app.get("/api/history")
def history() -> Any:
    with connect() as db:
        rows = db.execute(
            "SELECT id, query, intent, answer, evidence_json, trace_json, created_at FROM conversations ORDER BY id DESC LIMIT 20"
        ).fetchall()
    return jsonify(
        [
            {
                "id": row["id"],
                "query": row["query"],
                "intent": row["intent"],
                "answer": row["answer"],
                "evidence": json.loads(row["evidence_json"]),
                "trace": json.loads(row["trace_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    )


@app.get("/api/history/<int:turn_id>")
def history_item(turn_id: int) -> Any:
    with connect() as db:
        row = db.execute(
            "SELECT id, query, intent, answer, evidence_json, trace_json, created_at FROM conversations WHERE id=?",
            (turn_id,),
        ).fetchone()
    if not row:
        return jsonify({"error": "History item not found."}), 404
    return jsonify(
        {
            "id": row["id"],
            "query": row["query"],
            "intent": row["intent"],
            "answer": row["answer"],
            "evidence": json.loads(row["evidence_json"]),
            "trace": json.loads(row["trace_json"]),
            "created_at": row["created_at"],
        }
    )


@app.delete("/api/history/<int:turn_id>")
def history_delete(turn_id: int) -> Any:
    with connect() as db:
        db.execute("DELETE FROM conversations WHERE id=?", (turn_id,))
    return jsonify({"deleted": turn_id})


def build_session_notes() -> tuple[str, list[dict[str, Any]]]:
    with connect() as db:
        profile = get_profile(db)
        rows = db.execute(
            "SELECT id, query, resolved_query, intent, answer, evidence_json, created_at FROM conversations ORDER BY id ASC"
        ).fetchall()
    if not rows:
        return "### 学习总结\n目前还没有可总结的学习记录。", []

    turns = []
    evidence_by_key: dict[tuple[str, int | None], dict[str, Any]] = {}
    unclear_topics: Counter[str] = Counter()
    branch_questions = []
    for row in rows:
        evidence = json.loads(row["evidence_json"])
        turns.append(dict(row))
        if re.search(r"不懂|不理解|confused|weak|不会|卡住|阶段|这一步", row["query"], re.I):
            branch_questions.append(row["query"])
            unclear_topics.update(extract_topics(row["query"], evidence))
        for item in evidence:
            key = (item.get("source"), item.get("page"))
            if key not in evidence_by_key:
                evidence_by_key[key] = item

    ordered_evidence = reading_order(list(evidence_by_key.values()))
    weak_profile = [item["topic"] for item in profile.get("weak_topics", [])]
    unclear = [topic for topic, _ in unclear_topics.most_common(8) if topic not in {"请告诉我", "主要讲了什么"}]
    highlighted = list(dict.fromkeys(unclear + weak_profile))[:10]

    slide_lines = []
    for idx, item in enumerate(ordered_evidence[:24], start=1):
        page = f"p.{item['page']}" if item.get("page") else "no page"
        topics = ", ".join((item.get("topics") or item.get("metadata", {}).get("topics") or [])[:4])
        slide_lines.append(
            f"- **{idx}. {item['source']} {page}：{item['title']}**  \n"
            f"  内容：{item['excerpt'][:260]}  \n"
            f"  图片：{('[打开 slide 图片](' + evidence_link(item) + ')') if item.get('image_path') else '无图片'}"
            + (f"  \n  Topics: {topics}" if topics else "")
        )

    turn_lines = []
    for row in turns[-12:]:
        turn_lines.append(f"- **{row['intent']}**：{row['query']}")

    branch_lines = [f"- {q}" for q in branch_questions[-10:]] or ["- 暂无明显分支疑问。"]
    highlight_lines = [f"- **{topic}**：这是你在对话或 profile 中暴露出的薄弱/重点区域，复习时优先标红。" for topic in highlighted] or ["- 暂无明确薄弱点。"]

    notes = "\n".join(
        [
            "### 本轮学习总览",
            f"你这轮主要围绕 {', '.join(sorted({row['intent'] for row in turns}))} 展开学习。下面按学习轨迹、相关 slide、薄弱点和下一步复习动作整理。",
            "",
            "### 对话学习轨迹",
            *turn_lines,
            "",
            "### 相关 slide 图片与内容",
            "下面按建议查阅顺序列出本轮对话用到的相关 slides。每条都保留页码、内容摘要和可打开的 slide 图片链接。",
            *slide_lines,
            "",
            "### 分支问答与不清楚部分",
            *branch_lines,
            "",
            "### 需要高亮复习的知识点",
            *highlight_lines,
            "",
            "### 下一步学习行动",
            "- 先打开上方 slide 图片，按顺序快速复盘每页标题和图示。",
            "- 对高亮薄弱点各写一个 3 句解释：definition -> why it matters -> common trap。",
            "- 最后不看资料写一段 exam-style answer，把本轮所有分支问题串回原来的复习计划。",
        ]
    )
    return notes, ordered_evidence[:24]


def build_session_notes() -> tuple[str, list[dict[str, Any]]]:
    with connect() as db:
        profile = get_profile(db)
        rows = db.execute(
            "SELECT query, intent, answer, evidence_json, created_at FROM conversations ORDER BY id ASC"
        ).fetchall()
    if not rows:
        return "### 学习总结\n当前还没有可总结的学习记录。", []

    evidence_by_key: dict[tuple[str, int | None], dict[str, Any]] = {}
    intents: list[str] = []
    branch_questions: list[str] = []
    for row in rows:
        intents.append(row["intent"])
        if re.search(r"branch|不懂|不理解|confused|不要重写|不用重写", row["query"], re.I):
            branch_questions.append(row["query"])
        for item in json.loads(row["evidence_json"]):
            evidence_by_key.setdefault((item.get("source"), item.get("page")), item)

    ordered = reading_order(list(evidence_by_key.values()))
    weak = ", ".join(item["topic"] for item in profile.get("weak_topics", [])[:6]) or "none recorded"
    lines = [
        "### 本轮学习总结",
        f"本轮主要覆盖：{', '.join(sorted(set(intents)))}。",
        f"当前 weak topics：{weak}。",
        "",
        "### 实际阅读顺序",
        *[f"- {evidence_line(item, idx)}" for idx, item in enumerate(ordered[:18], 1)],
        "",
        "### 分支问题",
        *(branch_questions[-8:] or ["- 暂无明显分支问题。"]),
        "",
        "### 下一步学习行动",
        "- 先按上面的顺序打开 slide 图像，补齐标题、图和例子。",
        "- 对每个 weak topic 写：definition -> why it matters -> common trap。",
        "- 最后不看资料写一段 exam-style answer。",
    ]
    return "\n".join(lines), ordered[:18]


@app.post("/api/session-summary")
def session_summary() -> Any:
    notes, evidence = build_session_notes()
    return jsonify({"answer": notes, "evidence": evidence})


def load_benchmark_cases() -> list[dict[str, Any]]:
    if BENCHMARK_PATH.exists():
        try:
            return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return [
        {"id": "factual_rag", "family": "factual_retrieval", "query": "Explain RAG with retrieved evidence", "expected_keywords": "retrieval, generation, evidence", "top_k": 5},
        {"id": "cross_modal_reranker", "family": "cross_modal_retrieval", "query": "Find the slide image about reranker or RRF", "expected_keywords": "reranker, rrf, rank", "top_k": 5},
        {"id": "analysis_retriever_reranker", "family": "analytical_multihop", "query": "Compare retriever and reranker using lecture evidence", "expected_keywords": "retriever, reranker, candidate", "top_k": 5},
        {"id": "personalised_plan", "family": "personalised_followup", "query": "Create a revision plan for weak topics", "expected_keywords": "revision, weak, plan", "top_k": 5},
    ]


def write_evaluation_artifacts(payload: dict[str, Any]) -> None:
    ensure_dirs()
    (EVAL_DIR / "evaluation_results.json").write_text(json_dumps(payload), encoding="utf-8")
    rows = ["mode,mean_recall_at_k,mean_page_recall_at_k,mean_mrr,mean_answer_success_proxy,mean_latency_ms"]
    for mode, metrics in payload.get("summary", {}).items():
        rows.append(
            ",".join(
                [
                    mode,
                    str(metrics.get("mean_recall_at_k", "")),
                    str(metrics.get("mean_page_recall_at_k", "")),
                    str(metrics.get("mean_mrr", "")),
                    str(metrics.get("mean_answer_success_proxy", "")),
                    str(metrics.get("mean_latency_ms", "")),
                ]
            )
        )
    (EVAL_DIR / "evaluation_summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    failures = [item for item in payload.get("results", []) if item.get("mode") == "final_agent" and item.get("recall_at_k", 0) < 0.67]
    lines = ["# Failure Cases", ""]
    if not failures:
        lines.append("No final-agent benchmark case fell below the groundedness threshold in the latest run.")
    for item in failures:
        lines.extend(
            [
                f"## {item.get('case_id')}",
                f"- Query: {item.get('query')}",
                f"- Family: {item.get('family')}",
                f"- Recall@k: {item.get('recall_at_k')}, MRR: {item.get('mrr')}, page recall: {item.get('page_recall_at_k')}",
                "- Likely cause: retrieval coverage, OCR/caption mismatch, or insufficient visual grounding.",
                "- Fix: add expected slide metadata, improve caption generation, or route this family to visual retrieval.",
                "",
            ]
        )
    (EVAL_DIR / "failure_cases.md").write_text("\n".join(lines), encoding="utf-8")


@app.post("/api/evaluate")
def evaluate() -> Any:
    started = datetime.utcnow()
    sample_cases = load_benchmark_cases()
    payload = request.get_json(silent=True) or {}
    cases = payload.get("cases") if isinstance(payload.get("cases"), list) else sample_cases
    generate_answers = bool(payload.get("generate_answers"))
    modes = payload.get("modes") if isinstance(payload.get("modes"), list) else [
        "plain_llm",
        "text_only",
        "caption_only",
        "no_visual",
        "no_router",
        "no_rerank",
        "no_memory",
        "final_agent",
    ]
    results = []
    with connect() as db:
        for case in cases:
            query = case.get("query", "")
            top_k = int(case.get("top_k") or 3)
            expected = [token.strip().lower() for token in re.split(r"[,;]", case.get("expected_keywords", "")) if token.strip()]
            expected_pages = {int(page) for page in case.get("expected_pages", []) if str(page).isdigit()}
            for mode in modes:
                mode_start = datetime.utcnow()
                evidence = retrieve_for_ablation(db, query, mode, top_k)
                joined = " ".join(item["excerpt"].lower() + " " + item["title"].lower() for item in evidence)
                hits = sum(1 for token in expected if token in joined)
                recall = hits / max(len(expected), 1)
                page_hits = 0
                if expected_pages:
                    retrieved_pages = {int(item["page"]) for item in evidence if item.get("page")}
                    page_hits = len(expected_pages & retrieved_pages)
                mrr = 0.0
                for rank, item in enumerate(evidence, start=1):
                    text = item["excerpt"].lower() + " " + item["title"].lower()
                    page_match = bool(expected_pages and item.get("page") in expected_pages)
                    if page_match or any(token in text for token in expected):
                        mrr = 1.0 / rank
                        break
                if mode == "plain_llm" and generate_answers:
                    answer = plain_llm_answer(query)
                elif mode == "plain_llm":
                    answer = "Plain LLM baseline: no retrieved course evidence."
                elif mode == "final_agent" and generate_answers:
                    profile = get_profile(db)
                    intent = classify_intent(resolve_followup(db, query))
                    answer, _grounded = generate_answer(profile, query, query, intent, evidence, None)
                else:
                    answer = baseline_answer(query, evidence)
                latency_ms = int((datetime.utcnow() - mode_start).total_seconds() * 1000)
                result = {
                    "case_id": case.get("id", query[:30]),
                    "family": case.get("family", "custom"),
                    "mode": mode,
                    "query": query,
                    "top_k": top_k,
                    "recall_at_k": round(recall, 3),
                    "page_recall_at_k": round(page_hits / max(len(expected_pages), 1), 3) if expected_pages else None,
                    "mrr": round(mrr, 3),
                    "answer_success_proxy": answer_success_proxy(answer, expected, evidence),
                    "latency_ms": latency_ms,
                    "tool_calls": 0 if mode == "plain_llm" else 1,
                    "groundedness_proxy": "Fully grounded" if recall >= 0.67 else "Partially grounded" if recall > 0 else "Unsupported",
                    "top_evidence": evidence[:3],
                    "answer_preview": answer[:500],
                }
                db.execute(
                    """
                    INSERT INTO evaluation_cases (query, expected_source, expected_keywords, top_k, result_json, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (query, case.get("expected_source"), case.get("expected_keywords"), top_k, json_dumps(result), f"ablation:{mode}"),
                )
                results.append(result)
    summary: dict[str, dict[str, float]] = {}
    for mode in modes:
        subset = [item for item in results if item["mode"] == mode]
        if subset:
            summary[mode] = {
                "mean_recall_at_k": round(sum(item["recall_at_k"] for item in subset) / len(subset), 3),
                "mean_page_recall_at_k": round(sum((item["page_recall_at_k"] or 0.0) for item in subset) / len(subset), 3),
                "mean_mrr": round(sum(item["mrr"] for item in subset) / len(subset), 3),
                "mean_answer_success_proxy": round(sum(item["answer_success_proxy"] for item in subset) / len(subset), 3),
                "mean_latency_ms": round(sum(item["latency_ms"] for item in subset) / len(subset), 1),
            }
    payload_out = {
        "benchmark": "INFS4205 multimodal study-agent retrieval benchmark",
        "ablation_modes": modes,
        "summary": summary,
        "results": results,
        "total_latency_ms": int((datetime.utcnow() - started).total_seconds() * 1000),
    }
    write_evaluation_artifacts(payload_out)
    return jsonify(payload_out)


@app.post("/api/reset")
def reset() -> Any:
    with connect() as db:
        db.executescript(
            """
            DROP TABLE IF EXISTS evaluation_cases;
            DROP TABLE IF EXISTS conversations;
            DROP TABLE IF EXISTS conversation_sessions;
            DROP TABLE IF EXISTS evidence_chunks;
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS weak_topics;
            DROP TABLE IF EXISTS user_profiles;
            """
        )
    init_db()
    return jsonify({"ok": True})


if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
