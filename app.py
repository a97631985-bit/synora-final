import os
import json
import uuid
import random
from functools import wraps
from datetime import datetime, timedelta

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.mutable import MutableList, MutableDict
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
# SECURITY: always set SYNORA_SECRET to a long random value in production!
# The fallback below is ONLY for local dev — never rely on it in a live deploy.
app.secret_key = os.environ.get("SYNORA_SECRET") or os.urandom(32).hex()
# Session/cookie hardening
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Secure cookies only over HTTPS. Enabled by default on Render (has DATABASE_URL / prod),
# disabled for local HTTP testing unless explicitly forced on.
_is_prod = bool(os.environ.get("DATABASE_URL", "")) or os.environ.get("SYNORA_SECURE_COOKIES") == "1"
app.config["SESSION_COOKIE_SECURE"] = _is_prod
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ---------------------------------------------------------------------------
# Lightweight in-memory rate limiter (IP-based + action-based) to protect
# auth endpoints against brute-force / abuse. Uses a fixed-window counter.
# NOTE: resets on process restart (per-worker). Good enough for abuse defense;
# a Redis-backed limiter can be swapped in for high-scale production.
# ---------------------------------------------------------------------------
import threading
_ratelimit_lock = threading.Lock()
_ratelimit_buckets = {}  # key -> {"count": int, "window_start": epoch_seconds, "period": seconds, "limit": int}


def _rate_key(*parts):
    # Use IP + route identity; fall back consistently so proxies that strip
    # REMOTE_ADDR don't all share one bucket.
    ip = request.remote_addr or "unknown"
    return "|".join([ip] + [str(p) for p in parts])


def _rate_allowed(key, limit, period):
    """Return (allowed: bool, retry_after_seconds). Fixed-window counter."""
    import time as _t
    now = _t.time()
    with _ratelimit_lock:
        bucket = _ratelimit_buckets.get(key)
        if bucket is None or now - bucket["window_start"] >= bucket["period"]:
            _ratelimit_buckets[key] = {"count": 1, "window_start": now, "period": period, "limit": limit}
            return True, 0
        if bucket["count"] < bucket["limit"]:
            bucket["count"] += 1
            return True, 0
        retry_in = int(bucket["period"] - (now - bucket["window_start"])) + 1
        return False, retry_in


def rate_limit(limit, period, key_parts=None):
    """Decorator: deny with 429 when the IP/action bucket exceeds `limit`/`period` (seconds)."""
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            key = _rate_key(*(key_parts() if callable(key_parts) else (key_parts or [])))
            allowed, retry_in = _rate_allowed(key, limit, period)
            if not allowed:
                if request.path.startswith("/api/") or request.is_json:
                    return jsonify({"error": "Too many attempts. Please wait and try again.",
                                    "retry_after": retry_in}), 429
                return render_template("error.html", code=429, title="Too Many Requests",
                                       message=f"Please slow down and try again in about {retry_in} seconds."), 429
            return view(*args, **kwargs)
        return wrapped
    return decorator


# ---------------------------------------------------------------------------
# Input validation helpers (production-grade checks for auth fields)
# ---------------------------------------------------------------------------
import re as _re

_EMAIL_RE = _re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_PHONE_RE = _re.compile(r"^\+?[0-9]{7,15}$")
_USERNAME_RE = _re.compile(r"^[\w\s.\-]{2,50}$")
_PASSWORD_MIN = 6


def _valid_email(email):
    return bool(email) and len(email) <= 254 and bool(_EMAIL_RE.match(email))


def _valid_phone(phone):
    if not phone:
        return True  # phone is optional
    return bool(_PHONE_RE.match(phone))


def _valid_username(username):
    return bool(username) and bool(_USERNAME_RE.match(username))


def _password_ok(password):
    """Return (ok, reason). Enforce a minimum of 6 chars (no weird control chars)."""
    if isinstance(password, str) and len(password) >= _PASSWORD_MIN and len(password) <= 128:
        if any(c for c in password if ord(c) < 32):  # reject control characters
            return False, "Password contains invalid characters"
        return True, ""
    return False, f"Password must be at least {_PASSWORD_MIN} characters"


def _sanitize_email(email):
    return (email or "").strip().lower()

# --- Database Config: Postgres on Render, SQLite locally ---
db_url = os.environ.get("DATABASE_URL", "")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
if not db_url:
    # local SQLite file in instance folder
    os.makedirs(app.instance_path, exist_ok=True)
    db_url = "sqlite:///" + os.path.join(app.instance_path, "synora.db")
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Gemini Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-flash-lite-latest")
else:
    GEMINI_MODEL = None


def _gemini_generate(prompt, attempts=3):
    """Generate content with retry/backoff for transient errors (429/503)."""
    import time
    delay = 4
    last_err = None
    for i in range(attempts):
        try:
            return GEMINI_MODEL.generate_content(prompt)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "503" in msg or "quota" in msg.lower() or "overloaded" in msg.lower():
                last_err = e
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise last_err

# ---------------------------------------------------------------------------
# Database Models — persistent storage for every user
# ---------------------------------------------------------------------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(200), unique=True, nullable=False, index=True)
    username = db.Column(db.String(120), nullable=False)
    password = db.Column(db.String(300), default="")
    phone = db.Column(db.String(20), nullable=True)
    phone_verified = db.Column(db.Boolean, default=False)
    email_verified = db.Column(db.Boolean, default=False)
    exam_goal = db.Column(db.String(100), nullable=True)  # JEE, NEET etc
    study_level = db.Column(db.String(50), nullable=True)  # 11th, 12th, Dropper etc
    daily_hours = db.Column(db.Integer, nullable=True)
    school = db.Column(db.String(200), nullable=True)
    age = db.Column(db.Integer, nullable=True)
    onboarded = db.Column(db.Boolean, default=False)
    auth_provider = db.Column(db.String(20), default="email")  # email/google/phone
    photo_url = db.Column(db.String(500), nullable=True)
    password_reset_token = db.Column(db.String(200), nullable=True)
    password_reset_expires = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    current_energy = db.Column(db.String(20), default="Active")
    last_taunt = db.Column(db.Text, nullable=True)
    _history_seeded = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    # JSON blobs for flexible learning data — also stores tasks/focus/plans for legacy compat
    learning_data = db.Column(MutableDict.as_mutable(db.JSON), default=lambda: {
        "energy_reports": [], "completion_patterns": [], "focus_sessions_detailed": [],
        "sleep_wake": [], "routine_version": 0, "last_analyzed": None, "personal_routine": None, "energy_model": None
    })
    quiz_session = db.Column(MutableDict.as_mutable(db.JSON), nullable=True)
    quiz_history = db.Column(MutableList.as_mutable(db.JSON), default=list)
    # Persistent JSON stores (replaces in-memory lists) - mutable for auto-tracking
    tasks_data = db.Column(MutableList.as_mutable(db.JSON), default=list)
    focus_data = db.Column(MutableList.as_mutable(db.JSON), default=list)
    syllabus_data = db.Column(MutableList.as_mutable(db.JSON), default=list)

    # Relationships
    tasks = db.relationship("Task", backref="user", lazy=True, cascade="all, delete-orphan")
    focus_sessions = db.relationship("FocusSession", backref="user", lazy=True, cascade="all, delete-orphan")
    syllabus_plans = db.relationship("SyllabusPlan", backref="user", lazy=True, cascade="all, delete-orphan")

    # --- Dict-like compatibility for legacy code (JSON-backed) ---
    def __getitem__(self, key):
        if key == "tasks":
            return self.tasks_data if self.tasks_data is not None else []
        if key == "focus_sessions":
            return self.focus_data if self.focus_data is not None else []
        if key == "syllabus_plans":
            return self.syllabus_data if self.syllabus_data is not None else []
        if hasattr(self, key):
            return getattr(self, key)
        if key == "_history_seeded":
            return self._history_seeded
        if key == "learning_data":
            return self.learning_data
        raise KeyError(key)

    def __setitem__(self, key, val):
        if key == "tasks":
            self.tasks_data = val
            from sqlalchemy.orm.attributes import flag_modified; flag_modified(self, "tasks_data")
        elif key == "focus_sessions":
            self.focus_data = val
            from sqlalchemy.orm.attributes import flag_modified; flag_modified(self, "focus_data")
        elif key == "syllabus_plans":
            self.syllabus_data = val
            from sqlalchemy.orm.attributes import flag_modified; flag_modified(self, "syllabus_data")
        elif key == "_history_seeded":
            self._history_seeded = val
        elif key == "learning_data":
            self.learning_data = val
            from sqlalchemy.orm.attributes import flag_modified; flag_modified(self, "learning_data")
        elif hasattr(self, key):
            setattr(self, key, val)
        else:
            setattr(self, key, val)

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def setdefault(self, key, default):
        val = self.get(key, None)
        # treat empty list as existing? legacy setdefault checks None, but we mimic: if key missing or None
        if key in ("tasks", "focus_sessions", "syllabus_plans", "learning_data"):
            # if stored is None or empty and default is list, return stored
            existing = self.__getitem__(key)
            if existing is None:
                self.__setitem__(key, default)
                return default
            return existing
        if val is None:
            self.__setitem__(key, default)
            return default
        return val

    def pop(self, key, default=None):
        """Dict-compatible pop (clears the value for JSON-backed/column fields)."""
        current = self.get(key, default)
        if hasattr(self, key):
            try:
                setattr(self, key, None)
            except Exception:
                pass
        else:
            try:
                delattr(self, key)
            except AttributeError:
                pass
        return current

    def to_dict(self):
        return {
            "id": self.id, "email": self.email, "username": self.username,
            "current_energy": self.current_energy, "last_taunt": self.last_taunt,
            "learning_data": self.learning_data or {}, "quiz_session": self.quiz_session,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

# ---------------------------------------------------------------------------
# Community Models
# ---------------------------------------------------------------------------
class CommunityPost(db.Model):
    __tablename__ = "community_posts"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    content = db.Column(db.Text, nullable=False)
    media_urls = db.Column(db.JSON, default=list)  # list of {type: "image"|"video", url: "..."}
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_pinned = db.Column(db.Boolean, default=False)
    is_deleted = db.Column(db.Boolean, default=False)

    # Relationships
    author = db.relationship("User", backref=db.backref("posts", lazy=True))
    comments = db.relationship("CommunityComment", backref="post", lazy=True, cascade="all, delete-orphan")
    likes = db.relationship("CommunityLike", backref="post", lazy=True, cascade="all, delete-orphan")

    def to_dict(self, include_comments=False, current_user_id=None):
        like_count = len(self.likes)
        liked_by_current = False
        if current_user_id:
            liked_by_current = any(l.user_id == current_user_id for l in self.likes)
        d = {
            "id": self.id,
            "author": {"id": self.author.id, "username": self.author.username, "photo_url": self.author.photo_url} if self.author else None,
            "content": self.content,
            "media_urls": self.media_urls or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_pinned": self.is_pinned,
            "like_count": like_count,
            "liked_by_current": liked_by_current,
            "comment_count": len(self.comments),
        }
        if include_comments:
            d["comments"] = [c.to_dict(current_user_id=current_user_id) for c in self.comments if not c.is_deleted]
        return d


class CommunityComment(db.Model):
    __tablename__ = "community_comments"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = db.Column(db.String(36), db.ForeignKey("community_posts.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    content = db.Column(db.Text, nullable=False)
    media_urls = db.Column(db.JSON, default=list)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_deleted = db.Column(db.Boolean, default=False)

    author = db.relationship("User", backref=db.backref("comments", lazy=True))
    likes = db.relationship("CommunityCommentLike", backref="comment", lazy=True, cascade="all, delete-orphan")

    def to_dict(self, current_user_id=None):
        like_count = len(self.likes)
        liked_by_current = False
        if current_user_id:
            liked_by_current = any(l.user_id == current_user_id for l in self.likes)
        return {
            "id": self.id,
            "post_id": self.post_id,
            "author": {"id": self.author.id, "username": self.author.username, "photo_url": self.author.photo_url} if self.author else None,
            "content": self.content,
            "media_urls": self.media_urls or [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "like_count": like_count,
            "liked_by_current": liked_by_current,
            "is_deleted": self.is_deleted,
        }


class CommunityLike(db.Model):
    __tablename__ = "community_likes"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = db.Column(db.String(36), db.ForeignKey("community_posts.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("post_id", "user_id", name="uq_post_user_like"),)


class CommunityCommentLike(db.Model):
    __tablename__ = "community_comment_likes"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    comment_id = db.Column(db.String(36), db.ForeignKey("community_comments.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    __table_args__ = (db.UniqueConstraint("comment_id", "user_id", name="uq_comment_user_like"),)


class Task(db.Model):
    __tablename__ = "tasks"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    name = db.Column(db.String(300), nullable=False)
    start_time = db.Column(db.String(10), default="09:00")
    end_time = db.Column(db.String(10), default="10:00")
    date = db.Column(db.String(10), index=True)  # YYYY-MM-DD
    priority = db.Column(db.String(5), default="P2")
    energy = db.Column(db.String(10), default="Med")
    completed = db.Column(db.Boolean, default=False)
    completed_at = db.Column(db.String(30), nullable=True)
    is_syllabus = db.Column(db.Boolean, default=False)
    plan_id = db.Column(db.String(36), db.ForeignKey("syllabus_plans.id"), nullable=True)
    healed = db.Column(db.Boolean, default=False)
    no_time = db.Column(db.Boolean, default=False)
    duration = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "start_time": self.start_time, "end_time": self.end_time,
            "date": self.date, "priority": self.priority, "energy": self.energy,
            "completed": self.completed, "completed_at": self.completed_at,
            "is_syllabus": self.is_syllabus, "plan_id": self.plan_id,
            "healed": self.healed, "no_time": self.no_time, "duration": self.duration
        }

class FocusSession(db.Model):
    __tablename__ = "focus_sessions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    date = db.Column(db.String(10), index=True)
    seconds = db.Column(db.Integer, nullable=False)
    mode = db.Column(db.String(20), default="timer")
    hour = db.Column(db.Integer, nullable=True)
    start_hour = db.Column(db.Integer, nullable=True)
    duration_min = db.Column(db.Integer, nullable=True)

class SyllabusPlan(db.Model):
    __tablename__ = "syllabus_plans"
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    exam = db.Column(db.String(100))
    subject = db.Column(db.String(200))
    syllabus = db.Column(db.Text)
    exam_date = db.Column(db.String(10))
    daily_hours = db.Column(db.Integer, default=3)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    days_generated = db.Column(db.Integer, default=0)
    total_tasks = db.Column(db.Integer, default=0)
    days_left_total = db.Column(db.Integer, default=0)

    def to_dict(self):
        return {
            "id": self.id, "exam": self.exam, "subject": self.subject, "syllabus": self.syllabus,
            "exam_date": self.exam_date, "daily_hours": self.daily_hours,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "days_generated": self.days_generated, "total_tasks": self.total_tasks, "days_left_total": self.days_left_total
        }

# Create tables on startup
with app.app_context():
    db.create_all()

    # --- Lightweight auto-migration -------------------------------
    # db.create_all() only creates missing tables, NOT missing columns
    # on existing tables. Add new columns here when the model evolves so
    # existing SQLite/Postgres DBs get patched + backfilled automatically.
    # (Keeps zero-dependency deploys simple until a full Alembic setup.)
    def _ensure_column(table, column, ddl):
        from sqlalchemy import inspect as sa_inspect, text
        insp = sa_inspect(db.engine)
        if column not in [c["name"] for c in insp.get_columns(table)]:
            with db.engine.begin() as conn:
                conn.execute(text(ddl))
            print(f"[MIGRATE] added {table}.{column}")

    _ensure_column("users", "is_active", "ALTER TABLE users ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT TRUE")

    # Backfill is_active = True for any existing rows that might be NULL
    from sqlalchemy import text as _t
    try:
        with db.engine.begin() as conn:
            conn.execute(_t("UPDATE users SET is_active = TRUE WHERE is_active IS NULL"))
    except Exception as _e:
        print(f"[MIGRATE] backfill skipped: {_e}")

    # Ensure community tables exist (create_all handles this, but explicit for clarity)
    try:
        insp = sa_inspect(db.engine)
        existing_tables = set(insp.get_table_names())
        community_tables = {"community_posts", "community_comments", "community_likes", "community_comment_likes"}
        missing = community_tables - existing_tables
        if missing:
            # This will create only the missing tables
            db.create_all()
            print(f"[MIGRATE] created community tables: {missing}")
    except Exception as _e:
        print(f"[MIGRATE] community tables check skipped: {_e}")



def _hash_password(plain):
    """Hash a password using werkzeug (PBKDF2). Empty/short placeholders stay hashed too."""
    if not plain:
        return ""
    return generate_password_hash(plain)


def _check_password(stored, plain):
    """Verify a plaintext password against a stored hash. Handles legacy plaintext too."""
    if not stored:
        return False
    if not plain:
        return False
    # Already hashed (werkzeug hashes start with pbkdf2: or scrypt:)
    if stored.startswith(("pbkdf2:", "scrypt:", "sha256$")):
        try:
            return check_password_hash(stored, plain)
        except Exception:
            return False
    # Legacy plaintext stored before hashing was introduced
    return stored == plain


def create_user(username, email, password):
    existing = User.query.filter_by(email=email).first()
    if existing:
        return existing
    user = User(
        username=username, email=email, password=_hash_password(password),
        current_energy="Active",
        learning_data={
            "energy_reports": [], "completion_patterns": [], "focus_sessions_detailed": [],
            "sleep_wake": [], "routine_version": 0, "last_analyzed": None, "personal_routine": None, "energy_model": None
        },
        quiz_history=[]
    )
    db.session.add(user)
    db.session.commit()
    return user

def get_user_by_email(email):
    return User.query.filter_by(email=email).first()

def _user_tasks(user):
    return Task.query.filter_by(user_id=user.id).all()

def _user_tasks_dicts(user):
    return [t.to_dict() for t in _user_tasks(user)]


# ---------------------------------------------------------------------------
# AI LEARNING SYSTEM — Personal Energy Predictor & Routine Generator
# ---------------------------------------------------------------------------
import statistics
from collections import defaultdict, Counter

def record_energy_report(user, hour, energy_level, source="manual"):
    """Record user's energy level at a specific hour"""
    ld = user.learning_data or {}
    # ensure mutable tracking
    if user.learning_data is None:
        user.learning_data = ld
    ld.setdefault("energy_reports", []).append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "hour": hour,
        "energy": energy_level,
        "source": source,
        "timestamp": datetime.now().isoformat()
    })
    from sqlalchemy.orm.attributes import flag_modified; flag_modified(user, "learning_data")

def record_completion(user, task):
    """Record task completion pattern"""
    if not task.get("completed_at"):
        return
    try:
        dt = datetime.fromisoformat(task["completed_at"])
        ld = user.learning_data or {}
        if user.learning_data is None:
            user.learning_data = ld
        ld.setdefault("completion_patterns", []).append({
            "date": dt.strftime("%Y-%m-%d"),
            "hour": dt.hour,
            "task_type": task.get("priority", "P3"),
            "energy": task.get("energy", "Med"),
            "duration_min": int((datetime.fromisoformat(task.get("end_time", "18:00")).replace(year=dt.year, month=dt.month, day=dt.day) - dt).total_seconds() / 60) if task.get("end_time") else 60,
            "completed": True
        })
        from sqlalchemy.orm.attributes import flag_modified; flag_modified(user, "learning_data")
    except:
        pass

def analyze_user_patterns(user):
    """Analyze 3+ days of data to build personal energy model and routine"""
    ld = user.setdefault("learning_data", {})
    energy_reports = ld.get("energy_reports", [])
    completions = ld.get("completion_patterns", [])
    focus_sessions = ld.get("focus_sessions_detailed", user.get("focus_sessions", []))
    
    # Need at least 3 days of data
    unique_dates = set()
    for r in energy_reports:
        unique_dates.add(r["date"])
    for c in completions:
        unique_dates.add(c["date"])
    for f in focus_sessions:
        if isinstance(f, dict) and f.get("date"):
            unique_dates.add(f["date"])
    
    if len(unique_dates) < 3:
        return None  # Not enough data yet
    
    # ---- ENERGY MODEL: Hour -> average energy (0-1 scale) ----
    energy_map = {"Low": 0.3, "Med": 0.6, "High": 0.9}
    hour_energy = defaultdict(list)
    
    for r in energy_reports:
        hour_energy[r["hour"]].append(energy_map.get(r["energy"], 0.6))
    
    # Also infer from completions (completed tasks = energy was sufficient)
    for c in completions:
        hour_energy[c["hour"]].append(energy_map.get(c["energy"], 0.6))
    
    # Focus sessions = high energy
    for f in focus_sessions:
        if isinstance(f, dict):
            h = f.get("start_hour") or (datetime.fromisoformat(f["date"]).hour if "T" in f.get("date", "") else 10)
            hour_energy[h].append(0.85)
    
    # Build model: hour -> avg energy (0-1)
    energy_model = {}
    for h in range(24):
        vals = hour_energy.get(h, [])
        if vals:
            energy_model[h] = statistics.mean(vals)
        else:
            # Fallback to circadian
            circadian = {6:0.4,7:0.5,8:0.7,9:0.9,10:1.0,11:0.95,12:0.8,13:0.6,14:0.5,15:0.7,16:0.9,17:0.85,18:0.7,19:0.6,20:0.5,21:0.4,22:0.3}.get(h, 0.3)
            energy_model[h] = circadian
    
    # ---- SLEEP/WAKE PATTERN ----
    wake_hours = []
    sleep_hours = []
    for f in focus_sessions:
        if isinstance(f, dict):
            h = f.get("start_hour")
            if h is not None:
                wake_hours.append(h)
    # Infer from first activity of day
    daily_first_activity = defaultdict(list)
    for c in completions:
        daily_first_activity[c["date"]].append(c["hour"])
    for f in focus_sessions:
        if isinstance(f, dict) and f.get("date"):
            d = f["date"][:10]
            h = f.get("start_hour", 9)
            daily_first_activity[d].append(h)
    for d, hours in daily_first_activity.items():
        if hours:
            wake_hours.append(min(hours))
    
    avg_wake = statistics.mean(wake_hours) if wake_hours else 7
    avg_sleep = (avg_wake + 16) % 24  # Assume 16h awake
    
    # ---- PRODUCTIVE WINDOWS (top energy hours) ----
    sorted_hours = sorted(energy_model.items(), key=lambda x: x[1], reverse=True)
    peak_hours = [h for h, e in sorted_hours[:6] if e > 0.65]  # Top 6 hours above threshold
    
    # ---- BUILD PERSONAL ROUTINE ----
    routine = {
        "wake_time": f"{int(avg_wake):02d}:00",
        "sleep_time": f"{int(avg_sleep):02d}:00",
        "peak_windows": [],
        "work_blocks": [],
        "break_times": [],
        "generated_at": datetime.now().isoformat(),
        "data_days": len(unique_dates)
    }
    
    # Create 2-3 work blocks in peak hours
    peak_hours.sort()
    if peak_hours:
        # Group consecutive peak hours into blocks
        blocks = []
        current = [peak_hours[0]]
        for h in peak_hours[1:]:
            if h == current[-1] + 1:
                current.append(h)
            else:
                blocks.append(current)
                current = [h]
        blocks.append(current)
        
        for i, block in enumerate(blocks[:3]):
            start = block[0]
            end = block[-1] + 1
            routine["work_blocks"].append({
                "name": f"Deep Work Block {i+1}",
                "start": f"{start:02d}:00",
                "end": f"{end:02d}:00",
                "duration_min": len(block) * 60,
                "type": "P1" if i == 0 else "P2"
            })
    
    # Add breaks between blocks
    for i in range(len(routine["work_blocks"]) - 1):
        curr_end = int(routine["work_blocks"][i]["end"][:2])
        next_start = int(routine["work_blocks"][i+1]["start"][:2])
        if next_start - curr_end >= 1:
            routine["break_times"].append({
                "start": f"{curr_end:02d}:00",
                "end": f"{min(curr_end+1, next_start):02d}:00",
                "type": "Break"
            })
    
    # Save to user
    ld["energy_model"] = energy_model
    ld["personal_routine"] = routine
    ld["routine_version"] = ld.get("routine_version", 0) + 1
    ld["last_analyzed"] = datetime.now().isoformat()
    
    return routine

def get_personal_energy_curve(user, hour):
    """Get personalized energy for an hour (0-1 scale)"""
    ld = user.get("learning_data", {})
    model = ld.get("energy_model")
    if model and hour in model:
        return model[hour]
    # Fallback to circadian
    circadian = {6:0.4,7:0.5,8:0.7,9:0.9,10:1.0,11:0.95,12:0.8,13:0.6,14:0.5,15:0.7,16:0.9,17:0.85,18:0.7,19:0.6,20:0.5,21:0.4,22:0.3}.get(hour, 0.3)
    return circadian

def get_personal_routine(user):
    """Get or generate personal routine"""
    ld = user.get("learning_data", {})
    routine = ld.get("personal_routine")
    if not routine:
        routine = analyze_user_patterns(user)
    return routine

# Auto-record on task completion
def auto_record_completion(user, task):
    if task.get("completed") and task.get("completed_at"):
        record_completion(user, task)
    # Also record energy report from task's energy field
    if task.get("completed_at"):
        try:
            dt = datetime.fromisoformat(task["completed_at"])
            record_energy_report(user, dt.hour, task.get("energy", "Med"), source="completion")
        except:
            pass


# ---------------------------------------------------------------------------
# Exam Configurations with Marking Schemes
# ---------------------------------------------------------------------------
EXAM_CONFIGS = {
    "JEE Mains": {
        "description": "National level engineering entrance exam.",
        "icon": "calculate",
        "modules": "45 Modules",
        "priority": "High Priority",
        "category": "Engineering",
        "marking": {"correct": 4, "incorrect": -1, "unanswered": 0},
        "time_per_q": 60,
        "difficulty": "medium",
        "subjects": ["Physics", "Chemistry", "Mathematics"]
    },
    "JEE Advanced": {
        "description": "Premier engineering entrance for IITs.",
        "icon": "science",
        "modules": "32 Modules",
        "priority": "",
        "category": "Engineering",
        "marking": {"correct": 4, "incorrect": -1, "unanswered": 0},
        "time_per_q": 90,
        "difficulty": "hard",
        "subjects": ["Physics", "Chemistry", "Mathematics"]
    },
    "GATE": {
        "description": "Graduate Aptitude Test in Engineering.",
        "icon": "precision_manufacturing",
        "modules": "28 Modules",
        "priority": "",
        "category": "Engineering",
        "marking": {"correct": 2, "incorrect": -0.66, "unanswered": 0},
        "time_per_q": 120,
        "difficulty": "hard",
        "subjects": ["Core Engineering", "General Aptitude"]
    },
    "NEET UG": {
        "description": "National Eligibility cum Entrance Test.",
        "icon": "biotech",
        "modules": "50 Modules",
        "priority": "",
        "category": "Medical",
        "marking": {"correct": 4, "incorrect": -1, "unanswered": 0},
        "time_per_q": 60,
        "difficulty": "medium",
        "subjects": ["Physics", "Chemistry", "Biology"]
    },
    "NEET PG": {
        "description": "Postgraduate medical entrance exam.",
        "icon": "monitor_heart",
        "modules": "38 Modules",
        "priority": "High Priority",
        "category": "Medical",
        "marking": {"correct": 4, "incorrect": -1, "unanswered": 0},
        "time_per_q": 90,
        "difficulty": "hard",
        "subjects": ["Clinical", "Pre-clinical", "Para-clinical"]
    },
    "UPSC Prelims": {
        "description": "Civil services preliminary examination.",
        "icon": "gavel",
        "modules": "60 Modules",
        "priority": "High Priority",
        "category": "Government Exams",
        "marking": {"correct": 2, "incorrect": -0.66, "unanswered": 0},
        "time_per_q": 80,
        "difficulty": "hard",
        "subjects": ["GS Paper 1", "CSAT"]
    },
    "SSC CGL": {
        "description": "Staff Selection Commission Combined Graduate.",
        "icon": "fact_check",
        "modules": "42 Modules",
        "priority": "",
        "category": "Government Exams",
        "marking": {"correct": 2, "incorrect": -0.5, "unanswered": 0},
        "time_per_q": 60,
        "difficulty": "medium",
        "subjects": ["Quantitative Aptitude", "Reasoning", "English", "General Awareness"]
    },
    "IBPS PO": {
        "description": "Bank probationary officer exam.",
        "icon": "account_balance",
        "modules": "35 Modules",
        "priority": "",
        "category": "Government Exams",
        "marking": {"correct": 1, "incorrect": -0.25, "unanswered": 0},
        "time_per_q": 45,
        "difficulty": "medium",
        "subjects": ["Quantitative Aptitude", "Reasoning", "English", "General Awareness"]
    },
    "CBSE Class 12": {
        "description": "Central Board of Secondary Education finals.",
        "icon": "menu_book",
        "modules": "40 Modules",
        "priority": "",
        "category": "School Boards",
        "marking": {"correct": 1, "incorrect": 0, "unanswered": 0},
        "time_per_q": 90,
        "difficulty": "easy",
        "subjects": ["Physics", "Chemistry", "Mathematics", "Biology", "English"]
    },
    "ICSE Class 12": {
        "description": "Indian Certificate of Secondary Education.",
        "icon": "auto_stories",
        "modules": "36 Modules",
        "priority": "",
        "category": "School Boards",
        "marking": {"correct": 1, "incorrect": 0, "unanswered": 0},
        "time_per_q": 90,
        "difficulty": "easy",
        "subjects": ["Physics", "Chemistry", "Mathematics", "Biology", "English"]
    },
    "State Boards": {
        "description": "Class 12 board examinations.",
        "icon": "workspace_premium",
        "modules": "30 Modules",
        "priority": "",
        "category": "School Boards",
        "marking": {"correct": 1, "incorrect": 0, "unanswered": 0},
        "time_per_q": 90,
        "difficulty": "easy",
        "subjects": ["Physics", "Chemistry", "Mathematics", "Biology"]
    }
}


def current_user():
    email = session.get("email")
    if not email:
        return None
    user = User.query.filter_by(email=email).first()
    if user is None:
        return None
    # Enforce account status: a disabled user is treated as logged-out so the
    # session can't keep using protected routes after admin deactivation.
    if user.is_active is False:
        session.clear()
        return None
    return user


# ---------------------------------------------------------------------------
# Guest Auto-Login + Seed 12-Day History
# ---------------------------------------------------------------------------
GUEST_EMAIL = "student@synora.ai"


def _seed_guest_history(user):
    """Seed disabled — fresh start, no fake tasks."""
    if user.get("_history_seeded"):
        return
    user["_history_seeded"] = True
    db.session.commit()


@app.before_request
def ensure_guest_session():
    # Skip guest auto-login for public/auth routes
    public_paths = ("/login", "/signup", "/logout", "/auth/", "/static/", "/favicon", "/features", "/methodology")
    if any(request.path.startswith(p) for p in public_paths) or request.path == "/":
        return
    # If user is already logged in (not guest), skip
    if session.get("email") and session["email"] != GUEST_EMAIL:
        return
    # Auto-create guest user if needed
    guest = User.query.filter_by(email=GUEST_EMAIL).first()
    if not guest:
        guest = create_user("Student", GUEST_EMAIL, "")
    _seed_guest_history(guest)
    if not current_user():
        session["email"] = GUEST_EMAIL
        session["username"] = guest.username


@app.after_request
def _commit_after(response):
    # -- Security headers (defense-in-depth) --
    # Default-deny framing & MIME sniffing.
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-XSS-Protection", "0")  # modern XSS handling, avoid legacy filter
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    # A pragmatic CSP: allow our own styles/scripts, Tailwind CDN, and Google Identity (GIS) for OAuth.
    # 'unsafe-inline' is required by Tailwind's injected <style> and some inline scripts.
    if not response.headers.get("Content-Security-Policy"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://accounts.google.com https://apis.google.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://fonts.googleapis.com https://fonts.gstatic.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data: https:; "
            "frame-src https://accounts.google.com https://content.googleapis.com; "
            "connect-src 'self' https://cdn.tailwindcss.com https://accounts.google.com https://fonts.googleapis.com https://fonts.gstatic.com; "
            "base-uri 'self'; form-action 'self'"
        )
    try:
        # Ensure JSON mutations (including nested dict edits) are persisted
        # Touch all JSON columns for the current user to force dirty check
        u = None
        try:
            email = session.get("email")
            if email:
                u = User.query.filter_by(email=email).first()
                if u:
                    from sqlalchemy.orm.attributes import flag_modified
                    # flag all mutable JSON cols — cheap, ensures inner-dict edits commit
                    for col in ("tasks_data", "focus_data", "syllabus_data", "learning_data", "quiz_history", "quiz_session"):
                        try:
                            flag_modified(u, col)
                        except Exception:
                            pass
        except Exception:
            pass
        db.session.commit()
    except Exception:
        db.session.rollback()
    return response


# Rate limits applied to auth endpoints (configurable via env for tuning).
AUTH_RATE = {
    "per_ip": int(os.environ.get("RATE_IP", "30")),      # requests / window per IP
    "login": int(os.environ.get("RATE_LOGIN", "8")),     # login attempts / 60s
    "otp": int(os.environ.get("RATE_OTP", "5")),         # OTP sends or verifies / 60s
    "signup": int(os.environ.get("RATE_SIGNUP", "6")),   # signups / 60s
    "reset": int(os.environ.get("RATE_RESET", "5")),     # password reset requests / 60s
}
AUTH_WINDOW = 60  # seconds


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("email"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Login required"}), 401
            return redirect(url_for("login_page"))
        # If the session was cleared (e.g. account disabled/deleted mid-session),
        # treat as logged out instead of letting the view crash on a None user.
        if current_user() is None:
            session.clear()
            if request.path.startswith("/api/"):
                return jsonify({"error": "Login required"}), 401
            return redirect(url_for("login_page"))
        return view(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Root Route — landing page or dashboard if logged in
# ---------------------------------------------------------------------------
@app.route("/")
def landing():
    if session.get("email") and session["email"] != GUEST_EMAIL:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login_page"))


@app.route("/features")
def features():
    return render_template("features.html", active_tab="features")


@app.route("/methodology")
def methodology():
    return render_template("methodology.html", active_tab="methodology")


@app.route("/admin")
def admin_panel():
    # Simple key check — set ADMIN_KEY env var on Render, default for local
    key = request.args.get("key") or request.headers.get("X-Admin-Key")
    expected = os.environ.get("ADMIN_KEY", "synora-admin-2026")
    if key != expected:
        return f'''<div style="display:flex;min-height:100vh;align-items:center;justify-content:center;background:#111415;color:#e1e3e4;font-family:system-ui"><form style="background:#191c1d;padding:32px;border-radius:16px;border:1px solid rgba(255,255,255,0.08);text-align:center"><h2 style="color:#9ad3b6">Admin Access</h2><p style="color:#8a938c;font-size:13px">Enter admin key to view user data</p><input name="key" placeholder="ADMIN_KEY" style="width:100%;margin:12px 0;padding:10px 14px;border-radius:10px;background:#0c0f10;border:1px solid rgba(138,147,140,0.3);color:#e1e3e4"/><br><button type="submit" style="width:100%;padding:10px;background:#9ad3b6;color:#003825;border:none;border-radius:10px;font-weight:700">Unlock</button><p style="font-size:11px;color:#8a938c;margin-top:10px">Default local key: <code>synora-admin-2026</code></p></form></div>''', 401

    # --- Search + filter + pagination ---
    q = (request.args.get("q") or "").strip()
    f = (request.args.get("f") or "all").strip()
    page = max(1, int(request.args.get("page") or 1))
    PER = 25

    query = User.query
    if f == "active":
        query = query.filter((User.is_active.is_(None)) | (User.is_active.is_(True)))
    elif f == "disabled":
        query = query.filter(User.is_active.is_(False))

    if q:
        like = f"%{q}%"
        query = query.filter(
            (User.username.ilike(like)) | (User.email.ilike(like)) | (User.phone.ilike(like))
        )

    total = query.count()
    pages = max(1, (total + PER - 1) // PER)
    users = query.order_by(User.created_at.desc()).offset((page - 1) * PER).limit(PER).all()

    # Stats
    all_users = User.query.all()
    now = datetime.utcnow()
    new_today = sum(1 for u in all_users if u.created_at and u.created_at.date() == now.date())
    verified = sum(1 for u in all_users if u.email_verified)
    active_count = sum(1 for u in all_users if u.is_active is not False)
    disabled_count = sum(1 for u in all_users if u.is_active is False)
    total_tasks = sum(len(u.tasks_data or []) for u in all_users)

    urows = []
    for u in users:
        tasks = u.tasks_data or []
        urows.append({
            "id": u.id, "username": u.username, "email": u.email, "phone": u.phone or "-",
            "exam_goal": u.exam_goal or "-", "total_tasks": len(tasks),
            "done": len([t for t in tasks if t.get("completed")]),
            "focus_count": len(u.focus_data or []),
            "created_display": u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else "-",
            "email_verified": u.email_verified, "phone_verified": u.phone_verified,
            "is_active": u.is_active,
        })

    return render_template("admin.html", users=urows, total=total, pages=pages, page=page,
                           q=q, f=f, key=key, new_today=new_today, verified=verified,
                           active_count=active_count, disabled_count=disabled_count,
                           total_tasks=total_tasks)


@app.route("/admin/user/<int:user_id>/toggle", methods=["POST"])
def admin_toggle_user(user_id):
    key = request.args.get("key") or request.headers.get("X-Admin-Key")
    expected = os.environ.get("ADMIN_KEY", "synora-admin-2026")
    if key != expected:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    action = data.get("action")  # "disable" or "enable"
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    if action == "disable":
        user.is_active = False
        db.session.commit()
        return jsonify({"ok": True, "message": f"Disabled {user.username}"})
    elif action == "enable":
        user.is_active = True
        db.session.commit()
        return jsonify({"ok": True, "message": f"Enabled {user.username}"})
    return jsonify({"error": "Invalid action"}), 400

@app.route("/admin/export")
def admin_export():
    key = request.args.get("key")
    if key != os.environ.get("ADMIN_KEY", "synora-admin-2026"):
        return jsonify({"error": "Unauthorized"}), 401
    users = User.query.all()
    data = []
    for u in users:
        data.append({
            "id": u.id, "email": u.email, "username": u.username, "phone": u.phone,
            "exam_goal": u.exam_goal, "study_level": u.study_level, "daily_hours": u.daily_hours,
            "tasks": u.tasks_data, "focus_sessions": u.focus_data, "syllabus_plans": u.syllabus_data,
            "learning_data": u.learning_data, "created_at": u.created_at.isoformat() if u.created_at else None
        })
    return jsonify({"users": data, "count": len(data)})


# ---------------------------------------------------------------------------
# View Routes (Auth Required)
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        last_taunt=user.get("last_taunt") if user else None,
    )


@app.route("/calendar")
@login_required
def calendar():
    return render_template("calendar.html", active_page="calendar")


@app.route("/tasks")
@login_required
def tasks():
    return render_template("tasks.html", active_page="tasks")


@app.route("/quiz")
@login_required
def quiz():
    return render_template("quiz_selection.html", active_page="quiz")


@app.route("/quiz/start")
@login_required
def quiz_start():
    category = request.args.get("category", "General")
    user = current_user()
    quiz_session = user.get("quiz_session") if user else None
    questions = []
    exam_config = EXAM_CONFIGS.get(category, EXAM_CONFIGS["CBSE Class 12"])
    if quiz_session and quiz_session.get("category") == category:
        questions = quiz_session.get("questions", [])
        exam_config = quiz_session.get("exam_config", exam_config)
    return render_template(
        "quiz_interface.html",
        active_page="quiz",
        category=category,
        questions=questions,
        exam_config=exam_config
    )


@app.route("/api/quiz/generate", methods=["POST"])
@login_required
def api_quiz_generate():
    """Generate quiz questions using Gemini AI."""
    if not GEMINI_MODEL:
        return jsonify({"error": "AI model not configured."}), 503
    data = request.get_json(silent=True) or {}
    category = data.get("category", "General")
    topic = data.get("topic", "")
    num_questions = int(data.get("num_questions", 10))
    difficulty = data.get("difficulty") or EXAM_CONFIGS.get(category, {}).get("difficulty", "medium")
    exam_config = dict(EXAM_CONFIGS.get(category, EXAM_CONFIGS.get("CBSE Class 12")))
    exam_config["difficulty"] = difficulty
    subjects = exam_config.get("subjects", ["General"])
    topic_clean = (topic or "").strip()
    topic_line = f'STRICTLY and ONLY on the topic: "{topic_clean}"' if topic_clean else 'across the general syllabus of this exam (mixed subjects)'
    subject_line = f'Allowed subjects: {", ".join(subjects)}' if not topic_clean else f'Topic: "{topic_clean}" (ignore the general subject list and focus ONLY on this topic)'

    prompt = f"""
You are an expert question setter for the {category} exam.

TASK: Generate EXACTLY {num_questions} multiple-choice questions {topic_line}.
{subject_line}
Difficulty: {difficulty}
Marking: Correct +{exam_config["marking"]["correct"]}, Incorrect {exam_config["marking"]["incorrect"]}, Unanswered {exam_config["marking"]["unanswered"]}

CRITICAL RULES:
- EVERY question MUST be directly about "{topic_clean}" if a topic is given. Do NOT include questions from other topics.
- If topic is "Photosynthesis", ALL questions must be about Photosynthesis. If topic is "Kinematics", ALL about Kinematics. No exceptions.
- Return ONLY a valid JSON array — no prose, no markdown, no code fences.
- Each item MUST have exactly: "question" (string), "options" (array of 4 strings), "answer" (integer 0-3), "subject" (string = topic or subject name), "explanation" (string, plain text).
- Use ONLY plain ASCII. Do NOT use LaTeX backslashes (no \\, \\frac, \\theta). Write "alpha", "sqrt", "theta" as plain words. Never use backslashes in JSON strings.
- Exactly 4 options per question, answer 0-3.

JSON format:
[
  {{
    "question": "Question text? (plain text)",
    "options": ["A", "B", "C", "D"],
    "answer": 0,
    "subject": "{topic_clean or subjects[0]}",
    "explanation": "Plain text explanation."
  }}
]
"""
    try:
        response = _gemini_generate(prompt)
        text = response.text.strip()
        # Strip markdown fences if present
        if "```" in text:
            # Extract JSON array between first [ and last ]
            s, e = text.find("["), text.rfind("]")
            if s != -1 and e != -1:
                text = text[s:e+1]
        # Extract JSON array bounds
        s, e = text.find("["), text.rfind("]")
        if s != -1 and e != -1:
            text = text[s:e+1]
        # Try strict parse, then lenient fix for stray backslashes (e.g. LaTeX \theta)
        try:
            questions = json.loads(text)
        except json.JSONDecodeError as je:
            if "Invalid \\escape" in str(je) or "\\" in str(je):
                import re
                # Escape bare backslashes not part of valid JSON escapes
                fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
                questions = json.loads(fixed)
            else:
                raise
        if not isinstance(questions, list) or len(questions) == 0:
            raise ValueError("Invalid questions format")
        for i, q in enumerate(questions):
            if not all(k in q for k in ("question", "options", "answer", "subject", "explanation")):
                raise ValueError(f"Question {i} missing fields")
            q["id"] = i
        user = current_user()
        user["quiz_session"] = {
            "category": category,
            "topic": topic,
            "questions": questions,
            "exam_config": exam_config,
            "started_at": datetime.now().isoformat(),
        }
        return jsonify({"questions": questions, "exam_config": exam_config})
    except Exception as e:
        return jsonify({"error": f"Failed to generate quiz: {str(e)}"}), 500


@app.route("/api/quiz/submit", methods=["POST"])
@login_required
def api_quiz_submit():
    """Submit quiz answers and get detailed analysis."""
    user = current_user()
    quiz_session = user.get("quiz_session") if user else None
    if not quiz_session:
        return jsonify({"error": "No active quiz session."}), 400
    data = request.get_json(silent=True) or {}
    answers = data.get("answers", [])  # list of {question_id, answer, time_taken}
    questions = quiz_session["questions"]
    exam_config = quiz_session["exam_config"]
    marking = exam_config["marking"]
    correct = 0
    incorrect = 0
    unanswered = 0
    total_time = 0
    subject_stats = {}
    detailed = []
    for q in questions:
        ans_data = next((a for a in answers if a["question_id"] == q["id"]), None)
        subject = q.get("subject", "General")
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0, "time": 0}
        subject_stats[subject]["total"] += 1
        # Skipped or no answer => unanswered (do NOT penalize)
        if ans_data is None or ans_data.get("answer") is None:
            unanswered += 1
            is_correct = False
            time_taken = ans_data.get("time_taken", 0) if ans_data else 0
            total_time += time_taken
            subject_stats[subject]["time"] += time_taken
            your_ans = None
        else:
            time_taken = ans_data.get("time_taken", 0)
            total_time += time_taken
            subject_stats[subject]["time"] += time_taken
            is_correct = ans_data["answer"] == q["answer"]
            your_ans = ans_data["answer"]
            if is_correct:
                correct += 1
                subject_stats[subject]["correct"] += 1
            else:
                incorrect += 1
        detailed.append({
            "question_id": q["id"],
            "question": q["question"],
            "your_answer": your_ans,
            "correct_answer": q["answer"],
            "is_correct": is_correct,
            "time_taken": time_taken,
            "explanation": q.get("explanation", ""),
            "subject": subject,
            "options": q.get("options", [])
        })
    score = correct * marking["correct"] + incorrect * marking["incorrect"] + unanswered * marking["unanswered"]
    max_possible = len(questions) * marking["correct"]
    pct = round((score / max_possible * 100) if max_possible > 0 else 0)
    user.pop("quiz_session", None)

    # --- Gemini-powered deep analysis ---
    gemini_analysis = None
    if GEMINI_MODEL:
        try:
            # Build compact performance snapshot for Gemini
            perf_lines = []
            for d in detailed:
                status = "✓ Correct" if d["is_correct"] else ("○ Skipped" if d["your_answer"] is None else "✗ Wrong")
                perf_lines.append(f'Q{d["question_id"]+1} [{d["subject"]}] {status} | Time:{d["time_taken"]}s | Q: {d["question"][:120]}')
            subj_lines = ", ".join(f'{s}: {v["correct"]}/{v["total"]} ({round(v["correct"]/v["total"]*100) if v["total"] else 0}%)' for s, v in subject_stats.items())
            analysis_prompt = f"""
You are an expert exam coach for {quiz_session["category"]} (topic: {quiz_session["topic"] or "General"}).
A student just completed a {len(questions)}-question quiz:
- Score: {score}/{max_possible} ({pct}%, {correct} correct, {incorrect} wrong, {unanswered} skipped)
- Time: {total_time}s total, {round(total_time/len(questions)) if questions else 0}s avg per question
- Subject breakdown: {subj_lines or "N/A"}
- Per-question performance:
{chr(10).join(perf_lines)}

TASK: Provide a personalized, motivating, actionable analysis as STRICT JSON ONLY (no markdown, no fences) with this exact structure:
{{
  "overall_summary": "2-3 sentence overall performance summary in encouraging tone",
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "time_analysis": "1-2 sentences on time management (too fast/slow, per-question pacing)",
  "recommendations": ["actionable tip 1", "tip 2", "tip 3"],
  "study_plan": "1-2 sentence focused study plan for next 7 days",
  "motivational_message": "1 short uplifting line"
}}
Rules: Plain ASCII only, no backslashes, no LaTeX, no markdown. Keep each string concise. Return ONLY JSON.
"""
            a_resp = _gemini_generate(analysis_prompt)
            a_text = a_resp.text.strip()
            if "```" in a_text:
                s, e = a_text.find("{"), a_text.rfind("}")
                if s != -1 and e != -1:
                    a_text = a_text[s:e+1]
            try:
                gemini_analysis = json.loads(a_text)
            except json.JSONDecodeError:
                import re
                fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', a_text)
                gemini_analysis = json.loads(fixed)
        except Exception as _ea:
            gemini_analysis = None

    result = {
        "score": score,
        "max_score": max_possible,
        "percentage": pct,
        "correct": correct,
        "incorrect": incorrect,
        "unanswered": unanswered,
        "total_time_seconds": total_time,
        "avg_time_per_q": round(total_time / len(questions)) if questions else 0,
        "subject_breakdown": subject_stats,
        "detailed_analysis": detailed,
        "exam_config": exam_config,
        "verdict": "Excellent" if pct >= 80 else "Good" if pct >= 60 else "Average" if pct >= 40 else "Needs Improvement",
        "gemini_analysis": gemini_analysis
    }
    user = current_user()
    user.setdefault("quiz_history", []).append({
        "category": quiz_session["category"],
        "topic": quiz_session["topic"],
        "score": score,
        "max_score": max_possible,
        "percentage": pct,
        "correct": correct,
        "incorrect": incorrect,
        "unanswered": unanswered,
        "total_time": total_time,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "detailed": detailed
    })
    return jsonify(result)


@app.route("/api/quiz/history", methods=["GET"])
@login_required
def api_quiz_history():
    user = current_user()
    return jsonify({"history": user.get("quiz_history", [])})


@app.route("/planner")
@login_required
def planner():
    return render_template("planner.html", active_page="planner", exam_configs=EXAM_CONFIGS)


@app.route("/community")
@login_required
def community():
    return render_template("community.html", active_page="community")


@app.route("/api/syllabus/plans", methods=["GET"])
@login_required
def api_syllabus_list():
    user = current_user()
    return jsonify({"plans": user.get("syllabus_plans", [])})


@app.route("/api/syllabus/parse-pdf", methods=["POST"])
@login_required
def api_syllabus_parse_pdf():
    if not PYPDF2_AVAILABLE:
        return jsonify({"error": "PDF parsing not available."}), 500
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file uploaded. Use field name 'pdf'."}), 400
    f = request.files["pdf"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed."}), 400
    exam = (request.form.get("exam") or "").strip()
    subject_filter = (request.form.get("subject") or "").strip()
    try:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages[:8]:
            try:
                text += (page.extract_text() or "") + "\n"
            except:
                continue
        text = text.strip()
        if not text or len(text) < 30:
            return jsonify({"error": "Could not extract text from PDF. Try a text-based PDF."}), 400
        raw_text = text[:8000]
        # Ask Gemini to intelligently read and structure the syllabus
        if GEMINI_MODEL and raw_text:
            subj_line = f'Focus ONLY on subject "{subject_filter}" — extract only topics for this subject. Ignore other subjects.' if subject_filter else 'Extract all topics grouped by subject.'
            prompt = f"""
You are an expert syllabus analyzer for {exam or "General Exam"}.
PDF extracted text (first 8000 chars):
{raw_text}

Task: Intelligently read this syllabus PDF, clean OCR noise, and extract a structured syllabus.
{subj_line}

Return ONLY valid JSON (no markdown, no fences, plain ASCII, no backslashes):
{{
  "full_cleaned": "One-line cleaned syllabus summary (max 400 chars)",
  "topics": ["Topic 1: Subtopic", "Topic 2", ...],
  "subjects_found": ["SubjectA", "SubjectB"]
}}
Rules: Plain ASCII, no LaTeX, no backslashes, topics concise (3-8 words each), max 30 topics.
"""
            try:
                resp = _gemini_generate(prompt)
                t = resp.text.strip()
                if "```" in t:
                    s, e = t.find("{"), t.rfind("}")
                    if s != -1 and e != -1:
                        t = t[s:e+1]
                try:
                    parsed = json.loads(t)
                except json.JSONDecodeError:
                    import re
                    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', t)
                    parsed = json.loads(fixed)
                topics = parsed.get("topics", [])[:30]
                full_cleaned = parsed.get("full_cleaned", raw_text[:600])
                subjects_found = parsed.get("subjects_found", [])
                return jsonify({
                    "extracted_length": len(text),
                    "raw_preview": raw_text[:600],
                    "topics": topics,
                    "syllabus_text": "\n".join(topics) if topics else raw_text[:2000],
                    "full_cleaned": full_cleaned,
                    "subjects_found": subjects_found
                })
            except Exception as e:
                # Fallback: return raw text as syllabus
                return jsonify({
                    "extracted_length": len(text),
                    "raw_preview": raw_text[:600],
                    "topics": [l.strip() for l in raw_text.split("\n") if len(l.strip())>3][:20],
                    "syllabus_text": raw_text[:2000],
                    "full_cleaned": raw_text[:600],
                    "subjects_found": []
                })
        return jsonify({
            "extracted_length": len(text),
            "raw_preview": raw_text[:600] if 'raw_text' in locals() else text[:600],
            "topics": [l.strip() for l in text.split("\n") if len(l.strip())>3][:20],
            "syllabus_text": text[:2000],
            "full_cleaned": text[:600],
            "subjects_found": []
        })
    except Exception as e:
        return jsonify({"error": f"PDF parse failed: {str(e)}"}), 500


@app.route("/api/syllabus/generate", methods=["POST"])
@login_required
def api_syllabus_generate():
    if not GEMINI_MODEL:
        return jsonify({"error": "AI model not configured."}), 503
    data = request.get_json(silent=True) or {}
    exam = (data.get("exam") or "").strip()
    syllabus = (data.get("syllabus") or "").strip()
    exam_date_str = (data.get("exam_date") or "").strip()
    daily_hours = int(data.get("daily_hours", 3))
    subject = (data.get("subject") or "").strip()
    if not exam or not syllabus or not exam_date_str:
        return jsonify({"error": "Exam, syllabus and exam date are required."}), 400
    if exam not in EXAM_CONFIGS:
        return jsonify({"error": "Invalid exam."}), 400
    try:
        exam_date = datetime.strptime(exam_date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid exam date format. Use YYYY-MM-DD."}), 400
    today = datetime.now().date()
    if exam_date <= today:
        return jsonify({"error": "Exam date must be in the future."}), 400
    days_left = (exam_date - today).days
    # Cap planning horizon at 360 days for sanity
    gen_days = min(days_left, 360)
    syllabus_snippet = syllabus[:4000]
    subject_line = f'Subject focus: "{subject}" — ONLY schedule topics belonging to this subject. Ignore all other subjects.' if subject else 'Schedule across all subjects in the syllabus.'
    exam_subjects = ", ".join(EXAM_CONFIGS[exam].get("subjects", []))

    def build_batch_prompt(batch_start, batch_end, batch_no, total_batches):
        return f"""
You are an expert study planner for {exam} (exam date: {exam_date_str}, subjects: {exam_subjects}). Today is {today.isoformat()}.
{subject_line}
Syllabus to cover:
{syllabus_snippet}

Daily study capacity: {daily_hours} hours.
The FULL plan spans {gen_days} days ({(today + timedelta(days=1)).isoformat()} to {(today + timedelta(days=min(days_left, 360))).isoformat()}), divided into {total_batches} generation batches.
This is batch {batch_no} of {total_batches}: generate ONLY days from {batch_start.isoformat()} to {batch_end.isoformat()} ({((batch_end - batch_start).days + 1)} days).
Distribute the ENTIRE syllabus evenly across ALL {gen_days} days mentally, then output only this batch's slice — later batches will cover later portions. Harder topics get more days. Include 1 revision day per week.

Return ONLY valid JSON array with this exact structure (no markdown, no fences, plain ASCII, no backslashes):
[
  {{
    "date": "{batch_start.isoformat()}",
    "tasks": [
      {{"name": "Topic: Subtopic", "duration": 60, "priority": "P1"}}
    ]
  }}
]
Rules:
- EXACTLY one entry per date from {batch_start.isoformat()} to {batch_end.isoformat()}, no skipped dates.
- Each day: 1-3 topic tasks, total duration approx {daily_hours*60} minutes (±30 min).
- These are DAILY TOPIC TARGETS — no clock times needed, just what to finish that day.
- Priority P1 = hardest/most important, P2 = medium, P3 = revision/easy.
- Use plain ASCII only, no LaTeX, no backslashes.
- No extra text.
"""

    user = current_user()
    plan_id = str(uuid.uuid4())
    created_tasks = []

    # === CHUNKED GENERATION: batches run in PARALLEL so long plans stay fast ===
    from concurrent.futures import ThreadPoolExecutor
    BATCH = 40
    batch_ranges = []
    bs = today + timedelta(days=1)
    while bs <= min(today + timedelta(days=gen_days), exam_date):
        be = min(bs + timedelta(days=BATCH - 1), min(today + timedelta(days=gen_days), exam_date))
        batch_ranges.append((bs, be))
        bs = be + timedelta(days=1)
    total_batches = len(batch_ranges)

    def gen_batch(args):
        bi, (b_start, b_end) = args
        prompt = build_batch_prompt(b_start, b_end, bi + 1, total_batches)
        for attempt in range(2):  # one retry per batch
            try:
                resp = _gemini_generate(prompt)
                txt = resp.text.strip()
                if "```" in txt:
                    si, ei = txt.find("["), txt.rfind("]")
                    if si != -1 and ei != -1:
                        txt = txt[si:ei+1]
                si, ei = txt.find("["), txt.rfind("]")
                if si != -1 and ei != -1:
                    txt = txt[si:ei+1]
                try:
                    days = json.loads(txt)
                except json.JSONDecodeError:
                    import re
                    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', txt)
                    days = json.loads(fixed)
                if isinstance(days, list):
                    return days
            except Exception:
                if attempt == 1:
                    return []
        return []

    with ThreadPoolExecutor(max_workers=min(3, total_batches)) as pool:
        results = list(pool.map(gen_batch, enumerate(batch_ranges)))

    plan_days_all = []
    for chunk in results:
        plan_days_all.extend(chunk)

    try:
        plan_days = plan_days_all
        if not isinstance(plan_days, list) or not plan_days:
            raise ValueError("Invalid plan format")

        # Create tasks — daily targets WITHOUT fixed times
        seen_dates = set()
        for day in plan_days:
            d = day.get("date")
            # Validate date & dedupe (later batches could overlap boundary)
            try:
                dd = datetime.strptime(d, "%Y-%m-%d").date()
            except Exception:
                continue
            if dd in seen_dates or dd > today + timedelta(days=gen_days) or dd > exam_date:
                continue
            seen_dates.add(dd)
            for t in day.get("tasks", [])[:3]:
                name = (t.get("name") or "Study").strip()[:120]
                try:
                    dur = max(30, min(240, int(t.get("duration", 60))))
                except (TypeError, ValueError):
                    dur = 60
                prio = t.get("priority") if t.get("priority") in ("P1","P2","P3") else "P2"
                task = {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "start_time": "09:00",
                    "end_time": "18:00",
                    "no_time": True,
                    "duration": dur,
                    "energy": "High" if prio=="P1" else "Med" if prio=="P2" else "Low",
                    "priority": prio,
                    "completed": False,
                    "healed": False,
                    "date": d,
                    "is_syllabus": True,
                    "plan_id": plan_id,
                }
                user["tasks"].append(task)
                created_tasks.append(task)
        plan = {
            "id": plan_id,
            "exam": exam,
            "subject": subject or "All Subjects",
            "syllabus": syllabus[:600],
            "exam_date": exam_date_str,
            "daily_hours": daily_hours,
            "created_at": datetime.now().isoformat(),
            "days_generated": gen_days,
            "total_tasks": len(created_tasks),
            "days_left_total": days_left,
        }
        user.setdefault("syllabus_plans", []).append(plan)
        return jsonify({"plan": plan, "tasks": created_tasks})
    except Exception as e:
        return jsonify({"error": f"Failed to generate plan: {str(e)}"}), 500


@app.route("/api/syllabus/<plan_id>/reschedule", methods=["POST"])
@login_required
def api_syllabus_reschedule(plan_id):
    user = current_user()
    plan = next((p for p in user.get("syllabus_plans", []) if p["id"] == plan_id), None)
    if not plan:
        return jsonify({"error": "Plan not found."}), 404
    today_str = datetime.now().strftime("%Y-%m-%d")
    # Collect missed tasks (syllabus tasks, date < today, not completed)
    missed = [t for t in user["tasks"] if t.get("plan_id")==plan_id and t.get("date","") < today_str and not t.get("completed")]
    if not missed:
        return jsonify({"message": "No missed tasks to reschedule!", "rescheduled": 0})
    if not GEMINI_MODEL:
        # Simple fallback: push missed tasks to next 3 days
        future_dates = [(datetime.now().date() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(3)]
        for idx, t in enumerate(missed):
            t["date"] = future_dates[idx % len(future_dates)]
            t["healed"] = True
        return jsonify({"message": f"Rescheduled {len(missed)} missed tasks to next 3 days.", "rescheduled": len(missed)})
    # AI smart reschedule
    remaining = [t for t in user["tasks"] if t.get("plan_id")==plan_id and not t.get("completed") and t.get("date","") >= today_str]
    missed_names = [t["name"] for t in missed[:10]]
    remaining_names = [t["name"] for t in remaining[:10]]
    prompt = f"""
Exam: {plan["exam"]} on {plan["exam_date"]}. Today is {today_str}.
Missed tasks (not completed, overdue): {", ".join(missed_names) or "none"}
Remaining upcoming tasks: {", ".join(remaining_names) or "none"}

TASK: Redistribute the missed tasks smartly into the next 7 days, interleaving with remaining topics so load is balanced (max 3 tasks/day, prioritize missed high-priority first).

Return ONLY JSON array:
[
  {{"task_name": "Exact name from missed list", "new_date": "YYYY-MM-DD"}}
]
No markdown, no extra text, plain ASCII.
"""
    try:
        resp = _gemini_generate(prompt)
        text = resp.text.strip()
        if "```" in text:
            s, e = text.find("["), text.rfind("]")
            if s != -1 and e != -1:
                text = text[s:e+1]
        s, e = text.find("["), text.rfind("]")
        if s != -1 and e != -1:
            text = text[s:e+1]
        try:
            mapping = json.loads(text)
        except json.JSONDecodeError:
            import re
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
            mapping = json.loads(fixed)
        name_to_task = {t["name"]: t for t in missed}
        cnt = 0
        for m in mapping:
            nm = m.get("task_name")
            nd = m.get("new_date")
            if nm in name_to_task and nd:
                try:
                    datetime.strptime(nd, "%Y-%m-%d")
                    name_to_task[nm]["date"] = nd
                    name_to_task[nm]["healed"] = True
                    cnt += 1
                except:
                    continue
        # Fallback for any missed not mapped by AI
        if cnt < len(missed):
            fallback_dates = [(datetime.now().date() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(7)]
            for t in missed:
                if not t.get("healed"):
                    t["date"] = fallback_dates[0]
                    t["healed"] = True
                    cnt += 1
        return jsonify({"message": f"AI rescheduled {cnt} missed tasks smartly for the next week.", "rescheduled": cnt})
    except Exception as e:
        return jsonify({"error": f"Reschedule failed: {str(e)}"}), 500


@app.route("/api/syllabus/<plan_id>", methods=["DELETE"])
@login_required
def api_syllabus_delete(plan_id):
    user = current_user()
    plans = user.get("syllabus_plans", [])
    user["syllabus_plans"] = [p for p in plans if p["id"] != plan_id]
    # Also remove associated tasks
    user["tasks"] = [t for t in user["tasks"] if t.get("plan_id") != plan_id]
    return jsonify({"ok": True})


@app.route("/focus")
@login_required
def focus():
    return render_template("focus_timer.html", active_page="focus")


@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html", active_page="settings")


# ---------------------------------------------------------------------------
# REST API Routes
# ---------------------------------------------------------------------------
def user_tasks():
    user = current_user()
    return user["tasks"] if user else []


@app.route("/api/tasks", methods=["GET"])
@login_required
def api_tasks():
    tasks = sorted(user_tasks(), key=lambda t: t["start_time"])
    return jsonify({"tasks": tasks})


@app.route("/api/tasks", methods=["POST"])
@login_required
def api_add_task():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Task name is required."}), 400

    task = {
        "id": str(uuid.uuid4()),
        "name": name,
        "start_time": data.get("start_time") or "09:00",
        "end_time": data.get("end_time") or "10:00",
        "energy": data.get("energy") or "Med",
        "priority": data.get("priority") or "P2",
        "completed": False,
        "healed": False,
        "date": data.get("date") or datetime.now().strftime("%Y-%m-%d"),
        "is_syllabus": data.get("is_syllabus", False),
        "plan_id": data.get("plan_id"),
    }
    # Append via JSON column (MutableList tracks)
    user = current_user()
    user.tasks_data.append(task)
    from sqlalchemy.orm.attributes import flag_modified; flag_modified(user, "tasks_data")
    db.session.commit()
    return jsonify({"task": task}), 201


@app.route("/api/tasks/<task_id>", methods=["PUT"])
@login_required
def api_update_task(task_id):
    data = request.get_json(silent=True) or {}
    user = current_user()
    for task in user_tasks():
        if task["id"] == task_id:
            if "name" in data and (data.get("name") or "").strip():
                task["name"] = data["name"].strip()[:120]
            if "start_time" in data and data.get("start_time"):
                task["start_time"] = data["start_time"]
            if "end_time" in data and data.get("end_time"):
                task["end_time"] = data["end_time"]
            if "energy" in data and data.get("energy") in ("High", "Med", "Low"):
                task["energy"] = data["energy"]
            if "priority" in data and data.get("priority") in ("P1", "P2", "P3", "P4", "P5"):
                task["priority"] = data["priority"]
            if "date" in data and data.get("date"):
                try:
                    datetime.strptime(data["date"], "%Y-%m-%d")
                    task["date"] = data["date"]
                except ValueError:
                    pass
            from sqlalchemy.orm.attributes import flag_modified; flag_modified(user, "tasks_data")
            db.session.commit()
            return jsonify({"task": task})
    return jsonify({"error": "Task not found."}), 404


@app.route("/api/tasks/<task_id>/complete", methods=["POST"])
@login_required
def api_toggle_task(task_id):
    user = current_user()
    for task in user_tasks():
        if task["id"] == task_id:
            task["completed"] = not task["completed"]
            if task["completed"]:
                task["completed_at"] = datetime.now().isoformat()
                auto_record_completion(user, task)
            else:
                task.pop("completed_at", None)
            from sqlalchemy.orm.attributes import flag_modified; flag_modified(user, "tasks_data"); flag_modified(user, "learning_data")
            db.session.commit()
            return jsonify({"task": task})
    return jsonify({"error": "Task not found."}), 404


@app.route("/api/tasks/<task_id>", methods=["DELETE"])
@login_required
def api_delete_task(task_id):
    user = current_user()
    tasks = user_tasks()
    for i, task in enumerate(tasks):
        if task["id"] == task_id:
            removed = tasks.pop(i)
            from sqlalchemy.orm.attributes import flag_modified; flag_modified(user, "tasks_data")
            db.session.commit()
            return jsonify({"deleted": removed})
    return jsonify({"error": "Task not found."}), 404


def _hm_to_dt(hm, base):
    h, m = map(int, hm.split(":"))
    return base.replace(hour=h, minute=m, second=0, microsecond=0)


def _dt_to_hm(dt):
    return dt.strftime("%H:%M")


TAUNTS_LIGHT = [
    "A tiny scroll break? Cute. Your schedule forgives you this time.",
    "15 minutes? That was just a warm-up. Back to work now.",
]
TAUNTS_MED = [
    "Caught in the scroll-void again, huh? I'll pull you out. Again.",
    "Half an hour of reels — your tasks felt that. Let's rebuild.",
]
TAUNTS_HEAVY = [
    "An hour gone? Bhai, your day misses you. Let's save the rest.",
    "2 ghante? That's a whole lecture of distraction. Fine — I'll fix it. You're welcome.",
    "Your tasks are filing a complaint. I'll reschedule your guilt too.",
]
TAUNTS_EXTREME = [
    "Half your day vanished into thin air. Impressive dedication to procrastination.",
    "You really tested time itself. I'll rebuild your day from ashes. You're welcome.",
]
TAUNTS_ZERO = [
    "Zero time wasted? Main impressed. A disciplined human is rare these days.",
    "No waste detected — are you feeling okay? Let's just optimize anyway.",
]


def pick_taunt(wasted):
    if wasted <= 0:
        return random.choice(TAUNTS_ZERO)
    if wasted <= 15:
        return random.choice(TAUNTS_LIGHT)
    if wasted <= 45:
        return random.choice(TAUNTS_MED)
    if wasted <= 120:
        return random.choice(TAUNTS_HEAVY)
    return random.choice(TAUNTS_EXTREME)


@app.route("/api/reschedule", methods=["POST"])
@login_required
def api_reschedule():
    """
    Smart AI Reschedule:
    - Takes user's todo list (tasks with priorities, durations, preferred times)
    - AI predicts energy levels throughout the day
    - Creates optimal schedule matching energy peaks to high-priority work
    - On reschedule: AI redesigns based on current energy + missed tasks
    """
    data = request.get_json(silent=True) or {}
    
    # User inputs
    energy = data.get("current_energy")
    if energy in {"Low Energy", "Active", "Peak Focus"}:
        current_user()["current_energy"] = energy
    
    wasted = 0
    try:
        wasted = max(0, min(600, int(data.get("wasted_minutes", 0) or 0)))
    except (TypeError, ValueError):
        wasted = 0
    
    # Custom todo list from user (optional override)
    user_tasks_override = data.get("tasks")  # list of {name, duration, priority, preferred_start}
    
    user = current_user()
    now = datetime.now().replace(second=0, microsecond=0)
    
    # Get incomplete tasks (use override if provided, else existing tasks)
    if user_tasks_override:
        incomplete = user_tasks_override
    else:
        incomplete = [t for t in user["tasks"] if not t.get("completed")]
    
    # Only consider future, non-syllabus tasks (syllabus plans heal separately)
    remaining = [
        t for t in incomplete
        if _hm_to_dt(t.get("end_time", "23:59"), now) > now and not t.get("is_syllabus")
    ]
    
    if not remaining:
        return jsonify({
            "message": "Nothing left to heal - you're all done for today!",
            "healed": 0,
            "dropped": [],
            "taunt": pick_taunt(wasted),
            "wasted_minutes": wasted,
            "schedule": []
        })

    # Working window: from now (plus wasted time) until 22:00
    window_end = now.replace(hour=22, minute=0, second=0, microsecond=0)
    start = now + timedelta(minutes=wasted)
    if start >= window_end:
        return jsonify({
            "message": "The day is basically over - even AI can't create time from nothing.",
            "healed": 0,
            "dropped": [t["name"] for t in remaining],
            "taunt": pick_taunt(wasted),
            "wasted_minutes": wasted,
            "schedule": []
        })
    
    available = (window_end - start).total_seconds() / 60.0
    
    # Priority order: P1 Power > P2 Focus > P3 Quick Win > P4 Break > P5 Unproductive
    rank = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4}
    kept = sorted(remaining, key=lambda t: (rank.get(t.get("priority", "P3"), 2), t.get("start_time", "00:00")))
    
    # AI PREDICTED ENERGY CURVE for the day (based on circadian rhythm + user's current energy)
    def predict_energy_curve(current_energy, hours_from_now):
        """Returns energy multiplier 0.5-1.5 for each hour slot"""
        hour = (datetime.now().hour + hours_from_now) % 24
        # Base circadian rhythm (peaks at 10AM, 4PM; dips at 2PM, 10PM)
        circadian = {
            6: 0.6, 7: 0.7, 8: 0.9, 9: 1.1, 10: 1.2, 11: 1.15,
            12: 0.9, 13: 0.75, 14: 0.65, 15: 0.8, 16: 1.1, 17: 1.05,
            18: 0.9, 19: 0.8, 20: 0.7, 21: 0.6
        }.get(hour, 0.5)
        
        # Current energy modifier
        energy_mod = {"Low Energy": 0.85, "Active": 1.0, "Peak Focus": 1.15}.get(user.get("current_energy", "Active"), 1.0)
        
        return min(1.5, max(0.4, circadian * energy_mod))
    
    # Calculate demand
    demand = sum(
        (_hm_to_dt(t.get("end_time", "23:59"), now) - _hm_to_dt(t.get("start_time", "00:00"), now)).total_seconds() / 60.0
        for t in kept
    )

    # ========== GEMINI-POWERED RESCHEDULING (with heuristic fallback) ==========
    gemini_schedule = None
    gemini_cuts = []
    gemini_dropped = []
    if GEMINI_AVAILABLE and GEMINI_API_KEY and GEMINI_MODEL:
        try:
            # Build concise task list for prompt
            task_lines = []
            for t in kept:
                dur = int((_hm_to_dt(t.get("end_time","23:59"), now) - _hm_to_dt(t.get("start_time","00:00"), now)).total_seconds() / 60)
                task_lines.append(f'- id:{t["id"]} name:"{t["name"]}" priority:{t.get("priority","P3")} energy:{t.get("energy","Med")} duration:{dur}m orig:{t.get("start_time","?")}–{t.get("end_time","?")}')
            tasks_text = "\n".join(task_lines)
            energy_curve_hint = "Energy curve (circadian peaks 10AM & 4PM, dip 2PM): "
            # quick energy per hour 06-22
            tmp_curve = []
            for h in range(start.hour, 22):
                # simplified circadian
                circadian = {6:0.6,7:0.7,8:0.9,9:1.1,10:1.2,11:1.15,12:0.9,13:0.75,14:0.65,15:0.8,16:1.1,17:1.05,18:0.9,19:0.8,20:0.7,21:0.6}.get(h,0.5)
                emod = {"Low Energy":0.85,"Active":1.0,"Peak Focus":1.15}.get(user.get("current_energy","Active"),1.0)
                tmp_curve.append(f"{h:02d}:00={round(circadian*emod,2)}")
            energy_curve_hint += ", ".join(tmp_curve)

            prompt = f"""You are Synora, an expert productivity scheduler. Rebuild the user's remaining day as the MOST productive schedule.

Context:
- Today: {now.strftime("%Y-%m-%d %H:%M")}
- Reschedule window: {start.strftime("%H:%M")} to 22:00 (available {int(available)} min)
- Wasted minutes today: {wasted}
- Current energy: {user.get("current_energy","Active")}
- {energy_curve_hint}
- Rule: P1=Power (deep work, needs peak energy), P2=Focus (needs high energy), P3=Quick Win, P4=Break, P5=Unproductive (cut first if short on time). Minimum task length 15 min.

Tasks to schedule:
{tasks_text}

Instructions:
1. Place P1/P2 tasks in highest energy windows, earliest possible.
2. Fill remaining gaps chronologically with P3/P4/P5.
3. If demand > available: trim P5 up to 70%, P4 up to 50%, P3 up to 30% (min 15m) OR drop tasks that absolutely don't fit. Prefer trimming over dropping.
4. No overlaps, all times within window, 5-min granularity, end_time > start_time.
5. Be concise and realistic.

Return ONLY valid JSON with no markdown, no explanation, in this exact shape:
{{"schedule": [{{"id": "abc123", "start_time": "HH:MM", "end_time": "HH:MM"}}], "dropped": ["Task Name"], "cuts": [{{"name": "Task Name", "before": 60, "after": 30}}], "reasoning": "1-line why this is productive"}}
If every task fits, dropped=[] and cuts=[].
"""

            resp = GEMINI_MODEL.generate_content(prompt)
            raw = (getattr(resp, "text", "") or "").strip()
            # strip code fences if present
            if raw.startswith("```"):
                raw = raw.strip("`")
                # remove leading json marker
                if raw.lstrip().startswith("json"):
                    raw = raw.lstrip()[4:].strip()
            parsed = json.loads(raw)
            cand_schedule = parsed.get("schedule", [])
            # Validate
            seen_ids = set()
            sched_map = {}
            gemini_valid = True
            intervals = []
            orig_dur_by_id = {t["id"]: int((_hm_to_dt(t.get("end_time","23:59"), now) - _hm_to_dt(t.get("start_time","00:00"), now)).total_seconds()/60) for t in kept}
            name_by_id = {t["id"]: t["name"] for t in kept}
            for item in cand_schedule:
                tid = item.get("id")
                st = item.get("start_time")
                en = item.get("end_time")
                if tid not in name_by_id or tid in seen_ids:
                    gemini_valid = False; break
                try:
                    s_dt = _hm_to_dt(st, now)
                    e_dt = _hm_to_dt(en, now)
                except Exception:
                    gemini_valid = False; break
                if not (start <= s_dt < e_dt <= window_end):
                    gemini_valid = False; break
                if (e_dt - s_dt).total_seconds() < 15*60:
                    gemini_valid = False; break
                # overlap check
                for (os, oe) in intervals:
                    if not (e_dt <= os or s_dt >= oe):
                        gemini_valid = False; break
                if not gemini_valid:
                    break
                intervals.append((s_dt, e_dt))
                seen_ids.add(tid)
                sched_map[tid] = (s_dt, e_dt)
            if gemini_valid and len(seen_ids) > 0:
                # Build normalized schedule list sorted chronologically
                gemini_schedule = []
                for tid, (s_dt, e_dt) in sorted(sched_map.items(), key=lambda kv: kv[1][0]):
                    t = next(x for x in kept if x["id"] == tid)
                    zs_vals = []
                    cur = s_dt
                    while cur < e_dt:
                        h = cur.hour
                        circ = {6:0.6,7:0.7,8:0.9,9:1.1,10:1.2,11:1.15,12:0.9,13:0.75,14:0.65,15:0.8,16:1.1,17:1.05,18:0.9,19:0.8,20:0.7,21:0.6}.get(h,0.5)
                        emod = {"Low Energy":0.85,"Active":1.0,"Peak Focus":1.15}.get(user.get("current_energy","Active"),1.0)
                        zs_vals.append(min(1.5, max(0.4, circ*emod)))
                        cur += timedelta(minutes=15)
                    gemini_schedule.append({
                        "id": tid, "name": name_by_id[tid],
                        "start_time": _dt_to_hm(s_dt), "end_time": _dt_to_hm(e_dt),
                        "priority": t.get("priority","P3"),
                        "energy_slot": round(sum(zs_vals)/len(zs_vals),2) if zs_vals else 0.5,
                        "healed": True, "duration": int((e_dt - s_dt).total_seconds()/60)
                    })
                # Derive cuts & dropped
                gemini_dropped = parsed.get("dropped", []) or []
                # If Gemini didn't list dropped, infer from missing ids
                dropped_ids = set(name_by_id.keys()) - seen_ids
                if not gemini_dropped and dropped_ids:
                    gemini_dropped = [name_by_id[did] for did in dropped_ids]
                gemini_cuts = parsed.get("cuts", []) or []
                # Enrich cuts with emoji/label if Gemini omitted
                for c in gemini_cuts:
                    if "emoji" not in c:
                        c["emoji"] = {"P5":"⚪","P4":"🔵","P3":"🟢"}.get(next((x.get("priority","P3") for x in kept if x["name"]==c.get("name")), "P3"), "")
                    if "label" not in c:
                        c["label"] = {"P5":"Unproductive","P4":"Break","P3":"Quick Win"}.get(next((x.get("priority","P3") for x in kept if x["name"]==c.get("name")), "P3"), c.get("name",""))
                    # ensure before/after
                    if "before" not in c and c.get("name") in name_by_id.values():
                        tid2 = next((k for k,v in name_by_id.items() if v==c["name"]), None)
                        if tid2: c["before"] = orig_dur_by_id.get(tid2, 0)
                # If Gemini succeeded, persist and return immediately
                if gemini_schedule:
                    dropped_id_set = set(name_by_id[k] for k in dropped_ids)  # name set for filtering
                    # Persist onto user's real tasks (same Phase D as heuristic)
                    dropped_name_set = set(gemini_dropped)
                    # For cuts, update end_time implied by new schedule length already; no extra mutation needed
                    new_tasks = []
                    sched_by_id = {s["id"]: s for s in gemini_schedule}
                    dropped_ids_set = dropped_ids
                    for t in user["tasks"]:
                        if t.get("completed"):
                            new_tasks.append(t); continue
                        tid = t.get("id")
                        if tid in dropped_ids_set and _hm_to_dt(t.get("end_time","23:59"), now) > now:
                            continue
                        if tid in sched_by_id:
                            s = sched_by_id[tid]
                            t["start_time"] = s["start_time"]
                            t["end_time"] = s["end_time"]
                            t["healed"] = True
                        new_tasks.append(t)
                    user["tasks"] = new_tasks
                    user["last_taunt"] = pick_taunt(wasted)
                    # Filter cuts that were actually dropped
                    gemini_cuts = [c for c in gemini_cuts if c.get("name") not in dropped_name_set]
                    total_saved = sum((c.get("before",0)-c.get("after",0)) for c in gemini_cuts) if gemini_cuts else 0
                    parts = []
                    if gemini_cuts:
                        cut_desc = ", ".join(f"{c.get('emoji','')} {c['name']} ({c.get('before')}m → {c.get('after')}m)" for c in gemini_cuts)
                        parts.append(f"Gemini trimmed {len(gemini_cuts)} task(s): {cut_desc}.")
                    if gemini_dropped:
                        parts.append(f"Dropped: {', '.join(gemini_dropped)}.")
                    parts.append(f"{len(gemini_schedule)} tasks placed in your peak energy windows. {parsed.get('reasoning','')}".strip())
                    return jsonify({
                        "message": "Gemini rebuilt your day! " + " ".join(parts),
                        "healed": len(gemini_schedule),
                        "dropped": gemini_dropped,
                        "cuts": gemini_cuts,
                        "taunt": pick_taunt(wasted),
                        "wasted_minutes": wasted,
                        "current_energy": user.get("current_energy","Active"),
                        "schedule": sorted(gemini_schedule, key=lambda x: x["start_time"]),
                        "energy_curve": [{"hour": h, "energy": round(min(1.5, max(0.4, {6:0.6,7:0.7,8:0.9,9:1.1,10:1.2,11:1.15,12:0.9,13:0.75,14:0.65,15:0.8,16:1.1,17:1.05,18:0.9,19:0.8,20:0.7,21:0.6}.get(h,0.5)*{"Low Energy":0.85,"Active":1.0,"Peak Focus":1.15}.get(user.get("current_energy","Active"),1.0))), 2)} for h in range(start.hour, 22)],
                        "ai": True
                    })
        except Exception as _gem_e:
            # Fall through to heuristic; log for debugging
            print(f"[reschedule] Gemini failed, falling back to heuristic: {_gem_e}")

    # === SMART TIME-CUTTING: recover wasted minutes by compressing low-priority tasks ===
    # Order: Unproductive (P5, cut up to 70%) -> Breaks (P4, up to 50%) -> Quick Wins (P3, up to 30%)
    deficit = max(0.0, demand - available)
    cuts = []  # [{name, priority, before, after}] for animation
    compression_plan = [("P5", 0.70), ("P4", 0.50), ("P3", 0.30)]

    if deficit > 0:
        for prio, cut_ratio in compression_plan:
            if deficit <= 0:
                break
            prio_tasks = sorted(
                [t for t in kept if t.get("priority") == prio],
                key=lambda t: (_hm_to_dt(t["end_time"], now) - _hm_to_dt(t["start_time"], now)).total_seconds(),
                reverse=True,  # longest first — bigger cuts recover faster
            )
            for task in prio_tasks:
                if deficit <= 0:
                    break
                orig_dur = (_hm_to_dt(task["end_time"], now) - _hm_to_dt(task["start_time"], now)).total_seconds() / 60.0
                min_dur = 15.0
                max_recoverable = max(0.0, orig_dur - min_dur) * cut_ratio
                actual_cut = min(deficit, max_recoverable)
                if actual_cut < 5:  # not worth cutting less than 5 min
                    continue
                new_dur = orig_dur - actual_cut
                task["end_time"] = _dt_to_hm(_hm_to_dt(task["start_time"], now) + timedelta(minutes=new_dur))
                task["_cut"] = True
                deficit -= actual_cut
                demand -= actual_cut
                emoji = {"P5": "⚪", "P4": "🔵", "P3": "🟢"}.get(prio, "")
                label = {"P5": "Unproductive", "P4": "Break", "P3": "Quick Win"}.get(prio, prio)
                cuts.append({
                    "name": task["name"],
                    "priority": prio,
                    "label": label,
                    "emoji": emoji,
                    "before": round(orig_dur),
                    "after": round(new_dur),
                    "saved": round(actual_cut),
                })

    # AI SMART SCHEDULING — full productive redesign of the rest of the day
    # Strategy:
    #   Phase A: WORK tasks (P1/P2) placed at best ENERGY-matched windows (earliness-biased)
    #   Phase B: LIGHT/BREAK/UNPROD tasks fill remaining gaps CHRONOLOGICALLY
    #   Phase C: compaction — pull everything tight, no dead time
    #   Phase D: persist healed timetable onto user's real tasks

    slots = []
    cursor = start
    slot_idx = 0
    while cursor + timedelta(minutes=15) <= window_end:
        slots.append({
            "start": cursor,
            "energy": predict_energy_curve(user.get("current_energy", "Active"), slot_idx),
            "hour": cursor.hour
        })
        cursor += timedelta(minutes=15)
        slot_idx += 0.25

    req_energy_map = {"P1": 1.15, "P2": 1.0, "P3": 0.85, "P4": 0.6, "P5": 0.5}

    def task_duration(t):
        d = (_hm_to_dt(t.get("end_time", "23:59"), now) - _hm_to_dt(t.get("start_time", "00:00"), now)).total_seconds() / 60.0
        return max(15, round(d))

    def required_energy(t):
        prio = t.get("priority", "P3")
        e = req_energy_map.get(prio, 0.85)
        te = t.get("energy", "")
        if te == "High":
            e = max(e, 1.1)
        elif te == "Low":
            e = min(e, 0.7)
        return e

    used = [False] * len(slots)

    def place_best(task):
        """Energy-match placement with earliness bias. Returns index or -1."""
        dur = task_duration(task)
        needed = max(1, round(dur / 15))
        req_e = required_energy(task)
        prio = task.get("priority", "P3")
        pw = 5 - rank.get(prio, 2)
        best_score, best_i = -1e9, -1
        for i in range(len(slots) - needed + 1):
            if any(used[i + j] for j in range(needed)):
                continue
            avg_e = sum(slots[i + j]["energy"] for j in range(needed)) / needed
            match = max(0.3, 1.0 - abs(avg_e - req_e) * 1.5)
            earliness = (len(slots) - i) / len(slots)  # prefer sooner slots -> compact day
            score = avg_e * pw * match + earliness * 0.6
            if score > best_score:
                best_score, best_i = score, i
        if best_i == -1:
            return -1
        for j in range(needed):
            used[best_i + j] = True
        return best_i

    def place_earliest(task, from_idx=0):
        """Chronological first-fit into free gaps (from from_idx onward). Returns index or -1."""
        dur = task_duration(task)
        needed = max(1, round(dur / 15))
        for scan_start in (from_idx, 0):
            i = scan_start
            while i <= len(slots) - needed:
                if any(used[i + j] for j in range(needed)):
                    i += 1
                    continue
                for j in range(needed):
                    used[i + j] = True
                return i
            i += 1
        return -1

    work = [t for t in kept if rank.get(t.get("priority", "P3"), 2) <= 1]
    light = [t for t in kept if rank.get(t.get("priority", "P3"), 2) >= 2]

    # Phase A: hardest/longest work first -> grabs peak-energy windows
    placed = {}  # id -> (start_dt, end_dt, dur)
    dropped = []
    for t in sorted(work, key=lambda x: (-task_duration(x), required_energy(x)), reverse=False):
        idx = place_best(t)
        if idx == -1:
            dropped.append(t)
            continue
        dur = task_duration(t)
        s = slots[idx]["start"]
        placed[t["id"]] = (s, min(s + timedelta(minutes=dur), window_end), dur)

    # Phase B: light/break/unprod tasks fill gaps around the work blocks
    work_start_idx = None
    if placed:
        first_work_start = min(v[0] for v in placed.values())
        for i, sl in enumerate(slots):
            if sl["start"] >= first_work_start:
                work_start_idx = i
                break
    for t in sorted(light, key=lambda x: (rank.get(x.get("priority", "P3"), 2),)):
        idx = place_earliest(t, work_start_idx or 0)
        if idx == -1:
            dropped.append(t)
            continue
        dur = task_duration(t)
        s = slots[idx]["start"]
        placed[t["id"]] = (s, min(s + timedelta(minutes=dur), window_end), dur)

    # Phase C: compaction — pull tasks earlier INTO ORDER (no jumping ahead of prior task)
    timeline = sorted(placed.items(), key=lambda kv: kv[1][0])
    occupied = []  # finalized (start,end)

    def fits(st, en):
        return all(en <= os or st >= oe for os, oe in occupied)

    def zone_energy(a, b):
        vals = [sl["energy"] for sl in slots if a <= sl["start"] < b]
        return sum(vals) / len(vals) if vals else 0.5

    prev_end = start
    for tid, (s, e, dur) in timeline:
        t_obj = next((x for x in kept if x["id"] == tid), None)
        is_work = t_obj is not None and rank.get(t_obj.get("priority", "P3"), 2) <= 1
        orig_zone = zone_energy(s, e)
        cand = max(start, prev_end)
        while cand + timedelta(minutes=dur) <= e:
            ce = cand + timedelta(minutes=dur)
            if fits(cand, ce):
                # work tasks never slide into notably weaker energy zones
                if is_work and zone_energy(cand, ce) + 0.15 < orig_zone:
                    cand += timedelta(minutes=15)
                    continue
                s, e = cand, ce
                break
            cand += timedelta(minutes=15)
        occupied.append((s, e))
        prev_end = e
        placed[tid] = (s, e, dur)

    # Build response schedule
    schedule = []
    for t in kept:
        if t["id"] in placed:
            s, e, dur = placed[t["id"]]
            zs = [sl["energy"] for sl in slots if s <= sl["start"] < e]
            schedule.append({
                "id": t["id"],
                "name": t["name"],
                "start_time": _dt_to_hm(s),
                "end_time": _dt_to_hm(e),
                "priority": t.get("priority", "P3"),
                "energy_slot": round(sum(zs) / len(zs), 2) if zs else 0.5,
                "healed": True,
                "duration": dur
            })
    dropped = [t["name"] for t in dropped]

    # chronological order for display
    schedule.sort(key=lambda x: x["start_time"])

    # === PHASE D: PERSIST — the timetable actually shifts on the dashboard ===
    dropped_id_set = {t["id"] for t in kept if t["id"] not in placed}
    sched_by_id = {s["id"]: s for s in schedule}
    cut_by_name = {c["name"]: c for c in cuts}
    new_tasks = []
    for t in user["tasks"]:
        if t.get("completed"):
            new_tasks.append(t)
            continue
        tid = t.get("id")
        if tid in dropped_id_set and _hm_to_dt(t.get("end_time", "23:59"), now) > now:
            continue  # AI dropped this future task
        if tid in sched_by_id:
            s = sched_by_id[tid]
            t["start_time"] = s["start_time"]
            t["end_time"] = s["end_time"]
            t["healed"] = True
        elif t.get("_cut"):
            t["healed"] = True
        new_tasks.append(t)
    user["tasks"] = new_tasks

    taunt = pick_taunt(wasted)
    user["last_taunt"] = taunt

    # If a trimmed task still didn't fit, report it as dropped only (not both)
    dropped_names = set(dropped)
    cuts = [c for c in cuts if c["name"] not in dropped_names]

    # Smart message describing what was cut
    parts = []
    if cuts:
        total_saved = sum(c["saved"] for c in cuts)
        cut_desc = ", ".join(f"{c['emoji']} {c['name']} ({c['before']}m to {c['after']}m)" for c in cuts)
        parts.append(f"Recovered {round(total_saved)} min by trimming: {cut_desc}.")
    if dropped:
        parts.append(f"Dropped (couldn't fit): {', '.join(dropped)}.")
    parts.append(f"{len(schedule)} tasks matched to your best energy windows.")

    return jsonify({
        "message": "AI optimized your day! " + " ".join(parts),
        "healed": len(schedule),
        "dropped": dropped,
        "cuts": cuts,
        "taunt": pick_taunt(wasted),
        "wasted_minutes": wasted,
        "current_energy": user.get("current_energy", "Active"),
        "schedule": schedule,
        "energy_curve": [{"hour": s["hour"], "energy": round(s["energy"], 2)} for s in slots]
    })


def pick_taunt(wasted):
    if wasted <= 0:
        return random.choice(TAUNTS_ZERO)
    if wasted <= 15:
        return random.choice(TAUNTS_LIGHT)
    if wasted <= 45:
        return random.choice(TAUNTS_MED)
    if wasted <= 120:
        return random.choice(TAUNTS_HEAVY)
    return random.choice(TAUNTS_EXTREME)


@app.route("/api/focus/record", methods=["POST"])
@login_required
def api_focus_record():
    data = request.get_json(silent=True) or {}
    try:
        seconds = max(0, int(data.get("seconds", 0)))
    except (TypeError, ValueError):
        seconds = 0
    if seconds <= 0:
        return jsonify({"error": "Invalid duration."}), 400
    user = current_user()
    user.setdefault("focus_sessions", []).append(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "seconds": seconds,
            "mode": data.get("mode", "timer"),
        }
    )
    return jsonify({"ok": True})


def _fmt_hm(secs):
    h = secs // 3600
    m = (secs % 3600) // 60
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m"


@app.route("/api/focus/stats", methods=["GET"])
@login_required
def api_focus_stats():
    user = current_user()
    sessions = user.get("focus_sessions", [])
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    week_start = (now - timedelta(days=6)).strftime("%Y-%m-%d")
    month = now.strftime("%Y-%m")
    year = now.strftime("%Y")

    def total(pred):
        return sum(s["seconds"] for s in sessions if pred(s["date"]))

    today_secs = total(lambda d: d == today)
    return jsonify(
        {
            "today": _fmt_hm(today_secs),
            "week": _fmt_hm(total(lambda d: d >= week_start)),
            "month": _fmt_hm(total(lambda d: d.startswith(month))),
            "year": _fmt_hm(total(lambda d: d.startswith(year))),
            "today_seconds": today_secs,
        }
    )


def _get_active_dates(user):
    dates = set()
    for s in user.get("focus_sessions", []):
        if s.get("seconds", 0) >= 600:
            dates.add(s.get("date"))
    for t in user.get("tasks", []):
        ca = t.get("completed_at")
        if t.get("completed") and ca:
            dates.add(ca[:10])
    return dates


@app.route("/api/streak", methods=["GET"])
@login_required
def api_streak():
    user = current_user()
    active = _get_active_dates(user)
    today = datetime.now().date()
    # Current streak: consecutive days up to today (if today not active, count from yesterday)
    cur = 0
    d = today
    if d.strftime("%Y-%m-%d") not in active:
        d -= timedelta(days=1)
    while d.strftime("%Y-%m-%d") in active:
        cur += 1
        d -= timedelta(days=1)
    # Best streak
    if active:
        sorted_dates = sorted(active)
        best = cur_best = 1
        for i in range(1, len(sorted_dates)):
            prev = datetime.strptime(sorted_dates[i-1], "%Y-%m-%d").date()
            cur_d = datetime.strptime(sorted_dates[i], "%Y-%m-%d").date()
            if (cur_d - prev).days == 1:
                cur_best += 1
                best = max(best, cur_best)
            else:
                cur_best = 1
        best = max(best, cur_best)
    else:
        best = 0
    # Last 7 days history for UI dots
    last7 = []
    for i in range(6, -1, -1):
        dd = today - timedelta(days=i)
        ds = dd.strftime("%Y-%m-%d")
        last7.append({"date": ds, "active": ds in active, "label": dd.strftime("%a")[0]})
    return jsonify({
        "current_streak": cur,
        "best_streak": best,
        "total_active": len(active),
        "last7": last7
    })


@app.route("/api/predicted-energy", methods=["GET"])
@login_required
def api_predicted_energy():
    user = current_user()
    hours = ["8AM", "10AM", "12PM", "2PM", "4PM", "6PM"]
    hour_map = {"8AM": 8, "10AM": 10, "12PM": 12, "2PM": 14, "4PM": 16, "6PM": 18}
    
    # 1. Personal energy model (learned from user patterns)
    personal_vals = {}
    for label, h in hour_map.items():
        personal_vals[label] = int(get_personal_energy_curve(user, h) * 100)
    
    # 2. Focus-session based model (existing)
    hour_windows = {
        "8AM": (7, 9),    "10AM": (9, 11),  "12PM": (11, 13),
        "2PM": (13, 15),  "4PM": (15, 17),  "6PM": (17, 19)
    }
    focus_by_hour = {h: [] for h in hours}
    for sess in user.get("focus_sessions", []):
        h = sess.get("hour")
        if h is None: continue
        for label, (start, end) in hour_windows.items():
            if start <= h < end:
                focus_by_hour[label].append(sess.get("seconds", 0))
                break
    
    focus_vals = {}
    max_avg = 1
    for label in hours:
        secs = focus_by_hour[label]
        if secs:
            avg_min = sum(secs) / len(secs) / 60.0
        else:
            avg_min = 0
        focus_vals[label] = avg_min
        max_avg = max(max_avg, avg_min)
    
    for label in hours:
        if max_avg > 0:
            focus_vals[label] = int((focus_vals[label] / max_avg) * 80 + 15)
        else:
            focus_vals[label] = 40
    
    # Blend: 70% personal model, 30% focus sessions (personal model is more reliable with 3+ days)
    ld = user.get("learning_data", {})
    has_personal = bool(ld.get("energy_model"))
    blend_weight = 0.7 if has_personal else 0.3
    
    values = []
    ce = user.get("current_energy", "Active")
    mod = {"Low Energy": -12, "Active": 0, "Peak Focus": 10}.get(ce, 0)
    active = _get_active_dates(user)
    streak_boost = 5 if len([d for d in active if d >= (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")]) >= 3 else 0
    
    for label in hours:
        blended = blend_weight * personal_vals[label] + (1 - blend_weight) * focus_vals[label]
        v = max(12, min(98, blended + mod + streak_boost))
        values.append({"hour": label, "value": round(v), "label": label.replace("M", " M")})
    
    return jsonify({
        "hours": values, 
        "current_energy": ce,
        "personal_model_active": has_personal,
        "data_days": len(set(r["date"] for r in user.get("learning_data", {}).get("energy_reports", [])))
    })


@app.route("/api/energy", methods=["GET"])
@login_required
def api_energy_get():
    user = current_user()
    return jsonify({"current_energy": user["current_energy"]})


@app.route("/api/energy", methods=["POST"])
@login_required
def api_energy_set():
    data = request.get_json(silent=True) or {}
    level = data.get("current_energy")
    valid = {"Low Energy", "Active", "Peak Focus"}
    if level not in valid:
        return jsonify({"error": "Invalid energy level."}), 400
    user = current_user()
    user["current_energy"] = level
    return jsonify({"current_energy": level})


@app.route("/api/tasks", methods=["DELETE"])
@login_required
def api_tasks_clear():
    """Delete all tasks for the current user."""
    user = current_user()
    user["tasks"] = []
    return jsonify({"ok": True, "message": "All tasks cleared"})


@app.route("/api/focus/record", methods=["DELETE"])
@login_required
def api_focus_clear():
    user = current_user()
    user["focus_sessions"] = []
    return jsonify({"ok": True, "message": "Focus sessions cleared"})


@app.route("/api/quiz/history", methods=["DELETE"])
@login_required
def api_quiz_history_clear():
    user = current_user()
    user["quiz_history"] = []
    return jsonify({"ok": True, "message": "Quiz history cleared"})


@app.route("/api/syllabus/plans", methods=["DELETE"])
@login_required
def api_syllabus_plans_clear():
    user = current_user()
    user["syllabus_plans"] = []
    # Also delete associated tasks
    user["tasks"] = [t for t in user["tasks"] if not t.get("is_syllabus")]
    return jsonify({"ok": True, "message": "All plans cleared"})


@app.route("/api/streak", methods=["DELETE"])
@login_required
def api_streak_clear():
    user = current_user()
    user.pop("last_taunt", None)
    return jsonify({"ok": True, "message": "Streak data cleared"})


# ===== AI LEARNING ENDPOINTS =====

@app.route("/api/learning/energy-curve", methods=["GET"])
@login_required
def api_learning_energy_curve():
    """Get 24-hour personal energy curve"""
    user = current_user()
    curve = []
    for h in range(24):
        energy = get_personal_energy_curve(user, h)
        curve.append({"hour": h, "energy": round(energy, 2), "label": f"{h:02d}:00"})
    return jsonify({
        "curve": curve,
        "model_active": bool(user.get("learning_data", {}).get("energy_model")),
        "data_days": len(set(r["date"] for r in user.get("learning_data", {}).get("energy_reports", [])))
    })


@app.route("/api/learning/routine", methods=["GET"])
@login_required
def api_learning_routine():
    """Get personal routine (generates if enough data)"""
    user = current_user()
    routine = get_personal_routine(user)
    if not routine:
        return jsonify({
            "routine": None,
            "message": "Need 3+ days of data. Complete tasks and report energy to build your routine!",
            "days_collected": len(set(r["date"] for r in user.get("learning_data", {}).get("energy_reports", [])))
        })
    return jsonify({"routine": routine})


@app.route("/api/learning/report-energy", methods=["POST"])
@login_required
def api_learning_report_energy():
    """Manual energy report from user"""
    user = current_user()
    data = request.get_json(silent=True) or {}
    energy = data.get("energy")
    hour = data.get("hour", datetime.now().hour)
    if energy not in ("Low", "Med", "High"):
        return jsonify({"error": "Invalid energy level"}), 400
    record_energy_report(user, hour, energy, source="manual")
    return jsonify({"ok": True, "message": f"Recorded {energy} energy at {hour}:00"})


@app.route("/api/timetable/suggest-tomorrow", methods=["GET"])
@login_required
def api_suggest_tomorrow():
    """AI suggests tomorrow's timetable based on previous day (Gemini + fallback)."""
    user = current_user()
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    tomorrow_str = tomorrow.strftime("%Y-%m-%d")

    # Already has tasks for tomorrow? Tell frontend but still suggest
    existing_tomorrow = [t for t in user.get("tasks", []) if t.get("date") == tomorrow_str and not t.get("is_syllabus")]

    # Source day: most recent day with tasks (today else yesterday etc, last 7 days)
    source_tasks = []
    source_date = None
    for offset in range(0, 7):
        d = today - timedelta(days=offset)
        ds = d.strftime("%Y-%m-%d")
        cand = [t for t in user.get("tasks", []) if t.get("date") == ds and not t.get("is_syllabus")]
        if cand:
            source_tasks = cand
            source_date = ds
            break

    if not source_tasks:
        return jsonify({"suggestion": [], "source_date": None, "message": "No previous tasks found — create today's tasks first and AI will learn.", "has_existing": bool(existing_tomorrow)})

    # Try Gemini
    suggestion = []
    ai = False
    reasoning = ""
    if GEMINI_AVAILABLE and GEMINI_API_KEY and GEMINI_MODEL:
        try:
            lines = []
            for t in source_tasks:
                dur = int((_hm_to_dt(t.get("end_time","23:59"), datetime.now()) - _hm_to_dt(t.get("start_time","00:00"), datetime.now())).total_seconds()/60)
                lines.append(f'- "{t["name"]}" priority:{t.get("priority","P3")} energy:{t.get("energy","Med")} duration:{dur}m orig:{t.get("start_time")}–{t.get("end_time")} completed:{t.get("completed", False)}')
            prompt = f"""You are Synora. Based on the user's previous day ({source_date}), suggest a fresh but familiar timetable for tomorrow ({tomorrow_str}).

Previous day tasks:
{chr(10).join(lines)}

Goal: Keep productive habits (P1/P2 at peak hours ~10AM & 4PM), adapt slightly for freshness — vary order if useful, keep durations realistic.
If a task was completed quickly, keep it. If overrun, suggest realistic time.
Minimum 15 min per task.

Return ONLY JSON with no markdown:
{{"suggestion": [{{"name": "Task name", "start_time": "HH:MM", "end_time": "HH:MM", "priority": "P1|P2|P3|P4|P5", "energy": "High|Med|Low"}}], "reasoning": "short why"}}
Times must be within 06:30-22:00, no overlaps, 5-min granularity.
"""
            resp = GEMINI_MODEL.generate_content(prompt)
            raw = (getattr(resp, "text", "") or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lstrip().startswith("json"):
                    raw = raw.lstrip()[4:].strip()
            parsed = json.loads(raw)
            cand = parsed.get("suggestion", [])
            # Validate
            seen = set()
            intervals = []
            valid = []
            for it in cand:
                nm = (it.get("name") or "").strip()
                st = it.get("start_time"); en = it.get("end_time")
                if not nm or not st or not en: continue
                try:
                    s_dt = _hm_to_dt(st, datetime.now())
                    e_dt = _hm_to_dt(en, datetime.now())
                except Exception:
                    continue
                if not (s_dt < e_dt and (e_dt - s_dt).total_seconds() >= 15*60): continue
                # overlap
                overlap=False
                for (os, oe) in intervals:
                    if not (e_dt <= os or s_dt >= oe):
                        overlap=True; break
                if overlap: continue
                intervals.append((s_dt, e_dt))
                valid.append({"name": nm, "start_time": st, "end_time": en, "priority": it.get("priority","P2"), "energy": it.get("energy","Med")})
            if valid:
                suggestion = valid
                ai = True
                reasoning = parsed.get("reasoning","")
        except Exception as _e:
            print(f"[suggest] Gemini failed: {_e}")

    if not suggestion:
        # Fallback: copy previous day with same times but sort by priority
        suggestion = []
        for t in sorted(source_tasks, key=lambda x: ({"P1":0,"P2":1,"P3":2,"P4":3,"P5":4}.get(x.get("priority","P3"),2), x.get("start_time","00:00"))):
            suggestion.append({"name": t["name"], "start_time": t["start_time"], "end_time": t["end_time"], "priority": t.get("priority","P2"), "energy": t.get("energy","Med")})
        reasoning = "Copied previous day pattern (Gemini unavailable)."

    return jsonify({"suggestion": suggestion, "source_date": source_date, "tomorrow": tomorrow_str, "has_existing": bool(existing_tomorrow), "ai": ai, "reasoning": reasoning})


@app.route("/api/timetable/apply-tomorrow", methods=["POST"])
@login_required
def api_apply_tomorrow():
    """Apply the suggestion (or custom list) to tomorrow."""
    user = current_user()
    data = request.get_json(silent=True) or {}
    tomorrow = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    # Allow client to send custom edited list; else auto-generate via suggest logic
    to_create = data.get("suggestion")
    if to_create is None:
        # Call suggest logic inline (avoid HTTP round-trip)
        # Reuse same fallback: previous day copy
        today = datetime.now().date()
        source_tasks=[]
        for offset in range(0,7):
            ds = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
            cand=[t for t in user.get("tasks",[]) if t.get("date")==ds and not t.get("is_syllabus")]
            if cand:
                source_tasks=cand
                break
        to_create=[{"name":t["name"],"start_time":t["start_time"],"end_time":t["end_time"],"priority":t.get("priority","P2"),"energy":t.get("energy","Med")} for t in source_tasks]

    created=[]
    for it in (to_create or []):
        # skip if already exists tomorrow with same name+time to avoid dup on re-apply
        exists=False
        for t in user.get("tasks",[]):
            if t.get("date")==tomorrow and t.get("name")==it.get("name") and t.get("start_time")==it.get("start_time"):
                exists=True; break
        if exists: continue
        task={"id": str(uuid.uuid4())[:8], "name": it.get("name","Untitled"), "start_time": it.get("start_time","09:00"), "end_time": it.get("end_time","10:00"), "priority": it.get("priority","P2"), "energy": it.get("energy","Med"), "date": tomorrow, "completed": False, "healed": False}
        user.setdefault("tasks",[]).append(task)
        created.append(task)
    return jsonify({"ok": True, "created": len(created), "tomorrow": tomorrow, "tasks": created})


@app.route("/api/learning/status", methods=["GET"])
@login_required
def api_learning_status():
    """Get learning system status"""
    user = current_user()
    ld = user.get("learning_data", {})
    reports = ld.get("energy_reports", [])
    unique_dates = len(set(r["date"] for r in reports))
    return jsonify({
        "model_trained": bool(ld.get("energy_model")),
        "days_of_data": len(set(r["date"] for r in reports)),
        "total_reports": len(reports),
        "routine_version": ld.get("routine_version", 0),
        "last_analyzed": ld.get("last_analyzed"),
        "ready_for_routine": len(set(r["date"] for r in reports)) >= 3
    })


# ---------------------------------------------------------------------------
# Community API Routes
# ---------------------------------------------------------------------------
@app.route("/api/community/posts", methods=["GET"])
@login_required
def api_community_posts():
    """Get paginated feed of community posts"""
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(20, int(request.args.get("per_page", 10)))
    user = current_user()
    
    query = CommunityPost.query.filter_by(is_deleted=False).order_by(
        CommunityPost.is_pinned.desc(), CommunityPost.created_at.desc()
    )
    total = query.count()
    posts = query.offset((page - 1) * per_page).limit(per_page).all()
    
    return jsonify({
        "posts": [p.to_dict(current_user_id=user.id) for p in posts],
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": max(1, (total + per_page - 1) // per_page)
    })


@app.route("/api/community/posts", methods=["POST"])
@login_required
def api_community_create_post():
    """Create a new community post"""
    user = current_user()
    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    media_urls = data.get("media_urls") or []
    
    if not content and not media_urls:
        return jsonify({"error": "Post must have content or media"}), 400
    if len(content) > 5000:
        return jsonify({"error": "Content too long (max 5000 chars)"}), 400
    
    # Validate media URLs
    valid_media = []
    for m in media_urls:
        if isinstance(m, dict) and m.get("type") in ("image", "video") and m.get("url"):
            valid_media.append({"type": m["type"], "url": m["url"]})
    
    post = CommunityPost(
        user_id=user.id,
        content=content,
        media_urls=valid_media
    )
    db.session.add(post)
    db.session.commit()
    
    return jsonify({"ok": True, "post": post.to_dict(current_user_id=user.id)}), 201


@app.route("/api/community/posts/<post_id>", methods=["GET"])
@login_required
def api_community_get_post(post_id):
    """Get a single post with comments"""
    user = current_user()
    post = db.session.get(CommunityPost, post_id)
    if not post or post.is_deleted:
        return jsonify({"error": "Post not found"}), 404
    return jsonify({"post": post.to_dict(include_comments=True, current_user_id=user.id)})


@app.route("/api/community/posts/<post_id>", methods=["DELETE"])
@login_required
def api_community_delete_post(post_id):
    """Delete own post (or admin can delete any)"""
    user = current_user()
    post = db.session.get(CommunityPost, post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404
    # Allow if author or admin
    is_admin = user.email == os.environ.get("ADMIN_EMAIL", "admin@synora.app")
    if post.user_id != user.id and not is_admin:
        return jsonify({"error": "Not authorized"}), 403
    post.is_deleted = True
    db.session.commit()
    return jsonify({"ok": True})


@app.route("/api/community/posts/<post_id>/like", methods=["POST"])
@login_required
def api_community_toggle_like(post_id):
    """Like or unlike a post"""
    user = current_user()
    post = db.session.get(CommunityPost, post_id)
    if not post or post.is_deleted:
        return jsonify({"error": "Post not found"}), 404
    
    existing = CommunityLike.query.filter_by(post_id=post_id, user_id=user.id).first()
    if existing:
        db.session.delete(existing)
        db.session.commit()
        return jsonify({"ok": True, "liked": False, "like_count": len(post.likes)})
    else:
        like = CommunityLike(post_id=post_id, user_id=user.id)
        db.session.add(like)
        db.session.commit()
        return jsonify({"ok": True, "liked": True, "like_count": len(post.likes) + 1})


@app.route("/api/community/posts/<post_id>/comments", methods=["POST"])
@login_required
def api_community_add_comment(post_id):
    """Add a comment to a post"""
    user = current_user()
    post = db.session.get(CommunityPost, post_id)
    if not post or post.is_deleted:
        return jsonify({"error": "Post not found"}), 404
    
    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    media_urls = data.get("media_urls") or []
    
    if not content and not media_urls:
        return jsonify({"error": "Comment must have content or media"}), 400
    if len(content) > 2000:
        return jsonify({"error": "Comment too long (max 2000 chars)"}), 400
    
    valid_media = []
    for m in media_urls:
        if isinstance(m, dict) and m.get("type") in ("image", "video") and m.get("url"):
            valid_media.append({"type": m["type"], "url": m["url"]})
    
    comment = CommunityComment(
        post_id=post_id,
        user_id=user.id,
        content=content,
        media_urls=valid_media
    )
    db.session.add(comment)
    db.session.commit()
    
    return jsonify({"ok": True, "comment": comment.to_dict(current_user_id=user.id)}), 201


@app.route("/api/community/comments/<comment_id>", methods=["DELETE"])
@login_required
def api_community_delete_comment(comment_id):
    """Delete own comment (or admin/post author can delete)"""
    user = current_user()
    comment = db.session.get(CommunityComment, comment_id)
    if not comment:
        return jsonify({"error": "Comment not found"}), 404
    post = db.session.get(CommunityPost, comment.post_id)
    is_admin = user.email == os.environ.get("ADMIN_EMAIL", "admin@synora.app")
    if comment.user_id != user.id and (not post or post.user_id != user.id) and not is_admin:
        return jsonify({"error": "Not authorized"}), 403
    comment.is_deleted = True
    db.session.commit()
    return jsonify({"ok": True})


@app.route("/api/community/comments/<comment_id>/like", methods=["POST"])
@login_required
def api_community_toggle_comment_like(comment_id):
    """Like or unlike a comment"""
    user = current_user()
    comment = db.session.get(CommunityComment, comment_id)
    if not comment or comment.is_deleted:
        return jsonify({"error": "Comment not found"}), 404
    
    existing = CommunityCommentLike.query.filter_by(comment_id=comment_id, user_id=user.id).first()
    if existing:
        db.session.delete(existing)
        db.session.commit()
        return jsonify({"ok": True, "liked": False, "like_count": len(comment.likes)})
    else:
        like = CommunityCommentLike(comment_id=comment_id, user_id=user.id)
        db.session.add(like)
        db.session.commit()
        return jsonify({"ok": True, "liked": True, "like_count": len(comment.likes) + 1})


# ---------------------------------------------------------------------------
# Auth Routes — signup/login with Google, OTP, Skip
# ---------------------------------------------------------------------------
otp_store = {}  # {target: {otp, kind, expires}}

# ---------------------------------------------------------------------------
# Email OTP via Gmail SMTP (set env vars on Render / local shell):
#   MAIL_EMAIL   = your Gmail address
#   MAIL_PASSWORD= Gmail "App Password" (not your normal password)
#   MAIL_SMTP    = smtp.gmail.com (default)
#   MAIL_PORT    = 587 (default)
# If MAIL_EMAIL is not set, falls back to demo mode (OTP returned in response).
# ---------------------------------------------------------------------------
def send_otp_email(to_email, otp):
    sender = os.environ.get("MAIL_EMAIL", "").strip()
    password = os.environ.get("MAIL_PASSWORD", "").strip()
    smtp_host = os.environ.get("MAIL_SMTP", "smtp.gmail.com")
    smtp_port = int(os.environ.get("MAIL_PORT", "587"))
    if not sender or not password:
        return "not_configured"
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    body = f"""Hello from Synora AI!

Your one-time verification code is:

    {otp}

This code expires in 5 minutes. If you didn't request this, you can safely ignore this email.

— Synora AI, your adaptive study companion
"""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your Synora AI verification code"
    msg["From"] = f"Synora AI <{sender}>"
    msg["To"] = to_email
    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, password)
        server.sendmail(sender, [to_email], msg.as_string())
        server.quit()
        return "sent"
    except Exception as e:
        print(f"[OTP-EMAIL] failed: {e}")
        return "error"

@app.route("/signup", methods=["GET"])
def signup_page():
    return render_template("signup_new.html")

@app.route("/signup", methods=["POST"])
@rate_limit(AUTH_RATE["signup"], AUTH_WINDOW)
def signup_post():
    username = (request.form.get("username") or "").strip()
    email = _sanitize_email(request.form.get("email"))
    phone = (request.form.get("phone") or "").strip()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""
    exam_goal = request.form.get("exam_goal") or None
    study_level = request.form.get("study_level") or None
    daily_hours = request.form.get("daily_hours")
    school = request.form.get("school") or None
    age = request.form.get("age")
    # --- Validation ---
    if not _valid_username(username):
        return render_template("signup_new.html", error="Name should be 2–50 letters/numbers (no symbols)")
    if not _valid_email(email):
        return render_template("signup_new.html", error="Please enter a valid email address")
    if not _valid_phone(phone):
        return render_template("signup_new.html", error="Please enter a valid phone number")
    ok, reason = _password_ok(password)
    if not ok:
        return render_template("signup_new.html", error=reason)
    if password != confirm:
        return render_template("signup_new.html", error="Passwords do not match")
    if User.query.filter_by(email=email).first():
        return render_template("signup_new.html", error="Email already registered — try login")
    try:
        daily_hours = int(daily_hours) if daily_hours else None
    except:
        daily_hours = None
    try:
        age = int(age) if age else None
    except:
        age = None
    # Check verified flags from session (set via OTP verify)
    email_verified = session.get(f"otp_verified_email_{email}") is True
    phone_verified = session.get(f"otp_verified_phone_{phone}") is True if phone else False
    user = User(
        username=username, email=email, password=_hash_password(password), phone=phone,
        phone_verified=phone_verified, email_verified=email_verified,
        exam_goal=exam_goal, study_level=study_level, daily_hours=daily_hours,
        school=school, age=age, onboarded=bool(exam_goal or study_level),
        auth_provider="email"
    )
    db.session.add(user)
    db.session.commit()
    session["email"] = email
    session["username"] = username
    return redirect(url_for("dashboard"))

@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login_new.html")

@app.route("/login", methods=["POST"])
@rate_limit(AUTH_RATE["login"], AUTH_WINDOW)
def login_post():
    raw = (request.form.get("email") or "").strip()
    password = request.form.get("password") or ""
    if not raw or not password:
        return render_template("login_new.html", error="Please enter your email/phone and password")
    if password and not any(ord(c) >= 32 for c in password):
        return render_template("login_new.html", error="Invalid email/phone or password")
    email = raw.lower()
    # Allow phone login via email field
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User.query.filter_by(phone=raw).first()
    if not user or not _check_password(user.password, password):
        return render_template("login_new.html", error="Invalid email/phone or password")
    if user.is_active is False:
        return render_template("login_new.html", error="This account has been disabled by an administrator. Contact support@syntaxora.app.")
    session["email"] = user.email
    session["username"] = user.username
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))

@app.route("/auth/send-otp", methods=["POST"])
@rate_limit(AUTH_RATE["otp"], AUTH_WINDOW, key_parts=lambda: ["otp-send"])
def auth_send_otp():
    data = request.get_json(silent=True) or {}
    kind = data.get("kind")  # email/phone
    target = (data.get("target") or "").strip().lower()
    if kind == "email" and not _valid_email(target):
        return jsonify({"error": "Valid email required"}), 400
    if kind == "phone" and not _valid_phone(target):
        return jsonify({"error": "Valid phone required"}), 400
    if kind not in ("email", "phone") or not target:
        return jsonify({"error": "Invalid request"}), 400
    otp = f"{random.randint(100000, 999999)}"
    otp_store[target] = {"otp": otp, "kind": kind, "expires": datetime.now() + timedelta(minutes=5)}
    # Try to send a real email OTP
    delivery = "demo"
    if kind == "email":
        delivery = send_otp_email(target, otp)
    else:
        # Phone: real SMS needs a provider (Fast2SMS/Twilio). Demo for now.
        delivery = "demo"
    print(f"[OTP] {kind} {target} -> {otp} (delivery={delivery})")
    resp = {"ok": True, "message": f"OTP sent to {target}", "delivery": delivery}
    # If email wasn't actually sent (not configured or failed), include the OTP
    # so the user can still proceed in demo mode.
    if delivery in ("demo", "not_configured", "error"):
        resp["otp"] = otp
        resp["delivery"] = "demo"
    return jsonify(resp)

@app.route("/auth/verify-otp", methods=["POST"])
@rate_limit(AUTH_RATE["otp"], AUTH_WINDOW, key_parts=lambda: ["otp-verify"])
def auth_verify_otp():
    data = request.get_json(silent=True) or {}
    kind = data.get("kind")
    target = (data.get("target") or "").strip().lower()
    otp = (data.get("otp") or "").strip()
    login = data.get("login") is True
    if kind not in ("email", "phone") or not target:
        return jsonify({"error": "Invalid request"}), 400
    if not otp or not otp.isdigit() or len(otp) != 6:
        return jsonify({"error": "Invalid OTP format"}), 400
    rec = otp_store.get(target)
    if not rec or rec["otp"] != otp:
        return jsonify({"error": "Invalid OTP"}), 400
    if datetime.now() > rec["expires"]:
        return jsonify({"error": "OTP expired"}), 400
    # Mark verified in session
    session[f"otp_verified_{kind}_{target}"] = True
    # If login flow, actually log the user in
    if login:
        user = User.query.filter_by(email=target).first() if kind=="email" else User.query.filter_by(phone=target).first()
        if not user:
            # Auto-create user on OTP login (passwordless)
            username = target.split("@")[0] if "@" in target else target
            user = User(username=username, email=target if kind=="email" else f"{target}@phone.local", phone=target if kind=="phone" else None,
                        password="", phone_verified=(kind=="phone"), email_verified=(kind=="email"), auth_provider=kind)
            db.session.add(user)
            db.session.commit()
        else:
            if user.is_active is False:
                return jsonify({"error": "This account has been disabled by an administrator"}), 403
            if kind=="email":
                user.email_verified = True
            else:
                user.phone_verified = True
            db.session.commit()
        session["email"] = user.email
        session["username"] = user.username
        return jsonify({"ok": True, "message": "Logged in"})
    return jsonify({"ok": True, "message": "Verified"})

@app.route("/auth/google", methods=["POST"])
@rate_limit(AUTH_RATE["login"], AUTH_WINDOW, key_parts=lambda: ["google"])
def auth_google():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    name = (data.get("name") or "").strip()
    picture = (data.get("picture") or "").strip()
    if not _valid_email(email):
        return jsonify({"error": "Valid email required"}), 400
    if name and (len(name) > 50 or not _valid_username(name)):
        name = ""  # sanitize: drop invalid names rather than reject
    # Demo mode: allow a demo flag to skip strict verification.
    # In production, verify the Google ID token server-side.
    if picture and not picture.startswith("https://"):
        picture = ""  # avoid exfil/injection via arbitrary scheme
    user = User.query.filter_by(email=email).first()
    if user and user.is_active is False:
        return jsonify({"error": "This account has been disabled by an administrator"}), 403
    if not user:
        username = name or email.split("@")[0]
        if len(username) < 2:
            username = email.split("@")[0]
        user = User(username=username, email=email, password="",
                    email_verified=True, auth_provider="google")
        db.session.add(user)
        db.session.commit()
    # Store picture if provided
    if picture:
        try:
            user.photo_url = picture
            db.session.commit()
        except Exception:
            pass
    session["email"] = user.email
    session["username"] = user.username
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Error handlers — clean branded pages instead of ugly stack traces
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return render_template("error.html", code=404, title="Page Not Found",
                           message="The page you're looking for doesn't exist or was moved."), 404

@app.errorhandler(405)
def method_not_allowed(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Method not allowed"}), 405
    return render_template("error.html", code=405, title="Method Not Allowed",
                           message="This action isn't allowed here."), 405

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large (max 16MB)"}), 413

@app.errorhandler(500)
def server_error(e):
    # Log the real error for debugging, but never show details to the user
    import traceback
    traceback.print_exc()
    db.session.rollback()
    if request.path.startswith("/api/"):
        return jsonify({"error": "Something went wrong on our side. Please try again."}), 500
    return render_template("error.html", code=500, title="Something Went Wrong",
                           message="An unexpected error occurred. Please try again in a moment."), 500


# ---------------------------------------------------------------------------
# Legal pages — Terms of Service & Privacy Policy
# ---------------------------------------------------------------------------
@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


# ---------------------------------------------------------------------------
# Profile page — user can view/update their own data
# ---------------------------------------------------------------------------
@app.route("/profile")
@login_required
def profile_page():
    user = current_user()
    return render_template("profile.html", user=user)

@app.route("/profile", methods=["POST"])
@login_required
def profile_update():
    user = current_user()
    username = (request.form.get("username") or "").strip()
    school = (request.form.get("school") or "").strip() or None
    exam_goal = (request.form.get("exam_goal") or "").strip() or None
    study_level = (request.form.get("study_level") or "").strip() or None
    daily_hours = request.form.get("daily_hours")
    age = request.form.get("age")
    try:
        daily_hours = int(daily_hours) if daily_hours else None
    except Exception:
        daily_hours = None
    try:
        age = int(age) if age else None
    except Exception:
        age = None
    if username:
        user.username = username
    if school is not None:
        user.school = school
    if exam_goal is not None:
        user.exam_goal = exam_goal
    if study_level is not None:
        user.study_level = study_level
    if daily_hours is not None:
        user.daily_hours = daily_hours
    user.age = age
    user.onboarded = bool(user.exam_goal or user.study_level)
    db.session.commit()
    return redirect(url_for("profile_page"))


# ---------------------------------------------------------------------------
# Forgot / change password flow
# ---------------------------------------------------------------------------
@app.route("/forgot-password", methods=["GET"])
def forgot_password():
    return render_template("forgot_password.html")


@app.route("/forgot-password", methods=["POST"])
@rate_limit(AUTH_RATE["reset"], AUTH_WINDOW, key_parts=lambda: ["forgot"])
def forgot_password_post():
    email = _sanitize_email(request.form.get("email"))
    if not _valid_email(email):
        return render_template("forgot_password.html", error="Please enter a valid email")
    user = User.query.filter_by(email=email).first()
    # Always show a generic message to avoid revealing whether an email exists
    # (prevents account enumeration).
    if not user:
        return render_template("forgot_password_done.html", email=email, sent=False)
    # Generate a reset token tied to this user
    token = uuid.uuid4().hex
    user.password_reset_token = token
    user.password_reset_expires = datetime.utcnow() + timedelta(minutes=30)
    db.session.commit()
    # Build a reset link from the current host
    reset_link = url_for("reset_password", token=token, _external=True)
    delivery = "demo"
    try:
        # Send reset email (works with Gmail SMTP if configured, else demo)
        if os.environ.get("MAIL_EMAIL"):
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            sender = os.environ.get("MAIL_EMAIL")
            body = f"Hello {user.username},\n\nReset your Synora password using this link (valid 30 min):\n{reset_link}\n\nIf you didn't request this, ignore this email."
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "Reset your Synora password"
            msg["From"] = f"Synora AI <{sender}>"
            msg["To"] = email
            msg.attach(MIMEText(body, "plain"))
            server = smtplib.SMTP(os.environ.get("MAIL_SMTP", "smtp.gmail.com"), int(os.environ.get("MAIL_PORT", "587")))
            server.ehlo(); server.starttls(); server.ehlo()
            server.login(sender, os.environ.get("MAIL_PASSWORD", ""))
            server.sendmail(sender, [email], msg.as_string())
            server.quit()
            delivery = "sent"
        else:
            delivery = "demo"
    except Exception as e:
        print(f"[RESET-EMAIL] failed: {e}")
        delivery = "demo"
    print(f"[RESET] {email} token={token} delivery={delivery}")
    return render_template("forgot_password_done.html", email=email, sent=(delivery == "sent"),
                           demo_link=reset_link if delivery == "demo" else None)


# ---------------------------------------------------------------------------
# Change password (for logged-in users) — from /profile
# ---------------------------------------------------------------------------
@app.route("/password", methods=["GET"])
@login_required
def change_password_page():
    return render_template("change_password.html")

@app.route("/password", methods=["POST"])
@login_required
@rate_limit(AUTH_RATE["login"], AUTH_WINDOW, key_parts=lambda: ["chpwd"])
def change_password_post():
    user = current_user()
    current_pwd = request.form.get("current_password") or ""
    new_pwd = request.form.get("new_password") or ""
    confirm = request.form.get("confirm") or ""
    # Only enforce current password if the user actually has one set
    if user.password and not _check_password(user.password, current_pwd):
        return render_template("change_password.html", error="Current password is incorrect")
    ok, reason = _password_ok(new_pwd)
    if not ok:
        return render_template("change_password.html", error=reason)
    if new_pwd != confirm:
        return render_template("change_password.html", error="New passwords do not match")
    user.password = _hash_password(new_pwd)
    db.session.commit()
    return render_template("change_password.html", success=True)


@app.route("/reset-password/<token>", methods=["GET"])
def reset_password(token):
    user = User.query.filter_by(password_reset_token=token).first()
    if not user:
        return render_template("error.html", code=400, title="Invalid or Expired Link",
                               message="This password reset link is invalid or has expired. Please request a new one."), 400
    if not user.password_reset_expires or datetime.utcnow() > user.password_reset_expires:
        return render_template("error.html", code=400, title="Link Expired",
                               message="This password reset link has expired. Please request a new one."), 400
    return render_template("reset_password.html", token=token)

@app.route("/reset-password/<token>", methods=["POST"])
@rate_limit(AUTH_RATE["login"], AUTH_WINDOW, key_parts=lambda: ["reset"])
def reset_password_post(token):
    user = User.query.filter_by(password_reset_token=token).first()
    if not user:
        return render_template("error.html", code=400, title="Invalid or Expired Link",
                               message="This password reset link is invalid or has expired."), 400
    if not user.password_reset_expires or datetime.utcnow() > user.password_reset_expires:
        return render_template("error.html", code=400, title="Link Expired",
                               message="This password reset link has expired."), 400
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm") or ""
    ok, reason = _password_ok(password)
    if not ok:
        return render_template("reset_password.html", token=token, error=reason)
    if password != confirm:
        return render_template("reset_password.html", token=token, error="Passwords do not match")
    user.password = _hash_password(password)
    user.password_reset_token = None
    user.password_reset_expires = None
    db.session.commit()
    return render_template("reset_password_done.html")


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", os.environ.get("SYNORA_PORT", 5000)))
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)