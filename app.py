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

app = Flask(__name__)
app.secret_key = "synora-hackathon-demo-secret-key-2026"

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
# Mock Database (in-memory, hackathon demo)
# ---------------------------------------------------------------------------
users_db = {}


def create_user(username, email, password):
    user = {
        "username": username,
        "email": email,
        "password": password,
        "created_at": datetime.now().isoformat(),
        "current_energy": "Active",
        "focus_sessions": [],
        "tasks": [],
        "syllabus_plans": [],
    }
    users_db[email] = user
    return user


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
    return users_db.get(email)


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not current_user():
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Auth Routes
# ---------------------------------------------------------------------------
@app.route("/")
def landing():
    if current_user():
        return redirect(url_for("dashboard"))
    return render_template("landing.html")


@app.route("/features")
def features():
    return render_template("features.html", active_tab="features")


@app.route("/methodology")
def methodology():
    return render_template("methodology.html", active_tab="methodology")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = users_db.get(email)
        if user and user["password"] == password:
            session["email"] = email
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user():
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if not username or not email or not password:
            return render_template("signup.html", error="All fields are required.")
        if email in users_db:
            return render_template("signup.html", error="An account with this email already exists.")
        if password != confirm:
            return render_template("signup.html", error="Passwords do not match.")
        user = create_user(username, email, password)
        session["email"] = email
        session["username"] = username
        return redirect(url_for("dashboard"))
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


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
    # Limit generation to 30 days for token safety; if exam is farther, generate first 30 days
    gen_days = min(days_left, 30)
    syllabus_snippet = syllabus[:2500]
    subject_line = f'Subject focus: "{subject}" — ONLY schedule topics belonging to this subject. Ignore all other subjects.' if subject else 'Schedule across all subjects in the syllabus.'
    exam_subjects = ", ".join(EXAM_CONFIGS[exam].get("subjects", []))
    prompt = f"""
You are an expert study planner for {exam} (exam date: {exam_date_str}, subjects: {exam_subjects}). Today is {today.isoformat()}.
{subject_line}
Syllabus to cover:
{syllabus_snippet}

Daily study capacity: {daily_hours} hours.
Generate a day-wise study plan for the next {gen_days} days (from {(today + timedelta(days=1)).isoformat()} to {(today + timedelta(days=gen_days)).isoformat()}).
Distribute syllabus topics evenly, harder topics get more time, include 1 revision day per week.

Return ONLY valid JSON array with this exact structure (no markdown, no fences, plain ASCII, no backslashes):
[
  {{
    "date": "YYYY-MM-DD",
    "tasks": [
      {{"name": "Topic: Subtopic", "duration": 60, "priority": "P1"}}
    ]
  }}
]
Rules:
- Each day should have 1-3 tasks, total duration per day approx {daily_hours*60} minutes (±30 min).
- Priority P1 = hardest/most important, P2 = medium, P3 = revision/easy.
- Use plain ASCII only, no LaTeX, no backslashes.
- No extra text.
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
            plan_days = json.loads(text)
        except json.JSONDecodeError:
            import re
            fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
            plan_days = json.loads(fixed)
        if not isinstance(plan_days, list):
            raise ValueError("Invalid plan format")
        # Create tasks
        user = current_user()
        plan_id = str(uuid.uuid4())
        created_tasks = []
        for day in plan_days:
            d = day.get("date")
            # Validate date
            try:
                datetime.strptime(d, "%Y-%m-%d")
            except:
                continue
            start_hour = 9
            for t in day.get("tasks", [])[:3]:
                name = (t.get("name") or "Study").strip()[:120]
                dur = max(30, min(180, int(t.get("duration", 60))))
                prio = t.get("priority") if t.get("priority") in ("P1","P2","P3") else "P2"
                end_hour = start_hour + dur // 60
                end_min = dur % 60
                task = {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "start_time": f"{start_hour:02d}:00",
                    "end_time": f"{end_hour:02d}:{end_min:02d}",
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
                # Next slot
                start_hour = end_hour + 1
                if start_hour >= 18:
                    start_hour = 9
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
    user_tasks().append(task)
    return jsonify({"task": task}), 201


@app.route("/api/tasks/<task_id>/complete", methods=["POST"])
@login_required
def api_toggle_task(task_id):
    for task in user_tasks():
        if task["id"] == task_id:
            task["completed"] = not task["completed"]
            if task["completed"]:
                task["completed_at"] = datetime.now().isoformat()
            else:
                task.pop("completed_at", None)
            return jsonify({"task": task})
    return jsonify({"error": "Task not found."}), 404


@app.route("/api/tasks/<task_id>", methods=["DELETE"])
@login_required
def api_delete_task(task_id):
    tasks = user_tasks()
    for i, task in enumerate(tasks):
        if task["id"] == task_id:
            removed = tasks.pop(i)
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
    """AI 'healing' endpoint: user reports wasted time, AI replans the rest of the day."""
    data = request.get_json(silent=True) or {}
    energy = data.get("current_energy")
    if energy in {"Low Energy", "Active", "Peak Focus"}:
        current_user()["current_energy"] = energy
    try:
        wasted = max(0, min(600, int(data.get("wasted_minutes", 0) or 0)))
    except (TypeError, ValueError):
        wasted = 0

    user = current_user()
    now = datetime.now().replace(second=0, microsecond=0)
    incomplete = [t for t in user["tasks"] if not t["completed"]]
    remaining = [
        t for t in incomplete if _hm_to_dt(t["end_time"], now) > now
    ]

    if not remaining:
        return jsonify(
            {
                "message": "Nothing left to heal — you're all done for today!",
                "healed": 0,
                "dropped": [],
                "taunt": pick_taunt(wasted),
                "wasted_minutes": wasted,
            }
        )

    # Working window: from now (plus wasted time) until 22:00
    window_end = now.replace(hour=22, minute=0, second=0, microsecond=0)
    start = now + timedelta(minutes=wasted)
    if start >= window_end:
        return jsonify(
            {
                "message": "The day is basically over — even AI can't create time from nothing.",
                "healed": 0,
                "dropped": [t["name"] for t in remaining],
                "taunt": pick_taunt(wasted),
                "wasted_minutes": wasted,
            }
        )
    available = (window_end - start).total_seconds() / 60.0

    # Priority order: P1 (Power Task) first, then P2, then P3
    rank = {"P1": 0, "P2": 1, "P3": 2}
    kept = sorted(remaining, key=lambda t: (rank.get(t["priority"], 1), t["start_time"]))
    demand = sum(
        (_hm_to_dt(t["end_time"], now) - _hm_to_dt(t["start_time"], now)).total_seconds() / 60.0
        for t in kept
    )

    dropped = []
    if demand > available:
        # Drop the least important tasks (lowest priority first, longest last)
        for t in sorted(kept, key=lambda t: (-rank.get(t["priority"], 1), t["start_time"])):
            d = (_hm_to_dt(t["end_time"], now) - _hm_to_dt(t["start_time"], now)).total_seconds() / 60.0
            if demand - d >= available:
                demand -= d
                dropped.append(t["name"])
                kept.remove(t)
                continue
        # If still over, compress everything proportionally
        if demand > available:
            ratio = available / demand
            for t in kept:
                d = (_hm_to_dt(t["end_time"], now) - _hm_to_dt(t["start_time"], now)).total_seconds() / 60.0
                t["end_time"] = _dt_to_hm(_hm_to_dt(t["start_time"], now) + timedelta(minutes=max(15, round(d * ratio))))

    # Re-slot tasks back-to-back into the remaining window
    cursor = start
    healed = 0
    for t in kept:
        dur = (_hm_to_dt(t["end_time"], now) - _hm_to_dt(t["start_time"], now)).total_seconds() / 60.0
        t["start_time"] = _dt_to_hm(cursor)
        t["end_time"] = _dt_to_hm(cursor + timedelta(minutes=max(15, round(dur))))
        cursor += timedelta(minutes=max(15, round(dur)))
        t["healed"] = True
        healed += 1

    parts = [f"Day healed! {healed} task(s) replanned into your remaining window."]
    if dropped:
        parts.append(f"{len(dropped)} light task(s) dropped to make room: {', '.join(dropped)}.")
    parts.append("Highest-priority work now sits exactly where you'll do it best.")

    taunt = pick_taunt(wasted)
    user["last_taunt"] = taunt

    return jsonify(
        {
            "message": " ".join(parts),
            "healed": healed,
            "dropped": dropped,
            "taunt": taunt,
            "wasted_minutes": wasted,
            "current_energy": user["current_energy"],
        }
    )


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
    base = [40, 90, 75, 30, 50, 60]
    hours = ["8AM", "10AM", "12PM", "2PM", "4PM", "6PM"]
    # Deterministic daily offset per user (0-10) so chart feels personal but stable for the day
    email = user.get("email", "")
    today_str = datetime.now().strftime("%Y-%m-%d")
    seed = sum(ord(c) for c in (email + today_str)) % 100
    # Current energy modifier
    ce = user.get("current_energy", "Active")
    mod = {"Low Energy": -12, "Active": 0, "Peak Focus": 10}.get(ce, 0)
    # Activity boost: +5 if streak >2, +3 if any focus today
    active = _get_active_dates(user)
    streak_boost = 5 if len([d for d in active if d >= (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")]) >= 3 else 0
    values = []
    for i, b in enumerate(base):
        # pseudo-random per bar
        jitter = ((seed + i * 17) % 11) - 5  # -5..+5
        v = max(12, min(98, b + mod + jitter + streak_boost))
        values.append({"hour": hours[i], "value": v, "label": hours[i].replace("M", " M")})
    return jsonify({"hours": values, "current_energy": ce})


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


if __name__ == "__main__":
    import os

    port = int(os.environ.get("SYNORA_PORT", 5000))
    app.run(debug=True, port=port)