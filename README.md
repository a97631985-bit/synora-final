# Synora AI — Adaptive Calm Productivity

An AI-powered study planner built with Flask, Tailwind CSS and Google Gemini.

## Features
- **Dashboard** — Today's Flow, Quick Add (time range + task type: Quick Win / Focus Task / Power Task), Predicted Energy (dynamic), Streak
- **Study Planner** — Upload syllabus PDF (AI extracts topics intelligently), select exam + subject, pick exam date, AI generates day-wise schedule; missed tasks auto-reschedule
- **AI Quiz** — 3-step wizard (exam → topic/q-count/difficulty → Gemini generates quiz), per-question timer, exam-specific marking (JEE +4/-1, UPSC +2/-0.66 etc), detailed solutions + Gemini-powered analysis
- **Focus Mode** — Timer / Stopwatch, manual minutes input (shows HH:MM:SS), daily/weekly/monthly/yearly stats
- **Tasks & Calendar, Fix My Day** — AI heals schedule based on wasted time + energy level with taunt
- **Auth** — Login / Signup (in-memory mock DB for hackathon demo)

## Quick Start

```bash
git clone <your-repo>
cd Synora
pip install -r requirements.txt

# Set Gemini key (get free at https://aistudio.google.com/apikey)
set GEMINI_API_KEY=your_key   # Windows
export GEMINI_API_KEY=your_key # Mac/Linux

python app.py
# open http://localhost:5001
```

Optional: `SYNORA_PORT=5001` to change port.

## Project Structure
```
Synora/
  app.py              # Flask backend, APIs, Gemini integration
  templates/          # Jinja2 templates (dashboard, planner, quiz, focus, etc.)
  static/logo.png     # Brand logo
  requirements.txt
```

## Notes
- Mock DB is in-memory (`users_db`); data resets on restart. Replace with a real DB for production.
- Quick Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:5001`) can be used for a temporary shareable HTTPS link.
