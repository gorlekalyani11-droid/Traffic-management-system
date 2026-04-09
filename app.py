"""
╔══════════════════════════════════════════════════════════════╗
║     TRAFFIC VIOLATION DETECTION SYSTEM — SINGLE FILE        ║
║     Run: pip install fastapi uvicorn httpx python-multipart  ║
║           twilio pydantic                                    ║
║     Start: uvicorn app:app --reload --port 8000              ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, sqlite3, base64, json, smtplib
from contextlib import asynccontextmanager
from email.mime.text import MIMEText
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx

# ══════════════════════════════════════════════════════════════
#  CONFIG  (set via environment or .env file)
# ══════════════════════════════════════════════════════════════
DB_PATH          = "traffic.db"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TWILIO_SID       = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN     = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM      = os.getenv("TWILIO_FROM_NUMBER", "+10000000000")
SMTP_HOST        = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT        = int(os.getenv("SMTP_PORT", 587))
SMTP_USER        = os.getenv("SMTP_USER", "")
SMTP_PASS        = os.getenv("SMTP_PASS", "")
POLICE_EMAIL     = os.getenv("POLICE_EMAIL", "police@traffic.gov.in")
POLICE_PHONE     = os.getenv("POLICE_PHONE", "+919876543210")
UPLOAD_DIR       = "uploads"

FINE_TABLE = {
    "red_light_jump": 2000, "no_helmet": 1000, "triple_riding": 1500,
    "wrong_way": 2500,      "speeding": 3000,  "no_seatbelt": 1000,
    "mobile_use": 1500,     "illegal_parking": 500, "unknown": 500,
}

# In-memory signal state
_signals: dict[str, str] = {"JN-01": "red", "JN-02": "green", "JN-03": "yellow"}

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, plate TEXT NOT NULL,
        vehicle_type TEXT DEFAULT 'car', violation_type TEXT NOT NULL,
        junction_id TEXT DEFAULT 'JN-01', signal_state TEXT DEFAULT 'red',
        fine_amount INTEGER DEFAULT 0, fine_paid INTEGER DEFAULT 0,
        image_path TEXT, confidence REAL DEFAULT 0.0,
        created_at TEXT DEFAULT (datetime('now','localtime')))""")
    c.execute("""CREATE TABLE IF NOT EXISTS ambulance_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, junction_id TEXT DEFAULT 'JN-01',
        direction TEXT DEFAULT 'north', signal_overridden INTEGER DEFAULT 0,
        police_notified INTEGER DEFAULT 0, resolved INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now','localtime')))""")
    c.execute("""CREATE TABLE IF NOT EXISTS signal_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, junction_id TEXT DEFAULT 'JN-01',
        phase TEXT NOT NULL, duration_sec INTEGER DEFAULT 30,
        changed_at TEXT DEFAULT (datetime('now','localtime')))""")
    c.execute("""CREATE TABLE IF NOT EXISTS sms_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, recipient TEXT NOT NULL,
        message TEXT NOT NULL, alert_type TEXT DEFAULT 'fine',
        status TEXT DEFAULT 'sent', sent_at TEXT DEFAULT (datetime('now','localtime')))""")
    c.execute("""CREATE TABLE IF NOT EXISTS police_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, alert_type TEXT NOT NULL,
        plate TEXT, junction_id TEXT DEFAULT 'JN-01', description TEXT,
        status TEXT DEFAULT 'open', created_at TEXT DEFAULT (datetime('now','localtime')))""")
    conn.commit()
    conn.close()
    print("✅ Database initialised — traffic.db")


# ══════════════════════════════════════════════════════════════
#  AI SERVICE
# ══════════════════════════════════════════════════════════════
async def analyse_image(image_bytes: bytes, mime: str = "image/jpeg") -> dict:
    if not ANTHROPIC_API_KEY:
        return {"plate":"TS09AB1234","vehicle_type":"car","violation_type":"red_light_jump",
                "signal_state":"red","confidence":0.91,
                "description":"Vehicle crossed red light at junction.","fine_amount":2000}
    b64 = base64.standard_b64encode(image_bytes).decode()
    prompt = """You are a traffic-violation detection AI analysing a CCTV frame.
Respond ONLY with valid JSON (no markdown) with keys:
{"plate":"<plate or UNKNOWN>","vehicle_type":"car|bike|truck|auto|bus|unknown",
 "violation_type":"red_light_jump|no_helmet|triple_riding|wrong_way|speeding|no_seatbelt|mobile_use|illegal_parking|none|unknown",
 "signal_state":"red|green|yellow|unknown","confidence":0.0-1.0,"description":"one sentence"}"""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model":"claude-opus-4-5","max_tokens":300,"messages":[{
                "role":"user","content":[
                    {"type":"image","source":{"type":"base64","media_type":mime,"data":b64}},
                    {"type":"text","text":prompt}]}]})
        resp.raise_for_status()
        data = json.loads(resp.json()["content"][0]["text"])
    data["fine_amount"] = FINE_TABLE.get(data.get("violation_type","unknown"), 500)
    return data


# ══════════════════════════════════════════════════════════════
#  ALERT SERVICE
# ══════════════════════════════════════════════════════════════
def _log_sms(recipient: str, message: str, alert_type: str, status: str):
    conn = get_conn()
    conn.execute("INSERT INTO sms_log (recipient,message,alert_type,status) VALUES (?,?,?,?)",
                 (recipient, message, alert_type, status))
    conn.commit(); conn.close()

def send_sms(to: str, message: str, alert_type: str = "fine") -> str:
    if TWILIO_SID and TWILIO_TOKEN:
        try:
            from twilio.rest import Client
            Client(TWILIO_SID, TWILIO_TOKEN).messages.create(body=message, from_=TWILIO_FROM, to=to)
            status = "sent"
        except Exception as e:
            print(f"[SMS ERROR] {e}"); status = "failed"
    else:
        print(f"[SIMULATED SMS → {to}] {message}"); status = "simulated"
    _log_sms(to, message, alert_type, status)
    return status

def send_email(to: str, subject: str, body: str, alert_type: str = "fine") -> str:
    if SMTP_USER and SMTP_PASS:
        try:
            msg = MIMEText(body); msg["Subject"] = subject
            msg["From"] = SMTP_USER; msg["To"] = to
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls(); s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
            status = "sent"
        except Exception as e:
            print(f"[EMAIL ERROR] {e}"); status = "failed"
    else:
        print(f"[SIMULATED EMAIL → {to}] {subject}"); status = "simulated"
    _log_sms(to, f"{subject}: {body[:80]}", alert_type, status)
    return status

def notify_violation(plate: str, violation: str, fine: int, phone: str | None = None):
    msg = (f"Traffic Violation Alert!\nVehicle: {plate}\nViolation: {violation}\n"
           f"Fine: ₹{fine}\nPay at: https://traffic.gov.in/pay")
    send_sms(phone or POLICE_PHONE, msg, alert_type="fine")

def notify_ambulance(junction_id: str, direction: str):
    msg = f"🚨 AMBULANCE DETECTED\nJunction: {junction_id}\nDirection: {direction}\nSignal overridden to GREEN."
    send_sms(POLICE_PHONE, msg, alert_type="ambulance")
    send_email(POLICE_EMAIL, "Ambulance Signal Override",
               f"Ambulance at {junction_id} from {direction}. Signal forced green.", alert_type="ambulance")

def notify_police_alert(alert_type: str, plate: str | None, junction_id: str, desc: str):
    msg = (f"🚔 POLICE ALERT [{alert_type.upper()}]\nJunction: {junction_id}\n"
           f"Plate: {plate or 'N/A'}\n{desc}")
    send_sms(POLICE_PHONE, msg, alert_type="police")
    send_email(POLICE_EMAIL, f"Police Alert: {alert_type}", msg, alert_type="police")


# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="Traffic Violation Detection System",
    description="AI-powered traffic monitoring — single file edition",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ──────────────────────────────────────────────────────────────
#  FRONTEND — serve dashboard.html at /
# ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    html_path = os.path.join(os.path.dirname(__file__),"templates", "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h2>Put dashboard.html in templates  folder </h2>")


# ══════════════════════════════════════════════════════════════
#  DETECTION ROUTES  /api/detection
# ══════════════════════════════════════════════════════════════
class ManualViolation(BaseModel):
    plate: str; vehicle_type: str = "car"; violation_type: str
    junction_id: str = "JN-01"; signal_state: str = "red"

class AmbulanceLog(BaseModel):
    junction_id: str = "JN-01"; direction: str = "north"

@app.post("/api/detection/upload", tags=["Detection"])
async def upload_image(file: UploadFile = File(...), junction_id: str = "JN-01"):
    data = await file.read()
    mime = file.content_type or "image/jpeg"
    result = await analyse_image(data, mime)
    img_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(img_path, "wb") as f:
        f.write(data)
    if result.get("violation_type", "none") == "none":
        return {"message": "No violation detected", "result": result}
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO violations (plate,vehicle_type,violation_type,junction_id,signal_state,fine_amount,image_path,confidence) VALUES (?,?,?,?,?,?,?,?)",
        (result["plate"], result["vehicle_type"], result["violation_type"], junction_id,
         result["signal_state"], result["fine_amount"], img_path, result["confidence"]))
    conn.commit(); vid = cur.lastrowid; conn.close()
    notify_violation(result["plate"], result["violation_type"], result["fine_amount"])
    return {"message": "Violation recorded", "violation_id": vid, "result": result}

@app.post("/api/detection/manual", tags=["Detection"])
def manual_log(v: ManualViolation):
    fine = FINE_TABLE.get(v.violation_type, 500)
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO violations (plate,vehicle_type,violation_type,junction_id,signal_state,fine_amount,confidence) VALUES (?,?,?,?,?,?,?)",
        (v.plate, v.vehicle_type, v.violation_type, v.junction_id, v.signal_state, fine, 1.0))
    conn.commit(); vid = cur.lastrowid; conn.close()
    notify_violation(v.plate, v.violation_type, fine)
    return {"message": "Violation logged", "violation_id": vid, "fine_amount": fine}

@app.post("/api/detection/ambulance", tags=["Detection"])
def log_ambulance(a: AmbulanceLog):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO ambulance_alerts (junction_id,direction,signal_overridden,police_notified) VALUES (?,?,1,1)",
        (a.junction_id, a.direction))
    conn.execute("INSERT INTO signal_log (junction_id,phase,duration_sec) VALUES (?,?,?)",
                 (a.junction_id, "green_override", 60))
    conn.commit(); aid = cur.lastrowid; conn.close()
    _signals[a.junction_id] = "green"
    notify_ambulance(a.junction_id, a.direction)
    return {"message": "Ambulance logged, signal overridden", "alert_id": aid}


# ══════════════════════════════════════════════════════════════
#  FINES ROUTES  /api/fines
# ══════════════════════════════════════════════════════════════
@app.get("/api/fines/", tags=["Fines"])
def list_fines(paid: bool | None = None):
    conn = get_conn()
    if paid is None:
        rows = conn.execute("SELECT * FROM violations ORDER BY created_at DESC").fetchall()
    else:
        rows = conn.execute("SELECT * FROM violations WHERE fine_paid=? ORDER BY created_at DESC",
                            (1 if paid else 0,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/fines/stats", tags=["Fines"])
def fine_stats():
    conn = get_conn()
    total   = conn.execute("SELECT COALESCE(SUM(fine_amount),0) FROM violations").fetchone()[0]
    paid    = conn.execute("SELECT COALESCE(SUM(fine_amount),0) FROM violations WHERE fine_paid=1").fetchone()[0]
    pending = conn.execute("SELECT COALESCE(SUM(fine_amount),0) FROM violations WHERE fine_paid=0").fetchone()[0]
    count   = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
    conn.close()
    return {"total": total, "paid": paid, "pending": pending, "count": count}

@app.patch("/api/fines/{vid}/pay", tags=["Fines"])
def mark_paid(vid: int):
    conn = get_conn()
    r = conn.execute("SELECT id,fine_paid FROM violations WHERE id=?", (vid,)).fetchone()
    if not r: raise HTTPException(404, "Violation not found")
    if r["fine_paid"]: raise HTTPException(400, "Already paid")
    conn.execute("UPDATE violations SET fine_paid=1 WHERE id=?", (vid,))
    conn.commit(); conn.close()
    return {"message": "Fine marked as paid", "violation_id": vid}

@app.get("/api/fines/{plate}/history", tags=["Fines"])
def plate_history(plate: str):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM violations WHERE plate=? ORDER BY created_at DESC", (plate,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════
#  SIGNAL ROUTES  /api/signal
# ══════════════════════════════════════════════════════════════
class SignalUpdate(BaseModel):
    junction_id: str = "JN-01"; phase: str; duration_sec: int = 30

class AmbulanceOverride(BaseModel):
    junction_id: str = "JN-01"; direction: str = "north"

@app.get("/api/signal/", tags=["Signal"])
def get_signals():
    return _signals

@app.post("/api/signal/update", tags=["Signal"])
def update_signal(u: SignalUpdate):
    if u.phase not in ("red","green","yellow"):
        raise HTTPException(400, "phase must be red|green|yellow")
    _signals[u.junction_id] = u.phase
    conn = get_conn()
    conn.execute("INSERT INTO signal_log (junction_id,phase,duration_sec) VALUES (?,?,?)",
                 (u.junction_id, u.phase, u.duration_sec))
    conn.commit(); conn.close()
    return {"message": "Signal updated", "junction_id": u.junction_id, "phase": u.phase}

@app.post("/api/signal/override-ambulance", tags=["Signal"])
def override_ambulance(a: AmbulanceOverride):
    _signals[a.junction_id] = "green"
    conn = get_conn()
    conn.execute("INSERT INTO signal_log (junction_id,phase,duration_sec) VALUES (?,?,?)",
                 (a.junction_id, "green_override", 60))
    conn.commit(); conn.close()
    return {"message": "Signal forced GREEN for ambulance", "junction_id": a.junction_id}

@app.get("/api/signal/log/{junction_id}", tags=["Signal"])
def signal_log(junction_id: str):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM signal_log WHERE junction_id=? ORDER BY changed_at DESC LIMIT 50",
        (junction_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════
#  POLICE ROUTES  /api/police
# ══════════════════════════════════════════════════════════════
class NewAlert(BaseModel):
    alert_type: str; plate: str | None = None
    junction_id: str = "JN-01"; description: str = ""

@app.get("/api/police/dashboard", tags=["Police"])
def dashboard():
    conn = get_conn()
    total_violations = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
    unpaid_fines     = conn.execute("SELECT COUNT(*) FROM violations WHERE fine_paid=0").fetchone()[0]
    pending_amount   = conn.execute("SELECT COALESCE(SUM(fine_amount),0) FROM violations WHERE fine_paid=0").fetchone()[0]
    open_alerts      = conn.execute("SELECT COUNT(*) FROM police_alerts WHERE status='open'").fetchone()[0]
    ambulances_today = conn.execute("SELECT COUNT(*) FROM ambulance_alerts WHERE date(created_at)=date('now')").fetchone()[0]
    recent_violations = conn.execute("SELECT * FROM violations ORDER BY created_at DESC LIMIT 10").fetchall()
    conn.close()
    return {"total_violations": total_violations, "unpaid_fines": unpaid_fines,
            "pending_amount": pending_amount, "open_alerts": open_alerts,
            "ambulances_today": ambulances_today,
            "recent_violations": [dict(r) for r in recent_violations]}

@app.get("/api/police/alerts", tags=["Police"])
def get_alerts(status: str | None = None):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM police_alerts WHERE status=? ORDER BY created_at DESC" if status
        else "SELECT * FROM police_alerts ORDER BY created_at DESC",
        (status,) if status else ()).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.post("/api/police/alerts", tags=["Police"])
def create_alert(a: NewAlert):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO police_alerts (alert_type,plate,junction_id,description) VALUES (?,?,?,?)",
        (a.alert_type, a.plate, a.junction_id, a.description))
    conn.commit(); aid = cur.lastrowid; conn.close()
    notify_police_alert(a.alert_type, a.plate, a.junction_id, a.description)
    return {"message": "Alert created", "alert_id": aid}

@app.patch("/api/police/alerts/{aid}/resolve", tags=["Police"])
def resolve_alert(aid: int):
    conn = get_conn()
    r = conn.execute("SELECT id FROM police_alerts WHERE id=?", (aid,)).fetchone()
    if not r: raise HTTPException(404, "Alert not found")
    conn.execute("UPDATE police_alerts SET status='resolved' WHERE id=?", (aid,))
    conn.commit(); conn.close()
    return {"message": "Alert resolved", "alert_id": aid}

@app.get("/api/police/ambulance-alerts", tags=["Police"])
def ambulance_alerts():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM ambulance_alerts ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════
#  ALERTS / SMS ROUTES  /api/alerts
# ══════════════════════════════════════════════════════════════
@app.get("/api/alerts/sms", tags=["Alerts"])
def sms_log(alert_type: str | None = None):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM sms_log WHERE alert_type=? ORDER BY sent_at DESC" if alert_type
        else "SELECT * FROM sms_log ORDER BY sent_at DESC",
        (alert_type,) if alert_type else ()).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/alerts/sms/stats", tags=["Alerts"])
def sms_stats():
    conn = get_conn()
    rows = conn.execute("SELECT alert_type, COUNT(*) as count FROM sms_log GROUP BY alert_type").fetchall()
    conn.close()
    return {r["alert_type"]: r["count"] for r in rows}