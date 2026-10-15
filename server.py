"""Local mock-exam server. Run: python server.py, then open http://127.0.0.1:8000"""
import json
import os
import re
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
EXAMS_DIR = os.path.join(ROOT, "papers")
RESULTS_DIR = os.path.join(ROOT, "results")
STATIC_DIR = os.path.join(ROOT, "static")
BOOKMARKS = os.path.join(ROOT, "bookmarks.json")
PORT = 8000

Q_START = re.compile(r"^\*\*Q(\d+)\.\*\*\s*(.*)$")
OPTION = re.compile(r"^([A-D])\.(?:\s+(.*))?$")
EXPL_START = re.compile(r"^\*\*Q(\d+)\s*[—–-]\s*([A-D])\.\*\*\s*(.*)$")
ERRATUM = re.compile(r"\*\*Q(\d+)\s*(?:→|->)\s*([A-D])\b")


def find_heading(lines, name):
    for i, line in enumerate(lines):
        if re.match(r"^#\s+" + name + r"\s*$", line.strip(), re.I):
            return i
    return None


def section(lines, start):
    """Lines after a '# ' heading up to the next '# ' heading."""
    if start is None:
        return []
    out = []
    for line in lines[start + 1:]:
        if line.startswith("# "):
            break
        out.append(line)
    return out


def strip_block(block):
    while block and block[-1].strip() in ("", "---"):
        block.pop()
    while block and not block[0].strip():
        block.pop(0)
    return block


def parse_exam(text):
    lines = text.splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), "Exam")
    m = re.search(r"(\d+)\s*min", text)
    minutes = int(m.group(1)) if m else 150

    key_i = find_heading(lines, "ANSWER KEY")
    q_region = lines[:key_i] if key_i is not None else lines

    # Questions: split on **Qn.**; options are A.–D. lines outside code fences.
    starts = [i for i, l in enumerate(q_region) if Q_START.match(l)]
    questions = []
    for n, s in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(q_region)
        block = q_region[s:end]
        num = int(Q_START.match(block[0]).group(1))
        block[0] = Q_START.match(block[0]).group(2)
        body, options, cur, in_fence = [], {}, None, False
        for line in block:
            if line.strip().startswith("```"):
                in_fence = not in_fence
            om = None if in_fence else OPTION.match(line)
            if om:
                cur = om.group(1)
                options[cur] = [om.group(2) or ""]
            elif cur:
                options[cur].append(line)
            else:
                body.append(line)
        questions.append({
            "num": num,
            "body": "\n".join(strip_block(body)),
            "options": {k: "\n".join(strip_block(v)) for k, v in options.items()},
            "answer": None, "domain": None, "explanation": "", "erratum": None,
        })
    by_num = {q["num"]: q for q in questions}

    # Answer key table: repeating groups of Q | Ans | [Domain].
    for line in section(lines, key_i):
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if "Q" in cells:
            qs = [i for i, c in enumerate(cells) if c == "Q"]
            group = qs[1] - qs[0] if len(qs) > 1 else len(cells)
            continue
        for g in range(0, len(cells), group):
            chunk = cells[g:g + group]
            if len(chunk) >= 2 and chunk[0].isdigit() and int(chunk[0]) in by_num:
                q = by_num[int(chunk[0])]
                if chunk[1] in "ABCD" and chunk[1]:
                    q["answer"] = chunk[1]
                if len(chunk) >= 3 and chunk[2]:
                    q["domain"] = chunk[2]

    # Explanations: **Qn — X.** text ...
    cur = None
    for line in section(lines, find_heading(lines, "EXPLANATIONS")):
        em = EXPL_START.match(line)
        if em:
            cur = by_num.get(int(em.group(1)))
            if cur:
                cur["explanation"] = em.group(3)
                if not cur["answer"]:
                    cur["answer"] = em.group(2)
        elif cur and line.strip() == "---":
            cur = None
        elif cur:
            cur["explanation"] += "\n" + line
    for q in questions:
        q["explanation"] = q["explanation"].strip()

    # Errata override the answer key.
    for line in section(lines, find_heading(lines, "ERRATA")):
        for em in ERRATUM.finditer(line):
            q = by_num.get(int(em.group(1)))
            if q and q["answer"] != em.group(2):
                q["erratum"] = q["answer"]
                q["answer"] = em.group(2)

    return {"title": title, "minutes": minutes, "questions": questions}


def exam_files():
    return sorted(f for f in os.listdir(EXAMS_DIR) if f.lower().endswith(".md"))


def load_exam(name):
    with open(os.path.join(EXAMS_DIR, name), encoding="utf-8") as f:
        exam = parse_exam(f.read())
    for q in exam["questions"]:
        q["file"] = name
    return exam


def load_results():
    out = []
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, f), encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out


def load_bookmarks():
    if not os.path.isfile(BOOKMARKS):
        return []
    with open(BOOKMARKS, encoding="utf-8") as f:
        return json.load(f)


def question_pool():
    """Every question's tag and statuses (no question text). Statuses come only
    from full mock attempts, and a status seen in ANY attempt sticks."""
    full = [r for r in load_results() if not r.get("custom")]
    pool = []
    for name in exam_files():
        exam = load_exam(name)
        taken = [r for r in full if r.get("examFile") == name]
        statuses = {}
        for r in taken:
            for q, it in zip(r["exam"]["questions"], r["items"]):
                st = ("correct" if it["correct"] else "wrong" if it["status"] == "answered"
                      else "skipped" if it["status"] == "skipped" else "notReached")
                statuses.setdefault(q["num"], set()).add(st)
        for q in exam["questions"]:
            st = statuses.get(q["num"], set()) if taken else {"neverAttempted"}
            pool.append({"file": name, "title": exam["title"], "num": q["num"],
                         "domain": q["domain"], "statuses": sorted(st)})
    return pool


class Handler(SimpleHTTPRequestHandler):
    def send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path == "/api/exams":
            return self.send_json(exam_files())
        if url.path == "/api/exam":
            name = os.path.basename(qs.get("file", [""])[0])
            if not os.path.isfile(os.path.join(EXAMS_DIR, name)):
                return self.send_json({"error": "not found"}, 404)
            return self.send_json(load_exam(name))
        if url.path == "/api/results":
            out = [{k: r.get(k) for k in (
                "id", "examFile", "examTitle", "finishedAt", "endReason",
                "score", "total", "elapsedMs", "custom")} for r in load_results()]
            out.sort(key=lambda r: r["finishedAt"] or "", reverse=True)
            return self.send_json(out)
        if url.path == "/api/result":
            name = os.path.basename(qs.get("id", [""])[0]) + ".json"
            path = os.path.join(RESULTS_DIR, name)
            if not os.path.isfile(path):
                return self.send_json({"error": "not found"}, 404)
            with open(path, encoding="utf-8") as f:
                return self.send_json(json.load(f))
        if url.path == "/api/pool":
            return self.send_json(question_pool())
        if url.path == "/api/bookmarks":
            return self.send_json(load_bookmarks())
        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        if url.path == "/api/upload":
            body = self.read_json()
            name = os.path.basename(body.get("name", ""))
            if not name.lower().endswith(".md"):
                return self.send_json({"error": "File must be .md"}, 400)
            path = os.path.join(EXAMS_DIR, name)
            if os.path.exists(path):
                return self.send_json({"error": f"{name} already exists"}, 409)
            if not parse_exam(body.get("content", ""))["questions"]:
                return self.send_json({"error": "No questions found (expected **Q1.** format)"}, 400)
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(body["content"])
            return self.send_json({"file": name})
        if url.path == "/api/results":
            result = self.read_json()
            stem = os.path.splitext(os.path.basename(result.get("examFile", "exam")))[0]
            result["id"] = f"{stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            with open(os.path.join(RESULTS_DIR, result["id"] + ".json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=1)
            return self.send_json({"id": result["id"]})
        if url.path == "/api/delete-result":
            path = os.path.join(RESULTS_DIR, os.path.basename(self.read_json().get("id", "")) + ".json")
            if not os.path.isfile(path):
                return self.send_json({"error": "not found"}, 404)
            os.remove(path)
            return self.send_json({"ok": True})
        if url.path == "/api/bookmark":
            body = self.read_json()
            key = f"{os.path.basename(body['file'])}#{int(body['num'])}"
            marks = [k for k in load_bookmarks() if k != key] + ([key] if body.get("on") else [])
            with open(BOOKMARKS, "w", encoding="utf-8") as f:
                json.dump(marks, f, indent=1)
            return self.send_json(marks)
        if url.path == "/api/custom":
            body = self.read_json()
            exams, questions = {}, []
            for p in body["picks"]:
                name = os.path.basename(p["file"])
                if name not in exams:
                    exams[name] = load_exam(name)
                questions += [q for q in exams[name]["questions"] if q["num"] == int(p["num"])]
            return self.send_json({"title": f"Custom practice — {len(questions)} questions",
                                   "minutes": float(body["minutes"]), "custom": True,
                                   "questions": questions})
        self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Open http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=STATIC_DIR)).serve_forever()
