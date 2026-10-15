#!/usr/bin/env python3
"""Rebalance correct-answer letters across a paper.

    python scripts/rebalance.py <file.md> [--seed N] [--dry-run]

Rotates each question's options so the correct answer lands on a target
letter drawn from a seeded shuffle of a balanced multiset. Rejects any draw
with four identical letters in a row. Rewrites the file in place (questions,
answer key and explanation headers) unless --dry-run.

Never assign letters by a formula: a formulaic assignment once produced a
repeating A-B-C cycle with no D at all.
"""
import random
import re
import sys
import itertools
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from examparse import read, split_sections, parse_questions, parse_key  # noqa: E402

L = "ABCD"


def targets(n, seed):
    rnd = random.Random(seed)
    base = [L[i % 4] for i in range(n)]
    for _ in range(10000):
        rnd.shuffle(base)
        if not any(len(set(base[i:i + 4])) == 1 for i in range(n - 3)):
            return list(base)
    raise SystemExit("FAIL: could not find a balanced draw without a run of 4")


def main(path, seed=0, dry=False):
    s = read(path)
    qbody, key_text, expl_text = split_sections(s)
    qs, problems = parse_questions(qbody)
    if problems:
        raise SystemExit(f"FAIL: malformed questions {[p[0] for p in problems]}")
    key = parse_key(key_text)
    nums = [q for q, _, _ in qs]
    missing = [q for q in nums if q not in key]
    if missing:
        raise SystemExit(f"FAIL: no key entry for {missing}")

    tgt = dict(zip(nums, targets(len(nums), seed)))
    head = s[:s.index("**Q1.**")]
    out, newkey = [], {}

    for blk in re.split(r"(?=^\*\*Q\d+\.\*\*)", s[len(head):], flags=re.M):
        m = re.match(r"^\*\*Q(\d+)\.\*\*", blk)
        if not m:
            out.append(blk)
            continue
        q = int(m.group(1))
        if q not in tgt:
            out.append(blk)
            continue
        sp = [(o.group(1), o.start()) for o in
              re.finditer(r"^([A-D])\.(?: |$)", blk, flags=re.M)]
        ends = [sp[i + 1][1] for i in range(3)]
        dm = blk.find("\n---", sp[3][1])
        ends.append(dm if dm != -1 else len(blk))
        texts = [blk[p + 2:ends[i]].lstrip(" ").rstrip("\n").rstrip()
                 for i, (l, p) in enumerate(sp)]
        rot = (ord(key[q]) - 65 - (ord(tgt[q]) - 65)) % 4
        order = [texts[(i + rot) % 4] for i in range(4)]
        parts = [f"{L[i]}." + ("\n" + t if t.startswith("```") else " " + t)
                 for i, t in enumerate(order)]
        out.append(blk[:sp[0][1]] + "\n\n".join(parts) + "\n" + blk[ends[3]:])
        newkey[q] = tgt[q]

    s2 = head + "".join(out)
    s2 = re.sub(r"^([A-D])\. *\n```", r"\1.\n```", s2, flags=re.M)

    # rewrite key table cells and explanation headers
    def fix_row(mo):
        row = mo.group(0)
        cells = row.split("|")
        for i in range(len(cells) - 1):
            a, b = cells[i].strip(), cells[i + 1].strip()
            if a.isdigit() and re.fullmatch(r"[A-D]", b) and int(a) in newkey:
                cells[i + 1] = cells[i + 1].replace(b, newkey[int(a)])
        return "|".join(cells)

    kstart = s2.index("# ANSWER KEY") if "# ANSWER KEY" in s2 else None
    if kstart is not None:
        kend = s2.index("# EXPLANATIONS") if "# EXPLANATIONS" in s2 else len(s2)
        s2 = (s2[:kstart]
              + re.sub(r"^\|.+\|\s*$", fix_row, s2[kstart:kend], flags=re.M)
              + s2[kend:])
    s2 = re.sub(r"^\*\*Q(\d+)\s*—\s*[A-D]\.\*\*",
                lambda mo: f"**Q{mo.group(1)} — {newkey.get(int(mo.group(1)), '?')}.**",
                s2, flags=re.M)

    letters = [newkey[q] for q in sorted(newkey)]
    runs = max((len(list(g)) for g in itertools.groupby(letters)), default=0)
    print("=" * 62)
    print(f"REBALANCE  {path.rsplit('/', 1)[-1]}   seed={seed}"
          f"{'   DRY RUN' if dry else ''}")
    print("=" * 62)
    print(f"  questions rotated   {len(newkey)}")
    print(f"  distribution        {dict(sorted(Counter(letters).items()))}")
    print(f"  longest run         {runs}")
    if not dry:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s2)
        print(f"  written             {path}")
    print("\nRESULT: OK — re-run validate_set.py to confirm keys still agree")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Rebalance correct-answer letters")
    ap.add_argument("file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.exit(main(a.file, a.seed, a.dry_run))
