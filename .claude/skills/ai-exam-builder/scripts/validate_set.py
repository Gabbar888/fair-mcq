#!/usr/bin/env python3
"""Validate a topic set or assembled paper.

Checks format, option parity, hedge and absolute distribution, and agreement
between the answer key and the explanations.

    python scripts/validate_set.py <file.md>

Exits non-zero if anything fails.
"""
import re
import sys
import statistics as st
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from examparse import load, has_code, words, HEDGE, ABSOL  # noqa: E402


def main(path):
    s, qbody, qs, key, expl, problems = load(path)
    fails, warns = [], []

    print("=" * 62)
    print(f"VALIDATE  {path.rsplit('/', 1)[-1]}")
    print("=" * 62)

    # ---- structure -------------------------------------------------
    nq = len(qs)
    print(f"\nSTRUCTURE")
    print(f"  questions parsed      {nq}")
    print(f"  answer key entries    {len(key)}")
    print(f"  explanation entries   {len(expl)}")

    for q, why in problems:
        fails.append(f"Q{q} malformed options: {why}")
    if not key:
        fails.append("no answer key found")
    missing_key = [q for q, _, _ in qs if q not in key]
    if missing_key:
        fails.append(f"questions with no key entry: {missing_key}")
    missing_expl = [q for q, _, _ in qs if q not in expl]
    if missing_expl:
        fails.append(f"questions with no explanation: {missing_expl}")

    mism = [(q, expl[q], key[q]) for q in sorted(expl)
            if q in key and expl[q] != key[q]]
    if mism:
        fails.append("explanation letter != key letter: " +
                     ", ".join(f"Q{q} expl {e} key {k}" for q, e, k in mism))
    else:
        print("  key vs explanations   agree")

    # stray option-style lines in stems
    for q, stem, _ in qs:
        instem = re.sub(r"```.*?```", "", stem, flags=re.S)
        bad = [ln for ln in instem.split("\n") if re.match(r"^[A-D]\.(\s|$)", ln)]
        if bad:
            fails.append(f"Q{q} stem contains a line starting with an option letter")

    # ---- option statistics ----------------------------------------
    txt = [(q, key[q], o) for q, _, o in qs if q in key and not has_code(o)]
    ncode = nq - len(txt)
    if txt:
        ul = us = 0
        cw, dw, spread, wide = [], [], [], []
        for q, ck, o in txt:
            w = {l: words(t) for l, t in o.items()}
            mx, mn = max(w.values()), min(w.values())
            if w[ck] == mx and list(w.values()).count(mx) == 1:
                ul += 1
            if w[ck] == mn and list(w.values()).count(mn) == 1:
                us += 1
            cw.append(w[ck])
            dw.append(st.mean(w[l] for l in o if l != ck))
            spread.append(mx - mn)
            if mx - mn > 5:
                wide.append(q)
        n = len(txt)
        gap = st.mean(cw) - st.mean(dw)
        print(f"\nOPTION STATISTICS   ({n} text questions, {ncode} with code options)")
        print(f"  uniquely longest      {ul:>3}/{n}  {ul/n*100:5.1f}%   target 20-30")
        print(f"  uniquely shortest     {us:>3}/{n}  {us/n*100:5.1f}%   target 20-30")
        print(f"  mean correct words    {st.mean(cw):5.1f}")
        print(f"  mean distractor words {st.mean(dw):5.1f}")
        print(f"  gap                   {gap:+5.2f}            target within +/-1.0")
        print(f"  mean spread           {st.mean(spread):5.1f}            target <= 3")
        print(f"  spread > 5 words      {len(wide):>3}  {wide if wide else ''}")

        if n >= 20:
            if not 20 <= ul / n * 100 <= 30:
                fails.append(f"uniquely-longest {ul/n*100:.0f}% outside 20-30%")
            if not 20 <= us / n * 100 <= 30:
                fails.append(f"uniquely-shortest {us/n*100:.0f}% outside 20-30%")
        else:
            warns.append(f"only {n} text questions — length percentages are noisy, "
                         "judge by the gap and spread instead")
        if abs(gap) > 1.0:
            fails.append(f"mean word gap {gap:+.2f} exceeds +/-1.0")
        if st.mean(spread) > 3:
            fails.append(f"mean spread {st.mean(spread):.1f} exceeds 3 words")

        # ---- tells -------------------------------------------------
        hed = [q for q, ck, o in txt
               if re.search(HEDGE, o[ck].lower())
               and not any(re.search(HEDGE, o[l].lower()) for l in o if l != ck)]
        iso = [q for q, ck, o in txt
               if [l for l, t in o.items() if not re.search(ABSOL, t.lower())] == [ck]]
        print(f"\nTELLS")
        print(f"  correct is only hedged option      {len(hed):>3}  {hed if hed else ''}")
        print(f"  absolutes isolate correct answer   {len(iso):>3}  {iso if iso else ''}")
        if hed:
            fails.append(f"correct is the only hedged option on {hed}")
        if iso:
            fails.append(f"eliminating absolutes isolates the answer on {iso}")

    # ---- letters ---------------------------------------------------
    letters = [key[q] for q, _, _ in qs if q in key]
    c = Counter(letters)
    runs = max((len(list(g)) for g in __import__("itertools").groupby(letters)),
               default=0)
    print(f"\nANSWER LETTERS")
    print(f"  distribution          {dict(sorted(c.items()))}")
    print(f"  longest identical run {runs}")
    if runs >= 4:
        fails.append(f"{runs} identical answer letters in a row")
    if len(letters) >= 40:
        lo, hi = min(c.values(), default=0), max(c.values(), default=0)
        if hi - lo > 4:
            fails.append(f"letter distribution uneven: {dict(sorted(c.items()))}")

    # ---- verdict ---------------------------------------------------
    print("\n" + "-" * 62)
    for w in warns:
        print(f"  WARN  {w}")
    if fails:
        for f in fails:
            print(f"  FAIL  {f}")
        print(f"\nRESULT: FAILED ({len(fails)} issue"
              f"{'s' if len(fails) != 1 else ''})")
        return 1
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_set.py <file.md>")
    sys.exit(main(sys.argv[1]))
