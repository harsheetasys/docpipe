"""Turn detector boxes + OCR strings into the four structured key fields.

Rules are deliberately simple and transparent: pick the highest-confidence detection
per field, then normalise. For `total` and `date` we additionally regex-validate so a
mis-read like "TOTAL 1O.5O" (letter O) becomes 10.50.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MONEY = re.compile(r"(\d{1,3}(?:[,\d]{0,3})*\.\d{2})")
DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b"),
    re.compile(r"\b(\d{1,2}\s*[A-Za-z]{3,9}\s*\d{2,4})\b"),
    re.compile(r"\b(\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b"),
]
_O_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple[int, int, int, int]
    text: str = ""
    ocr_conf: float = 0.0


@dataclass
class Extraction:
    company: str | None = None
    date: str | None = None
    address: str | None = None
    total: str | None = None
    confidences: dict = field(default_factory=dict)


def normalise_total(s: str) -> str | None:
    s2 = s.translate(_O_FIX)
    m = MONEY.findall(s2.replace(" ", ""))
    if not m:
        return None
    return m[-1].replace(",", "")  # the right-most amount on a TOTAL line is the figure


def _plausible_date(s: str) -> bool:
    nums = re.findall(r"\d+", s)
    if len(nums) == 3 and all(n.isdigit() for n in nums):
        a, b, c = (int(n) for n in nums)
        if len(nums[0]) == 4:                      # yyyy-mm-dd
            return 2000 <= a <= 2035 and 1 <= b <= 12 and 1 <= c <= 31
        year_ok = (len(nums[2]) == 2 and 0 <= c <= 35) or (len(nums[2]) == 4 and 2000 <= c <= 2035)
        return year_ok and 1 <= a <= 31 and 1 <= b <= 31 and (a <= 12 or b <= 12)
    return bool(re.search(r"[A-Za-z]{3}", s))     # "05 MAR 2018"


def normalise_date(s: str) -> str | None:
    for p in DATE_PATTERNS:
        m = p.search(s)
        if m and _plausible_date(m.group(1)):
            return m.group(1)
    return None


def extract(dets: list[Detection]) -> Extraction:
    out = Extraction()
    by_cls: dict[str, list[Detection]] = {}
    for d in dets:
        by_cls.setdefault(d.cls, []).append(d)

    if "company" in by_cls:
        best = max(by_cls["company"], key=lambda d: d.conf)
        out.company, out.confidences["company"] = best.text.strip() or None, best.conf
    if "date" in by_cls:
        for d in sorted(by_cls["date"], key=lambda d: -d.conf):
            v = normalise_date(d.text)
            if v:
                out.date, out.confidences["date"] = v, d.conf
                break
    if "address" in by_cls:
        # address spans several lines: concatenate top-to-bottom
        lines = sorted(by_cls["address"], key=lambda d: d.box[1])
        out.address = " ".join(ln.text.strip() for ln in lines if ln.text.strip()) or None
        out.confidences["address"] = min(ln.conf for ln in lines)
    if "total" in by_cls:
        # prefer lines that literally say TOTAL, then highest detector confidence
        def rank(d: Detection):
            u = d.text.upper()
            says_total = bool(re.search(r"(?<!SUB)(?<!SUB )TOTAL", u)) and "SUB" not in u
            return (not says_total, -d.conf)

        cands = sorted(by_cls["total"], key=rank)
        for d in cands:
            v = normalise_total(d.text)
            if v:
                out.total, out.confidences["total"] = v, d.conf
                break
    return out
