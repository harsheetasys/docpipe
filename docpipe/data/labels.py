"""Turn SROIE's weak supervision (document-level key fields) into line-level labels.

SROIE gives us, per receipt, every text line's box + transcription, and separately the
four key fields (company, date, address, total). It does NOT say which line holds which
field. We derive that with fuzzy matching, which is exactly the kind of label-engineering
step an ML engineer does when a delivery pod hands over labelled data in a format the
model can't consume directly.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from docpipe.config import FIELD_CLASSES

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def normalise(s: str) -> str:
    return _NON_ALNUM.sub("", s.upper())


def assign_field(line_text: str, key_fields: dict[str, str], threshold: int = 85) -> str:
    """Return the field class for one text line.

    Rules (ordered, first match wins):
      total   -> exact numeric match on the total value AND the line mentions a total-ish word
                 or is only the number (keeps us from labelling every sub-total as "total").
      date    -> the date string is contained in the line.
      company -> fuzzy partial-ratio against the company name.
      address -> the line is a (fuzzy) substring of the address.
    """
    t = line_text.strip()
    nt = normalise(t)
    if not nt:
        return "other"

    total = normalise(key_fields.get("total", "") or "")
    if total and total in nt:
        if re.search(r"TOTAL|AMOUNT|NET|PAYABLE|RM", t.upper()) or nt == total or nt == "RM" + total:
            return "total"

    date = normalise(key_fields.get("date", "") or "")
    if date and len(date) >= 6 and date in nt:
        return "date"

    company = normalise(key_fields.get("company", "") or "")
    if company and len(nt) >= 4 and fuzz.partial_ratio(nt, company) >= threshold and len(nt) >= 0.5 * min(len(company), 20):
        return "company"

    address = normalise(key_fields.get("address", "") or "")
    if address and len(nt) >= 5 and fuzz.partial_ratio(nt, address) >= threshold:
        return "address"

    return "other"


def class_id(name: str) -> int:
    return FIELD_CLASSES.index(name)
