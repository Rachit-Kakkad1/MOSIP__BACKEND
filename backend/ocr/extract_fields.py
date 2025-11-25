# backend/ocr/extract_fields.py
"""
Production-ready field extraction for OCR text.

Features:
- spaCy NER when available (en_core_web_sm)
- Robust regexes for email, phone, date, age, gender
- Heuristic address extraction using keywords + scoring
- Name extraction: spaCy PERSON > heuristics
- Candidate scoring using rapidfuzz (optional)
- Returns dict: { field: { value, confidence (0-1), candidates: [{value,score}, ...] } }
"""

import re
from typing import Dict, Any, List, Tuple
from math import ceil

# Attempt imports with graceful fallbacks
try:
    import spacy
    _HAS_SPACY = True
except Exception:
    spacy = None
    _HAS_SPACY = False

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# Load spaCy model lazily; if it fails, nlp will be None
nlp = None
if _HAS_SPACY:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None

# Regex patterns
EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_RE = re.compile(r"(\+?\d{1,3}[\s\-\.\(]?\d{2,4}[\d\-\s\.\(\)]{4,}\d)")
DATE_RE = re.compile(
    r"\b(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})\b"
)  # simple dd/mm/yyyy or variants
AGE_RE = re.compile(r"\b(?:age|aged)[:\s]*([0-9]{1,3})\b", re.IGNORECASE)
GENDER_RE = re.compile(r"\b(male|female|other|m|f)\b", re.IGNORECASE)

ADDRESS_KEYWORDS = [
    "street", "st", "road", "rd", "lane", "ln", "avenue", "ave", "block", "sector",
    "colony", "nagar", "wing", "house", "apt", "apartment", "flat", "city", "village",
    "district", "state", "zipcode", "pincode"
]

CAPITALIZED_NAME_RE = re.compile(r"\b([A-Z][a-z]{1,}\s+(?:[A-Z]\w+)(?:\s+[A-Z]\w+)?)\b")


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def score_candidate(base: str, candidate: str) -> float:
    """
    Compute similarity score in [0,1] between base text and candidate.
    Uses rapidfuzz if available, else falls back to simple ratio of lengths and substring.
    """
    if not base or not candidate:
        return 0.0
    base = normalize_whitespace(base).lower()
    candidate = normalize_whitespace(candidate).lower()
    if _HAS_RAPIDFUZZ:
        # token_sort_ratio is good for name-like comparisons
        s = fuzz.token_sort_ratio(base, candidate) / 100.0
        return max(0.0, min(1.0, s))
    # fallback heuristic
    if base in candidate or candidate in base:
        return 0.9
    # length-normalized overlap
    common = sum((1 for ch in base if ch in candidate))
    return max(0.0, min(1.0, common / max(1, len(base))))


def candidate_list(values: List[str], base: str = "") -> List[Dict[str, Any]]:
    """
    Return list of dicts with candidate values and confidence score (0-1).
    base: optional original text to score against.
    """
    out = []
    for v in values:
        s = score_candidate(base, v) if base else 0.5
        out.append({"value": normalize_whitespace(v), "confidence": round(float(s), 3)})
    return out


def best_candidate(values: List[str], base: str = "") -> Tuple[str, float]:
    if not values:
        return "", 0.0
    scored = [(v, score_candidate(base, v) if base else 0.5) for v in values]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_val, best_score = scored[0]
    return normalize_whitespace(best_val), round(float(best_score), 3)


def extract_emails(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    emails = EMAIL_RE.findall(text) or []
    emails = [e.strip() for e in emails]
    primary, score = best_candidate(emails, text)
    return primary, candidate_list(emails, text)


def extract_phones(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    raw = PHONE_RE.findall(text) or []
    cleaned = []
    for r in raw:
        c = re.sub(r"[^\d\+]", "", r)
        # normalize common errors: drop leading 00 -> +
        if c.startswith("00"):
            c = "+" + c[2:]
        cleaned.append(c)
    cleaned = [c for c in cleaned if len(re.sub(r"\D", "", c)) >= 7]
    primary, score = best_candidate(cleaned, text)
    return primary, candidate_list(cleaned, text)


def extract_dates(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    dates = DATE_RE.findall(text) or []
    dates = [d.strip() for d in dates]
    primary, _ = best_candidate(dates, text)
    return primary, candidate_list(dates, text)


def extract_age(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    ages = AGE_RE.findall(text) or []
    ages = [a.strip() for a in ages]
    primary, _ = best_candidate(ages, text)
    return primary, candidate_list(ages, text)


def extract_gender(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    genders = GENDER_RE.findall(text) or []
    genders = [g.lower() for g in genders]
    # normalize to 'male' or 'female' or 'other'
    normalized = []
    for g in genders:
        if g.startswith("m"):
            normalized.append("male")
        elif g.startswith("f"):
            normalized.append("female")
        else:
            normalized.append("other")
    primary, _ = best_candidate(normalized, text)
    return primary, candidate_list(normalized, text)


def extract_name(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    # Try spaCy first
    candidates = []
    if nlp:
        try:
            doc = nlp(text)
            ents = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
            candidates.extend(ents)
        except Exception:
            # spaCy can occasionally fail on weird input; continue to heuristics
            pass

    # Heuristic: first occurrence of 2 capitalized words (First Last)
    if not candidates:
        m = CAPITALIZED_NAME_RE.search(text)
        if m:
            candidates.append(m.group(1).strip())

    # Heuristic: lines that look like names (short line, letters only)
    if not candidates:
        for line in text.splitlines():
            ln = line.strip()
            if 3 <= len(ln) <= 60 and re.match(r"^[A-Za-z\s\.\-']+$", ln):
                # skip lines with many lowercase words (likely sentence)
                words = ln.split()
                if 1 <= len(words) <= 4 and sum(1 for w in words if w[0].isupper()) >= 1:
                    candidates.append(ln)
    # deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c and c.lower() not in seen:
            unique.append(c)
            seen.add(c.lower())

    primary, _ = best_candidate(unique, text)
    return primary, candidate_list(unique, text)


def extract_address(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract address-like lines using keywords and scoring.
    Returns best candidate and a list.
    """
    candidates = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        lw = line.lower()
        # If line contains digits and a keyword -> strong candidate
        if any(k in lw for k in ADDRESS_KEYWORDS) and re.search(r"\d", line):
            candidates.append(line)
        # If line contains keyword alone (no digits), consider as weaker candidate
        elif any(k in lw for k in ADDRESS_KEYWORDS):
            candidates.append(line)
    # If still no candidates, fallback to long lines (addresses are often long)
    if not candidates:
        candidates = [l for l in lines if len(l) >= 30][:5]
    primary, _ = best_candidate(candidates, text)
    return primary, candidate_list(candidates, text)


def extract_fields(text: str) -> Dict[str, Any]:
    text = normalize_whitespace(text or "")
    out: Dict[str, Any] = {}

    email_val, email_cands = extract_emails(text)
    out["email"] = {
        "value": email_val,
        "confidence": email_cands[0]["confidence"] if email_cands else 0.0,
        "candidates": email_cands,
    }

    phone_val, phone_cands = extract_phones(text)
    out["phone"] = {
        "value": phone_val,
        "confidence": phone_cands[0]["confidence"] if phone_cands else 0.0,
        "candidates": phone_cands,
    }

    dob_val, dob_cands = extract_dates(text)
    out["date_of_birth"] = {
        "value": dob_val,
        "confidence": dob_cands[0]["confidence"] if dob_cands else 0.0,
        "candidates": dob_cands,
    }

    age_val, age_cands = extract_age(text)
    out["age"] = {
        "value": age_val,
        "confidence": age_cands[0]["confidence"] if age_cands else 0.0,
        "candidates": age_cands,
    }

    gender_val, gender_cands = extract_gender(text)
    out["gender"] = {
        "value": gender_val,
        "confidence": gender_cands[0]["confidence"] if gender_cands else 0.0,
        "candidates": gender_cands,
    }

    name_val, name_cands = extract_name(text)
    out["name"] = {
        "value": name_val,
        "confidence": name_cands[0]["confidence"] if name_cands else 0.0,
        "candidates": name_cands,
    }

    addr_val, addr_cands = extract_address(text)
    out["address"] = {
        "value": addr_val,
        "confidence": addr_cands[0]["confidence"] if addr_cands else 0.0,
        "candidates": addr_cands,
    }

    return out
