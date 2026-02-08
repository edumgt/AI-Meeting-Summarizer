import os
import re
import time
import json
import hashlib
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer

from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

# -----------------------------
# Config (env)
# -----------------------------
SUM_MODEL_ID = os.getenv("SUM_MODEL_ID", "gogamza/kobart-summarization")
CLS_MODEL_ID = os.getenv("CLS_MODEL_ID", "Seonghaa/korean-emotion-classifier-roberta")
EMB_MODEL_ID = os.getenv("EMB_MODEL_ID", "upskyy/bge-m3-korean")

CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "900"))
REDUCE_MAX_TOKENS = int(os.getenv("REDUCE_MAX_TOKENS", "1100"))

DEFAULT_MAX_LENGTH = int(os.getenv("DEFAULT_MAX_LENGTH", "180"))
DEFAULT_MIN_LENGTH = int(os.getenv("DEFAULT_MIN_LENGTH", "50"))

# repetition control (요약 품질)
DEFAULT_NUM_BEAMS = int(os.getenv("DEFAULT_NUM_BEAMS", "4"))
DEFAULT_NO_REPEAT_NGRAM = int(os.getenv("DEFAULT_NO_REPEAT_NGRAM", "3"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("DEFAULT_REPETITION_PENALTY", "1.12"))
DEFAULT_LENGTH_PENALTY = float(os.getenv("DEFAULT_LENGTH_PENALTY", "1.0"))

# CPU optimization (큐잉)
SUM_MAX_CONCURRENCY = int(os.getenv("SUM_MAX_CONCURRENCY", "1"))
EMB_MAX_CONCURRENCY = int(os.getenv("EMB_MAX_CONCURRENCY", "2"))
CLS_MAX_CONCURRENCY = int(os.getenv("CLS_MAX_CONCURRENCY", "2"))

# Cache
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "1800"))  # 30분
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "256"))

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Korean Meeting Minutes Summarizer (CPU)", version="2.0.0")

# ✅ CORS 허용 (FE → API 호출 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

summarizer = None
sum_tokenizer = None
classifier = None
embedder = None

sum_sem = asyncio.Semaphore(SUM_MAX_CONCURRENCY)
emb_sem = asyncio.Semaphore(EMB_MAX_CONCURRENCY)
cls_sem = asyncio.Semaphore(CLS_MAX_CONCURRENCY)

# -----------------------------
# Tiny TTL Cache
# -----------------------------
@dataclass
class _CacheItem:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self, max_items: int, ttl_sec: int):
        self.max_items = max_items
        self.ttl_sec = ttl_sec
        self._store: Dict[str, _CacheItem] = {}
        self._order: List[str] = []  # simple LRU-ish

    def _purge_expired(self):
        now = time.time()
        expired = [k for k, v in self._store.items() if v.expires_at <= now]
        for k in expired:
            self._store.pop(k, None)
        if expired:
            self._order = [k for k in self._order if k in self._store]

    def get(self, key: str):
        self._purge_expired()
        item = self._store.get(key)
        if not item:
            return None
        if key in self._order:
            self._order.remove(key)
        self._order.insert(0, key)
        return item.value

    def set(self, key: str, value: Any):
        self._purge_expired()
        if key in self._store:
            self._store[key] = _CacheItem(value=value, expires_at=time.time() + self.ttl_sec)
            if key in self._order:
                self._order.remove(key)
            self._order.insert(0, key)
            return

        while len(self._store) >= self.max_items and self._order:
            old = self._order.pop()
            self._store.pop(old, None)

        self._store[key] = _CacheItem(value=value, expires_at=time.time() + self.ttl_sec)
        self._order.insert(0, key)

    def stats(self):
        self._purge_expired()
        return {"items": len(self._store), "max_items": self.max_items, "ttl_sec": self.ttl_sec}


sum_cache = TTLCache(CACHE_MAX_ITEMS, CACHE_TTL_SEC)
emb_cache = TTLCache(CACHE_MAX_ITEMS, CACHE_TTL_SEC)

# -----------------------------
# DTOs
# -----------------------------
class HealthOut(BaseModel):
    ok: bool
    sum_model: str
    cls_model: str
    emb_model: str
    cache: Dict[str, Any]
    concurrency: Dict[str, int]


class SummarizeIn(BaseModel):
    text: str = Field(..., description="원문 텍스트(긴 문서 가능)")

    max_length: int = Field(DEFAULT_MAX_LENGTH, ge=16, le=512)
    min_length: int = Field(DEFAULT_MIN_LENGTH, ge=4, le=256)
    do_sample: bool = False

    chunk_max_tokens: int = Field(CHUNK_MAX_TOKENS, ge=200, le=2000)
    reduce_max_tokens: int = Field(REDUCE_MAX_TOKENS, ge=300, le=2500)

    num_beams: int = Field(DEFAULT_NUM_BEAMS, ge=1, le=8)
    no_repeat_ngram_size: int = Field(DEFAULT_NO_REPEAT_NGRAM, ge=0, le=6)
    repetition_penalty: float = Field(DEFAULT_REPETITION_PENALTY, ge=0.8, le=2.0)
    length_penalty: float = Field(DEFAULT_LENGTH_PENALTY, ge=0.1, le=2.0)

    normalize_whitespace: bool = True
    collapse_repeats: bool = True
    max_repeat: int = Field(3, ge=2, le=10)


class SummarizeOut(BaseModel):
    chunks: List[str]
    chunk_summaries: List[str]
    final_summary: str
    meta: Dict[str, Any]


class ActionItem(BaseModel):
    task: str
    owner: Optional[str] = None
    due: Optional[str] = None  # ISO date string "YYYY-MM-DD" (가능하면)
    due_text: Optional[str] = None  # 원문 표현 (예: "차주 수요일까지")


class ExtractIn(BaseModel):
    text: str = Field(..., description="회의록 원문(대화체 가능)")
    meeting_date_hint: Optional[str] = Field(
        None, description="기준일 힌트(YYYY-MM-DD). 문서에 날짜가 없을 때 사용."
    )
    use_summary_hint: bool = True  # 원문에서 거의 못 찾으면 요약을 힌트로 2차 추출


class ExtractOut(BaseModel):
    meeting_date: Optional[str] = None
    attendees: List[str] = []
    decisions: List[str] = []
    action_items: List[ActionItem] = []
    issues: List[str] = []
    next_agenda: List[str] = []
    meta: Dict[str, Any]


class ReportIn(BaseModel):
    text: str
    meeting_title: Optional[str] = "회의록"
    meeting_date_hint: Optional[str] = None
    include_summary: bool = True
    summary_max_length: int = Field(DEFAULT_MAX_LENGTH, ge=16, le=512)
    summary_min_length: int = Field(DEFAULT_MIN_LENGTH, ge=4, le=256)


class ReportOut(BaseModel):
    markdown: str
    extracted: ExtractOut


class ClassifyIn(BaseModel):
    text: str


class EmbedIn(BaseModel):
    texts: List[str]
    normalize: bool = True


class EmbedOut(BaseModel):
    dim: int
    vectors: List[List[float]]


# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def _startup():
    global summarizer, sum_tokenizer, classifier, embedder
    summarizer = pipeline("summarization", model=SUM_MODEL_ID, device=-1)
    sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_ID)
    classifier = pipeline("text-classification", model=CLS_MODEL_ID, device=-1)
    embedder = SentenceTransformer(EMB_MODEL_ID, device="cpu")


# -----------------------------
# Helpers
# -----------------------------
def _sha_key(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _normalize_text(text: str, normalize_whitespace: bool, collapse_repeats: bool, max_repeat: int) -> str:
    t = text.replace("\r\n", "\n")
    if normalize_whitespace:
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
    if collapse_repeats:
        pattern = re.compile(rf"(\b[0-9A-Za-z가-힣]+\b)(\s+\1){{{max_repeat},}}")
        while True:
            new_t = pattern.sub(lambda m: " ".join([m.group(1)] * max_repeat), t)
            if new_t == t:
                break
            t = new_t
    return t.strip()


def _split_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").strip()
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if len(blocks) >= 2:
        return blocks
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines if lines else [text]


def _num_tokens(s: str) -> int:
    return len(sum_tokenizer.encode(s, add_special_tokens=False))


def _make_chunks(paragraphs: List[str], chunk_max_tokens: int) -> List[str]:
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    for p in paragraphs:
        t = _num_tokens(p)
        if t > chunk_max_tokens:
            if cur:
                chunks.append("\n".join(cur).strip())
                cur, cur_tokens = [], 0

            ids = sum_tokenizer.encode(p, add_special_tokens=False)
            for i in range(0, len(ids), chunk_max_tokens):
                part_ids = ids[i : i + chunk_max_tokens]
                part = sum_tokenizer.decode(part_ids, skip_special_tokens=True).strip()
                if part:
                    chunks.append(part)
            continue

        if cur_tokens + t <= chunk_max_tokens:
            cur.append(p)
            cur_tokens += t
        else:
            if cur:
                chunks.append("\n".join(cur).strip())
            cur = [p]
            cur_tokens = t

    if cur:
        chunks.append("\n".join(cur).strip())

    return [c for c in chunks if c]


def _summarize_sync(
    text: str,
    *,
    max_length: int,
    min_length: int,
    do_sample: bool,
    num_beams: int,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
    length_penalty: float,
) -> str:
    out = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        truncation=True,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        early_stopping=True,
    )
    if isinstance(out, list) and out:
        return (out[0].get("summary_text") or "").strip()
    return ""


async def _summarize_full(req: SummarizeIn) -> Dict[str, Any]:
    normalized = _normalize_text(req.text, req.normalize_whitespace, req.collapse_repeats, req.max_repeat)
    cache_key = _sha_key(
        "summarize",
        normalized,
        req.max_length,
        req.min_length,
        req.do_sample,
        req.chunk_max_tokens,
        req.reduce_max_tokens,
        req.num_beams,
        req.no_repeat_ngram_size,
        req.repetition_penalty,
        req.length_penalty,
    )

    cached = sum_cache.get(cache_key)
    if cached:
        cached["meta"]["cache_hit"] = True
        return cached

    paragraphs = _split_paragraphs(normalized)
    chunks = _make_chunks(paragraphs, req.chunk_max_tokens)

    async with sum_sem:
        chunk_summaries: List[str] = []
        for ch in chunks:
            ch_sum = await asyncio.to_thread(
                _summarize_sync,
                ch,
                max_length=min(req.max_length, 220),
                min_length=min(req.min_length, 90),
                do_sample=req.do_sample,
                num_beams=req.num_beams,
                no_repeat_ngram_size=req.no_repeat_ngram_size,
                repetition_penalty=req.repetition_penalty,
                length_penalty=req.length_penalty,
            )
            if not ch_sum:
                ch_sum = ch[:300]
            chunk_summaries.append(ch_sum)

        joined = "\n".join(chunk_summaries).strip()
        reduce_chunks = _make_chunks(_split_paragraphs(joined), req.reduce_max_tokens)

        if len(reduce_chunks) == 1:
            final_summary = await asyncio.to_thread(
                _summarize_sync,
                reduce_chunks[0],
                max_length=req.max_length,
                min_length=req.min_length,
                do_sample=req.do_sample,
                num_beams=req.num_beams,
                no_repeat_ngram_size=req.no_repeat_ngram_size,
                repetition_penalty=req.repetition_penalty,
                length_penalty=req.length_penalty,
            )
            if not final_summary:
                final_summary = reduce_chunks[0]
        else:
            reduce_summaries = []
            for rc in reduce_chunks:
                s = await asyncio.to_thread(
                    _summarize_sync,
                    rc,
                    max_length=req.max_length,
                    min_length=req.min_length,
                    do_sample=req.do_sample,
                    num_beams=req.num_beams,
                    no_repeat_ngram_size=req.no_repeat_ngram_size,
                    repetition_penalty=req.repetition_penalty,
                    length_penalty=req.length_penalty,
                )
                reduce_summaries.append(s or rc)

            final_summary = await asyncio.to_thread(
                _summarize_sync,
                "\n".join(reduce_summaries),
                max_length=req.max_length,
                min_length=req.min_length,
                do_sample=req.do_sample,
                num_beams=req.num_beams,
                no_repeat_ngram_size=req.no_repeat_ngram_size,
                repetition_penalty=req.repetition_penalty,
                length_penalty=req.length_penalty,
            )
            if not final_summary:
                final_summary = "\n".join(reduce_summaries)

    meta = {
        "cache_hit": False,
        "paragraphs": len(paragraphs),
        "chunks": len(chunks),
        "chunk_max_tokens": req.chunk_max_tokens,
        "reduce_max_tokens": req.reduce_max_tokens,
        "summary_params": {
            "max_length": req.max_length,
            "min_length": req.min_length,
            "num_beams": req.num_beams,
            "no_repeat_ngram_size": req.no_repeat_ngram_size,
            "repetition_penalty": req.repetition_penalty,
            "length_penalty": req.length_penalty,
        },
        "models": {"summarization": SUM_MODEL_ID, "classification": CLS_MODEL_ID, "embedding": EMB_MODEL_ID},
    }

    payload = {
        "chunks": chunks,
        "chunk_summaries": chunk_summaries,
        "final_summary": final_summary.strip(),
        "meta": meta,
    }
    sum_cache.set(cache_key, payload)
    return payload


# -------- Dialogue parsing (발언자: ...)
_SPEAKER_RE = re.compile(r"^\s*([가-힣A-Za-z0-9_]{2,20})\s*[:：]\s*(.+)\s*$")


def _parse_dialogue_lines(text: str) -> List[Tuple[Optional[str], str]]:
    """
    대화체를 (speaker, utterance)로 변환.
    speaker 없는 라인은 (None, line)
    """
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
    out: List[Tuple[Optional[str], str]] = []
    cur_speaker: Optional[str] = None

    for ln in lines:
        m = _SPEAKER_RE.match(ln)
        if m:
            cur_speaker = m.group(1).strip()
            out.append((cur_speaker, m.group(2).strip()))
        else:
            # 이전 speaker의 연속 발언으로 붙일지, 일반 라인으로 둘지
            # 회의록은 보통 연속 발언이 많아서 speaker가 있으면 이어붙임
            if cur_speaker and len(ln) < 200:
                out.append((cur_speaker, ln))
            else:
                out.append((None, ln))
    return out


# -------- Date parsing (기한 강화)
_KO_WEEKDAY = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}


def _parse_base_date(meeting_date_str: Optional[str], hint: Optional[str]) -> date:
    # 우선순위: meeting_date_str > hint > today
    for s in [meeting_date_str, hint]:
        if s:
            try:
                return datetime.strptime(s, "%Y-%m-%d").date()
            except Exception:
                pass
    return date.today()


def _next_weekday(d: date, weekday: int) -> date:
    delta = (weekday - d.weekday()) % 7
    if delta == 0:
        delta = 7
    return d + timedelta(days=delta)


def _this_weekday(d: date, weekday: int) -> date:
    # "이번주 수요일"은 이번 주(월~일) 기준으로 해당 요일 날짜
    start = d - timedelta(days=d.weekday())  # Monday
    return start + timedelta(days=weekday)


def _next_week_weekday(d: date, weekday: int) -> date:
    start_next = (d - timedelta(days=d.weekday())) + timedelta(days=7)
    return start_next + timedelta(days=weekday)


def _parse_due_expression(expr: str, base: date) -> Tuple[Optional[str], Optional[str]]:
    """
    return (iso_date, due_text)
    지원: "~까지", "D+3", "내일/모레", "이번주 금", "차주 수요일", "다음주 월요일", "2/12", "2026-02-12"
    """
    if not expr:
        return (None, None)

    raw = expr.strip()
    raw = raw.replace("까지", "").strip()

    # D+N
    m = re.search(r"\bD\s*\+\s*(\d{1,2})\b", raw, re.I)
    if m:
        dd = int(m.group(1))
        return ((base + timedelta(days=dd)).isoformat(), expr.strip())

    # 내일/모레
    if "내일" in raw:
        return ((base + timedelta(days=1)).isoformat(), expr.strip())
    if "모레" in raw:
        return ((base + timedelta(days=2)).isoformat(), expr.strip())

    # YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD
    m = re.search(r"\b(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})\b", raw)
    if m:
        y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return (date(y, mm, dd).isoformat(), expr.strip())
        except Exception:
            pass

    # M/D or M월 D일
    m = re.search(r"\b(\d{1,2})\s*[./월]\s*(\d{1,2})\s*(?:일)?\b", raw)
    if m:
        mm, dd = int(m.group(1)), int(m.group(2))
        y = base.year
        # 회의가 연말인데 기한이 다음 해로 넘어갈 가능성 보정
        try:
            dt = date(y, mm, dd)
            if dt < base - timedelta(days=30):  # 지나치게 과거면 다음 해로
                dt = date(y + 1, mm, dd)
            return (dt.isoformat(), expr.strip())
        except Exception:
            pass

    # 이번주/차주/다음주 + 요일
    # 예: "차주 수요일", "다음주 금요일", "이번주 금"
    m = re.search(r"(이번주|금주|차주|다음주)\s*([월화수목금토일])(?:요일)?", raw)
    if m:
        scope = m.group(1)
        wd = _KO_WEEKDAY[m.group(2)]
        if scope in ["이번주", "금주"]:
            return (_this_weekday(base, wd).isoformat(), expr.strip())
        if scope in ["차주", "다음주"]:
            return (_next_week_weekday(base, wd).isoformat(), expr.strip())

    # 요일만 있으면 "다음 해당 요일"로 해석 (ex: "수요일까지")
    m = re.search(r"([월화수목금토일])(?:요일)?", raw)
    if m and ("주" not in raw):
        wd = _KO_WEEKDAY[m.group(1)]
        return (_next_weekday(base, wd).isoformat(), expr.strip())

    return (None, expr.strip())


def _strip_due_from_task(text: str) -> str:
    # 기한 표현을 task에서 최대한 제거
    t = text
    t = re.sub(r"(까지)\s*$", "", t).strip()
    t = re.sub(r"\bD\s*\+\s*\d{1,2}\b", "", t, flags=re.I).strip()
    t = re.sub(r"(이번주|금주|차주|다음주)\s*[월화수목금토일](요일)?", "", t).strip()
    t = re.sub(r"\b(내일|모레)\b", "", t).strip()
    t = re.sub(r"\b(20\d{2})[.\-/]\d{1,2}[.\-/]\d{1,2}\b", "", t).strip()
    t = re.sub(r"\b\d{1,2}\s*[./월]\s*\d{1,2}\s*(일)?\b", "", t).strip()
    return t


def _clean_bullet(s: str) -> str:
    s = re.sub(r"^\s*[-•*\d\)\.]+\s*", "", s).strip()
    return s


def _extract_attendees_and_date(text: str) -> Tuple[Optional[str], List[str]]:
    # 날짜 패턴
    date_match = re.search(r"\b(20\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})\b", text)
    meeting_date = None
    if date_match:
        y, m, d = date_match.group(1), int(date_match.group(2)), int(date_match.group(3))
        meeting_date = f"{y}-{m:02d}-{d:02d}"

    # 참석자
    attendees: List[str] = []
    m = re.search(r"(참석자?|참석)\s*[:：]\s*(.+)", text)
    if m:
        raw = m.group(2).split("\n")[0]
        parts = re.split(r"[,/·]|및|\s{2,}", raw)
        attendees = [p.strip() for p in parts if p.strip() and len(p.strip()) <= 20]
    return meeting_date, attendees


def _extract_structured_dialogue(text: str, meeting_date_hint: Optional[str]) -> ExtractOut:
    """
    대화체(발언자:) + 일반 회의록 라인 모두에서
    - 결정사항/이슈/다음 안건: 키워드 기반 + 섹션 기반
    - 액션아이템: (1) '담당/기한' 라인, (2) 발언자 기반 "제가/우리/조치/확인/수정/정리" + 기한 표현
    """
    meeting_date, attendees = _extract_attendees_and_date(text)
    base = _parse_base_date(meeting_date, meeting_date_hint)

    pairs = _parse_dialogue_lines(text)

    decisions: List[str] = []
    issues: List[str] = []
    next_agenda: List[str] = []
    actions: List[ActionItem] = []

    def add_unique(lst: List[str], item: str):
        item = item.strip()
        if item and item not in lst:
            lst.append(item)

    # 섹션 힌트
    current = None

    # action 후보를 잡기 위한 키워드
    action_verb = re.compile(r"(하겠습니다|진행(하겠|합)|처리(하겠|합)|조치(하겠|합)|수정(하겠|합)|추가(하겠|합)|정리(하겠|합)|확인(하겠|합)|공유(하겠|합)|작성(하겠|합))")

    for speaker, utter in pairs:
        ln = utter.strip()
        if not ln:
            continue

        # 섹션 헤더
        head = ln.lower()
        if re.search(r"(결정사항|결정\s*:|결정\s*사항)", ln):
            current = "decisions"
            continue
        if re.search(r"(액션\s*아이템|action\s*item|todo|to-do|할\s*일|담당\s*업무)", head, re.I):
            current = "actions"
            continue
        if re.search(r"(이슈|문제|리스크|blocker)", head, re.I):
            current = "issues"
            continue
        if re.search(r"(다음\s*(회의|미팅)|다음\s*안건|next\s*agenda)", head, re.I):
            current = "next"
            continue

        # 결정/이슈/다음안건: 섹션 없이도 짧은 문장 위주로 캐치
        if re.search(r"(결정|확정|합의)", ln) and len(ln) <= 140:
            add_unique(decisions, _clean_bullet(ln))
        if re.search(r"(이슈|문제|리스크|지연|장애|막힘)", ln) and len(ln) <= 160:
            add_unique(issues, _clean_bullet(ln))
        if re.search(r"(다음|차주|다음주).*(안건|회의|미팅|논의)", ln) and len(ln) <= 180:
            add_unique(next_agenda, _clean_bullet(ln))

        # 섹션 기반 수집
        if current == "decisions" and len(ln) <= 180:
            add_unique(decisions, _clean_bullet(ln))
        elif current == "issues" and len(ln) <= 220:
            add_unique(issues, _clean_bullet(ln))
        elif current == "next" and len(ln) <= 220:
            add_unique(next_agenda, _clean_bullet(ln))

        # 액션아이템 (패턴1) "담당/기한" 라인
        if re.search(r"(담당|owner|기한|due|까지|D\+\d+|이번주|차주|다음주|내일|모레)", ln, re.I):
            owner = None
            due_iso = None
            due_text = None

            mo = re.search(r"(담당자?|owner)\s*[:：]\s*([가-힣A-Za-z0-9_]{2,20})", ln, re.I)
            if mo:
                owner = mo.group(2).strip()
            else:
                # 대화체면 speaker가 owner가 되는 경우가 많음
                owner = speaker or None

            md = re.search(r"(기한|due)\s*[:：]\s*(.+)$", ln, re.I)
            if md:
                due_iso, due_text = _parse_due_expression(md.group(2).strip(), base)
            else:
                # "차주 수요일까지" 같이 라인 어딘가에 있는 경우
                m2 = re.search(r"(D\s*\+\s*\d{1,2}|내일|모레|(이번주|금주|차주|다음주)\s*[월화수목금토일](?:요일)?|\b20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}\b|\b\d{1,2}\s*[./월]\s*\d{1,2}\s*(?:일)?\b|[월화수목금토일](?:요일)?\s*까지?)", ln)
                if m2:
                    due_iso, due_text = _parse_due_expression(m2.group(1).strip(), base)

            # task 정리
            task = _clean_bullet(ln)
            task = _strip_due_from_task(task)
            task = re.sub(r"(담당자?|owner)\s*[:：]\s*[가-힣A-Za-z0-9_]{2,20}", "", task, flags=re.I).strip()
            task = re.sub(r"(기한|due)\s*[:：]\s*.*$", "", task, flags=re.I).strip()

            # 액션성 문장만 등록 (너무 일반 문장 방지)
            if len(task) >= 8 and (action_verb.search(task) or current == "actions" or "TODO" in ln.upper() or "할 일" in ln):
                actions.append(ActionItem(task=task, owner=owner, due=due_iso, due_text=due_text))

        # 액션아이템 (패턴2) 대화체: "제가 ... 하겠습니다" + 기한 표현
        if speaker and action_verb.search(ln):
            m_due = re.search(r"(D\s*\+\s*\d{1,2}|내일|모레|(이번주|금주|차주|다음주)\s*[월화수목금토일](?:요일)?|\b20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}\b|\b\d{1,2}\s*[./월]\s*\d{1,2}\s*(?:일)?\b|[월화수목금토일](?:요일)?\s*까지?)", ln)
            due_iso, due_text = (None, None)
            if m_due:
                due_iso, due_text = _parse_due_expression(m_due.group(1).strip(), base)

            task = _strip_due_from_task(ln)
            task = _clean_bullet(task)
            # 너무 짧거나 의미 없는 건 제외
            if len(task) >= 8:
                actions.append(ActionItem(task=task, owner=speaker, due=due_iso, due_text=due_text))

    # 중복 제거
    seen_task = set()
    uniq_actions: List[ActionItem] = []
    for a in actions:
        key = a.task.strip()
        if key and key not in seen_task:
            seen_task.add(key)
            uniq_actions.append(a)

    return ExtractOut(
        meeting_date=meeting_date or base.isoformat(),
        attendees=attendees,
        decisions=decisions,
        action_items=uniq_actions,
        issues=issues,
        next_agenda=next_agenda,
        meta={
            "base_date": base.isoformat(),
            "models": {"summarization": SUM_MODEL_ID, "classification": CLS_MODEL_ID, "embedding": EMB_MODEL_ID},
        },
    )


def _markdown_report(title: str, extracted: ExtractOut, summary: Optional[str]) -> str:
    md = []
    md.append(f"# {title}")
    md.append("")
    md.append(f"- 날짜: {extracted.meeting_date or '-'}")
    md.append(f"- 참석자: {', '.join(extracted.attendees) if extracted.attendees else '-'}")
    md.append("")

    if summary:
        md.append("## 요약")
        md.append("")
        md.append(summary.strip())
        md.append("")

    md.append("## 결정사항")
    md.append("")
    if extracted.decisions:
        for d in extracted.decisions:
            md.append(f"- {d}")
    else:
        md.append("- (없음)")
    md.append("")

    md.append("## 액션아이템")
    md.append("")
    if extracted.action_items:
        md.append("| 작업 | 담당 | 기한 |")
        md.append("|---|---|---|")
        for a in extracted.action_items:
            owner = a.owner or "-"
            due = a.due or (a.due_text or "-")
            md.append(f"| {a.task} | {owner} | {due} |")
    else:
        md.append("- (없음)")
    md.append("")

    md.append("## 이슈")
    md.append("")
    if extracted.issues:
        for it in extracted.issues:
            md.append(f"- {it}")
    else:
        md.append("- (없음)")
    md.append("")

    md.append("## 다음 회의 안건")
    md.append("")
    if extracted.next_agenda:
        for it in extracted.next_agenda:
            md.append(f"- {it}")
    else:
        md.append("- (없음)")
    md.append("")

    return "\n".join(md).strip() + "\n"


# -----------------------------
# Routes
# -----------------------------
@app.get("/health", response_model=HealthOut)
def health():
    return HealthOut(
        ok=True,
        sum_model=SUM_MODEL_ID,
        cls_model=CLS_MODEL_ID,
        emb_model=EMB_MODEL_ID,
        cache={"summarize": sum_cache.stats(), "embed": emb_cache.stats()},
        concurrency={"summarize": SUM_MAX_CONCURRENCY, "embed": EMB_MAX_CONCURRENCY, "classify": CLS_MAX_CONCURRENCY},
    )


@app.get("/summarize")
def summarize_help():
    return {"message": "Use POST /summarize with JSON body. Try /docs for Swagger UI."}


@app.post("/summarize", response_model=SummarizeOut)
async def summarize(req: SummarizeIn):
    if not req.text.strip():
        return SummarizeOut(chunks=[], chunk_summaries=[], final_summary="", meta={"note": "empty input"})

    payload = await _summarize_full(req)
    return SummarizeOut(
        chunks=payload["chunks"],
        chunk_summaries=payload["chunk_summaries"],
        final_summary=payload["final_summary"],
        meta=payload["meta"],
    )


@app.get("/extract")
def extract_help():
    return {"message": "Use POST /extract with JSON body: { text, meeting_date_hint, use_summary_hint }"}


@app.post("/extract", response_model=ExtractOut)
async def extract(req: ExtractIn):
    raw = _normalize_text(req.text, True, True, 3)
    if not raw:
        return ExtractOut(meta={"note": "empty input"})

    extracted = _extract_structured_dialogue(raw, req.meeting_date_hint)

    # 원문에서 너무 sparse하면 요약을 힌트로 한 번 더(옵션)
    total = len(extracted.decisions) + len(extracted.issues) + len(extracted.next_agenda) + len(extracted.action_items)
    if req.use_summary_hint and total < 2:
        sreq = SummarizeIn(text=raw)
        payload = await _summarize_full(sreq)
        hinted = _extract_structured_dialogue(payload["final_summary"], extracted.meeting_date)

        # merge (원문 우선)
        def merge_list(a: List[str], b: List[str]):
            for x in b:
                if x not in a:
                    a.append(x)

        merge_list(extracted.decisions, hinted.decisions)
        merge_list(extracted.issues, hinted.issues)
        merge_list(extracted.next_agenda, hinted.next_agenda)

        seen = {ai.task for ai in extracted.action_items}
        for ai in hinted.action_items:
            if ai.task not in seen:
                extracted.action_items.append(ai)
                seen.add(ai.task)

        extracted.meta["used_summary_hint"] = True
        extracted.meta["hint_summary_preview"] = payload["final_summary"][:200]
    else:
        extracted.meta["used_summary_hint"] = False

    return extracted


@app.get("/report")
def report_help():
    return {"message": "Use POST /report with JSON body: { text, meeting_title, include_summary, meeting_date_hint }"}


@app.post("/report", response_model=ReportOut)
async def report(req: ReportIn):
    raw = _normalize_text(req.text, True, True, 3)
    extracted = await extract(ExtractIn(text=raw, meeting_date_hint=req.meeting_date_hint, use_summary_hint=True))

    summary = None
    if req.include_summary:
        sreq = SummarizeIn(text=raw, max_length=req.summary_max_length, min_length=req.summary_min_length)
        payload = await _summarize_full(sreq)
        summary = payload["final_summary"]

    md = _markdown_report(req.meeting_title or "회의록", extracted, summary)
    return ReportOut(markdown=md, extracted=extracted)


@app.post("/classify")
async def classify(req: ClassifyIn):
    async with cls_sem:
        return await asyncio.to_thread(classifier, req.text)


@app.post("/embed", response_model=EmbedOut)
async def embed(req: EmbedIn):
    vectors: List[List[float]] = []
    dim = 0

    async with emb_sem:
        for t in req.texts:
            t_norm = t.strip()
            key = _sha_key("embed", EMB_MODEL_ID, t_norm, req.normalize)
            cached = emb_cache.get(key)
            if cached:
                vec = cached
            else:
                vec_np = await asyncio.to_thread(embedder.encode, [t_norm], req.normalize)
                vec = vec_np[0].tolist()
                emb_cache.set(key, vec)
            vectors.append(vec)
            if not dim:
                dim = len(vec)

    return EmbedOut(dim=dim, vectors=vectors)
