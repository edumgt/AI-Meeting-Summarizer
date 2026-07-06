"""
AICC 콜 요약 어시스턴트 (실제 텔레포니/IVR/CTI/실시간 스트리밍/상담원 배분은 제외).

상담 녹음을 업로드하면 전사 -> 문의유형/주문요청/컴플라이언스/감정 태깅 -> QA
리포트를 생성한다. 컴플라이언스 체크는 키워드 매칭 수준의 참고용 휴리스틱이며,
금융소비자보호법 등 실제 규제 요건을 담보하지 않는다. 운영 적용 전 컴플라이언스
팀 검토가 필요하다.
"""
import asyncio
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import db
import pipeline as pipeline_module
from pipeline import (
    MP3_OUTPUT_DIR,
    TRANSCRIBE_LANGUAGE,
    _convert_audio_to_mp3,
    _summarize_with_ollama,
    _summarize_with_openai,
    _transcribe_mp3,
)
from tickers import extract_tickers

router = APIRouter()


# -----------------------------
# DTOs
# -----------------------------
class ComplianceFlag(BaseModel):
    type: str  # "banned_phrase" | "missing_disclosure"
    phrase: str
    context: str


class CallExtractOut(BaseModel):
    inquiry_type: str  # "계좌문의"|"주문/매매"|"상품문의"|"불만/민원"|"기타"
    order_requested: bool
    order_detail: Optional[str] = None
    tickers_mentioned: List[str] = []
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    compliance_flags: List[ComplianceFlag] = []
    risk_level: str = "낮음"  # "낮음"|"중간"|"높음"


class CallSummaryOut(BaseModel):
    call_id: int
    transcript: str
    summary: Optional[str] = None
    markdown: str
    extracted: CallExtractOut
    mp3_download_url: str
    mp3_file_name: str
    meta: Dict[str, Any]


# -----------------------------
# 규칙 기반 추출 (참고용 휴리스틱)
# -----------------------------
_INQUIRY_KEYWORDS = [
    (re.compile(r"계좌|비밀번호|인증서|공동인증|로그인|아이디"), "계좌문의"),
    (re.compile(r"매수|매도|주문|체결|수량|호가"), "주문/매매"),
    (re.compile(r"수수료|펀드|ETF|상품\s*설명|이자율|금리"), "상품문의"),
    (re.compile(r"불편|항의|화가|환불|민원|답답"), "불만/민원"),
]

_ORDER_KEYWORDS_RE = re.compile(r"매수|매도")

# 아래 목록은 참고용 시드일 뿐이며, 실제 컴플라이언스 요건은 법무/컴플라이언스 팀이 정의해야 한다.
BANNED_PHRASES = ["무조건 수익", "100% 보장", "손실 없이", "확정 수익", "원금 보장"]
REQUIRED_DISCLOSURES = ["원금손실", "투자위험", "예금자보호"]


def _detect_inquiry_type(text: str) -> str:
    for pattern, label in _INQUIRY_KEYWORDS:
        if pattern.search(text):
            return label
    return "기타"


def _detect_order(text: str):
    m = _ORDER_KEYWORDS_RE.search(text)
    if not m:
        return False, None
    idx = m.start()
    snippet = text[max(0, idx - 20) : idx + 40].strip()
    return True, snippet


def _detect_compliance(text: str, order_requested: bool) -> List[ComplianceFlag]:
    flags: List[ComplianceFlag] = []
    for phrase in BANNED_PHRASES:
        idx = text.find(phrase)
        if idx >= 0:
            flags.append(
                ComplianceFlag(
                    type="banned_phrase",
                    phrase=phrase,
                    context=text[max(0, idx - 20) : idx + 40].strip(),
                )
            )
    if order_requested and not any(d in text for d in REQUIRED_DISCLOSURES):
        flags.append(
            ComplianceFlag(
                type="missing_disclosure",
                phrase="(필수 고지 문구 미발견)",
                context=text[:80].strip(),
            )
        )
    return flags


async def _detect_sentiment(text: str):
    if pipeline_module.classifier is None:
        return None, None
    async with pipeline_module.cls_sem:
        try:
            result = await asyncio.to_thread(pipeline_module.classifier, text[:512])
        except Exception:
            return None, None
    if not result:
        return None, None
    top = result[0] if isinstance(result, list) else result
    return top.get("label"), top.get("score")


async def extract_call_insights(transcript: str) -> CallExtractOut:
    inquiry_type = _detect_inquiry_type(transcript)
    tickers = extract_tickers(transcript)
    order_requested, order_detail = _detect_order(transcript)
    compliance_flags = _detect_compliance(transcript, order_requested)
    sentiment, sentiment_score = await _detect_sentiment(transcript)

    is_negative = bool(sentiment) and ("부정" in sentiment or "neg" in sentiment.lower())
    if compliance_flags or (is_negative and (sentiment_score or 0) > 0.8):
        risk_level = "높음"
    elif is_negative:
        risk_level = "중간"
    else:
        risk_level = "낮음"

    return CallExtractOut(
        inquiry_type=inquiry_type,
        order_requested=order_requested,
        order_detail=order_detail,
        tickers_mentioned=tickers,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        compliance_flags=compliance_flags,
        risk_level=risk_level,
    )


def call_markdown_report(
    agent_name: Optional[str],
    call_date: Optional[str],
    extracted: CallExtractOut,
    summary: Optional[str],
) -> str:
    md = []
    md.append("# 상담 통화 QA 리포트")
    md.append("")
    md.append(f"- 상담일: {call_date or '-'}")
    md.append(f"- 상담원: {agent_name or '-'}")
    md.append(f"- 문의유형: {extracted.inquiry_type}")
    md.append(f"- 리스크 등급: {extracted.risk_level}")
    md.append("")
    md.append("## 상담 요약")
    md.append("")
    md.append(summary.strip() if summary else "- 요약이 생성되지 않았습니다.")
    md.append("")
    md.append("## 컴플라이언스 체크 (참고용 휴리스틱)")
    md.append("")
    if extracted.compliance_flags:
        for f in extracted.compliance_flags:
            md.append(f'- [{f.type}] "{f.phrase}" — ...{f.context}...')
    else:
        md.append("- (이상 없음)")
    md.append("")
    if extracted.order_requested:
        md.append("## 주문 요청 상세")
        md.append("")
        md.append(f"- 감지된 내용: {extracted.order_detail or '-'}")
        tickers_label = ", ".join(extracted.tickers_mentioned) if extracted.tickers_mentioned else "-"
        md.append(f"- 언급된 종목: {tickers_label}")
        md.append("")
    return "\n".join(md).strip() + "\n"


# -----------------------------
# DB
# -----------------------------
def _save_call_summary(
    agent_name: Optional[str],
    call_date: Optional[str],
    extracted: CallExtractOut,
    transcript: str,
    mp3_file_name: str,
) -> int:
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO call_summaries
                    (agent_name, call_date, inquiry_type, order_requested, order_detail,
                     tickers_mentioned, sentiment, sentiment_score, compliance_flags,
                     risk_level, transcript, mp3_file_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    agent_name or None,
                    call_date or None,
                    extracted.inquiry_type,
                    extracted.order_requested,
                    extracted.order_detail,
                    json.dumps(extracted.tickers_mentioned, ensure_ascii=False),
                    extracted.sentiment,
                    extracted.sentiment_score,
                    json.dumps([f.model_dump() for f in extracted.compliance_flags], ensure_ascii=False),
                    extracted.risk_level,
                    transcript,
                    mp3_file_name,
                ),
            )
            return cur.fetchone()[0]


def _load_call_summary(call_id: int) -> Optional[CallSummaryOut]:
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT agent_name, call_date, inquiry_type, order_requested, order_detail,
                       tickers_mentioned, sentiment, sentiment_score, compliance_flags,
                       risk_level, transcript, mp3_file_name
                FROM call_summaries
                WHERE id = %s
                """,
                (call_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None

    (
        agent_name,
        call_date,
        inquiry_type,
        order_requested,
        order_detail,
        tickers_mentioned,
        sentiment,
        sentiment_score,
        compliance_flags,
        risk_level,
        transcript,
        mp3_file_name,
    ) = row

    extracted = CallExtractOut(
        inquiry_type=inquiry_type,
        order_requested=order_requested,
        order_detail=order_detail,
        tickers_mentioned=tickers_mentioned or [],
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        compliance_flags=[ComplianceFlag(**f) for f in (compliance_flags or [])],
        risk_level=risk_level,
    )
    markdown = call_markdown_report(
        agent_name,
        call_date.isoformat() if call_date else None,
        extracted,
        None,
    )
    return CallSummaryOut(
        call_id=call_id,
        transcript=transcript,
        summary=None,
        markdown=markdown,
        extracted=extracted,
        mp3_download_url=f"/audio/{mp3_file_name}",
        mp3_file_name=mp3_file_name,
        meta={"loaded_from": "db"},
    )


# -----------------------------
# Routes
# -----------------------------
@router.get("/call-summary")
def call_summary_help():
    return {
        "message": (
            "Use POST /call-summary (multipart: audio, agent_name, call_date_hint, "
            "include_summary, ai_provider[openai|ollama], language)"
        )
    }


@router.post("/call-summary", response_model=CallSummaryOut)
async def call_summary(
    audio: UploadFile = File(...),
    agent_name: str = Form(""),
    call_date_hint: str = Form(""),
    include_summary: str = Form("true"),
    ai_provider: str = Form("openai"),
    language: str = Form(TRANSCRIBE_LANGUAGE),
):
    ext = Path(audio.filename or "input").suffix or ".webm"
    output_name = f"call-{uuid.uuid4().hex}.mp3"
    output_path = MP3_OUTPUT_DIR / output_name

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_input:
        temp_bytes = await audio.read()
        if not temp_bytes:
            raise HTTPException(status_code=400, detail="audio file is empty")
        temp_input.write(temp_bytes)
        temp_input_path = temp_input.name

    try:
        await asyncio.to_thread(_convert_audio_to_mp3, temp_input_path, str(output_path))
        transcript = await asyncio.to_thread(_transcribe_mp3, str(output_path), language)
        if not transcript.strip():
            raise HTTPException(status_code=400, detail="transcript is empty")

        use_summary = include_summary.strip().lower() in {"1", "true", "yes", "on"}
        summary = None
        if use_summary:
            if ai_provider == "ollama":
                summary = await asyncio.to_thread(_summarize_with_ollama, transcript, 200, 60)
            else:
                summary = await asyncio.to_thread(_summarize_with_openai, transcript, 200, 60)

        extracted = await extract_call_insights(transcript)
        markdown = call_markdown_report(agent_name or None, call_date_hint or None, extracted, summary)
        call_id = await asyncio.to_thread(
            _save_call_summary, agent_name, call_date_hint, extracted, transcript, output_name
        )

        return CallSummaryOut(
            call_id=call_id,
            transcript=transcript,
            summary=summary,
            markdown=markdown,
            extracted=extracted,
            mp3_download_url=f"/audio/{output_name}",
            mp3_file_name=output_name,
            meta={"ai_provider": ai_provider, "language": language or TRANSCRIBE_LANGUAGE},
        )
    finally:
        try:
            os.remove(temp_input_path)
        except OSError:
            pass


@router.get("/call-summary/{call_id}", response_model=CallSummaryOut)
async def get_call_summary(call_id: int):
    result = await asyncio.to_thread(_load_call_summary, call_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Call summary not found")
    return result
