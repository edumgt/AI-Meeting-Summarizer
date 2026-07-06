"""
증권사 애널리스트 리포트 추천 추출 + 종목별 컨센서스 집계.

기존 _extract_structured_dialogue(app.py)와 동일한 스타일로 정규식/키워드
기반 추출을 사용한다(별도 ML 모델 없음). 추출 결과는 Postgres
(analyst_recommendations 테이블, db.py)에 누적 저장되어 컨센서스 계산에 쓰인다.
"""
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import asyncio
import db
from pipeline import _extract_text_from_pdf, _normalize_text
from tickers import code_to_name, extract_tickers

router = APIRouter()


# -----------------------------
# DTOs
# -----------------------------
class AnalystRecommendation(BaseModel):
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    opinion: Optional[str] = None  # "매수" | "중립" | "매도"
    target_price: Optional[int] = None  # 원
    prior_target_price: Optional[int] = None
    rationale: List[str] = []
    analyst_firm: Optional[str] = None
    report_date: Optional[str] = None  # YYYY-MM-DD


class AnalystReportIn(BaseModel):
    text: str
    analyst_firm: Optional[str] = None
    report_date_hint: Optional[str] = None


class AnalystExtractOut(BaseModel):
    recommendations: List[AnalystRecommendation]
    meta: Dict[str, Any]


class ConsensusOut(BaseModel):
    ticker: str
    report_count: int
    opinion_distribution: Dict[str, int]
    avg_target_price: Optional[float] = None
    min_target_price: Optional[int] = None
    max_target_price: Optional[int] = None
    latest_reports: List[AnalystRecommendation]


# -----------------------------
# 추출 로직
# -----------------------------
_OPINION_PATTERNS = [
    (re.compile(r"매수|BUY|비중\s*확대", re.I), "매수"),
    (re.compile(r"중립|HOLD|보유", re.I), "중립"),
    (re.compile(r"매도|SELL|비중\s*축소", re.I), "매도"),
]

_TARGET_PRICE_RE = re.compile(r"목표\s*주가[^\d]{0,10}([\d,]+)\s*원")
_RATIONALE_HEADER_RE = re.compile(r"(투자\s*포인트|주요\s*내용|핵심\s*요약|key\s*takeaways?)", re.I)


def _detect_opinion(text: str) -> Optional[str]:
    for pattern, label in _OPINION_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _detect_target_price(text: str) -> Optional[int]:
    m = _TARGET_PRICE_RE.search(text)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def _detect_rationale(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
    for i, ln in enumerate(lines):
        if _RATIONALE_HEADER_RE.search(ln):
            bullets = []
            for follow in lines[i + 1 : i + 6]:
                cleaned = re.sub(r"^[-•*\d\)\.]+\s*", "", follow).strip()
                if cleaned and not _RATIONALE_HEADER_RE.search(cleaned):
                    bullets.append(cleaned)
            if bullets:
                return bullets[:5]

    sentences = re.split(r"(?<=[.!?다])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()][:3]


def extract_analyst_recommendations(
    text: str,
    analyst_firm: Optional[str],
    report_date_hint: Optional[str],
) -> List[AnalystRecommendation]:
    matched_tickers = extract_tickers(text)
    opinion = _detect_opinion(text)
    target_price = _detect_target_price(text)
    rationale = _detect_rationale(text)

    if not matched_tickers:
        return [
            AnalystRecommendation(
                ticker=None,
                company_name=None,
                opinion=opinion,
                target_price=target_price,
                rationale=rationale,
                analyst_firm=analyst_firm,
                report_date=report_date_hint,
            )
        ]

    return [
        AnalystRecommendation(
            ticker=code,
            company_name=code_to_name(code),
            opinion=opinion,
            target_price=target_price,
            rationale=rationale,
            analyst_firm=analyst_firm,
            report_date=report_date_hint,
        )
        for code in matched_tickers
    ]


# -----------------------------
# DB
# -----------------------------
def _save_recommendation(rec: AnalystRecommendation, source_file: Optional[str] = None):
    if not rec.ticker:
        return
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO analyst_recommendations
                    (ticker, company_name, analyst_firm, opinion, target_price,
                     prior_target_price, rationale, report_date, source_file)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    rec.ticker,
                    rec.company_name,
                    rec.analyst_firm,
                    rec.opinion,
                    rec.target_price,
                    rec.prior_target_price,
                    json.dumps(rec.rationale, ensure_ascii=False),
                    rec.report_date,
                    source_file,
                ),
            )


def get_consensus(ticker: str) -> Optional[ConsensusOut]:
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT opinion, target_price, company_name, analyst_firm, rationale, report_date, created_at
                FROM analyst_recommendations
                WHERE ticker = %s
                ORDER BY created_at DESC
                """,
                (ticker,),
            )
            rows = cur.fetchall()

    if not rows:
        return None

    opinion_dist: Dict[str, int] = {}
    prices: List[float] = []
    latest: List[AnalystRecommendation] = []

    for opinion, target_price, company_name, analyst_firm, rationale, report_date, _created_at in rows:
        if opinion:
            opinion_dist[opinion] = opinion_dist.get(opinion, 0) + 1
        if target_price is not None:
            prices.append(float(target_price))
        if len(latest) < 5:
            latest.append(
                AnalystRecommendation(
                    ticker=ticker,
                    company_name=company_name,
                    opinion=opinion,
                    target_price=int(target_price) if target_price is not None else None,
                    rationale=rationale or [],
                    analyst_firm=analyst_firm,
                    report_date=report_date.isoformat() if report_date else None,
                )
            )

    return ConsensusOut(
        ticker=ticker,
        report_count=len(rows),
        opinion_distribution=opinion_dist,
        avg_target_price=(sum(prices) / len(prices)) if prices else None,
        min_target_price=int(min(prices)) if prices else None,
        max_target_price=int(max(prices)) if prices else None,
        latest_reports=latest,
    )


# -----------------------------
# Routes
# -----------------------------
@router.get("/analyst-report")
def analyst_report_help():
    return {"message": "Use POST /analyst-report with JSON body: { text, analyst_firm, report_date_hint }"}


@router.post("/analyst-report", response_model=AnalystExtractOut)
async def analyst_report(req: AnalystReportIn):
    text = _normalize_text(req.text, True, True, 3)
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    recs = extract_analyst_recommendations(text, req.analyst_firm, req.report_date_hint)
    for rec in recs:
        await asyncio.to_thread(_save_recommendation, rec)

    return AnalystExtractOut(recommendations=recs, meta={"source": "text"})


@router.post("/analyst-report/pdf", response_model=AnalystExtractOut)
async def analyst_report_pdf(
    pdf: UploadFile = File(...),
    analyst_firm: str = Form(""),
    report_date_hint: str = Form(""),
):
    ext = Path(pdf.filename or "input.pdf").suffix.lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_bytes = await pdf.read()
        if not temp_bytes:
            raise HTTPException(status_code=400, detail="pdf file is empty")
        temp_pdf.write(temp_bytes)
        temp_pdf_path = temp_pdf.name

    try:
        text = await asyncio.to_thread(_extract_text_from_pdf, temp_pdf_path)
        recs = extract_analyst_recommendations(text, analyst_firm or None, report_date_hint or None)
        for rec in recs:
            await asyncio.to_thread(_save_recommendation, rec, pdf.filename)

        return AnalystExtractOut(
            recommendations=recs,
            meta={"source": "pdf", "file_name": pdf.filename},
        )
    finally:
        try:
            os.remove(temp_pdf_path)
        except OSError:
            pass


@router.get("/analyst-consensus/{ticker}", response_model=ConsensusOut)
async def analyst_consensus(ticker: str):
    consensus = await asyncio.to_thread(get_consensus, ticker)
    if consensus is None:
        raise HTTPException(status_code=404, detail=f"No analyst reports found for ticker {ticker}")
    return consensus
