"""
공유 Postgres(Docker) 접속 헬퍼.

analyst.py(애널리스트 추천 컨센서스), callcenter.py(콜 QA 이력)가 사용하는
테이블을 이 모듈이 초기화한다. 여러 랩 프로젝트가 같은 Postgres 인스턴스를
공유하므로, 기본 `postgres` DB에 테이블을 얹지 않고 전용 DB(`meeting_agent`)를
따로 만들어 사용한다.
"""
import logging
import os
from contextlib import contextmanager

import psycopg2

logger = logging.getLogger("meeting_agent.db")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_MAINTENANCE_DB = os.getenv("POSTGRES_MAINTENANCE_DB", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "meeting_agent")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analyst_recommendations (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    company_name TEXT,
    analyst_firm TEXT,
    opinion TEXT,
    target_price NUMERIC,
    prior_target_price NUMERIC,
    rationale JSONB,
    report_date DATE,
    source_file TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_analyst_reco_ticker ON analyst_recommendations (ticker);

CREATE TABLE IF NOT EXISTS call_summaries (
    id SERIAL PRIMARY KEY,
    agent_name TEXT,
    call_date DATE,
    inquiry_type TEXT,
    order_requested BOOLEAN,
    order_detail TEXT,
    tickers_mentioned JSONB,
    sentiment TEXT,
    sentiment_score NUMERIC,
    compliance_flags JSONB,
    risk_level TEXT,
    transcript TEXT,
    mp3_file_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_call_summaries_risk ON call_summaries (risk_level);
"""


def _connect(dbname: str):
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=dbname,
    )


def _ensure_database_exists():
    conn = _connect(POSTGRES_MAINTENANCE_DB)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (POSTGRES_DB,))
            if not cur.fetchone():
                cur.execute(f'CREATE DATABASE "{POSTGRES_DB}"')
    finally:
        conn.close()


@contextmanager
def get_conn():
    conn = _connect(POSTGRES_DB)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """앱 시작 시 1회 호출. Postgres에 연결할 수 없으면 경고만 남기고 넘어간다.

    (핵심 회의록 기능은 DB 없이도 동작해야 하므로, 신규 기능의 DB 연결 실패가
    전체 서버 기동을 막아서는 안 된다.)
    """
    try:
        _ensure_database_exists()
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_SQL)
    except Exception as exc:
        logger.warning(
            "Postgres 초기화 실패 (애널리스트/콜QA 기능은 비활성화됨): %s", exc
        )
