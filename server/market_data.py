"""
환율(KRW 기준) / 주요국 주가지수 조회 (무료 공개 API, 키 불필요).

- 환율: open.er-api.com (exchangerate-api.com의 무료 공개 미러)
- 주가지수: Yahoo Finance 비공식 chart 엔드포인트

주의: 두 API 모두 문서화된 공식 계약이 아니며(특히 Yahoo chart 엔드포인트는
비공식/undocumented), 사전 통보 없이 응답 형식이 바뀌거나 접근이 막힐 수 있다.
이 모듈은 개별 항목 실패가 전체 응답을 깨지 않도록 방어적으로 동작한다.
"""
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

FX_URL = "https://open.er-api.com/v6/latest/USD"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
_USER_AGENT = "Mozilla/5.0 (compatible; meeting-agent-market-data/1.0)"

CACHE_TTL_SEC = 300

# (country_label, currency_code_or_None, index_symbol, index_label)
COUNTRIES: List[tuple] = [
    ("한국", None, "^KS11", "KOSPI"),
    ("한국", None, "^KQ11", "KOSDAQ"),
    ("미국", "USD", "^GSPC", "S&P500"),
    ("일본", "JPY", "^N225", "Nikkei225"),
    ("중국", "CNY", "000001.SS", "상하이종합"),
    ("유럽(독일)", "EUR", "^GDAXI", "DAX"),
    ("영국", "GBP", "^FTSE", "FTSE100"),
    ("홍콩", "HKD", "^HSI", "항셍지수"),
]

# JPY는 관례상 100엔 기준으로 표기
_UNIT_SCALE = {"JPY": 100}

_cache: Dict[str, Any] = {"expires_at": 0.0, "payload": None}


def _http_get_json(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _fetch_fx_rates() -> Dict[str, float]:
    data = _http_get_json(FX_URL)
    rates = data.get("rates") or {}
    if "USD" not in rates or "KRW" not in rates:
        raise ValueError("unexpected FX response shape")
    return rates


def _build_fx_rows(rates: Dict[str, float]) -> List[Dict[str, Any]]:
    rows = []
    seen_currencies = {code for _label, code, _sym, _idx in COUNTRIES if code}
    for currency in sorted(seen_currencies):
        country = next(label for label, code, _sym, _idx in COUNTRIES if code == currency)
        row: Dict[str, Any] = {"country": country, "currency": currency}
        try:
            krw_per_unit = rates["KRW"] / rates[currency]
            scale = _UNIT_SCALE.get(currency, 1)
            row["unit_label"] = f"{scale} {currency}"
            row["krw_per_unit"] = round(krw_per_unit * scale, 2)
        except Exception as exc:  # noqa: BLE001 - 개별 통화 실패는 이 행만 에러 처리
            row["error"] = str(exc)
        rows.append(row)
    return rows


def _fetch_index(symbol: str) -> Dict[str, Any]:
    data = _http_get_json(YAHOO_CHART_URL.format(symbol=symbol))
    result = (data.get("chart") or {}).get("result") or []
    if not result:
        raise ValueError("empty chart result")
    meta = result[0].get("meta") or {}
    price = meta.get("regularMarketPrice")
    prev_close = meta.get("chartPreviousClose")
    if price is None or prev_close is None:
        raise ValueError("missing price fields")
    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else None
    return {
        "price": price,
        "prev_close": prev_close,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "currency": meta.get("currency"),
        "market_time": meta.get("regularMarketTime"),
    }


def _build_index_rows() -> List[Dict[str, Any]]:
    rows = []
    for country, _currency, symbol, index_label in COUNTRIES:
        row: Dict[str, Any] = {"country": country, "index_label": index_label, "symbol": symbol}
        try:
            row.update(_fetch_index(symbol))
        except Exception as exc:  # noqa: BLE001 - 개별 지수 실패는 이 행만 에러 처리
            row["error"] = str(exc)
        rows.append(row)
    return rows


def get_market_overview(force_refresh: bool = False) -> Dict[str, Any]:
    now = time.time()
    if not force_refresh and _cache["payload"] is not None and now < _cache["expires_at"]:
        return _cache["payload"]

    try:
        rates = _fetch_fx_rates()
        fx_rows = _build_fx_rows(rates)
        fx_error = None
    except Exception as exc:  # noqa: BLE001
        fx_rows = []
        fx_error = str(exc)

    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "fx": fx_rows,
        "fx_error": fx_error,
        "indices": _build_index_rows(),
    }

    _cache["payload"] = payload
    _cache["expires_at"] = now + CACHE_TTL_SEC
    return payload
