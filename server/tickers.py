"""
종목명 <-> 코드 매핑 (analyst.py, callcenter.py 공용).

실시간 시세/종목마스터 API 연동 없이 정적 시드 파일(data/tickers_kr.json)만
사용하는 간단한 사전이다. 실제 운영에서는 KRX 등 공식 종목마스터로 교체/확장 필요.
"""
import json
import re
from pathlib import Path
from typing import List, Optional

_TICKERS_PATH = Path(__file__).parent / "data" / "tickers_kr.json"
_CODE_RE = re.compile(r"\b\d{6}\b")


def _load_name_to_code() -> dict:
    with open(_TICKERS_PATH, encoding="utf-8") as f:
        return json.load(f)


_NAME_TO_CODE = _load_name_to_code()
_CODE_SET = set(_NAME_TO_CODE.values())
_CODE_TO_NAME = {code: name for name, code in _NAME_TO_CODE.items()}


def extract_tickers(text: str) -> List[str]:
    found = set()
    for code in _CODE_RE.findall(text):
        if code in _CODE_SET:
            found.add(code)
    for name, code in _NAME_TO_CODE.items():
        if name in text:
            found.add(code)
    return sorted(found)


def code_to_name(code: str) -> Optional[str]:
    return _CODE_TO_NAME.get(code)
