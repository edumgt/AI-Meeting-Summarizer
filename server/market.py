"""
글로벌 마켓 대시보드 라우터 (환율 / 주요국 주가지수).

세계 시간대는 백엔드 호출 없이 프론트엔드(Intl.DateTimeFormat)에서 계산하므로
이 라우터는 다루지 않는다.
"""
from fastapi import APIRouter, Query

from market_data import get_market_overview

router = APIRouter()


@router.get("/market-overview")
def market_overview(refresh: bool = Query(False, description="true면 캐시를 무시하고 즉시 재조회")):
    return get_market_overview(force_refresh=refresh)
