"""
공용 오디오/텍스트 처리 헬퍼 (STT 변환, 요약 API 호출, PDF 텍스트 추출).

app.py, analyst.py, callcenter.py가 공통으로 재사용한다.
순환 임포트를 피하기 위해 이 모듈은 app.py를 임포트하지 않는다.
"""
import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from openai import OpenAI

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except ImportError:
        PdfReader = None  # type: ignore


CLS_MODEL_ID = os.getenv("CLS_MODEL_ID", "Seonghaa/korean-emotion-classifier-roberta")
CLS_MAX_CONCURRENCY = int(os.getenv("CLS_MAX_CONCURRENCY", "2"))

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
TRANSCRIBE_LANGUAGE = os.getenv("TRANSCRIBE_LANGUAGE", "ko")
MP3_OUTPUT_DIR = Path(os.getenv("MP3_OUTPUT_DIR", "/tmp/ai_meeting_audio"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11435")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")

# 감정분류 모델(및 로딩 시점 세팅용) - app.py의 _startup()에서 load_classifier() 호출로 채워짐
classifier = None
cls_sem = asyncio.Semaphore(CLS_MAX_CONCURRENCY)

openai_client: Optional[OpenAI] = None
ollama_client: Optional[OpenAI] = None


def load_classifier():
    global classifier
    from transformers import pipeline as hf_pipeline

    classifier = hf_pipeline("text-classification", model=CLS_MODEL_ID, device=-1)


def _get_openai_client() -> OpenAI:
    global openai_client
    if openai_client is not None:
        return openai_client

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")

    openai_client = OpenAI(api_key=api_key)
    return openai_client


def _get_ollama_client() -> OpenAI:
    global ollama_client
    if ollama_client is not None:
        return ollama_client
    ollama_client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
    return ollama_client


def _convert_audio_to_mp3(input_path: str, output_path: str):
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise HTTPException(status_code=500, detail="ffmpeg is not installed on server.")

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "128k",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Audio conversion failed: {result.stderr[-500:]}",
        )


def _transcribe_mp3(mp3_path: str, language: str) -> str:
    client = _get_openai_client()
    with open(mp3_path, "rb") as fp:
        text = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=fp,
            language=language or TRANSCRIBE_LANGUAGE,
            response_format="text",
        )
    if isinstance(text, str):
        return text.strip()
    return str(text).strip()


def _summarize_with_ollama(text: str, max_length: int, min_length: int, model: Optional[str] = None) -> str:
    client = _get_ollama_client()
    use_model = model or OLLAMA_MODEL
    prompt = (
        "다음 내용을 한국어로 간결하게 요약하세요. "
        "중요 결정사항, 핵심 이슈, 후속 조치가 드러나야 합니다. "
        f"분량은 대략 {min_length}자 이상 {max_length}자 이하로 맞추세요."
    )
    res = client.chat.completions.create(
        model=use_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "당신은 한국어 텍스트를 정리하는 실무 비서입니다. 요청한 분량에 맞게 요약만 출력하세요."},
            {"role": "user", "content": f"{prompt}\n\n[원문]\n{text}"},
        ],
    )
    return (res.choices[0].message.content or "").strip()


def _summarize_with_openai(text: str, max_length: int, min_length: int) -> str:
    client = _get_openai_client()
    prompt = (
        "다음 내용을 한국어로 간결하게 요약하세요. "
        "중요 결정사항, 핵심 이슈, 후속 조치가 드러나야 합니다. "
        f"분량은 대략 {min_length}자 이상 {max_length}자 이하로 맞추세요."
    )
    res = client.chat.completions.create(
        model=OPENAI_SUMMARY_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "당신은 한국어 텍스트를 정리하는 실무 비서입니다."},
            {"role": "user", "content": f"{prompt}\n\n[원문]\n{text}"},
        ],
    )
    return (res.choices[0].message.content or "").strip()


def _extract_text_from_pdf(pdf_path: str) -> str:
    if PdfReader is None:
        raise HTTPException(
            status_code=500,
            detail="PDF support is not installed on server. Install pypdf or PyPDF2.",
        )

    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"PDF open failed: {exc}") from exc

    pages = []
    for page in reader.pages:
        try:
            pages.append((page.extract_text() or "").strip())
        except Exception:
            continue

    text = "\n\n".join(part for part in pages if part).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")
    return text


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
