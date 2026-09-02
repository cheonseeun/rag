"""
Gemini 2.5 Pro 호출 래퍼: 디스크 캐시 + 지수 백오프 재시도 + JSON 파싱.

캐시가 있으면 중간에 죽어도 이어서 돌릴 수 있습니다.
7개 모델 x 쿼리 수 x 3콜(생성+분해+검증)이라 캐시가 없으면 재실행이 아픕니다.

  pip install google-genai
  export GOOGLE_API_KEY=...
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from config import CACHE_DIR, ROOT

# 프로젝트 루트의 .env를 자동 로드 -> 터미널에서 키를 칠 필요가 없습니다.
load_dotenv(ROOT / ".env")

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY 환경변수를 설정하세요.")
        _client = genai.Client(api_key=key)
    return _client


def _cache_path(model, system, prompt, as_json):
    h = hashlib.sha256(
        json.dumps([model, system, prompt, as_json], ensure_ascii=False).encode()
    ).hexdigest()[:24]
    d = CACHE_DIR / model.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.json"


def call(prompt: str, model: str, system: str = "", as_json: bool = False,
         temperature: float = 0.0, max_retries: int = 5, use_cache: bool = True):
    """as_json=True면 dict/list를 반환, 아니면 문자열."""
    cp = _cache_path(model, system, prompt, as_json)
    if use_cache and cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))["result"]

    def _make_config(full: bool):
        """full=True면 temperature/thinking까지, False면 최소 설정만.

        Gemini 3.x부터 temperature 등 샘플링 파라미터가 deprecated 되어
        모델 세대별로 받는 인자가 다릅니다. 실패하면 최소 설정으로 재시도합니다.
        """
        kw = {
            "system_instruction": system or None,
            "response_mime_type": "application/json" if as_json else "text/plain",
        }
        if full:
            kw["temperature"] = temperature
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=128)
        return types.GenerateContentConfig(**kw)

    last = None
    for attempt in range(max_retries):
        for full in (True, False):
            try:
                resp = client().models.generate_content(
                    model=model, contents=prompt, config=_make_config(full))
                text = (resp.text or "").strip()
                if not text:
                    raise ValueError("빈 응답 (safety block 가능)")
                result = _parse_json(text) if as_json else text
                if use_cache:
                    cp.write_text(json.dumps({"result": result}, ensure_ascii=False),
                                  encoding="utf-8")
                return result
            except Exception as e:
                last = e
                msg = str(e)
                # 파라미터 문제면 최소 설정으로 즉시 재시도
                if full and any(s in msg for s in (
                        "temperature", "thinking", "INVALID_ARGUMENT",
                        "Unknown field", "not supported")):
                    print("  파라미터 비호환 -> 최소 설정으로 재시도")
                    continue
                break

        msg = str(last)
        fatal = any(s in msg for s in (
            "API_KEY_INVALID", "API key not valid", "PERMISSION_DENIED",
            "NOT_FOUND", "no longer available", "limit: 0",
        ))
        if fatal or attempt == max_retries - 1:
            raise RuntimeError(f"Gemini 호출 실패 ({type(last).__name__}): {msg}") from last
        wait = min(2 ** attempt + random.random(), 30)
        print(f"  retry {attempt + 1}/{max_retries} ({msg[:150]}) {wait:.1f}s")
        time.sleep(wait)
    raise RuntimeError(f"Gemini 호출 실패: {last}")


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    return json.loads(text.strip())