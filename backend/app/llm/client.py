"""
LLM client with dual-provider support: Google Gemini (primary) + Groq (fallback).
Both are free-tier APIs. If both fail, returns a structured fallback message.
"""
import logging
import asyncio
from typing import Optional

import google.generativeai as genai

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

# Configure Gemini on import
if settings.google_api_key:
    genai.configure(api_key=settings.google_api_key)

_gemini_model = None


def _get_gemini_model():
    global _gemini_model
    if _gemini_model is None and settings.google_api_key:
        _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    return _gemini_model


async def call_gemini(prompt: str, timeout: float = 30.0) -> Optional[str]:
    """Call Google Gemini API. Returns response text or None on failure."""
    model = _get_gemini_model()
    if model is None:
        log.warning("Gemini model not configured (missing GOOGLE_API_KEY)")
        return None

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=timeout,
        )
        if response and response.text:
            log.info("Gemini responded (%d chars)", len(response.text))
            return response.text
        log.warning("Gemini returned empty response")
        return None
    except asyncio.TimeoutError:
        log.error("Gemini call timed out after %.0fs", timeout)
        return None
    except Exception as e:
        log.error("Gemini API error: %s", e)
        return None


async def call_groq(prompt: str, timeout: float = 30.0) -> Optional[str]:
    """Call Groq API as fallback. Returns response text or None on failure."""
    if not settings.groq_api_key:
        return None

    try:
        # Lazy import — only loaded if Groq is actually needed
        from groq import Groq

        def _call():
            client = Groq(api_key=settings.groq_api_key)
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3,
            )
            return response.choices[0].message.content

        result = await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout)
        if result:
            log.info("Groq responded (%d chars)", len(result))
        return result
    except asyncio.TimeoutError:
        log.error("Groq call timed out after %.0fs", timeout)
        return None
    except ImportError:
        log.warning("groq package not installed — skipping Groq fallback")
        return None
    except Exception as e:
        log.error("Groq API error: %s", e)
        return None


async def generate_report(prompt: str) -> tuple[Optional[str], str]:
    """
    Try Gemini first, fall back to Groq, then to a canned fallback.
    Returns (response_text, provider_name).
    """
    # Primary: Gemini
    text = await call_gemini(prompt)
    if text:
        return text, "gemini"

    # Fallback: Groq
    log.info("Falling back to Groq for report generation")
    text = await call_groq(prompt)
    if text:
        return text, "groq"

    log.warning("All LLM providers failed — using fallback message")
    return None, "fallback"
