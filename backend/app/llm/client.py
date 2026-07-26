"""
LLM client with dual-provider support: Groq (primary, ultra-fast Llama 3.3 70B) + Google Gemini (fallback).
Both are free-tier APIs. If both fail, returns a structured fallback message.
"""
import logging
import asyncio
from typing import Optional

import google.generativeai as genai

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

# Configure Gemini if key is provided
if settings.google_api_key:
    try:
        genai.configure(api_key=settings.google_api_key)
    except Exception as e:
        log.warning("Gemini configuration error: %s", e)


async def call_groq(prompt: str, timeout: float = 30.0) -> Optional[str]:
    """Call Groq API using Llama 3.3 70B versatile. Returns response text or None on failure."""
    if not settings.groq_api_key:
        log.warning("Groq not configured (missing GROQ_API_KEY)")
        return None

    try:
        from groq import Groq

        def _call():
            client = Groq(api_key=settings.groq_api_key)
            # Try active Groq model names
            for model_name in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192"]:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2000,
                        temperature=0.3,
                    )
                    if response.choices and response.choices[0].message.content:
                        log.info("Groq (%s) responded successfully", model_name)
                        return response.choices[0].message.content
                except Exception as err:
                    log.warning("Groq model %s error: %s", model_name, err)
                    continue
            return None

        result = await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout)
        return result
    except asyncio.TimeoutError:
        log.error("Groq call timed out after %.0fs", timeout)
        return None
    except ImportError:
        log.warning("groq package not installed — skipping Groq")
        return None
    except Exception as e:
        log.error("Groq API error: %s", e)
        return None


async def call_gemini(prompt: str, timeout: float = 30.0) -> Optional[str]:
    """Call Google Gemini API with multi-model fallback. Returns response text or None on failure."""
    if not settings.google_api_key:
        return None

    try:
        def _call():
            for model_name in ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    res = model.generate_content(prompt)
                    if res and res.text:
                        log.info("Gemini (%s) responded (%d chars)", model_name, len(res.text))
                        return res.text
                except Exception as err:
                    log.warning("Gemini model %s error: %s", model_name, err)
                    continue
            return None

        response_text = await asyncio.wait_for(
            asyncio.to_thread(_call),
            timeout=timeout,
        )
        return response_text
    except asyncio.TimeoutError:
        log.error("Gemini call timed out after %.0fs", timeout)
        return None
    except Exception as e:
        log.error("Gemini API error: %s", e)
        return None


async def generate_report(prompt: str) -> tuple[Optional[str], str]:
    """
    Try Groq first (fastest, high rate limit), fall back to Gemini, then to canned fallback.
    Returns (response_text, provider_name).
    """
    # Primary: Groq (Llama 3.3 70B)
    text = await call_groq(prompt)
    if text:
        return text, "groq"

    # Secondary: Gemini
    log.info("Falling back to Gemini for report generation")
    text = await call_gemini(prompt)
    if text:
        return text, "gemini"

    log.warning("All LLM providers failed — using fallback message")
    return None, "fallback"
