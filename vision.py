# vision.py - Gemini Vision Semantic Image & Diagram Summarizer
import os
import sys
from pathlib import Path
from typing import Optional

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config


VISION_SYSTEM_PROMPT = """You are a technical document analyst inspecting an embedded figure, diagram, chart, or table from a document.
Provide a clear, detailed, and highly searchable textual description of this image.
Include:
1. Core Subject: What this image represents (architecture diagram, flow chart, financial graph, UI screenshot, data table, etc.).
2. Key Data & Labels: Specific numbers, metrics, axis labels, legends, entities, and flow connections visible.
3. Summary / Key Takeaway: The primary insight, trend, or system flow depicted.
Keep the explanation objective, factual, and compact (2-4 paragraphs). Do not invent details not visible."""


def describe_image(
    image_bytes: bytes,
    mime_type: str = "image/png",
    source_hint: Optional[str] = None
) -> str:
    """
    Sends an image to Gemini Vision API and returns an in-depth semantic description.
    Gracefully catches errors to ensure ingestion pipeline never crashes.
    """
    if not config.GEMINI_API_KEY:
        print("[Vision] Warning: GEMINI_API_KEY not configured. Skipping vision description.")
        return f"[Image: {source_hint or 'Embedded Visual Asset'} - Vision API key not provided]"

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.GEMINI_API_KEY)

        prompt = VISION_SYSTEM_PROMPT
        if source_hint:
            prompt += f"\n\nContext Hint: This visual appears in document '{source_hint}'."

        response = client.models.generate_content(
            model=config.VISION_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt
            ]
        )

        desc = response.text.strip() if response.text else ""
        if desc:
            return desc
        return f"[Image: {source_hint or 'Embedded Visual Asset'} - No visual description generated]"

    except Exception as e:
        print(f"[Vision] Error generating description for image ({source_hint}): {e}")
        return f"[Image from {source_hint or 'document'}: Visual asset extracted, description unavailable ({e})]"
