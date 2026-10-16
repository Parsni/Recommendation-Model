"""Layer 1: Ingestion & Preprocessing Layer - Request Sanitizer.

Accepts inbound user prompt payloads, strips markdown/prompt injection artifacts,
evaluates payload bounds (payload size, character length), and flags structural compliance.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SanitizedRequest:
    """Validated and sanitized inbound request payload."""
    user_id: str
    original_prompt: str
    sanitized_prompt: str
    char_length: int
    byte_size: int
    is_compliant: bool
    violations: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequestSanitizer:
    """Enterprise-grade request sanitizer and bounds evaluator."""

    # Patterns for markdown injection, HTML script tags, and prompt injection delimiters
    DANGEROUS_PATTERNS = [
        (re.compile(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", re.IGNORECASE | re.DOTALL), "[STRIPPED_SCRIPT]"),
        (re.compile(r"<\s*iframe[^>]*>.*?<\s*/\s*iframe\s*>", re.IGNORECASE | re.DOTALL), "[STRIPPED_IFRAME]"),
        (re.compile(r"!\[.*?\]\(javascript:.*?\)", re.IGNORECASE), "[STRIPPED_MD_XSS]"),
        (re.compile(r"\[.*?\]\(javascript:.*?\)", re.IGNORECASE), "[STRIPPED_MD_LINK]"),
        (re.compile(r"(?i)\b(system\s*override|ignore\s*previous\s*instructions|ignore\s*all\s*prior)\b"), "[FLAGGED_PROMPT_INJECTION]"),
    ]

    def __init__(
        self,
        min_chars: int = 3,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB bound
        strip_markdown: bool = True
    ) -> None:
        self.min_chars = min_chars
        self.max_bytes = max_bytes
        self.strip_markdown = strip_markdown

    def sanitize(self, payload: Dict[str, Any]) -> SanitizedRequest:
        """Validate and sanitize an inbound JSON request.
        
        Args:
            payload: Dict containing at least 'prompt', and optionally 'user_id' and 'metadata'.
            
        Returns:
            SanitizedRequest with sanitization details and compliance status.
        """
        violations: list[str] = []
        
        # 1. Validate structure and user_id
        if not isinstance(payload, dict):
            return SanitizedRequest(
                user_id="anonymous",
                original_prompt="",
                sanitized_prompt="",
                char_length=0,
                byte_size=0,
                is_compliant=False,
                violations=["Payload must be a JSON dictionary."]
            )

        raw_user_id = str(payload.get("user_id", "anonymous")).strip()
        user_id = re.sub(r"[^\w\-\.@]", "_", raw_user_id)[:64] or "anonymous"

        raw_prompt = payload.get("prompt")
        if raw_prompt is None:
            return SanitizedRequest(
                user_id=user_id,
                original_prompt="",
                sanitized_prompt="",
                char_length=0,
                byte_size=0,
                is_compliant=False,
                violations=["Missing required 'prompt' field."]
            )

        if not isinstance(raw_prompt, str):
            raw_prompt = str(raw_prompt)

        # 2. Check byte size bounds (rejecting multi-megabyte payloads)
        byte_size = len(raw_prompt.encode("utf-8", errors="replace"))
        if byte_size > self.max_bytes:
            violations.append(f"Payload size ({byte_size} bytes) exceeds maximum limit ({self.max_bytes} bytes).")
            return SanitizedRequest(
                user_id=user_id,
                original_prompt=raw_prompt[:500] + "... [TRUNCATED]",
                sanitized_prompt="",
                char_length=len(raw_prompt),
                byte_size=byte_size,
                is_compliant=False,
                violations=violations
            )

        # 3. Unicode normalization and strip control characters
        normalized = unicodedata.normalize("NFKC", raw_prompt)
        # Strip null bytes and non-printable characters except newlines/tabs
        cleaned = "".join(ch for ch in normalized if ch in "\n\r\t" or unicodedata.category(ch)[0] != "C")

        # 4. Strip dangerous markdown and injection patterns
        sanitized = cleaned
        if self.strip_markdown:
            for pattern, replacement in self.DANGEROUS_PATTERNS:
                if pattern.search(sanitized):
                    violations.append(f"Sanitizer detected and neutralized suspicious pattern: {pattern.pattern}")
                    sanitized = pattern.sub(replacement, sanitized)

        # Clean excess whitespace
        sanitized = sanitized.strip()
        char_length = len(sanitized)

        # 5. Evaluate basic bounds
        if char_length < self.min_chars:
            violations.append(f"Prompt length ({char_length} chars) is below minimum threshold ({self.min_chars}).")

        is_compliant = len(violations) == 0 or all("neutralized" in v for v in violations)
        # If the prompt itself is empty or too small, it is not compliant
        if char_length < self.min_chars:
            is_compliant = False

        return SanitizedRequest(
            user_id=user_id,
            original_prompt=raw_prompt,
            sanitized_prompt=sanitized,
            char_length=char_length,
            byte_size=byte_size,
            is_compliant=is_compliant,
            violations=violations,
            metadata=payload.get("metadata", {})
        )
