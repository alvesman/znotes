"""Prompt construction for grounded answers (PLAN.md §9 step 7).

The system prompt carries the answering rules (adapted for citations and
current-versus-historical documents); the user message carries the assembled
context, the question and a language directive. Two rules are conditional on
the context shape: the table rule when the bundle contains a table, and the
completeness rule for large contexts.
"""

from __future__ import annotations

from datetime import datetime

from mynotes.llm.chat import ChatMessage
from mynotes.retrieval.assemble import ContextBundle

_PROMPT_INTRO = (
    "You answer questions about the user's notes using the context provided below. Rules:"
)

_BASE_RULES: tuple[str, ...] = (
    "1. Answer using ONLY the context. If it doesn't contain the answer, say briefly that the "
    "information isn't in the notes. Never invent.",
    "2. Answer exactly what is asked; don't add unrequested warnings, procedures or background.",
    "3. The context may contain several documents; use only the relevant ones and don't mix "
    "details from different documents or dates.",
    "4. Reproduce critical values exactly (numbers, dates, names, units, qualifiers such as "
    '"at least" or "up to"). When the answer is a list or a set of values, give every item '
    "the source lists for what is asked, each as written, and nothing from other topics or "
    "sections. Copy identifiers in full (prefix, name, version, extension), never a fragment.",
    '5. Prefer information from current documents. Context groups marked "historical" are '
    "older versions: use them only if the question asks about history or changes, and then "
    "say how and when it changed.",
    "6. When sources disagree, prefer the most recent current document; mention the "
    "disagreement only if relevant.",
    "7. Cite the sources you use with their bracket numbers from the context headers, e.g. "
    "`[1]` or `[2][3]`, placed right after the sentence they support. Don't mention file names "
    "or page numbers in the text.",
    "8. Complete sentences; no cut-off endings.",
)

SYSTEM_PROMPT = _PROMPT_INTRO + "\n" + "\n".join(_BASE_RULES)

_TABLE_RULE = (
    "9. The context contains tables; when asked about their contents, reproduce the relevant "
    "values exactly, keeping numbers and units."
)

_DETAILS_RULE_TOKENS = 200
"""Contexts above this many tokens get the completeness rule."""

_DETAILS_RULE = (
    "10. If a complete answer needs several details (steps, settings, quantities), include all "
    "of them."
)

_AS_OF_RULE = (
    "11. The question asks for the state of things as of {date}. Answer with the information "
    "that was valid on that date: use the sources dated on or before it (each context header "
    "shows the source date), even when they are marked historical, and ignore later values "
    "except to note that they changed afterwards."
)
"""Added when the question carries an ``as-of:`` date (history intent)."""

LANGUAGE_DIRECTIVES: dict[str, str] = {
    "pt": "Responde em português europeu.",
    "en": "Answer in English.",
}
"""The per-language directive appended to the user message."""

_DEFAULT_LANGUAGE_DIRECTIVE = "Answer in the same language as the question."

NOT_FOUND: dict[str, str] = {
    "pt": "Não encontrei essa informação nas tuas notas.",
    "en": "I couldn't find that information in your notes.",
}
"""The gated answer when nothing relevant was retrieved, per language."""


def language_directive(language: str | None) -> str:
    """The user-message directive that pins the answer language.

    Args:
        language: The detected question language, or None when unknown.

    Returns:
        The directive for a known language, else the "same language" default.
    """
    if language is None:
        return _DEFAULT_LANGUAGE_DIRECTIVE
    return LANGUAGE_DIRECTIVES.get(language, _DEFAULT_LANGUAGE_DIRECTIVE)


def build_messages(
    question: str,
    bundle: ContextBundle,
    language: str | None,
    *,
    as_of: datetime | None = None,
) -> list[ChatMessage]:
    """Build the chat messages for one grounded answer.

    The system message holds the base rules plus the table rule when the
    context contains a table, the completeness rule when it exceeds 200
    tokens and the as-of rule when the question names a date. The user message
    holds the context, the question and the language directive.

    Args:
        question: The user's question as typed (inline filter tokens stay in
            the prompt; only the retrieval text is stripped of them).
        bundle: The assembled context from :func:`assemble_context`.
        language: The question language for the answer directive.
        as_of: The question's ``as-of:`` date, when it has one.

    Returns:
        One system message followed by one user message.
    """
    rules = [SYSTEM_PROMPT]
    if bundle.has_table:
        rules.append(_TABLE_RULE)
    if bundle.tokens > _DETAILS_RULE_TOKENS:
        rules.append(_DETAILS_RULE)
    if as_of is not None:
        rules.append(_AS_OF_RULE.format(date=f"{as_of:%Y-%m-%d}"))
    system = "\n".join(rules)
    user = f"Context:\n{bundle.text}\n\nQuestion: {question}\n{language_directive(language)}"
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
