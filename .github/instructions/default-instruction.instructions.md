---
description: 'Enforce English for all generated code and code comments, regardless of the language used in the request, unless the user explicitly specifies a different response language.'
applyTo: '**'
---

# Language Guidelines

## Code and Comments

- Always write generated code, identifiers (variable/function/class names), and inline comments in English.
- This applies even if the user's question, prompt, or surrounding conversation is in another language (e.g. Chinese, French).
- Do not translate existing English code comments into another language unless explicitly asked to.

## Natural-Language Responses

- Respond to the user in the same language they used to ask the question.
- If the user explicitly requests a specific response language (e.g. "answer in English" or "回答用中文"), follow that instruction instead of mirroring the question's language.

## Exceptions

- If the user explicitly asks for comments or code in a specific non-English language, follow that explicit instruction for that request only; revert to English afterward unless told otherwise.
- Documentation files (e.g. README) follow the language explicitly requested by the user for that file, defaulting to English if unspecified.