# Agentic RAG with Anthropic

This example shows a minimal Agentic RAG loop using Anthropic tool use.

## What it demonstrates

Instead of forcing retrieval before every answer, the model can:

- decide whether to search
- write its own search query
- search multiple times
- stop when it has enough information

That is the core idea of Agentic RAG.

## Files

- `agentic_rag_anthropic.py`
- `requirements.txt`

## Setup

Export your API key:

```bash
export ANTHROPIC_API_KEY="your_key_here"
