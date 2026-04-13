"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re

# Common English stopwords: improves scoring/guardrails (weak matches on "the", "what", …).
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "as",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "it",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "if",
        "than",
        "so",
        "too",
        "very",
        "just",
        "not",
        "no",
        "any",
        "each",
        "few",
        "more",
        "most",
        "some",
        "such",
        "all",
        "both",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "there",
        "here",
    }
)


def _tokenize(text):
    """Lowercase word tokens; strips punctuation at word boundaries."""
    return re.findall(r"\b\w+\b", text.lower())


def _substantive_tokens(query):
    """
    Query tokens used for relevance and guardrails.
    Falls back to all tokens when the query is only stopwords.
    """
    raw = _tokenize(query)
    substantive = [t for t in raw if t not in _STOPWORDS]
    return substantive if substantive else raw


# Tiny synonym/variant lists so obvious doc wording (e.g. "connection") still scores.
_TOKEN_VARIANTS = {
    "connect": ("connection", "connections", "connected", "connecting"),
    "list": ("lists",),
    "return": ("returns", "returned", "returning"),
    "generated": ("generate", "generates", "generation"),
}

# Count identifiers like generate_access_token when the query uses "generated".
_EXTRA_TOKEN_REGEX = {
    "generated": re.compile(r"\bgenerate\w*"),
}


def _evidence_score_threshold(query):
    """
    Minimum cumulative keyword score required on the best snippet to treat
    retrieval as grounded. Refuse when the best match is weak.
    """
    tokens = _substantive_tokens(query)
    if not tokens:
        return None
    n = len(tokens)
    return max(2, (n + 1) // 2)


def _split_into_chunks(text, max_chunk_chars=1400):
    """
    Split one document into smaller units: blank-line-separated paragraphs.
    Oversized blocks are sliced so no single chunk dominates retrieval.
    """
    parts = re.split(r"\n\s*\n", text.strip())
    raw = [p.strip() for p in parts if p.strip()]
    chunks = []
    for p in raw:
        if len(p) <= max_chunk_chars:
            chunks.append(p)
        else:
            for i in range(0, len(p), max_chunk_chars):
                chunks.append(p[i : i + max_chunk_chars])
    merged = []
    for c in chunks:
        if merged and len(c) < 60:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)
    return merged


class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Paragraph-level chunks for retrieval (filename, snippet text)
        self.chunks = self._documents_to_chunks(self.documents)
        # Inverted index: token -> sorted chunk indices
        self.index = self.build_index(self.chunks)

    def _documents_to_chunks(self, documents):
        out = []
        for filename, text in documents:
            for chunk in _split_into_chunks(text):
                out.append((filename, chunk))
        return out

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, chunks):
        """
        Build an inverted index mapping lowercase words to chunk indices
        (each chunk is one retrieval unit: a paragraph-sized snippet).

        Example for a token appearing in chunks from two files:
        { "token": [0, 4, 7] }
        """
        index_sets = {}
        for i, (_, text) in enumerate(chunks):
            for token in set(_tokenize(text)):
                index_sets.setdefault(token, set()).add(i)
        return {t: sorted(indices) for t, indices in index_sets.items()}

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Relevance score: sum of whole-word hit counts for substantive query tokens.
        """
        tokens = _substantive_tokens(query)
        if not tokens:
            return 0
        lowered = text.lower()
        total = 0
        for w in tokens:
            n = len(re.findall(r"\b" + re.escape(w) + r"\b", lowered))
            for v in _TOKEN_VARIANTS.get(w, ()):
                n += len(re.findall(r"\b" + re.escape(v) + r"\b", lowered))
            rx = _EXTRA_TOKEN_REGEX.get(w)
            if rx is not None:
                n += len(rx.findall(lowered))
            total += min(n, 4)
        return total

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select top_k relevant snippets.

        Returns a list of (filename, snippet_text) sorted by score descending.
        Empty list when there is insufficient evidence (guardrail).
        """
        threshold = _evidence_score_threshold(query)
        if threshold is None:
            return []

        query_tokens = _tokenize(query)
        candidates = set()
        for t in query_tokens:
            if t in self.index:
                candidates.update(self.index[t])
        if not candidates:
            candidates = set(range(len(self.chunks)))

        scored = []
        for i in candidates:
            filename, chunk_text = self.chunks[i]
            score = self.score_document(query, chunk_text)
            scored.append((score, filename, chunk_text))

        scored.sort(key=lambda x: (-x[0], x[1], x[2][:80]))

        if not scored or scored[0][0] < threshold:
            return []

        # One best chunk per file so top_k spans sources and stays easier to read.
        out = []
        used_files = set()
        for score, filename, chunk_text in scored:
            if score < threshold:
                break
            if filename in used_files:
                continue
            used_files.add(filename)
            out.append((filename, chunk_text))
            if len(out) >= top_k:
                break
        return out

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.

        Orchestration (draft + placeholder validation + logging) lives in
        ``rag_pipeline.run_rag_pipeline``.
        """
        from pipeline_models import UserQuery
        from rag_pipeline import run_rag_pipeline

        return run_rag_pipeline(self, UserQuery(text=query, top_k=top_k))

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
