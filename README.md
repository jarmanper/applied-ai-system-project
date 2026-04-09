## Reflection (TA prep)

*Notes from running through the project before students.*

**Core concept students needed to understand:**  
I want them to leave knowing that documentation Q&A is not "just ask the model." **Context is engineered:** naive mode dumps the **entire** corpus (cost, limits, noise); retrieval-only shows **which evidence** the system would surface without generation; RAG pairs **targeted snippets** with Gemini so answers are easier to check against sources. I'll push them to connect **retrieval choices** (chunking, scoring, stopwords, top-k in [docubot.py](docubot.py)) to **answer quality**, and to treat a clean **"I do not know"** as a win when the docs do not support a claim.

**Where students may struggle:**  
I'm expecting friction when the **top chunk looks wrong** but scores high (token overlap vs real relevance), when they tune **chunk boundaries** or **scoring** and accidentally break behaviors like **one chunk per file**, and when **fluent model output** hides a **retrieval** mistake. I'll flag that [evaluation.py](evaluation.py) is a **coarse** sanity check, not ground truth. On my prep run I'll double-check **`.env` / `GEMINI_API_KEY`** so "mode 1 vs 3" issues in lab are not silently **missing-key** problems.

**Where AI was helpful vs misleading:**  
It was useful for me to **sanity-check** RAG vs full-context tradeoffs, brainstorm **test queries**, and reason about **precision/recall** in this intentionally minimal setup. It steered wrong when it suggested **vector DBs / big frameworks** that skip the point of the hand-built indexer, or when it implied **prompt tweaks alone** fix bad top-k without walking the **retrieval trace** first.

**One way you'd guide a student without giving the answer:**  
I'd have them pick **one query**, run **all three modes** on the same docs, and write down **what text reached the model** each time. Then we'd walk **which files and chunks** ranked and **which query tokens** drove the score, form one **hypothesis** (synonym gap, chunk split, scoring guardrail), and change **a single knob** in [docubot.py](docubot.py) or [llm_client.py](llm_client.py) before re-running.

---

# DocuBot

DocuBot is a small documentation assistant that helps answer developer questions about a codebase.  
It can operate in three different modes:

1. **Naive LLM mode**  
   Sends the entire documentation corpus to a Gemini model and asks it to answer the question.

2. **Retrieval only mode**  
   Uses a simple indexing and scoring system to retrieve relevant snippets without calling an LLM.

3. **RAG mode (Retrieval Augmented Generation)**  
   Retrieves relevant snippets, then asks Gemini to answer using only those snippets.

The docs folder contains realistic developer documents (API reference, authentication notes, database notes), but these files are **just text**. They support retrieval experiments and do not require students to set up any backend systems.

---

## Setup

### 1. Install Python dependencies

    pip install -r requirements.txt

### 2. Configure environment variables

Copy the example file:

    cp .env.example .env

Then edit `.env` to include your Gemini API key:

    GEMINI_API_KEY=your_api_key_here

If you do not set a Gemini key, you can still run retrieval only mode.

---

## Running DocuBot

Start the program:

    python main.py

Choose a mode:

- **1**: Naive LLM (Gemini reads the full docs)  
- **2**: Retrieval only (no LLM)  
- **3**: RAG (retrieval + Gemini)

You can use built in sample queries or type your own.

---

## Running Retrieval Evaluation (optional)

    python evaluation.py

This prints simple retrieval hit rates for sample queries.

---

## Modifying the Project

You will primarily work in:

- `docubot.py`  
  Implement or improve the retrieval index, scoring, and snippet selection.

- `llm_client.py`  
  Adjust the prompts and behavior of LLM responses.

- `dataset.py`  
  Add or change sample queries for testing.

---

## Requirements

- Python 3.9+
- A Gemini API key for LLM features (only needed for modes 1 and 3)
- No database, no server setup, no external services besides LLM calls
