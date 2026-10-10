# Credit Card RAG Assistant

A conversational Retrieval-Augmented Generation (RAG) application that answers questions about a credit-card agreement using retrieved document passages.

Built as a hands-on project to understand document extraction, chunking, embeddings, vector storage, retrieval, and grounded answer generation.

## Features

- Extracts text from a credit-card agreement PDF.
- Splits text into chunks while preserving source page numbers.
- Creates embeddings using OpenAI.
- Stores chunk text, embeddings, and metadata locally in Chroma.
- Reuses stored embeddings when the PDF and indexing settings are unchanged.
- Uses recent conversation history to interpret follow-up questions.
- Generates answers with source and page references.
- Displays the search question and retrieved chunk IDs for inspection.

## How It Works

### Document preparation

1. Read the PDF and extract text from each page.
2. Split page text into overlapping chunks.
3. Create an embedding for each chunk.
4. Store the text, vector, and source metadata in Chroma.

### Question answering

1. Rewrite follow-up questions into standalone search questions.
2. Create an embedding for the search question.
3. Retrieve the highest-ranked chunks.
4. Supply the retrieved evidence and question to the LLM.
5. Generate an answer with citations.

Embeddings help locate relevant content. The original chunk text supplies evidence for the answer.

## Technology

- Python
- pypdf
- LangChain Text Splitters
- OpenAI API
- Chroma
- python-dotenv

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Suresh-18-11/credit-card-rag.git
cd credit-card-rag
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the API key

The current code loads `.env` from the parent directory:

```python
load_dotenv(PROJECT_FOLDER.parent / ".env")
```

Create that file with:

```dotenv
OPENAI_API_KEY=your_api_key_here
CHAT_MODEL=gpt-4.1-mini
```

Alternatively, to keep configuration inside the project, change the code to:

```python
load_dotenv(PROJECT_FOLDER / ".env")
```

Then create `.env` beside `load_document.py`.

Never commit an actual API key. OpenAI API usage requires separate API billing.

### 5. Run the application

```bash
python load_document.py
```

The first run creates embeddings and saves the index in `chroma_db/`. Later runs reuse the index when the source document and indexing settings are unchanged.

Type `exit` or `quit` to stop.

## Example Questions

- What does the agreement say about minimum payments?
- What are the rules for late payments?
- How does the agreement describe interest charges?
- Explain that more simply.
- Which pages support your answer?

Answers depend on the evidence retrieved from the included agreement.

## Configuration

The following settings are defined in `load_document.py`:

| Setting | Default |
|---|---|
| Embedding model | `text-embedding-3-small` |
| Chat model | `gpt-4.1-mini`, overridable through `CHAT_MODEL` |
| Chunk size | 1,000 characters |
| Overlap target | 150 characters |
| Top-k retrieval | Up to 4 chunks |

These are starting settings for experimentation.

## Current Limitations

- Supports PDFs with extractable text; OCR is not implemented.
- Splits each page separately, so overlap does not cross page boundaries.
- Conversation history retains up to 12 messages and resets on restart.
- Uses basic vector retrieval without reranking or a relevance threshold.
- Citations and factual accuracy are not automatically validated.
- Document extraction can lose formatting or table relationships.
- The public agreement is not necessarily the agreement for a user's own account.

The application is a learning prototype. Generated answers should be checked against the cited document.

## Planned Improvements

- Evaluate retrieval and answer quality.
- Improve chunk boundaries and cross-page context.
- Add hybrid search and reranking.
- Validate citations.
- Integrate retrieval into a LangGraph workflow.
- Support additional document formats.

## Source Document

The project uses Capital One's publicly available consumer credit-card agreement:

https://ecm.capitalone.com/WCM/card/credit-card-agreements-q1-2026/credit-card-agreement-for-consumer-cards-in-capital-one-n.a.pdf

The source document belongs to its publisher. This project is not affiliated with Capital One.
