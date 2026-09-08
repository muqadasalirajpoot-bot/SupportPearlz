LIVE DEMO 👉👉👉:https://support-pearlz-itrarwx3fpdixmkxgrhbqj.streamlit.app/
# SupportPearlz — Simple Single-File RAG Project

SupportPearlz is a Retrieval-Augmented Generation (RAG) customer-support assistant built with Python, LangChain, OpenAI, Chroma and Streamlit.

The Python logic is intentionally kept in **one file: `app.py`** so the project is easier to run and explain.

## Project structure

```text
SupportPearlz_Simple/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    ├── knowledge_base/
    └── vector_store/     # generated automatically on first run
```

## Recommended Python version

Use **Python 3.11**.

## First-time setup on Windows

Open PowerShell inside this folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, run this once in that PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Using the app

After `streamlit run app.py`:

1. The browser opens SupportPearlz.
2. Enter your OpenAI API key in the password field.
3. On the first run, the app reads the knowledge-base documents, chunks them, creates embeddings and saves a persistent Chroma vector index.
4. Later runs load that saved index instead of embedding everything again.
5. Ask questions about Pearlz warranty, returns, shipping, installation, filters, pricing and troubleshooting.

The API key is stored only in the current Streamlit session. It is **not written into the code or committed to Git**.

## Main features

- PDF, DOCX, Markdown, TXT and CSV loading
- Recursive text chunking
- OpenAI embeddings
- Persistent Chroma vector database
- Semantic retrieval with a relevance threshold
- Conversation-aware query rewriting
- Grounded generation
- Refusal when evidence is missing
- Partial-answer handling
- Prompt-injection resistance
- Pydantic structured model output
- Validated source labels and source snippets
- Local logging
- Rebuild-index button
- Reset-chat button

## Important GitHub note

Do not upload an API key or `.env` file. The local vector database is also excluded by `.gitignore` and is rebuilt automatically.

## Run again later

Every time you open a new PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Troubleshooting

### Dependency-conflict messages
Do not install this project into your global Python environment. Create and activate `.venv` first, then install `requirements.txt`.

### `streamlit` is not recognized
Your virtual environment is probably not active. Activate it and retry.

### Invalid API key
Use the **Change API key** button in the app and enter a valid OpenAI API key.

### First run takes longer
That is expected because the knowledge base is embedded once. Later runs load the saved Chroma index.
