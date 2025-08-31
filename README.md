# StudySpace
Multi-Modal Study Agentic AI Assistant with RAG for fast and efficient learning.

### 🚀 Features

📄 Multi-format ingestion: Supports .pdf, .docx, .pptx, .csv, .txt, and video and audio.

🔎 Chunking & Vector Storage: Automatically splits documents into chunks and stores them in a Chroma Vector Database for retrieval.

🧠 Retrieval-Augmented Generation (RAG): Uses LLMs to answer user questions based on uploaded content.

✍️ Summarization: Generates concise summaries of uploaded documents.

🎓 Study Aids: Creates flashcards and structured study notes.

🗂 Knowledge Persistence: Content stays indexed in the vector DB for future queries.

### 🛠️ LangGraph Agent Capabilities

This project uses LangGraph to orchestrate multi-tool agents for flexible reasoning:

🌐 Web Search – Fetches live, up-to-date information.

💻 Code Execution – Runs Python code snippets for calculations & data processing.

➗ Calculator – Quick and precise arithmetic operations.

📚 RAG Pipeline – Context-aware Q&A over uploaded documents.

### 🔥 Why LangGraph?

Stateful Agents – Maintains memory across interactions for contextual awareness.

Multi-tool Orchestration – Dynamically routes queries to the right tool.

Graph-based Workflows – Provides clear control over agent reasoning paths.

Scalable & Modular – Easy to extend with new tools and workflows.


### ⚡ Quick Start

Clone the repo:
```
git clone https://github.com/yourusername/studyspace.git
cd studyspace
```

Install dependencies:
```
pip install -r requirements.txt
```

Run the app:
```
python app.py
```
