# LLM AI Assistant — Full-Stack Application

This is my first full-stack LLM AI Assistant application. The main goal of this project is to learn how Large Language Models (LLMs) work and how to use tools, Retrieval-Augmented Generation (RAG), open-source models, and evaluation techniques to customize LLM-powered applications for real-world business use cases.

The application provides a simple environment for experimenting with different LLM features, comparing paid and free models, processing audio files, retrieving information from custom knowledge bases, and evaluating the quality of RAG responses.

## Technologies Used

- **LLMs:** OpenAI API and open-source models running locally with Ollama
- **Embeddings:** OpenAI Embeddings and Hugging Face Sentence Transformers
- **Frontend:** Vue 3
- **Backend:** FastAPI
- **LLM Framework:** LangChain
- **Open-Source Models:** Qwen2.5 and Whisper
- **Vector Database:** PostgreSQL with pgvector
- **Containerization:** Docker and Docker Compose
- **Model Development:** Hugging Face Transformers

## Getting Started

### 1. Start the Application

Clone the repository, navigate to the API directory, and start the application using Docker Compose:

```bash
cd api
docker compose up
```

The Vue frontend is available at:

http://localhost:5173/

### 2. Choose Between Paid and Free Models

The application supports two model configurations. You can switch between them by setting the `USE_PAID_MODELS` environment variable.

**Paid models — OpenAI**

```env
USE_PAID_MODELS=true
OPENAI_API_KEY=your_openai_api_key
```

This configuration uses:

- **Chat model:** `gpt-4.1-nano`
- **Embedding model:** `text-embedding-3-large`
- **Concurrency:** Up to 8 evaluation tests at a time

**Free models — Ollama and Hugging Face**

```env
USE_PAID_MODELS=false
```

This configuration uses:

- **Chat model:** `qwen2.5:3b`, running locally through Ollama
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Concurrency:** Up to 2 evaluation tests at a time, to limit resource usage on a CPU-only machine

To use the free configuration, install Ollama on your machine and download the required model:

```bash
ollama pull qwen2.5:3b
```

Make sure the Ollama service is running and that the FastAPI container can reach it. The Ollama base URL should be configured appropriately for your Docker environment.

The embedding model `sentence-transformers/all-MiniLM-L6-v2` is downloaded during the Docker image build, so it can be loaded locally without calling a paid embedding API.

The application uses separate PostgreSQL tables for the two embedding configurations:

- `documents` — OpenAI embeddings
- `documents_free` — Hugging Face embeddings

This separation is necessary because the two embedding models produce different vector representations. Documents must be ingested using the same embedding configuration that will be used to retrieve them.

The model configuration is centralized in the application's model profile, which selects the appropriate embedding model, chat model, evaluation judge, database table, and concurrency limit.

## Features

### 1. LLM Tools

**Frontend route:** [http://localhost:5173/](http://localhost:5173/)

This page is for experimenting with tools and LLM interactions. It provides a starting point for learning how LLMs can use external functionality to perform tasks beyond generating text.

### 2. Meeting Minutes from Audio

**Frontend route:** [http://localhost:5173/meeting-minutes](http://localhost:5173/meeting-minutes)

This feature generates meeting minutes from an uploaded MP3 audio file.

The processing pipeline consists of two steps:

1. **Audio transcription:** The API endpoint `/audio/upload` uses the `openai/whisper-small.en` model to convert speech into text.
2. **Meeting summarization:** The application uses `Qwen/Qwen2.5-0.5B-Instruct` to summarize the transcript and generate meeting minutes.

This feature helped me learn how to use Hugging Face Transformers pipelines with open-source models for audio transcription and text summarization.

For experimentation, I downloaded a portion of a Denver City Council meeting recording and used it to test the transcription and summarization pipeline.

**Sample audio:** [Denver City Council meeting excerpt](https://drive.google.com/file/d/1N_kpSojRR5RYzupz6nqM8hMSoEF_R7pU/view?usp=sharing)

**Original dataset:** [MeetingBank dataset on Hugging Face](https://huggingface.co/datasets/huuuyeah/meetingbank)

**Audio dataset:** [MeetingBank Audio on Hugging Face](https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio/tree/main)

### 3. Retrieval-Augmented Generation (RAG)

**Frontend route:** [http://localhost:5173/rag](http://localhost:5173/rag)

This feature explores how RAG can improve LLM responses by grounding them in custom business data.

I generated synthetic data for a fictional company and stored the source documents in the `knowledge-base` directory.

Before using the RAG feature, populate the PostgreSQL vector database with the relevant documents:

```bash
cd api
uv run python -m app.rag.ingest
```

The ingestion process prepares the documents, splits them into chunks, generates embeddings, and stores the resulting vectors in PostgreSQL using pgvector.

When a user submits a question, the application:

1. Converts the question into a vector using the configured embedding model.
2. Retrieves relevant document chunks from the vector database.
3. Adds the retrieved context to the LLM prompt.
4. Generates an answer grounded in the retrieved information.

This feature demonstrates how chunking strategies, embedding models, vector search, and prompt construction affect the accuracy, completeness, and relevance of generated answers.

**Important:** Run ingestion with the same model configuration used for retrieval. If you switch between paid and free models, populate the corresponding vector database table before evaluating or querying that configuration.

### 4. RAG Evaluation

**Frontend route:** [http://localhost:5173/rag-eval](http://localhost:5173/rag-eval)

This feature evaluates the quality of the RAG implementation by running a curated test dataset against the RAG endpoint and measuring retrieval and answer quality.

Instead of relying only on manual testing, I created a test set containing questions and their expected answers. The evaluation pipeline uses an LLM-as-a-judge approach to compare generated responses with the expected answers.

The dashboard provides two groups of evaluation metrics.

#### Retrieval Evaluation

Retrieval evaluation measures whether the RAG system retrieves relevant information for a given question.

- **Mean Reciprocal Rank (MRR):** Measures how highly the first relevant result is ranked. Higher values indicate that relevant results tend to appear near the top.
- **Normalized Discounted Cumulative Gain (nDCG):** Measures the quality of the ranking while accounting for the relevance of retrieved results and their positions.
- **Keyword Coverage:** Measures how much of the expected keyword set is covered by the retrieved information.

The dashboard also displays average MRR by question category, helping identify which types of questions the retrieval system handles well and which need improvement.

#### Answer Evaluation

Answer evaluation measures the quality of the generated answers against the expected answers.

- **Accuracy:** Measures how factually correct the generated answer is.
- **Completeness:** Measures whether the answer covers the important information required to answer the question.
- **Relevance:** Measures how well the answer addresses the user's question.

An LLM judge grades the generated answers using the evaluation criteria. The dashboard displays the average scores and breaks them down by question category to help identify weaknesses in the system.

#### Evaluation Results

The following screenshot shows an example of the evaluation dashboard after running 150 tests.

![RAG retrieval and answer evaluation dashboard](images/rag-eval.png)

The dashboard makes it easier to identify areas for improvement, compare model configurations, and evaluate the impact of changes to the retrieval pipeline.

For example, lower retrieval scores in a particular category may indicate that the document chunking, embeddings, or retrieval strategy needs improvement. Lower answer scores may indicate that the retrieved context or prompt construction needs further work.

The goal is to use these metrics to guide iterative improvements to the RAG pipeline rather than relying exclusively on subjective assessments of individual responses.

## Project Goals and What I Learned

Through this project, I have been exploring:

- How to integrate LLMs into a full-stack application.
- How to use tools and extend LLM capabilities.
- How to run open-source models locally with Ollama and Hugging Face.
- How to switch between paid API-based models and locally hosted models.
- How to generate embeddings and store them in a PostgreSQL vector database.
- How to implement RAG using document chunking, vector retrieval, and prompt construction.
- How to process audio files using speech-to-text models and generate meeting summaries.
- How to evaluate retrieval quality using MRR, nDCG, and keyword coverage.
- How to evaluate answer accuracy, completeness, and relevance using an LLM judge.
- How to use evaluation results to identify weaknesses and improve an LLM application.

This project is an ongoing learning exercise focused on understanding the practical components of building, testing, and improving LLM-powered business applications.