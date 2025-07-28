# Persona-Driven Document Intelligence

## Overview

This project provides an intelligent system for extracting, ranking, and summarizing the most relevant information from a set of PDF documents, tailored to a specific persona and task. It is designed for scenarios where a user (persona) needs actionable insights from multiple documents, such as planning a trip or conducting research.

## Approach

### 1. Input Configuration

- The system reads an input configuration from [`input/input.json`](input/input.json), specifying:
  - The persona (e.g., "Travel Planner")
  - The job to be done (e.g., "Plan a trip of 4 days for a group of 10 college friends.")
  - The list of PDF documents to analyze

### 2. PDF Text Extraction

- Each PDF in [`input/PDFs/`](input/PDFs/) is processed using [`pypdf`](requirements.txt).
- Text is extracted page-wise, preserving page numbers for context.

### 3. Section Extraction

- The extracted text is split into meaningful sections using enhanced heading detection:
  - Recognizes major headings, numbered sections, all-caps headers, and topic introducers.
  - If no headings are found, the text is chunked into larger, context-rich segments.

### 4. Semantic Query Generation

- A dynamic semantic query is generated for the persona and task using [`sentence-transformers`](requirements.txt):
  - The system builds a context string by finding professional vocabulary terms most semantically similar to the persona's role.
  - This query guides relevance scoring.

### 5. Section Ranking

- Each section is scored for relevance using cosine similarity between its embedding and the persona query.
- Scores are boosted for longer, practical, and actionable content.
- The top sections are selected, ensuring diversity across documents.

### 6. Subsection Analysis

- For each top section, the system extracts the most meaningful sentences using semantic clustering.
- These refined text pieces provide concise, actionable insights.

### Approach

1. **Input Configuration**: The user specifies the persona (e.g., "Travel Planner"), the job to be done (e.g., "Plan a trip for 10 college friends"), and the list of PDFs in `input/input.json`.
2. **PDF Extraction**: Text is extracted from each PDF using `pypdf`, preserving page numbers for context.
3. **Section Detection**: The system splits text into meaningful sections using advanced heading detection. If headings are missing, it creates context-rich chunks.
4. **Semantic Query Generation**: A dynamic query is built for the persona and task using `sentence-transformers`, leveraging professional vocabulary most relevant to the role.
5. **Section Ranking**: Each section is scored for relevance using cosine similarity and boosted for practical, actionable content. Top sections are selected, ensuring diversity across documents.
6. **Subsection Analysis**: The most meaningful sentences are extracted from each top section using semantic clustering.
7. **Output**: Results are saved to `output/output.json`, including metadata, ranked sections, and refined sentences.

## Build and Run with Docker

1. **Build the Docker image**:
   ```sh
   docker build -t persona-doc-intel .
   docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output persona-doc-intel

## Key Technologies

- [`sentence-transformers`](requirements.txt): For semantic embeddings and similarity.
- [`pypdf`](requirements.txt): For PDF text extraction.
- [`scikit-learn`](requirements.txt), [`numpy`](requirements.txt): For similarity calculations and clustering.


## Example

Given a travel planner persona and several South of France travel PDFs, the system extracts and ranks the most relevant sections and sentences to help plan a trip.

## Extensibility

- Easily adapt to other personas and tasks by changing the input configuration.
- The section extraction and ranking logic can be tuned for different document types.

## License

MIT License. See [`LICENSE`](LICENSE) for details.
