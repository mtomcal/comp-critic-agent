# **PRD: Comp Critic Agent

* **Document Status:** V1.0 - Draft
* **Author:** Michael Tomcal
* **Objective:** To build an AI agent that provides expert-level compositional critiques of landscape photographs. The agent will ground its analysis in a curated knowledge base of professional video transcripts, combining its own visual understanding with retrieved, domain-specific advice.

### 1. Problem Statement

Photographers, especially those retraining in landscape composition, lack a way to get immediate, expert feedback on their images. While generic AI can describe a photo, it cannot reference specific techniques taught by experts the user trusts. This project aims to bridge that gap by creating an agent that "watches" a library of expert videos (via their transcripts) and uses that knowledge to critique a user's new photo.

### 2. User Persona

* **"The Photographer" (User):** A frontend engineer and photographer retraining in AI and landscape photography. They have a specific set of video-based learning materials and want to build a tool to "consult" those materials for feedback on their own work.

### 3. Core Features (User Stories)

* **Story 1 (Data Ingestion):** As the system administrator, I need to ingest a directory of `.txt` video transcripts into a persistent vector database, so that the agent can perform semantic searches on this knowledge base.
* **Story 2 (Core Loop):** As "The Photographer," I want to submit an image file to the agent system, so that I can receive a detailed, text-based compositional critique.
* **Story 3 (Agent Behavior):** As "The Photographer," I want the agent's critique to be a *synthesis* of two things:
    1.  The agent's own multimodal analysis of the image (e.g., "I see leading lines...").
    2.  Specific, relevant advice retrieved from the video transcript database (e.g., "...and the transcripts mention that leading lines should guide the eye to a key subject.").

### 4. Functional Requirements

#### 4.1. System 1: Data Ingestion Pipeline (One-Time Setup)

This system indexes the knowledge base. It is a standalone Python script.

* **R-1.1: Load Documents:** The system must load all `.txt` files from a specified directory (e.g., `./transcripts`).
* **R-1.2: Chunk Documents:** The loaded documents must be split into smaller, overlapping text chunks (e.g., 1000-character chunks with a 200-character overlap).
* **R-1.3: Embed & Store:** The system must use an embedding model (e.g., `OpenAIEmbeddings`) to convert chunks into vectors.
* **R-1.4: Persist:** These vectors and their metadata must be saved to a persistent vector database on disk (e.g., **ChromaDB** at `./chroma_db`).

#### 4.2. System 2: The Multimodal RAG Agent (Runtime)

This is the main, user-facing agent.

* **R-2.1: Agent Brain:** The agent's core logic must be a multimodal LLM (e.g., **`gpt-4.1`** or **`gpt-4o`**) orchestrated via the **LangChain** framework.
* **R-2.2: Input:** The agent executor must accept an `input` that is a list containing two message types:
    1.  A `text` message (e.g., "Critique the composition of this photo").
    2.  An `image_url` message (a base64-encoded data URI of the user's photo).
* **R-2.3: Agent Prompt:** The agent's system prompt must instruct it to follow a specific reasoning process:
    1.  First, visually analyze the provided image to identify key compositional elements (e.g., rule of thirds, subject placement, leading lines, framing, foreground/background).
    2.  Second, based on that analysis, formulate one or more search queries to find relevant advice.
    3.  Third, execute a search for those queries using the `composition_rag_tool`.
    4.  Fourth, synthesize its *own visual analysis* with the *retrieved text* into a single, comprehensive answer.
* **R-2.4: Tool Definition (`composition_rag_tool`):**
    * The agent must be given access to one and only one tool, defined with the `@tool` decorator.
    * **Tool Name:** `composition_rag_tool`
    * **Tool Description:** "Searches the landscape photography video transcripts for advice. Use this tool to find specific techniques, tips, or critiques related to a compositional query."
    * **Tool Input:** `query: str`
    * **Tool Logic:**
        1.  Loads the persistent ChromaDB from `./chroma_db`.
        2.  Performs a similarity search on the vector store using the `query`.
        3.  Retrieves the top `k` (e.g., `k=3`) most relevant document chunks.
        4.  Formats and returns these chunks as a single string.
* **R-2.5: Output:** The final output of the `agent_executor.invoke()` call must be a text string (`output["output"]`) containing the synthesized critique.

### 5. Non-Functional Requirements

* **NFR-1 (Performance):** The agent's response time should be reasonable, ideally under 30 seconds for a complete critique. (The RAG lookup should be < 2 seconds).
* **NFR-2 (Reliability):** The agent must *always* attempt to use the RAG tool. It should be heavily penalized in its prompt if it attempts to answer from general knowledge alone.

### 6. Assumptions & Dependencies

* **A-1:** A directory of `.txt` transcripts already exists, generated by a service like **Deepgram**.
* **D-1:** Python 3.10+
* **D-2 (Libraries):** `langchain`, `langchain-openai`, `chromadb`, `pillow` (for image encoding helper).
* **D-3 (Services):** An **OpenAI API Key** with access to a multimodal model (`gpt-4.1` or `gpt-4o`).

### 7. Acceptance Criteria (How to Test)

* **AC-1 (Ingestion):** Given a `./transcripts` folder, running the ingestion script successfully creates a populated `./chroma_db` directory on disk.
* **AC-2 (RAG Tool):** Given an image of a waterfall using a long exposure, the agent's internal "thought" process shows it formulates and calls the RAG tool with a query like `"long exposure"`, `"waterfall composition"`, or `"motion blur"`.
* **AC-3 (Synthesis):** The agent's final output text *explicitly* quotes or references content from the retrieved transcripts *in addition* to its own visual analysis.
* **AC-4 (Failure Mode):** If the RAG tool returns no relevant documents, the agent states that it "found no specific advice in the knowledge base" for that element but still provides its own visual analysis.

