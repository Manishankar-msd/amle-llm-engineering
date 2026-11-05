# LLM Engineering

A language model on its own invents what it does not know: asked about a company it has never seen, it answers anyway, in a confident tone. In these notebooks you build a support assistant for Tretwerk, a fictional company that rents e-bikes by subscription, with a model on Groq: first by writing and improving prompts, then by retrieving answers from the company's support handbook, and finally by deciding what goes into the model's context in a conversation.

## Learning Objectives

By the end of this repository, you should be able to:

- Call a language model through an API and track its token usage.
- Build prompts from their ingredients and chain them with LangChain.
- Teach a model an output format with an example.
- Retrieve the passages closest to a question with embeddings.
- Build a RAG chain with LangChain and Chroma.
- Score a RAG answer for faithfulness and its passages for recall with a model as judge.

## Learning Path

Work through the notebooks in this order.

**Scope** marks the core path every student is expected to complete. Optional lessons stay in
the repository and are worth returning to, but the day does not depend on them. **Units** are
indicative pacing, where one unit is 45 minutes.

| File / Folder | Description | Scope | Units |
|---|---|---|---|
| [**01 - Prompt Engineering**](01_prompt_engineering.ipynb) | Call `gpt-oss-20b` on Groq, read its reasoning and token usage, build prompts from their ingredients, and chain prompts with LangChain. | Core | 1.75 |
| [**02 - Retrieval-Augmented Generation**](02_rag.ipynb) | Split a support handbook into chunks, retrieve passages with embeddings, see where retrieval fails, generate grounded answers with citations, rebuild the pipeline with LangChain and Chroma, and score the answer with a model as judge. | Core | 3.5 |
| [**03 - Context Engineering**](03_context_engineering.ipynb) | Compare the whole handbook with retrieved passages, teach a format with an example in the context, keep a conversation history short with a window or a summary, and rewrite follow-up questions before retrieval. | Core | 2.75 |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Solutions**](solutions/) | Worked solutions for every notebook, with an explanation after each exercise. |
| [**Assets**](assets/) | The figures used in the notebooks, and [`figures.py`](assets/figures.py), which draws `context_parts.png`. |
| [**Data**](data/) | Tretwerk's support handbook (20 pages) and 14 customer emails, all fictional. |
| [**.env.example**](.env.example) | Template for the `.env` file with your Groq API key. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a
> **placeholder**. Replace it, including the `< >` brackets, with your own
> value. For example, `cd <repo-name>` becomes `cd amle-llm-engineering`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies, including LangChain, Chroma and sentence-transformers, and creates a virtual environment in `.venv/`. sentence-transformers installs PyTorch, a large package, so the first install takes a few minutes.

```bash
cd <repo-name>
uv sync
```

---

### 5. Add Your Groq API Key

The language model runs on [Groq](https://groq.com/), which needs an API key. The key is free:

1. Sign up or log in at [console.groq.com](https://console.groq.com/).
2. Open the [API Keys page](https://console.groq.com/keys) and click **Create API Key**.
3. Give the key a name, for example `amle`, and copy it. Groq shows the key only once.

Then copy the example file:

```bash
cp .env.example .env
```

Open `.env` and paste your key after `GROQ_API_KEY=`. The notebooks read it with `load_dotenv()`.

> [!CAUTION]
> Never commit your `.env` file. It contains your personal API key, and anyone who can read the repository could use it. `.env` is listed in `.gitignore`, so Git ignores it as long as you do not add it by force.

---

### 6. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## Data and Credits

The handbook and the emails are part of the repository, in `data/`. Tretwerk, its handbook and its customers are fictional, written for these notebooks, so the language model cannot know them from its training. The embedding model `sentence-transformers/all-MiniLM-L6-v2` (90 MB) is downloaded from Hugging Face on the first run and stored in `~/.cache/huggingface`. The free Groq plan allows about 8,000 tokens per minute, enough for every notebook; if a cell hits the limit, wait a minute and run it again.

- **`gpt-oss-20b`**: OpenAI (2025), [gpt-oss-120b and gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925), Apache-2.0, served by [Groq](https://console.groq.com/docs/model/openai/gpt-oss-20b).
- **`all-MiniLM-L6-v2`**: [Sentence Transformers](https://sbert.net/), Apache-2.0.

## References & Further Reading

- [**Prompt Engineering Guide**](https://www.promptingguide.ai/): Prompting techniques from zero-shot to chain-of-thought, with examples.
- [**Groq: Reasoning**](https://console.groq.com/docs/reasoning): How `reasoning_effort` and the reasoning output work for models such as `gpt-oss-20b`.
- [**LangChain: ChatGroq**](https://docs.langchain.com/oss/python/integrations/chat/groq): Using models hosted on Groq in LangChain.
- [**LangChain: Build a semantic search engine**](https://docs.langchain.com/oss/python/langchain/knowledge-base): Loading documents, splitting them into chunks, embedding them and searching a vector store.
- [**Anthropic: Effective context engineering for AI agents**](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): How to decide what goes into a model's context, from system prompts to memory.

### The papers behind the notebooks

- Wei, J. et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903): why writing out the steps improves answers to reasoning questions.
- Lewis, P. et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401): the paper that introduced RAG.
- Liu, N. F. et al. (2023). [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172): why more context is not always better.
