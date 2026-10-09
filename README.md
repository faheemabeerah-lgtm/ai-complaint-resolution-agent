# AI Research Agent

A beginner-friendly research application built with:

- Streamlit
- CrewAI
- Groq
- OpenAI GPT-OSS 120B
- DuckDuckGo Search

## Model

The app uses:

`openai/gpt-oss-120b`

through Groq.

## Project structure

```text
ai-research-agent/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally

Use Python 3.12.

Create and activate a virtual environment, then install:

```bash
pip install -r requirements.txt
```

Set your Groq API key as `GROQ_API_KEY`.

Then run:

```bash
streamlit run app.py
```

## Streamlit Cloud

1. Upload these files to a GitHub repository.
2. Create a new app on Streamlit Community Cloud.
3. Select the repository and `app.py`.
4. Open the app's Secrets settings.
5. Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Never upload your real API key to GitHub.
