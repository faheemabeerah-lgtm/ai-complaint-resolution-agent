import os

import streamlit as st
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from duckduckgo_search import DDGS


MODEL_NAME = "openai/gpt-oss-120b"


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Search the web with DuckDuckGo. Use this to find current information, "
        "facts, and useful sources about the research topic."
    )

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(query, max_results=5)
            if not results:
                return "No search results were found."

            lines = []
            for i, result in enumerate(results, start=1):
                title = result.get("title", "Untitled")
                body = result.get("body", "")
                href = result.get("href", "")
                lines.append(f"{i}. {title}\n{body}\nSource: {href}")

            return "\n\n".join(lines)
        except Exception as exc:
            return f"Search failed: {exc}"


def create_research_crew(topic: str) -> Crew:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    llm = LLM(
        model=f"groq/{MODEL_NAME}",
        api_key=api_key,
        temperature=0.2,
        max_tokens=6000,
    )

    search_tool = DuckDuckGoSearchTool()

    researcher = Agent(
        role="AI Research Analyst",
        goal=(
            "Research the user's topic using reliable web sources and produce "
            "an accurate, balanced, well-organized research report."
        ),
        backstory=(
            "You are a careful research analyst. You search for current information, "
            "compare sources, avoid unsupported claims, and clearly identify sources."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=f"""
Research the following topic:

{topic}

Use the web search tool to gather information from multiple relevant sources.

Write a beginner-friendly research report with these sections:
1. Title
2. Executive Summary
3. Introduction
4. Key Findings
5. Detailed Discussion
6. Advantages / Benefits (when relevant)
7. Challenges / Limitations (when relevant)
8. Conclusion
9. Sources

Important:
- Prefer recent and credible sources.
- Do not invent facts or citations.
- Include source URLs in the Sources section when available.
- Clearly distinguish established facts from interpretation.
- Keep the report informative and easy to understand.
""",
        expected_output="A complete, well-structured research report with a source list.",
        agent=researcher,
    )

    return Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )


st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI Research Agent")
st.write(
    "Enter a topic and let a CrewAI research agent search the web and create a report."
)

topic = st.text_area(
    "Research topic",
    placeholder="Example: The impact of artificial intelligence on education",
    height=120,
)

if st.button("🚀 Start Research", type="primary"):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    elif not os.getenv("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY is missing. Add it to your environment variables "
            "or Streamlit Cloud Secrets."
        )
    else:
        with st.spinner("Researching the topic and writing your report..."):
            try:
                crew = create_research_crew(topic.strip())
                result = crew.kickoff()
                report = str(result)

                st.success("Research completed!")
                st.markdown(report)

                st.download_button(
                    "⬇️ Download Report",
                    data=report,
                    file_name="research_report.md",
                    mime="text/markdown",
                )
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.info(
                    "Check that your GROQ_API_KEY is valid and that all packages "
                    "from requirements.txt were installed successfully."
                )
