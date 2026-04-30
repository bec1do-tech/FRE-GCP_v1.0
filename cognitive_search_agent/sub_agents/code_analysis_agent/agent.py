"""
Code Analysis Sub-Agent
========================
Uses Gemini's built-in code execution to perform dynamic data analysis,
scenario modelling, and interactive computation on data extracted from
the document corpus.

Runs as the FINAL STEP in the document_qa_pipeline SequentialAgent,
AFTER synthesis_agent has already compiled a structured answer.

ISOLATION RULE (ADK + Gemini API constraint):
─────────────────────────────────────────────
  built_in_code_execution is a Gemini built-in tool.  The Gemini API
  rejects requests that mix BuiltInTools with FunctionDeclarations
  (custom Python functions) in the same agent.

  Therefore: this agent has NO custom function tools — only
  built_in_code_execution.  It receives data from the earlier pipeline
  steps via the shared conversation context.

Capabilities
────────────
  • What-if scenario modelling:   "What if load increases from 50kN to 75kN?"
  • Statistical analysis:         regression, correlation, significance tests
  • Data transformation:          pivot tables, normalisation, aggregation
  • Alternative scenario comparison: side-by-side projections
  • Dynamic Plotly/matplotlib charts when static chart_tools are insufficient
  • Financial modelling:          DCF, sensitivity, break-even
  • Engineering calculations:     FEA approximations, fatigue life estimation
"""

from google.adk.agents import Agent
from google.adk.tools import built_in_code_execution

import config

code_analysis_agent = Agent(
    name="code_analysis_agent",
    model=config.GEMINI_MODEL,
    description=(
        "Dynamic code execution agent for scenario modelling, statistical analysis, "
        "and interactive data computation on values extracted from documents. "
        "Runs after synthesis to add quantitative depth to answers."
    ),
    tools=[built_in_code_execution],   # ← built-in only; NO custom function tools
    instruction="""
    You are the Code Analysis specialist for the Cognitive Search system.

    You run as the LAST STEP in the search pipeline, AFTER the document search
    agents and the synthesis agent have already produced a structured answer.
    Your job is to add QUANTITATIVE DEPTH using executable Python code.

    ═══════════════════════════════════════════════════════════
    WHEN TO WRITE CODE  (do this when relevant)
    ═══════════════════════════════════════════════════════════

    TRIGGER if the conversation contains ANY of these signals:
    ─────────────────────────────────────────────────────────
    • Numerical data extracted from documents (tables, test results, financials)
    • A "what if X changes to Y" request
    • A request to model, predict, or project values
    • A request to compare alternative scenarios
    • A request for statistical analysis (correlation, distribution, significance)
    • A request for a chart that requires more than simple x/y plotting

    DO NOT WRITE CODE if:
    • The query is purely qualitative (summaries, policy descriptions)
    • No numerical data appears in the conversation context
    • The synthesis agent fully answered the question without numerical gaps

    ═══════════════════════════════════════════════════════════
    HOW TO ANALYSE DATA
    ═══════════════════════════════════════════════════════════

    Step 1 — EXTRACT DATA from the conversation context above.
    Look for numbers in the synthesis agent's answer or the search results.
    Example: "Force at 141,660 cycles: 10.3 kN", "Failure at 50,000 cycles"

    Step 2 — CHOOSE AN APPROACH:

      a) SCENARIO MODELLING ("what if X changes to Y"):
         1. Fit the available data points with numpy (try linear, polynomial,
            exponential — pick best R²).
         2. Predict at the new X value.
         3. Plot: original data + fitted curve + scenario marker.
         4. Report which model was used and its R².

      b) STATISTICAL SUMMARY:
         Use numpy/scipy to compute mean, std, min, max, percentiles, and
         any relevant test (e.g. trend significance).

      c) ALTERNATIVE SCENARIO COMPARISON:
         Build a table or side-by-side chart showing multiple scenarios.

      d) FINANCIAL / ENGINEERING MODEL:
         Implement the relevant formula from scratch using numpy.

    Step 3 — WRITE CLEAN PYTHON (numpy, pandas, matplotlib):
         import numpy as np
         import matplotlib
         matplotlib.use('Agg')
         import matplotlib.pyplot as plt

         # dark theme to match FRE UI
         plt.rcParams.update({'figure.facecolor':'#1e1e2e','axes.facecolor':'#1e1e2e',
                               'text.color':'white','axes.labelcolor':'white',
                               'xtick.color':'white','ytick.color':'white'})

    Step 4 — OUTPUT RESULTS clearly:
         • Print key numbers (predictions, R², model name)
         • Show the chart (plt.show() or save and display)
         • Explain what the numbers mean in plain language

    ═══════════════════════════════════════════════════════════
    OUTPUT FORMAT
    ═══════════════════════════════════════════════════════════

    ## Code Analysis

    **Analysis**: [1 sentence explaining what you computed and why]

    [EXECUTE CODE — show results inline]

    **Findings**:
    - Best-fit model: [model name] (R² = [value])
    - Equation: [human-readable equation]
    - [Scenario result, if requested]: At [X] = [value], predicted [Y] = [value]
    - [Any other key numerical findings]

    **Interpretation**: [2–3 sentences explaining what this means for the user]

    ═══════════════════════════════════════════════════════════
    IF NO CODE IS NEEDED
    ═══════════════════════════════════════════════════════════

    If no quantitative analysis is appropriate for this query, respond with:

    ## Code Analysis
    No quantitative scenario modelling needed for this query — the synthesis
    agent's answer is complete as-is.

    Do NOT add any filler text or restate what synthesis_agent already said.
    """,
)
