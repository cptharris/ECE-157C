Your report should include user-interface screenshots and text-based intermediate agent traces for at least 6 representative examples. Among these examples, include at least 2 generic-domain questions using DuckDuckGo web search, at least 2 analytics-oriented examples involving iterative reasoning/code-generation/execution, and at least 2 examples where the validator agent requests retry or detects suspicious/incomplete analysis.

For analytics-oriented examples, create a realistic analysis scenario and interact with your system through multiple questions and follow-up questions. There should be at least 5 questions/interactions used within the scenario. At least part of the scenario should involve cross-year or temporal analysis using multiple yearly CSV files from the dataset. The interaction should demonstrate how the system handles comparisons, trends, or changes across different years.

Based on the findings from the interaction, create a PowerPoint presentation (.pptx) summarizing the overall analysis results. The presentation should focus on high-level insights, conclusions, visualizations, and analysis summaries rather than simply listing the original user questions.

Your report should clearly explain the overall system design, including the orchestration node design, analytics agent design, validator agent design, execution/reasoning loop, and overall agent workflow. Include architecture/agent-design diagrams illustrating how different components interact with each other.

For each major node/component, explain its role, how it operates, what inputs/outputs it receives, and how prompts are designed or structured.

For the scenarios used in your presentation, choose 2 questions from the scenario along with the corresponding final answers, generated visualizations/plots, and the iterative reasoning/execution process used by your system.

Explain how many reasoning/execution iterations were performed, what actions were taken at each step, what observations were made from execution results, and how those observations influenced the next reasoning or analysis step.

Each analytics-oriented example should include at least 3 reasoning/execution iterations. These examples are separate from the 6 examples described above.

Your report should also briefly discuss stopping conditions, retry behavior, validation strategies, limitations/failure cases, lessons learned, implementation challenges, and possible future improvements.

## Synthesized Requirements

### Dataset
- **Source:** Kaggle — *200 Financial Indicators of US Stocks (2014–2018)*; multiple yearly CSV files must be used for temporal analysis

---

### 6 Representative Examples (Report Section)
Each must include a UI screenshot and a text-based intermediate agent trace.

| Constraint                                                      | Count | Notes                           |
| --------------------------------------------------------------- | ----- | ------------------------------- |
| Generic-domain (DuckDuckGo web search)                          | ≥ 2   | No dataset involved             |
| Analytics-oriented (iterative reasoning / code-gen / execution) | ≥ 2   | ≥ 3 reasoning iterations each   |
| Validator requests retry or flags suspicious/incomplete result  | ≥ 2   | Must show actual retry behavior |

---

### Analytics Scenario (Separate from the 6 Examples)
- ≥ 5 questions/interactions in a **single coherent narrative**
- At least part must use **cross-year or temporal analysis** across multiple yearly CSVs
- Must demonstrate comparisons, trends, or year-over-year changes
- **2 questions** from this scenario get a dedicated deep-dive write-up:
  - Final answer
  - Generated visualizations/plots
  - Step-by-step iterative reasoning/execution process (≥ 3 iterations each), documenting: what action was taken, what was observed, and how the observation changed the next step

---

### PowerPoint Presentation
- Generated **from the findings of the analytics scenario**
- Focus: high-level insights, conclusions, visualizations, analysis summaries
- **Not** a list of the original questions

---

### System Design Documentation
For each of these components, explain: role, operation, inputs/outputs, prompt structure.

1. **Orchestration node** — routing, task dispatch, flow control
2. **Analytics agent** — code generation, execution, iterative reasoning loop
3. **Validator agent** — output verification, retry triggering, anomaly/suspicion detection
4. **Execution/reasoning loop** — how iterations proceed and terminate
5. **Overall agent workflow** — end-to-end flow with architecture diagram(s)

---

### Stopping Conditions & Robustness (Discussion Section)
- Stopping conditions for the reasoning loop
- Retry behavior and how the validator triggers it
- Validation strategies
- Limitations and failure cases
- Lessons learned / implementation challenges
- Possible future improvements

---

## Recommended Questions

### Web Search Examples (×2)
| #   | Question                                                                                    | Why it works                                                                               |
| --- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| W1  | *What caused the 2016 oil-price crash?*                                                     | **Given.** Broad, factual, no dataset needed; tests DuckDuckGo retrieval and summarization |
| W2  | *What US macroeconomic events (rate hikes, policy changes) occurred between 2014 and 2018?* | Provides contextual backdrop for the dataset's time window; still pure web search          |

### Validator-Retry Examples (×2)
| #   | Question                                                     | Why it triggers retry                                                                                                                                    |
| --- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V1  | *Find undervalued technology companies from the dataset.*    | **Given.** Likely returns an incomplete or threshold-undefined first pass — validator should flag missing justification or suspiciously short stock list |
| V2  | *What is the average P/E ratio across all sectors for 2017?* | Easy to produce a result with silent NaN-drops or division errors; validator catches implausible values or missing sectors                               |

### Analytics-Oriented Standalone Examples (×2, ≥3 iterations each)
| #   | Question                                                                                                 | Why it works                                                                                                 |
| --- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| A1  | *Which sectors have the most companies represented, and how does that distribution change year to year?* | **Given/tested.** Requires loading all 5 CSVs, aggregating, plotting — naturally produces 3+ code iterations |
| A2  | *Which sector saw the largest improvement in median EPS over the 5-year window?*                         | **Given/tested.** Cross-year merge, outlier handling, trend fitting — multi-step reasoning                   |

### Analytics Scenario (≥5 interactions, temporal narrative)
Framing: *"Sector Health & Earnings Quality Across the 2014–2018 Market Cycle"*

| #   | Question                                                                                                                          | Role in scenario                              |
| --- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| S1  | *Which sectors have the most companies represented, and how does that distribution change year to year?*                          | Orientation — establish the dataset landscape |
| S2  | *Which sector saw the largest improvement in median EPS over the 5-year window?*                                                  | Core temporal finding                         |
| S3  | *For the top-performing EPS sector, which individual companies drove that improvement — and were any consistent outliers?*        | Drill-down follow-up on S2                    |
| S4  | *How did the median debt-to-equity ratio evolve across all sectors from 2014 to 2018?*                                            | Cross-metric, cross-year enrichment           |
| S5  | *Based on EPS growth and leverage trends, which sector appears most financially resilient entering a hypothetical 2019 downturn?* | Synthesis/conclusion — motivates the PPTX     |

> **Deep-dive the two bolded questions:** **S2** and **S3** are the best candidates for the ≥3-iteration write-up since they involve data loading decisions, outlier handling, and visualization refinement.
