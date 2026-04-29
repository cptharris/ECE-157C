a query arrives, containing the PROMPT and the DATESET to evaluate on
- the PROMPT may be a question (which would result in a text response and could result in a visualization)
- the PROMPT may be a visualization request (which would result in a visualization)
- the PROMPT may be a follow-up question (which would result in a text response and could result in a visualization)

# DESCRIPTION OF NODES

## *`history`*
- adds recent context (Q/A pairs)
- adds previous data and previous data description

## *`describe_dataset`*
- adds dataset description
    - deterministically generates it or pulls it from previous artifact
    - this helps LLM calls understand the available dataset


## *`planner`* (calls LLM)
- given recent_context and previous_data_description, is this a follow-up prompt (boolean)?
    - a follow-up is not allowed to access the original dataset again
- should we generate a text output (boolean)?
    - this is almost always true, unless the prompt is specifically asking for a visualization
    - clearly and concisely state the QUESTION that the text output must answer (considering recent_context if this is a follow-up question)
- should we generate a VISUALIZATION (boolean)?
    - what is our VISUALIZATION GOAL?
- in 1-2 sentences, what DATA are necessary to answer the current question, generate a visualization, and answer potential follow-up questions?
    - make sure to be broad in the captured information so we can generate a visualization (if necessary) and answer follow-up  questions

## *`codegen`* (calls LLM)
- given a dataframe called `df` and its description
    - if this is a follow-up, the dataframe will be the previous run's data
- must create a variable called `data`, which should be a JSON-safe dict (not print statements)
- do NOT include explanations; output ONLY Python code
- generate code that captures necessary DATA

## *`data`*
- execute the Python code created by codegen
- capture the data
- handle errors
    - retry from codegen once
    - otherwise, stop work and return errors here, no sense in continuing with no data

## *`describe_data`*
- adds data_description
    - this helps LLM calls understand the captured data

## *`visualize-codegen`* (calls LLM) (conditional node)
- given the captured data and the visualization goal, generate code for the visualization
    - result is preloaded
    - use plotly.express as px
    - assign figure to `fig`
    - avoid non-JSON serializable objects
    - do not call fig.show()
    - output ONLY Python code

## *`visualize-execute`*
- execute the code created by visualize-codegen
- capture the figure plotly json
- handle errors
    - retry from visualize-codegen
    - otherwise, give up on visualization

## *`respond`* (calls LLM) (conditional node)
- given the captured data and the question, answer the question
    - Answer only from the execution result — do not invent numbers.
    - Be specific: include key values, percentages, or rankings.
    - Keep the answer under 150 words.
    - Do not mention Python, pandas, or code.


# CONCERNS

*`visualize`* and *`respond`* can run concurrently

generally, we want to keep enough information from the dataset to answer the question, generate the visualization, and answer potential follow-up questions
however, we also want to avoid passing a lot of data to the LLM (this will be a problem for the respond node, which must synthesize the data into a text response to the question)

# STATE DICT

```json
{
    "metadata": {
        "session_id": str,
        "run_id": str,
        "prompt": str,
        "dataset": str,
        "step": str,
    },
    "recent_context": str,
    "previous": {
        "data": {},
        "data_description": {
            "column_types": {},
        }
    },
    "plan": {
        "is_follow_up": bool,
        "do_response": bool,
        "do-visualization": bool,
        "question": str,
        "visualization_goal": str,
        "need_data": str,
    },
    "information": {
        "data_code": str,
        "data": {},
        "data_description": {},
        "error": str,
    },
    "visualization": {
        "visualize_code": str,
        "fig": {},
        "error": str,
    },
    "response": str,
}
```
