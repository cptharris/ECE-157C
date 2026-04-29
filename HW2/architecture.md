# INPUT

a query arrives, containing the PROMPT and the DATESET to evaluate on
- the PROMPT may be a question (which would result in a text response and could result in a visualization)
- the PROMPT may be a visualization request (which would result in a visualization)
- the PROMPT may be a follow-up question (which would result in a text response and could result in a visualization)


# DESCRIPTION OF NODES


## *`history`*
- adds `["recent_context"]` (Q/A pairs)
- adds `["previous"]["data"]` and `["previous"]["data_desc"]`


## *`describe_dataset`*
- adds `["dataset_desc"]`
    - deterministically generates it or pulls it from previous artifact
      - serialize the dataset as a formatted string, include columns types and a summary
    - this helps LLM calls understand the available dataset


## *`planner`* (calls LLM)
- given `["recent_context"]` and `["previous"]["data_desc"]`, is this a follow-up prompt (boolean, `["plan"]["is_follow_up"]`)?
    - a follow-up is not allowed to access the original dataset again
- should we generate a text output (boolean, `["plan"]["do_response"]`)?
    - this is almost always true, unless the prompt is specifically asking for a visualization
    - clearly and concisely state the QUESTION that the text output must answer (considering recent_context if this is a follow-up question) (`["plan"]["question"]`)
- should we generate a VISUALIZATION (boolean, `["plan"]["do_vis"]`)?
    - what is our VISUALIZATION specification (`["plan"]["viz_spec"]`)?
- in 1-2 sentences, what data are necessary to answer the current question, generate a visualization, and answer potential follow-up questions (`["plan"]["data_spec"]`)?
    - make sure to be broad in the captured information so we can generate a visualization (if necessary) and answer follow-up  questions


## *`codegen`* (calls LLM)
- given a dataframe called `df` and its description
    - this will be the contents of the dataset CSV
    - unless! if this is a follow-up, the dataframe will be the previous run's data
- must create a variable called `data`, which must be a JSON-safe dict or a list of dicts (not print statements)
- do NOT include explanations; output ONLY Python code
- generate code that captures necessary data
    - code is stored in `["execution"]["data_code"]`


## *`execute`*
- execute the Python code created by codegen
- capture the `data` variable to `["execution"]["data"]`
    - make sure `data` is a plain dict or a list of dicts
    - if it is a DataFrame, use `to_dict(orient="records")` to convert
- handle errors
    - retry from codegen once
    - otherwise, stop work and return errors here (`["execution"]["error"]`), no sense in continuing with no data
- add `["execution"]["data_desc"]`
    - serialize `data` as a formatted string, include columns types and a summary
    - this helps LLM calls understand the captured data


## *`visualize-codegen`* (calls LLM) (conditional node)
- given the captured data and the visualization goal, generate code for the visualization
    - result is preloaded
    - use plotly.express as px
    - assign figure to `fig`
    - avoid non-JSON serializable objects
    - do not call fig.show()
    - output ONLY Python code
- stored in `["vis"]["vis_code"]`


## *`visualize-execute`*
- execute the code created by visualize-codegen
- capture the `fig` variable and convert to plotly json
    - store as `["vis"]["fig"]`
- handle errors
    - retry from visualize-codegen
    - otherwise, give up on visualization and store error in `["vis"]["error"]`


## *`respond`* (calls LLM) (conditional node)
- given the captured data and the question, answer the question
    - Answer only from the execution result — do not invent numbers.
    - Be specific: include key values, percentages, or rankings.
    - Keep the answer under 150 words.
    - Do not mention Python, pandas, or code.
- store in `["response"]`


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
    },
    "recent_context": str,
    "previous": {
        "data": {},
        "data_desc": str,
    },
    "dataset_desc": str,
    "plan": {
        "is_follow_up": bool,
        "do_response": bool,
        "do_vis": bool,
        "question": str,
        "vis_spec": str,
        "data_spec": str,
    },
    "execution": {
        "data_code": str,
        "data": {},
        "data_desc": {},
        "error": str,
    },
    "vis": {
        "vis_code": str,
        "fig": {},
        "error": str,
    },
    "response": str,
}
```
