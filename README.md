# EvoMonitor — Phase 1

A Windows dashboard for **Pre-Commitment Impact & Risk Prediction for Self-Evolving Multi-Agent Workflows**. Phase 1 focuses on **Workflow Evolution monitoring**: EvoAgentX automatically generates a workflow from the user's goal, the app captures that real graph, executes it, collects LangSmith traces, extracts available features, and stores datasets.

## Requirements
- Windows 10/11
- Python 3.12 recommended for this project environment
- LangSmith account and API key
- OpenAI API key with available usage for a real EvoAgentX run
- Internet connection

## Run
1. Extract the ZIP.
2. Open the `PreCommit_AI_Phase1` folder in VS Code.
3. Activate your Python 3.12 virtual environment.
4. Install dependencies with `python -m pip install -r requirements.txt`.
5. Start the dashboard with `streamlit run app.py`.
6. Open `http://localhost:8501` if the browser does not open automatically. Keep the terminal running.

## UI flow
1. In the sidebar, enter your LangSmith API key, project name (default `evoagentx-phase1`), OpenAI API key, and model (default `gpt-4o-mini`). Do not paste API keys into chat or commit them to Git.
2. Open **Run EvoAgentX**, enter a goal, and run it. The app calls `WorkFlowGenerator.generate_workflow(goal=...)`; it does not create a manually authored task list. EvoAgentX generates the graph and assigns agents.
3. Before execution, the app captures the generated graph's actual nodes, edges, inputs, outputs, and assigned agents.
4. The app loads the generated agents into `AgentManager` and executes the returned graph using `WorkFlow`.
5. Open **LangSmith Monitor**, test the connection, then refresh traces from the configured project.
6. Open **Feature Extraction** to create trace-level numerical features.
7. Open **Dataset** to view/download raw LangSmith runs, extracted features, and generated workflow graph records.

## Captured data
- **Generated graph dataset (`data/generated_workflows.jsonl`):** goal, actual generated nodes and edges, node inputs/outputs, assigned agent names, declared workflow inputs/outputs, and execution metadata.
- **Raw LangSmith dataset (`data/langsmith_raw_runs.csv`):** run hierarchy, run type, timing, errors, token usage/cost when supplied, and input/output sizes.
- **Feature dataset (`data/phase1_features.csv`):** run count, LLM/tool call counts, errors, failure rate, retry-tag count, trace depth, duration, token counts, cost when supplied, and input/output character counts.

## Important limitations
- Graph structure is captured from the `WorkFlowGraph` returned by EvoAgentX. LangSmith traces are collected separately; LangSmith does not replace graph capture.
- The app creates traceable spans for workflow generation and workflow execution. Individual internal agent/model/tool spans appear only if those calls are separately instrumented by EvoAgentX or their integrations. The app does not fabricate child spans.
- EvoAgentX resets graph node states after execution. This implementation saves the generated graph structure before execution; it does not claim to preserve per-node runtime states from the reset graph.
- Token and cost fields are populated only when the trace contains usage metadata.
- Ordinary traces do not automatically provide benchmark quality scores or explicit evolution diffs. Those require evaluator/evolution-event instrumentation.
- Phase 1 monitors automatically generated workflow structure and execution. It does **not** yet claim to detect or implement Version 1 → Version 2 evolution, train prediction models, or make ACCEPT/MODIFY/REJECT decisions.
- API keys are not written to the datasets.

## Troubleshooting
If dependency installation or EvoAgentX import fails, capture the complete terminal error and resolve the actual dependency conflict rather than randomly changing package versions. If LangSmith shows no traces, verify the project name and API key, wait a few seconds, and refresh.
