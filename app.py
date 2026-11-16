"""EvoMonitor: EvoAgentX execution, LangSmith traces, and trace-derived monitoring."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from langsmith import Client

from src.evo_runner import run_evoagentx
from src.activity_monitor import load_activity_events
from src.pipeline import activity_events_to_frame, extract_activity_features, extract_features, runs_to_frame
from src.store import append_jsonl, load_csv, save_csv


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_PATH = DATA_DIR / "langsmith_raw_runs.csv"
FEATURE_PATH = DATA_DIR / "phase1_features.csv"
ACTIVITY_FEATURE_PATH = DATA_DIR / "activity_features.csv"
ACTIVITY_PATH = DATA_DIR / "activity_events.jsonl"
GRAPH_PATH = DATA_DIR / "generated_workflows.jsonl"

st.set_page_config(
    page_title="EvoMonitor | Phase 1",
    page_icon="🛰️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem;}
    .stMetric {border: 1px solid rgba(128,128,128,.25); padding: 12px; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛰️ EvoMonitor")
st.caption(
    "Phase 1 | Automatically generated EvoAgentX workflow • LangSmith traces "
    "• Tool and memory activity inspection • Feature dataset"
)


# -------------------- Sidebar --------------------

with st.sidebar:
    st.header("Connections")
    ls_key = st.text_input("LangSmith API key", type="password")
    project = st.text_input("LangSmith project", value="evoagentx-phase1")
    endpoint = st.text_input(
        "LangSmith endpoint",
        value="https://api.smith.langchain.com",
    )

    st.divider()
    st.header("EvoAgentX LLM")
    llm_key = st.text_input("Gemini API key", type="password")
    model = st.text_input("Model", value="gemini/gemini-3.8-flash")
    st.caption("Keys are used for this running app session and are not written to datasets.")

    page = st.radio(
        "Pages",
        [
            "Overview",
            "Run EvoAgentX",
            "Workflow Monitor",
            "LangSmith Monitor",
            "Tools Monitor",
            "Memory Monitor",
            "Feature Extraction",
            "Dataset",
        ],
    )


# -------------------- Helpers --------------------

def langsmith_client():
    if not ls_key.strip():
        raise ValueError("Enter your LangSmith API key in the sidebar.")
    return Client(api_key=ls_key.strip(), api_url=endpoint.strip())


def set_trace_environment():
    if not ls_key.strip():
        raise ValueError("Enter your LangSmith API key in the sidebar.")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = ls_key.strip()
    os.environ["LANGSMITH_PROJECT"] = project.strip()
    os.environ["LANGSMITH_ENDPOINT"] = endpoint.strip()


def get_column(frame: pd.DataFrame, names):
    """Return the first existing column name from a list."""
    for name in names:
        if name in frame.columns:
            return name
    return None


def classify_activity(frame: pd.DataFrame, keywords):
    """
    Find trace rows whose available metadata indicates the requested activity.
    This is trace inspection, not proof that untraced activity did not happen.
    """
    if frame.empty:
        return frame.copy()

    searchable = [
        c for c in [
            "name", "run_type", "tags", "serialized", "extra",
            "metadata", "inputs", "outputs"
        ] if c in frame.columns
    ]
    if not searchable:
        return frame.iloc[0:0].copy()

    pattern = "|".join(keywords)
    mask = pd.Series(False, index=frame.index)
    for column in searchable:
        mask = mask | frame[column].astype(str).str.contains(
            pattern,
            case=False,
            regex=True,
            na=False,
        )
    return frame[mask].copy()


def show_download(frame: pd.DataFrame, filename: str, label: str):
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def read_activity_frame(activity_type=None):
    events = load_activity_events(ACTIVITY_PATH)
    if activity_type:
        events = [event for event in events if event.get("activity_type") == activity_type]
    return pd.DataFrame(events)


def read_graph_records():
    records = []
    if GRAPH_PATH.exists():
        with GRAPH_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


# -------------------- Overview --------------------

if page == "Overview":
    raw = load_csv(RAW_PATH)
    features = load_csv(FEATURE_PATH)
    graphs = read_graph_records()

    st.subheader("Phase 1 overview")
    st.write(
        "The application asks EvoAgentX to generate a workflow from a goal, "
        "captures the generated graph before execution, and uses LangSmith "
        "to inspect available execution traces."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw trace records", len(raw))
    c2.metric("Feature rows", len(features))
    c3.metric("Saved workflow graphs", len(graphs))
    c4.metric("LLM runner", "Ready" if llm_key else "Needs API key")

    st.markdown("### Project stages")
    stages = [
        ("1. Automatic workflow generation", "WorkFlowGenerator creates the graph from the user goal."),
        ("2. Graph capture", "The actual nodes, edges, inputs, outputs and assigned agents are saved."),
        ("3. Execution tracing", "LangSmith receives the instrumented generation and execution traces."),
        ("4. Activity inspection", "Tool and memory-related trace records are inspected when available."),
        ("5. Feature extraction", "Trace records are converted into a feature dataset."),
    ]
    for heading, description in stages:
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.caption(description)

    st.warning(
        "Tool and memory pages only show activity present in the collected traces. "
        "The current runner does not itself configure an EvoAgentX memory module "
        "or register external tools. Missing trace records must not be reported "
        "as proof that memory or tools were used."
    )


# -------------------- Run EvoAgentX --------------------

elif page == "Run EvoAgentX":
    st.subheader("Run the automatically generated EvoAgentX workflow")

    with st.form("run_workflow_form"):
        label = st.text_input("Run label", value="phase1-auto-generated-workflow")
        version = st.text_input("Workflow version", value="baseline-v1")
        pathway = st.selectbox(
            "Monitoring focus label",
            ["Workflow", "Tools", "Memory", "Model", "None / baseline"],
        )
        goal = st.text_area(
            "Goal for EvoAgentX",
            value=(
                "Design a multi-agent workflow that analyzes customer feedback "
                "and produces a concise report with key themes and recommendations."
            ),
            height=120,
        )
        submitted = st.form_submit_button(
            "Run EvoAgentX and trace with LangSmith",
            type="primary",
        )

    if submitted:
        if not ls_key.strip() or not llm_key.strip():
            st.error("Enter both LangSmith and Gemini API keys in the sidebar.")
        else:
            try:
                set_trace_environment()
                os.environ["GEMINI_API_KEY"] = llm_key.strip()

                with st.spinner(
                    "Generating the workflow and executing it. This may take several minutes..."
                ):
                    result = run_evoagentx(
                        goal=goal,
                        model_name=model,
                        run_label=label,
                        evolution_type=pathway,
                        workflow_version=version,
                    )

                graph = result.get("generated_graph", {})
                graph_record = {
                    key: value
                    for key, value in result.items()
                    if key != "output"
                }
                append_jsonl(graph_record, GRAPH_PATH)

                st.success("Workflow generated and executed successfully. Actual graph saved.")
                st.text_area(
                    "Workflow output",
                    value=result.get("output", ""),
                    height=220,
                )

                st.markdown("### Actual generated graph")
                c1, c2 = st.columns(2)
                c1.metric("Generated nodes", graph.get("node_count", 0))
                c2.metric("Generated edges", graph.get("edge_count", 0))

                node_rows = []
                for node in graph.get("nodes", []):
                    if not isinstance(node, dict):
                        continue
                    node_rows.append(
                        {
                            "Node": node.get("name", ""),
                            "Description": node.get("description", ""),
                            "Agents": str(node.get("agents", [])),
                            "Inputs": str(node.get("inputs", [])),
                            "Outputs": str(node.get("outputs", [])),
                        }
                    )
                st.dataframe(pd.DataFrame(node_rows), use_container_width=True)

                st.markdown("**Generated edges**")
                st.dataframe(
                    pd.DataFrame(graph.get("edges", [])),
                    use_container_width=True,
                )
                with st.expander("Full generated graph JSON"):
                    st.json(graph)

            except Exception as exc:
                st.error(f"EvoAgentX execution failed: {exc}")
                st.exception(exc)


# -------------------- Workflow Monitor --------------------

elif page == "Workflow Monitor":
    st.subheader("Workflow lifecycle monitoring")
    events = load_activity_events(ACTIVITY_PATH)
    workflow_events = [event for event in events if event.get("activity_type") == "workflow"]
    frame = activity_events_to_frame(workflow_events)

    c1, c2, c3 = st.columns(3)
    c1.metric("Workflow events", len(frame))
    c2.metric("Generation events", int((frame["operation"] == "generate").sum()) if not frame.empty else 0)
    c3.metric("Execution failures", int((frame["status"] == "error").sum()) if not frame.empty else 0)

    if frame.empty:
        st.info("No workflow lifecycle events recorded yet. Run an automatically generated EvoAgentX workflow first.")
    else:
        st.dataframe(frame.sort_values("timestamp", ascending=False), use_container_width=True, height=380)
        show_download(frame, "workflow_activity_events.csv", "Download workflow activity events")

    st.markdown("### Captured generated workflow graphs")
    graphs = read_graph_records()
    if graphs:
        graph_rows = [{
            "Run label": row.get("run_label"),
            "Workflow version": row.get("workflow_version"),
            "Status": row.get("workflow_status"),
            "Nodes": row.get("generated_graph", {}).get("node_count", 0),
            "Edges": row.get("generated_graph", {}).get("edge_count", 0),
        } for row in graphs]
        st.dataframe(pd.DataFrame(graph_rows), use_container_width=True)
    else:
        st.info("No generated graph records saved yet.")

# -------------------- LangSmith Monitor --------------------

elif page == "LangSmith Monitor":
    st.subheader("LangSmith trace monitoring")

    if st.button("Test LangSmith connection", type="primary"):
        try:
            client = langsmith_client()
            projects = list(client.list_projects(limit=5))
            st.success(f"Connected. Found {len(projects)} accessible project(s).")
        except Exception as exc:
            st.error(f"LangSmith connection failed: {exc}")

    limit = st.select_slider(
        "Maximum trace records to retrieve",
        options=[50, 100, 250, 500, 1000],
        value=250,
    )

    if st.button("Refresh traces and collect data", type="primary"):
        try:
            client = langsmith_client()
            with st.spinner(f"Retrieving runs from project '{project}'..."):
                runs = list(client.list_runs(project_name=project.strip(), limit=limit))
                raw_frame = runs_to_frame(runs)
                save_csv(raw_frame, RAW_PATH)
            st.success(f"Retrieved and saved {len(raw_frame)} LangSmith run records.")
        except Exception as exc:
            st.error(f"Could not retrieve LangSmith runs: {exc}")
            st.exception(exc)

    raw = load_csv(RAW_PATH)
    if not raw.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Run records", len(raw))
        trace_col = get_column(raw, ["trace_id"])
        c2.metric(
            "Unique traces",
            raw[trace_col].nunique() if trace_col else "Not available",
        )
        error_col = get_column(raw, ["has_error", "error"])
        c3.metric(
            "Records with error",
            int(raw[error_col].fillna(False).astype(str).str.lower().isin(["true", "1"]).sum())
            if error_col else "Not available",
        )
        st.dataframe(raw, use_container_width=True, height=420)
        show_download(raw, "langsmith_raw_runs.csv", "Download raw trace CSV")
    else:
        st.info("No traces collected yet. Run EvoAgentX, then refresh traces here.")


# -------------------- Tools Monitor --------------------

elif page == "Tools Monitor":
    st.subheader("Tool activity monitoring")
    raw = load_csv(RAW_PATH)
    run_type_col = get_column(raw, ["run_type"])
    traced_tools = (
        raw[raw[run_type_col].astype(str).str.lower().eq("tool")].copy()
        if run_type_col else pd.DataFrame()
    )
    logged_tools = read_activity_frame("tool")

    c1, c2 = st.columns(2)
    c1.metric("LangSmith tool spans", len(traced_tools))
    c2.metric("Instrumented tool events", len(logged_tools))
    st.caption("Instrumented events are written only when a real tool adapter calls log_tool_activity() or track_activity('tool', ...).")
    if not traced_tools.empty:
        st.markdown("**LangSmith tool spans**")
        st.dataframe(traced_tools, use_container_width=True, height=260)
    if not logged_tools.empty:
        st.markdown("**Structured tool activity log**")
        st.dataframe(logged_tools, use_container_width=True, height=260)
        show_download(logged_tools, "tool_activity_events.csv", "Download tool activity events")
    if traced_tools.empty and logged_tools.empty:
        st.info("No real tool calls have been recorded yet. Register/invoke a tool in an EvoAgentX agent and wrap its actual operation with the activity logger. No sample or fabricated tool calls are shown.")


# -------------------- Memory Monitor --------------------

elif page == "Memory Monitor":
    st.subheader("Memory activity monitoring")
    raw = load_csv(RAW_PATH)
    memory_rows = classify_activity(
        raw, ["memory", "retrieve", "retrieval", "vector.?store", "embedding.?store"]
    ) if not raw.empty else pd.DataFrame()
    logged_memory = read_activity_frame("memory")

    c1, c2 = st.columns(2)
    c1.metric("LangSmith memory-related spans", len(memory_rows))
    c2.metric("Instrumented memory events", len(logged_memory))
    st.caption("Instrumented events are written only when a real memory backend calls log_memory_activity() or track_activity('memory', ...).")
    if not memory_rows.empty:
        st.markdown("**LangSmith memory-related traces**")
        st.dataframe(memory_rows, use_container_width=True, height=260)
    if not logged_memory.empty:
        st.markdown("**Structured memory activity log**")
        st.dataframe(logged_memory, use_container_width=True, height=260)
        show_download(logged_memory, "memory_activity_events.csv", "Download memory activity events")
    if memory_rows.empty and logged_memory.empty:
        st.info("No memory reads, writes or retrievals have been recorded. The runner still needs a real memory backend configured; the logger alone does not create or simulate memory operations.")


# -------------------- Feature Extraction --------------------

elif page == "Feature Extraction":
    st.subheader("Feature extraction")
    st.write(
        "The existing feature extractor converts collected LangSmith run records "
        "into trace-level numerical features."
    )

    if st.button("Extract features from stored traces", type="primary"):
        raw = load_csv(RAW_PATH)
        if raw.empty:
            st.error("First collect runs from the LangSmith Monitor page.")
        else:
            try:
                features = extract_features(raw)
                save_csv(features, FEATURE_PATH)
                activity_features = extract_activity_features(load_activity_events(ACTIVITY_PATH))
                save_csv(activity_features, ACTIVITY_FEATURE_PATH)
                st.success(
                    f"Saved {len(features)} LangSmith feature records and "
                    f"{len(activity_features)} structured activity feature records."
                )
            except Exception as exc:
                st.error(f"Feature extraction failed: {exc}")
                st.exception(exc)

    features = load_csv(FEATURE_PATH)
    activity_features = load_csv(ACTIVITY_FEATURE_PATH)
    tab_trace_features, tab_activity_features = st.tabs(["LangSmith trace features", "Structured activity features"])
    with tab_trace_features:
        if not features.empty:
            st.metric("Trace feature records", len(features))
            st.dataframe(features, use_container_width=True, height=360)
            show_download(features, "phase1_features.csv", "Download trace feature dataset")
        else:
            st.info("No trace features extracted yet.")
    with tab_activity_features:
        if not activity_features.empty:
            st.metric("Activity feature records", len(activity_features))
            st.dataframe(activity_features, use_container_width=True, height=360)
            show_download(activity_features, "activity_features.csv", "Download activity feature dataset")
        else:
            st.info("No structured activity features yet. Run the workflow and instrument real tool/memory calls first.")


# -------------------- Dataset --------------------

elif page == "Dataset":
    st.subheader("Phase 1 datasets")
    tab_features, tab_raw, tab_graphs, tab_activity = st.tabs(
        ["Extracted features", "Raw LangSmith traces", "Generated workflow graphs", "Activity events"]
    )

    with tab_features:
        features = load_csv(FEATURE_PATH)
        if features.empty:
            st.info("Feature dataset is empty. Run Feature Extraction first.")
        else:
            st.dataframe(features, use_container_width=True, height=400)
            show_download(features, "phase1_features.csv", "Export feature dataset")

    with tab_raw:
        raw = load_csv(RAW_PATH)
        if raw.empty:
            st.info("No raw trace data collected yet.")
        else:
            st.dataframe(raw, use_container_width=True, height=400)
            show_download(raw, "langsmith_raw_runs.csv", "Export raw trace dataset")

    with tab_graphs:
        records = read_graph_records()
        if not records:
            st.info("No generated workflow graphs saved yet. Run EvoAgentX first.")
        else:
            summary = [
                {
                    "Run label": record.get("run_label"),
                    "Goal": record.get("goal"),
                    "Status": record.get("workflow_status"),
                    "Nodes": record.get("generated_graph", {}).get("node_count"),
                    "Edges": record.get("generated_graph", {}).get("edge_count"),
                }
                for record in records
            ]
            st.metric("Saved workflow graphs", len(records))
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
            st.download_button(
                "Download generated workflow dataset",
                data=GRAPH_PATH.read_text(encoding="utf-8"),
                file_name="generated_workflows.jsonl",
                mime="application/jsonl",
            )

    with tab_activity:
        events = load_activity_events(ACTIVITY_PATH)
        if events:
            activity_frame = activity_events_to_frame(events)
            st.dataframe(activity_frame, use_container_width=True, height=400)
            show_download(activity_frame, "activity_events.csv", "Export structured activity events")
            activity_features = load_csv(ACTIVITY_FEATURE_PATH)
            if not activity_features.empty:
                st.markdown("**Aggregated activity features**")
                st.dataframe(activity_features, use_container_width=True)
                show_download(activity_features, "activity_features.csv", "Export activity features")
        else:
            st.info("No structured activity events have been recorded yet.")

    st.markdown("### What the datasets represent")
    st.markdown(
        "- **Execution traces:** runs and timing fields returned by LangSmith.\n"
        "- **Tool activity:** only explicitly recorded tool spans.\n"
        "- **Memory activity:** only trace records identified as memory-related.\n"
        "- **Generated graphs:** actual nodes, edges, workflow inputs/outputs and agents "
        "captured from EvoAgentX before execution."
    )
    st.warning(
        "This Phase 1 implementation does not claim to implement memory storage, "
        "register external tools, detect Version 1 to Version 2 evolution, or produce "
        "pre-commitment risk predictions. Those require separate implementation."
    )
