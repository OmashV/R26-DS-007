"""
Meta-Agent Dashboard — interactive demo interface.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st

from bkt.predict import BKTPredictor
from core.knowledge_graph import KnowledgeGraph
from core.concept_extractor import ConceptExtractor

st.set_page_config(
    page_title="Meta-Agent Dashboard",
    page_icon="🎓",
    layout="wide",
)


# Cache expensive objects so they're only loaded once per Streamlit session
@st.cache_resource
def load_predictor() -> BKTPredictor:
    return BKTPredictor.load()


@st.cache_resource
def load_knowledge_graph() -> KnowledgeGraph:
    return KnowledgeGraph(predictor=load_predictor())


@st.cache_resource
def load_extractor() -> ConceptExtractor:
    predictor = load_predictor()
    return ConceptExtractor(allowed_skills=list(predictor.params.keys()))


# Header
st.title("🎓 Meta-Agent Dashboard")
st.caption(
    "AI tutoring system — Meta-Agent component. "
    "Processes session transcripts, tracks per-student mastery via BKT, "
    "and generates personalised learning paths."
)

# Three main tabs
tab_session, tab_profile, tab_system = st.tabs([
    "📝 Process Session",
    "👤 Student Profile",
    "ℹ️ System Info",
])

# Preset transcripts — useful for live demos so you don't have to type
PRESET_TRANSCRIPTS = {
    "(Custom — paste your own)": "",
    "Percent Of — student struggling then learning": """\
TUTOR: Today we're working on percentages. What's 25% of 80?
STUDENT: Um... 25?
TUTOR: Not quite. Percent means "per hundred". So 25% is 0.25. Try 0.25 × 80.
STUDENT: Oh! 20.
TUTOR: Exactly. Now what's 15% of 60?
STUDENT: 9.
TUTOR: Perfect. 40% of 50?
STUDENT: 20.
TUTOR: Brilliant.""",
    "Mixed skills — equation solving and percentages": """\
TUTOR: Let's try equation solving. Solve 2x + 4 = 10.
STUDENT: x equals 3.
TUTOR: Yes. Try 5x - 7 = 18.
STUDENT: 5.
TUTOR: Correct. Now percentages — what's 20% of 90?
STUDENT: 18.
TUTOR: Right. 75% of 40?
STUDENT: 30.
TUTOR: Excellent.""",
    "Student bombing a previously known skill (regression)": """\
TUTOR: Let's revisit fractions. What's 1/2 of 20?
STUDENT: I don't know.
TUTOR: Try thinking of it as half of 20.
STUDENT: 5?
TUTOR: Not quite — half of 20 is 10. Try 1/4 of 16.
STUDENT: 8.
TUTOR: That's not right either. 1/4 of 16 is 4.
STUDENT: I'm confused.
TUTOR: It's okay. Let's slow down.""",
}


def parse_transcript_text(text: str) -> list[dict]:
    """Parse the freeform transcript text into the structured format."""
    turns = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("TUTOR:"):
            turns.append({"role": "tutor", "text": line[6:].strip()})
        elif line.upper().startswith("STUDENT:"):
            turns.append({"role": "student", "text": line[8:].strip()})
    return turns


with tab_session:
    st.header("Process a tutoring session")
    st.markdown(
        "Paste a session transcript below (or pick a preset). "
        "The Meta-Agent will extract attempts, run BKT, update the knowledge graph, "
        "and generate an updated learning path."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Input")
        student_id = st.text_input("Student ID", value="alice_demo")

        preset = st.selectbox("Choose a preset transcript", list(PRESET_TRANSCRIPTS.keys()))
        transcript_text = st.text_area(
            "Transcript",
            value=PRESET_TRANSCRIPTS[preset],
            height=300,
            help="Use 'TUTOR:' and 'STUDENT:' prefixes on each line.",
        )

        process_clicked = st.button("🚀 Process session", type="primary", use_container_width=True)

    with col_right:
        st.subheader("Output")

        if process_clicked:
            transcript = parse_transcript_text(transcript_text)
            if not transcript:
                st.error("Couldn't parse any turns from the transcript. "
                         "Make sure each line starts with 'TUTOR:' or 'STUDENT:'.")
            else:
                with st.spinner("Extracting concepts via LLM..."):
                    extractor = load_extractor()
                    extraction = extractor.extract(transcript)

                st.success(f"Extracted {len(extraction['attempts'])} attempts")

                # Show extracted attempts
                with st.expander("📋 Extracted attempts", expanded=True):
                    for a in extraction["attempts"]:
                        marker = "✅" if a["correct"] == 1 else "❌"
                        st.write(f"{marker}  **{a['skill']}**")
                    if extraction["misconceptions"]:
                        st.markdown("**Misconceptions detected:**")
                        for m in extraction["misconceptions"]:
                            st.write(f"- {m}")

                # Run through knowledge graph
                with st.spinner("Updating knowledge graph (running BKT)..."):
                    kg = load_knowledge_graph()
                    session = kg.process_session(
                        student_id=student_id,
                        attempts=extraction["attempts"],
                    )

                st.success(f"Session {session['session_id']} processed")

                # Show updated mastery
                with st.expander("🧠 Updated knowledge graph", expanded=True):
                    for entry in session["graph"]:
                        p = entry["mastery_probability"]
                        label = entry["mastery_label"]
                        if label == "strong":
                            colour = "🟢"
                        elif label == "weak":
                            colour = "🔴"
                        else:
                            colour = "🟡"
                        st.write(
                            f"{colour}  **{entry['skill']}** — "
                            f"P(mastery) = {p:.3f} ({label})"
                        )

                st.info("👤 Switch to **Student Profile** tab to see the full graph "
                        "and learning path with charts.")
        else:
            st.write("Click **Process session** to run the pipeline.")

with tab_profile:
    st.header("Student profile")
    st.markdown(
        "Browse a student's complete knowledge state — current mastery across all "
        "skills, the current learning path, and per-skill mastery trajectories."
    )

    kg = load_knowledge_graph()
    predictor = load_predictor()

    # Get all students who have data
    from db.database import get_connection
    with get_connection() as conn:
        student_rows = conn.execute(
            "SELECT DISTINCT student_id FROM mastery ORDER BY student_id"
        ).fetchall()
    student_ids = [r["student_id"] for r in student_rows]

    if not student_ids:
        st.warning("No students with mastery data yet. Process a session first in the previous tab.")
    else:
        selected_student = st.selectbox("Select student", student_ids)

        graph = kg.get_student_graph(selected_student)

        if not graph:
            st.warning(f"No mastery data for {selected_student}.")
        else:
            # Top-level summary metrics
            from core.learning_path import generate_learning_path
            path = generate_learning_path(graph)
            summary = path["summary"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total skills", len(graph))
            col2.metric("Strong", summary["strong_count"])
            col3.metric("Partial", summary["learn_next_count"])
            col4.metric(
                "Revise urgently",
                summary["revise_count"],
                delta=f"{summary['regression_count']} regressions" if summary["regression_count"] else None,
                delta_color="inverse",
            )

            st.divider()

            # Two-column layout: knowledge graph chart + learning path
            col_graph, col_path = st.columns([1.2, 1])

            with col_graph:
                st.subheader("Knowledge graph")
                from core.visualisations import plot_knowledge_graph
                fig = plot_knowledge_graph(graph=graph, student_id=selected_student)
                st.pyplot(fig)

            with col_path:
                st.subheader("Learning path")

                if path["revise_urgently"]:
                    st.markdown("##### 🔴 Revise urgently")
                    for e in path["revise_urgently"]:
                        flag = "  ↓ regression" if e.get("is_regression") else ""
                        st.write(f"- **{e['skill']}** (P={e['mastery_probability']:.3f}){flag}")

                if path["learn_next"]:
                    st.markdown("##### 🟡 Learn next")
                    for e in path["learn_next"]:
                        st.write(f"- **{e['skill']}** (P={e['mastery_probability']:.3f})")

                if path["already_strong"]:
                    st.markdown("##### 🟢 Already strong")
                    for e in path["already_strong"]:
                        st.write(f"- **{e['skill']}** (P={e['mastery_probability']:.3f})")

            st.divider()

            # Per-skill trajectory drill-down
            st.subheader("Mastery trajectory — drill into a specific skill")
            skills_with_data = [e["skill"] for e in graph]
            selected_skill = st.selectbox(
                "Select skill",
                skills_with_data,
                help="Shows how this student's mastery evolved attempt by attempt.",
            )

            attempts = kg.get_attempts(selected_student, selected_skill)
            if not attempts:
                st.warning("No attempts recorded.")
            else:
                trajectory = predictor.predict_trajectory(selected_skill, attempts)
                from core.visualisations import plot_mastery_trajectory
                fig = plot_mastery_trajectory(
                    skill=selected_skill,
                    attempts=attempts,
                    trajectory=trajectory,
                    student_id=selected_student,
                )
                st.pyplot(fig)

                with st.expander("Raw attempt history"):
                    for i, a in enumerate(attempts, 1):
                        marker = "✅" if a == 1 else "❌"
                        st.write(f"{i}. {marker}")

with tab_system:
    st.header("System info")
    st.markdown(
        "Architecture overview, model details, and runtime statistics for the Meta-Agent."
    )

    predictor = load_predictor()

    # Top-level stats
    from db.database import get_connection
    with get_connection() as conn:
        n_students = conn.execute("SELECT COUNT(*) AS c FROM students").fetchone()["c"]
        n_sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
        n_attempts = conn.execute("SELECT COUNT(*) AS c FROM attempts").fetchone()["c"]
        n_mastery = conn.execute("SELECT COUNT(*) AS c FROM mastery").fetchone()["c"]

    st.subheader("Runtime statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students", n_students)
    col2.metric("Sessions processed", n_sessions)
    col3.metric("Attempts logged", n_attempts)
    col4.metric("Mastery records", n_mastery)

    st.divider()

    # Model info
    st.subheader("BKT model")
    col_a, col_b = st.columns(2)
    col_a.metric("Skills modelled", len(predictor.params))
    col_b.metric("Trained on", "ASSISTments 2009-2010")

    st.markdown("##### Evaluation results (held-out test students)")
    eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
    eval_col1.metric("AUC", "0.8095", delta="+0.06 vs benchmark")
    eval_col2.metric("RMSE", "0.3944")
    eval_col3.metric("Accuracy", "77.1%", delta="+7.6% vs baseline")
    eval_col4.metric("F1 Score", "0.8501")
    st.caption(
        "Reference: published BKT results on ASSISTments report AUC 0.73 – 0.78. "
        "Test set: 94,280 attempts from 734 unseen students."
    )

    st.divider()

    # Browsable parameter table
    st.subheader("Learned parameters per skill")
    st.markdown(
        "BKT learned 4 parameters for each of the 95 skills via Expectation-Maximisation. "
        "Each row below describes the 'personality' of a skill."
    )

    import pandas as pd
    params_df = pd.DataFrame.from_dict(predictor.params, orient="index")
    params_df.index.name = "skill"
    params_df = params_df[["prior", "learns", "guesses", "slips", "forgets"]]
    params_df = params_df.sort_values("prior", ascending=False)

    st.dataframe(
        params_df.style.format({
            "prior":   "{:.4f}",
            "learns":  "{:.4f}",
            "guesses": "{:.4f}",
            "slips":   "{:.4f}",
            "forgets": "{:.4f}",
        }).background_gradient(subset=["prior", "learns", "guesses", "slips"], cmap="RdYlGn"),
        height=400,
        use_container_width=True,
    )

    st.caption(
        "**P(L₀) prior** — prior probability students know it before practice. "
        "**P(T) learns** — probability of learning from each attempt. "
        "**P(G) guesses** — probability of getting it right by guessing. "
        "**P(S) slips** — probability of getting it wrong despite knowing."
    )

    st.divider()

    # Architecture summary
    st.subheader("Architecture")
    st.markdown("""
    The Meta-Agent operates between sessions, not during them. The pipeline is:

    1. **Concept extractor** — LLM (Gemini) reads transcript, outputs structured `(skill, correct)` attempts
    2. **Knowledge graph** — SQLite-backed per-student concept map
    3. **BKT inference** — pure-Python implementation, computes calibrated mastery probabilities
    4. **Regression detector** — flags concepts whose mastery has dropped meaningfully
    5. **Learning path generator** — produces revise / learn next / already strong categories

    All five components are exposed via REST API endpoints and through this dashboard.
    """)