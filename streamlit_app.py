"""
Meta-Agent Dashboard — interactive demo interface.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st

from bkt.predict import BKTPredictor
from core.knowledge_graph import KnowledgeGraph
from core.concept_extractor import ConceptExtractor
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent

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
    # ───── Pseudo-observation showcase presets ─────

    "🟢 [Signals demo] Clean correct answers + explanation": """\
TUTOR: Let's work on percentages. What's 25% of 80?
STUDENT: 20.
TUTOR: Right. What's 15% of 60?
STUDENT: 9.
TUTOR: How did you work that out?
STUDENT: 15 over 100 times 60. Same as 0.15 times 60.
TUTOR: Exactly. What's 40% of 50?
STUDENT: 20.
TUTOR: Perfect.""",

    "🟡 [Signals demo] Mixed — incorrect, partial, then correct": """\
TUTOR: What's 30% of 60?
STUDENT: Um... 30?
TUTOR: Not quite. 30% means 30 out of every 100. So it's 0.3 times 60.
STUDENT: Oh, 18.
TUTOR: Right. Now try 25% of 40.
STUDENT: Is it 10?
TUTOR: Yes! How did you get there?
STUDENT: I just figured 25% is a quarter, so a quarter of 40 is 10.
TUTOR: Great thinking. Try 50% of 90.
STUDENT: 45.
TUTOR: Perfect.""",

    "🔴 [Signals demo] Confusion and clarification requests": """\
TUTOR: What's 20% of 50?
STUDENT: I don't get percentages.
TUTOR: Let me explain. Percent means out of 100. So 20% is 20 out of 100.
STUDENT: Can you explain that again?
TUTOR: Sure. If you have 100 sweets, 20% is 20 of them.
STUDENT: I'm still confused.
TUTOR: Let's try a smaller example. What's 10% of 100?
STUDENT: 100?
TUTOR: Not quite. 10% of 100 is 10.
STUDENT: Wait, what?""",

    "🟠 [Signals demo] Repeated misunderstanding (subtraction error)": """\
TUTOR: What's 25% of 80?
STUDENT: 25.
TUTOR: Not quite. Remember, percent means out of 100, so 25% is 0.25 of the number.
STUDENT: Oh, so 25% of 80 is 80 minus 25?
TUTOR: No — it's 80 times 0.25. So one quarter of 80.
STUDENT: 20.
TUTOR: Right. Try 30% of 50.
STUDENT: 50 minus 30, so 20?
TUTOR: Hmm, that's the wrong approach. 30% of 50 is 0.3 times 50, which is 15.
STUDENT: Oh. So it's like multiplication, not subtraction.
TUTOR: Yes — percents are about parts of a whole, not what's left over.""",

    "🔵 [Signals demo] Mixed skills — equations and percentages": """\
TUTOR: Let's start with equation solving. Solve 2x + 4 = 10.
STUDENT: x equals 3.
TUTOR: Right. Try 5x - 7 = 18.
STUDENT: 5.
TUTOR: Correct. Now let's switch to percentages. What's 20% of 90?
STUDENT: Um... 18?
TUTOR: Yes. How did you get there?
STUDENT: 20 percent is one fifth, and one fifth of 90 is 18.
TUTOR: Excellent. Try 75% of 40.
STUDENT: 30.
TUTOR: Perfect.""",

    "🟣 [Signals demo] Student-initiated question (no upfront signals)": """\
STUDENT: What's 30% of 50?
TUTOR: Let me walk you through it. 30% means 30 out of 100. What's half of 30?
STUDENT: 15.
TUTOR: Right — so 30% of 50 is 15.
STUDENT: Oh that makes sense.
TUTOR: Try 20% of 80.
STUDENT: 16.
TUTOR: Perfect.""",
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
    "🔵 [Cold-start demo 1A] Master Adding/Subtracting Fractions": """\
TUTOR: Let's work on adding fractions. What's 1/4 + 1/4?
STUDENT: 2/4
TUTOR: Right. Try 1/3 + 1/3.
STUDENT: 2/3
TUTOR: Good. What's 3/8 minus 1/8?
STUDENT: 2/8
TUTOR: Excellent. Try 5/9 - 2/9.
STUDENT: 3/9
TUTOR: Perfect. One more — 2/7 + 4/7?
STUDENT: 6/7
TUTOR: Brilliant, you've got this.""",
    "⭐ [Cold-start demo 1B] First encounter: Equivalent Fractions": """\
TUTOR: Now let's look at equivalent fractions. Is 2/4 the same as 1/2?
STUDENT: I think so.
TUTOR: Yes. Is 3/9 equivalent to 1/3?
STUDENT: yes
TUTOR: Good. What about 4/10 and 2/5?
STUDENT: yes
TUTOR: Right. Is 6/8 the same as 3/4?
STUDENT: yes
TUTOR: Excellent.""",
    "🔵 [Cold-start demo 2A] Master Equation Solving (≤2 steps)": """\
TUTOR: Let's start with simple equations. Solve x + 5 = 12.
STUDENT: x = 7
TUTOR: Right. Try 2x = 10.
STUDENT: x = 5
TUTOR: Good. Solve 2x + 3 = 11.
STUDENT: x = 4
TUTOR: Perfect. What about 3y - 4 = 8?
STUDENT: y = 4
TUTOR: Excellent. One more — 5z + 2 = 17.
STUDENT: z = 3
TUTOR: Brilliant.""",
    "⭐ [Cold-start demo 2B] First encounter: Equation Solving (>2 steps)": """\
TUTOR: Let's try harder ones. Solve 3x + 4 - x = 10.
STUDENT: x = 3
TUTOR: Yes. Try 2(x + 3) = 14.
STUDENT: x = 4
TUTOR: Good. Solve 4x - 2 = 2x + 6.
STUDENT: x = 4
TUTOR: Excellent.""",
    "🔵 [Cold-start demo 3A] Master Volume of Rectangular Prism": """\
TUTOR: Find the volume of a box that is 3 by 4 by 5.
STUDENT: 60
TUTOR: Right. What about a box 2 by 6 by 4?
STUDENT: 48
TUTOR: Good. Find the volume of a 5 by 5 by 3 box.
STUDENT: 75
TUTOR: Perfect. Now a 7 by 2 by 4 box.
STUDENT: 56
TUTOR: Excellent.""",
    "⭐ [Cold-start demo 3B] First encounter: Surface Area of Rectangular Prism": """\
TUTOR: Now let's find surface area. For a box 2 by 3 by 4, what's the surface area?
STUDENT: 52
TUTOR: Yes. Try a 3 by 3 by 5 box.
STUDENT: 78
TUTOR: Good. What about 4 by 5 by 6?
STUDENT: 148
TUTOR: Excellent.""",
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
                with st.spinner("Extracting conversational signals via LLM..."):
                    extractor = load_extractor()
                    extraction = extractor.extract(transcript)

                st.success(f"Extracted {len(extraction['signals'])} learning signals")

                # Show extracted signals with confidence levels
                with st.expander("📋 Extracted conversational signals", expanded=True):
                    st.caption(
                        "Each signal is a piece of evidence about student mastery, "
                        "weighted by confidence. Standard binary attempts have confidence 1.0; "
                        "softer evidence (confusion, partial reasoning) has lower confidence."
                    )
                    for s in extraction["signals"]:
                        marker = "✅" if s["label"] == 1 else "❌"
                        sig_type = s.get("signal_type", "unknown")
                        confidence = s.get("confidence", 1.0)
                        # Visual confidence bar
                        bar_width = int(confidence * 100)
                        bar_color = "#10b981" if s["label"] == 1 else "#ef4444"
                        st.markdown(
                            f"{marker}  **{s['skill']}**  "
                            f"`{sig_type}`  &nbsp; "
                            f"<span style='background: linear-gradient(to right, "
                            f"{bar_color} 0%, {bar_color} {bar_width}%, "
                            f"#e5e7eb {bar_width}%, #e5e7eb 100%); "
                            f"padding: 2px 8px; border-radius: 4px; "
                            f"color: white; font-size: 0.75em; font-weight: 600;'>"
                            f"conf {confidence:.1f}</span>",
                            unsafe_allow_html=True,
                        )
                    if extraction["misconceptions"]:
                        st.markdown("**Misconceptions detected:**")
                        for m in extraction["misconceptions"]:
                            st.write(f"- {m}")

                # Run through knowledge graph
                with st.spinner("Updating knowledge graph (running BKT with confidence-weighted signals)..."):
                    kg = load_knowledge_graph()
                    session = kg.process_session(
                        student_id=student_id,
                        signals=extraction["signals"],
                    )

                st.success(f"Session {session['session_id']} processed")

                # Show updated mastery, highlighting cold-start transfers
                with st.expander("🧠 Updated knowledge graph", expanded=True):
                    # First, the skills affected this session (with cold-start details)
                    cold_start_skills = [s for s in session["skills_updated"] if s.get("cold_start_used")]
                    if cold_start_skills:
                        st.markdown("**⭐ Cold-start transfer applied** to skills the student encountered for the first time:")
                        for s in cold_start_skills:
                            details = s["cold_start_details"]
                            related = ", ".join(details["related_skills_used"])
                            st.info(
                                f"**{s['skill']}** — prior boosted "
                                f"{details['population_prior']:.3f} → {details['transferred_prior']:.3f} "
                                f"based on student's mastery of: *{related}*"
                            )

                    st.markdown("**Current mastery for affected skills:**")
                    for s in session["skills_updated"]:
                        p = s["probability"]
                        label = s["label"]
                        if label == "strong":
                            colour = "🟢"
                        elif label == "weak":
                            colour = "🔴"
                        else:
                            colour = "🟡"
                        cold_start_marker = "  ⭐" if s.get("cold_start_used") else ""
                        st.write(
                            f"{colour}  **{s['skill']}** — "
                            f"P(mastery) = {p:.3f} ({label}){cold_start_marker}"
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

                # Extract just the labels for the plot's marker rendering
                attempt_labels = [a[0] if isinstance(a, tuple) else a for a in attempts]

                from core.visualisations import plot_mastery_trajectory
                fig = plot_mastery_trajectory(
                    skill=selected_skill,
                    attempts=attempt_labels,
                    trajectory=trajectory,
                    student_id=selected_student,
                )
                st.pyplot(fig)

                with st.expander("Raw signal history"):
                    for i, a in enumerate(attempts, 1):
                        if isinstance(a, tuple):
                            label, confidence = a
                            marker = "✅" if label == 1 else "❌"
                            conf_text = f" (confidence {confidence:.1f})" if confidence < 1.0 else ""
                            st.write(f"{i}. {marker}{conf_text}")
                        else:
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

    st.divider()

    # Skill similarity explorer (cold-start novelty)
    st.subheader("Skill similarity explorer")
    st.markdown(
        "The system uses sentence embeddings to compute semantic similarity between "
        "skill names. Skills with similarity ≥ 0.5 are considered related and used "
        "for **cold-start knowledge transfer** — when a new student first encounters "
        "a skill, the system transfers evidence from related skills they've already "
        "mastered, rather than starting from the population prior."
    )

    # Load similarity data
    import json
    similarity_path = PROJECT_ROOT / "models" / "skill_similarity.json"
    if not similarity_path.exists():
        st.warning(
            "Skill similarity matrix not found. Run `notebooks/skill_similarity.ipynb` "
            "to generate it. The system falls back to population priors without it."
        )
    else:
        with open(similarity_path) as f:
            similarity_data = json.load(f)

        col_picker, col_stats = st.columns([1.5, 1])

        with col_picker:
            picked_skill = st.selectbox(
                "Choose a skill to see its related neighbours",
                sorted(similarity_data.keys()),
            )

        related = similarity_data.get(picked_skill, {})
        related_sorted = sorted(related.items(), key=lambda kv: -kv[1])

        with col_stats:
            st.metric("Related skills", len(related_sorted))
            if related_sorted:
                st.metric("Strongest similarity", f"{related_sorted[0][1]:.3f}")

        if not related_sorted:
            st.info(
                f"**{picked_skill}** has no related skills above the 0.5 threshold. "
                f"It is treated as conceptually isolated for cold-start purposes — "
                f"new students encountering this skill use the population prior."
            )
        else:
            st.markdown(f"**Skills related to *{picked_skill}*:**")
            for related_skill, sim in related_sorted:
                # Visual bar showing similarity strength
                bar_width = int(sim * 100)
                st.markdown(
                    f"`{sim:.3f}`  **{related_skill}**  \n"
                    f"<div style='background: linear-gradient(to right, "
                    f"#4CAF50 0%, #4CAF50 {bar_width}%, #eee {bar_width}%, #eee 100%); "
                    f"height: 8px; border-radius: 4px; margin-bottom: 8px;'></div>",
                    unsafe_allow_html=True,
                )

        # Global stats
        st.divider()
        st.markdown("**Across the full skill set:**")

        n_skills = len(similarity_data)
        n_with_relations = sum(1 for r in similarity_data.values() if r)
        n_isolated = n_skills - n_with_relations
        total_pairs = sum(len(r) for r in similarity_data.values()) // 2
        all_sims = [s for r in similarity_data.values() for s in r.values()]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total skills", n_skills)
        col2.metric("With related skills", n_with_relations)
        col3.metric("Isolated skills", n_isolated)
        col4.metric("Total related pairs", total_pairs)

        if all_sims:
            st.caption(
                f"Mean similarity (above threshold): {sum(all_sims) / len(all_sims):.3f}.  "
                f"Maximum similarity: {max(all_sims):.3f}.  "
                f"Embeddings via sentence-transformers (all-MiniLM-L6-v2)."
            )

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