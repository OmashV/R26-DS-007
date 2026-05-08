"""
FAPR-LB Live Demo Dashboard — Deployment Mode
==============================================
Walks the panel through the deployment-flow pipeline:
  Memory → TSRP → Failure Detector → Context vector
       → LinTS picks → Tutor Agent generates → PPS scores chosen response
       → Reward → Posterior update

In deployment, PPS scores ONLY the chosen action's actual response
(not all 7 candidates upfront). For PP1 evaluation we used fixed
templates as a Tutor-Agent placeholder.

Run with:
    streamlit run app/demo_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from src.simulator.student_simulator import StudentSimulator, REPAIR_ACTIONS
from src.policy.lin_ts import LinTSBandit
from src.policy.action_templates import template_for, ACTION_TEMPLATES
from src.models.preference_scorer import PreferenceScorer
from src.pipeline.context import build_context, CONTEXT_DIM
from src.pipeline.closed_loop import compose_student_context_text, reward_function


# =============================================================
# Page config
# =============================================================
st.set_page_config(
    page_title="FAPR-LB Live Demo",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 FAPR-LB Live Demo (Deployment Mode)")
st.markdown("""
**Failure-Aware Pedagogical Repair via Contextual Bandit Strategy Selection.**
This dashboard shows the deployment-flow pipeline: the bandit picks first,
the Tutor Agent (placeholder) generates a response, then PPS scores only
that response, then the reward updates the bandit.
""")

# Pipeline overview banner
st.markdown("""
<div style='background-color:#f0f4f8; padding:10px; border-radius:5px; font-size:14px;'>
<b>Deployment Flow:</b> Memory input → TSRP struggle vector → Failure detector →
Context vector → <b>LinTS picks repair strategy</b> → Tutor Agent generates response →
PPS scores chosen response → Outcome → Reward → Posterior update
</div>
""", unsafe_allow_html=True)

st.markdown("")


# =============================================================
# Cached resource loaders
# =============================================================
@st.cache_resource
def load_pps():
    return PreferenceScorer()


@st.cache_resource
def load_simulator():
    return StudentSimulator(rng=np.random.default_rng(7))


def fresh_bandit():
    return LinTSBandit(context_dim=CONTEXT_DIM, seed=42)


# =============================================================
# Session state
# =============================================================
if "bandit" not in st.session_state:
    st.session_state.bandit = fresh_bandit()
    st.session_state.history = []
    st.session_state.current_obs = None
    st.session_state.prev_action = None
    st.session_state.prev_outcome = None
    st.session_state.turn_count = 0
    st.session_state.last_decision = None


# =============================================================
# Sidebar
# =============================================================
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Pick a new student", use_container_width=True):
        sim = load_simulator()
        obs = sim.reset_with_min_failures(min_failures=3)
        st.session_state.current_obs = obs
        st.session_state.prev_action = None
        st.session_state.prev_outcome = None
        st.session_state.turn_count = 0
        st.session_state.history = []
        st.session_state.last_decision = None
        st.rerun()

    if st.button("♻️ Reset bandit", use_container_width=True):
        st.session_state.bandit = fresh_bandit()
        st.session_state.history = []
        st.session_state.last_decision = None
        st.success("Bandit reset to uninformed prior.")

    st.markdown("---")
    st.subheader("System info")
    total_pulls = sum(a.n_pulls for a in st.session_state.bandit.arms.values())
    st.caption(f"Context dim: {CONTEXT_DIM}")
    st.caption(f"Repair actions: {len(REPAIR_ACTIONS)}")
    st.caption(f"Total bandit pulls so far: {total_pulls}")

    st.markdown("---")
    st.subheader("How to read this demo")
    st.caption(
        "Each turn runs the deployment pipeline once. PPS scores only the "
        "Tutor Agent's chosen response, NOT all 7 candidates — matching how "
        "the system would behave in production."
    )


# =============================================================
# Load models
# =============================================================
pps = load_pps()
sim = load_simulator()

if st.session_state.current_obs is None:
    st.info("👈 Click **Pick a new student** in the sidebar to begin.")
    st.stop()

obs = st.session_state.current_obs


# =============================================================
# SECTION 1 — Memory input + Failure detector
# =============================================================
st.markdown("## Step 1 — What Memory sends, and what we detect")
st.caption(
    "The Memory component sends the student's recent history. "
    "Our **rule-based failure detector** labels the current attempt."
)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    st.markdown("**Student identifiers**")
    st.text(f"Student ID: {obs['user_id']}")
    st.text(f"Skill ID:   {obs['skill_id']}")
    st.text(f"Turn # in this session: {st.session_state.turn_count + 1}")
with c2:
    st.markdown("**Recent behavioural signals**")
    st.text(f"Skill rolling correct rate: {obs['skill_correct_rate']:.2f}")
    st.text(f"Help dependency: {obs['help_dependency']:.2f}")
    st.text(f"Recent correct rate: {obs['recent_correct_rate']:.2f}")
with c3:
    failure_color = {
        "low_mastery_failure": "🟠",
        "repair_needed_failure": "🔴",
        "no_failure": "🟢",
    }.get(obs["failure_type"], "⚪")
    st.markdown("**Failure detector output**")
    st.markdown(f"### {failure_color} `{obs['failure_type']}`")
    explain = {
        "low_mastery_failure": "Student is wrong AND has weak history on this skill — needs foundation work.",
        "repair_needed_failure": "Student is struggling but has decent history — needs strategy adjustment, not foundation.",
        "no_failure": "Clean success.",
    }
    st.caption(explain.get(obs["failure_type"], ""))


# =============================================================
# SECTION 2 — TSRP outputs
# =============================================================
st.markdown("## Step 2 — Struggle Vector from TSRP (Model 1)")
st.caption(
    "Three LightGBM heads read the recent history and predict how this turn will struggle."
)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Repair-need score", f"{obs['repair_need_score']:.3f}",
              help="P(this turn requires pedagogical repair). High = student likely to fail.")
    st.progress(min(1.0, max(0.0, obs["repair_need_score"])))
with c2:
    st.metric("Effort cost", f"{obs['effort_cost_score']:.3f}",
              help="Predicted hint+attempt count. High = student will struggle effortfully.")
    norm_effort = min(1.0, obs["effort_cost_score"] / 5.0)
    st.progress(min(1.0, max(0.0, norm_effort)))
with c3:
    st.metric("Disengagement risk", f"{obs['disengagement_risk']:.3f}",
              help="Z-scored expected response time. Positive = slowing down.")
    norm_diseng = min(1.0, max(0.0, (obs["disengagement_risk"] + 1.0) / 2.0))
    st.progress(norm_diseng)


# =============================================================
# SECTION 3 — Run one turn
# =============================================================
st.markdown("## Step 3 — Run one repair turn")
st.caption(
    "Click below to run the deployment flow: build context vector, "
    "let LinTS pick, Tutor Agent generates response, PPS scores only that response, "
    "compute reward, update bandit posterior."
)

if st.button("▶️ Run next turn", type="primary", use_container_width=True):
    # --- Build context vector ---
    x = build_context(obs, st.session_state.prev_action, st.session_state.prev_outcome)

    # --- Bandit picks FIRST ---
    chosen = st.session_state.bandit.select(x)

    # --- "Tutor Agent" generates response ---
    # In PP1 this is the canned template for the chosen action.
    # In real deployment this would be replaced by the actual Tutor Agent
    # (a generative LLM that produces fresh text per turn).
    tutor_response = template_for(chosen)

    # --- PPS scores ONLY the chosen response ---
    student_text = compose_student_context_text(obs)
    chosen_pps = pps.score(student_text, tutor_response)

    # --- Simulator returns student outcome ---
    next_obs, info = sim.step(chosen)
    if info.get("skipped"):
        st.warning("Episode finished — pick a new student.")
        st.session_state.current_obs = None
    else:
        # --- Reward ---
        r = reward_function(info["sim_correct"], info["sim_hint"], chosen_pps)

        # --- Posterior update ---
        st.session_state.bandit.update(chosen, x, r)

        # Save details for display
        st.session_state.last_decision = {
            "context_vector": x.tolist(),
            "student_text": student_text,
            "chosen": chosen,
            "tutor_response": tutor_response,
            "chosen_pps": chosen_pps,
            "info": info,
            "reward": r,
        }

        st.session_state.history.append({
            "turn": st.session_state.turn_count + 1,
            "failure": obs["failure_type"],
            "action": chosen,
            "correct": info["sim_correct"],
            "hint": info["sim_hint"],
            "pps": round(chosen_pps, 3),
            "reward": round(r, 3),
        })

        st.session_state.prev_action = chosen
        st.session_state.prev_outcome = info
        st.session_state.turn_count += 1

        if info["done"]:
            st.success("Student finished all failure rows. Pick a new student to continue.")
            st.session_state.current_obs = None
        else:
            st.session_state.current_obs = next_obs

    st.rerun()


# =============================================================
# SECTION 4–8 — What happened on the last turn (rich detail)
# =============================================================
if st.session_state.last_decision is not None:
    d = st.session_state.last_decision

    st.markdown("---")

    # ---------------------------------------------------------
    # STEP 4 — Context vector built
    # ---------------------------------------------------------
    st.markdown("## Step 4 — Context vector built for the bandit")
    st.caption(
        "The bandit reads only this 10-dim vector. It cannot see raw text or history — "
        "only these numbers."
    )

    ctx_labels = [
        "0: bias (constant 1)",
        "1: repair_need (TSRP)",
        "2: effort_cost (TSRP, z-scored)",
        "3: disengagement_risk (TSRP)",
        "4: skill_correct_rate",
        "5: failure=low_mastery (one-hot)",
        "6: failure=repair_needed (one-hot)",
        "7: prev_action_was_repair",
        "8: prev_outcome_correct",
        "9: prev_outcome_used_hint",
    ]
    ctx_df = pd.DataFrame({
        "Dimension": ctx_labels,
        "Value": [round(v, 3) for v in d["context_vector"]],
    })
    st.dataframe(ctx_df, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # STEP 5 — LinTS picks
    # ---------------------------------------------------------
    st.markdown("## Step 5 — LinTS picked a repair strategy")
    st.caption(
        "The bandit selects ONE of the 7 repair strategies based on its learned "
        "context-conditioned posterior. No PPS scores were consulted at this stage."
    )

    st.success(f"**Chosen strategy:** `{d['chosen']}`")
    st.caption(
        "The bandit picked this because, based on past observations, this strategy "
        "is predicted to give the highest reward in the current context."
    )

    # ---------------------------------------------------------
    # STEP 6 — Tutor Agent generates response
    # ---------------------------------------------------------
    st.markdown("## Step 6 — Tutor Agent generates the actual response")
    st.caption(
        "In deployment, the Tutor Agent (a generative LLM) would produce a fresh, "
        "student-tailored response based on the chosen strategy. In this PP1 demo, "
        "we use a fixed canned template as a Tutor-Agent placeholder."
    )

    st.info(d["tutor_response"])
    st.caption(
        "🛈 Future work: Replace this placeholder with the group's live Tutor Agent. "
        "PPS will then score the agent's actual generated text, not a template."
    )

    # ---------------------------------------------------------
    # STEP 7 — PPS scores the chosen response
    # ---------------------------------------------------------
    st.markdown("## Step 7 — PPS scores the chosen response")
    st.caption(
        "PPS (DeBERTa-v3 fine-tuned on MathDial) reads the student-context snippet "
        "and the Tutor Agent's actual response. It returns one pedagogical-quality "
        "score in [0, 1]."
    )

    with st.expander("📄 Student-context snippet given to PPS"):
        st.code(d["student_text"])

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("PPS score (chosen response)", f"{d['chosen_pps']:.3f}")
        st.caption("Higher = pedagogically better teaching.")
    with c2:
        st.markdown("**What PPS evaluated:**")
        st.markdown(f"- Student context: *(see expandable above)*")
        st.markdown(f"- Tutor response: *(from Step 6)*")
        st.caption(
            "PPS does NOT influence which action was chosen. It is a quality "
            "signal that enters the reward formula afterwards."
        )

    # ---------------------------------------------------------
    # STEP 8 — Student outcome + Reward + Posterior update
    # ---------------------------------------------------------
    st.markdown("## Step 8 — Student outcome, reward, and bandit update")
    st.caption(
        "The simulator (held-out ASSISTments students) returns the student's "
        "response. Reward combines correctness, PPS quality, and hint-cost. "
        "LinTS posterior updates for the chosen arm — this is the self-reflection step."
    )

    info = d["info"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Correct?", "✅ Yes" if info["sim_correct"] else "❌ No")
    with c2:
        st.metric("Hint used?", "Yes" if info["sim_hint"] > 0 else "No")
    with c3:
        st.metric("PPS for chosen action", f"{d['chosen_pps']:.3f}")
    with c4:
        st.metric("**Reward**", f"{d['reward']:.3f}")

    st.markdown(
        f"**Reward formula:** "
        f"`{d['reward']:.3f} = 0.5 × {info['sim_correct']} "
        f"+ 0.3 × {d['chosen_pps']:.3f} "
        f"− 0.2 × {1 if info['sim_hint'] > 0 else 0}`"
    )

    st.success(
        f"✅ Bandit posterior updated for arm `{d['chosen']}`. "
        f"Next time a similar context appears, the bandit's belief about this "
        f"action is more confident."
    )


# =============================================================
# Episode history
# =============================================================
if st.session_state.history:
    st.markdown("---")
    st.markdown("## Episode history (this student's session so far)")
    st.dataframe(pd.DataFrame(st.session_state.history),
                 use_container_width=True, hide_index=True)


# =============================================================
# Bandit posterior live view
# =============================================================
st.markdown("---")
st.markdown("## Bandit posterior — pull counts per action")
st.caption(
    "Updates live as the bandit acts. Shows what the bandit has been favouring "
    "across all turns this session."
)
arm_counts = {a: arm.n_pulls for a, arm in st.session_state.bandit.arms.items()}
arm_df = pd.DataFrame({
    "action": list(arm_counts.keys()),
    "pulls": list(arm_counts.values()),
}).sort_values("pulls", ascending=False)
st.bar_chart(arm_df.set_index("action"))