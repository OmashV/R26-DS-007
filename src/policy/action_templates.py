"""
Canned tutor-utterance templates per repair action.

Each template represents what the action "looks like" when realised
as text. The PPS scorer evaluates these templates against the current
student context to give the bandit an action-conditioned pedagogical
quality score.

The templates are uniform across policies (random / fixed / rule-based /
bandit) so no policy is favoured by template choice. In deployment,
these are replaced by the actual Tutor Agent's generated utterances.
"""

ACTION_TEMPLATES = {
    "worked_example": (
        "Let me show you a similar example step by step. "
        "If we had to solve a problem like this, here is how we'd think about it: "
        "first, identify what we know; then, write down each step clearly. "
        "Now you try the same approach on your problem."
    ),
    "direct_correction": (
        "I see where the difficulty is. The step you're missing is to set up "
        "the relationship between the quantities before computing. Let me point "
        "out exactly where the reasoning needs adjusting, then walk it through with you."
    ),
    "scaffolded_question": (
        "Let's break this down with a small question first. "
        "What is the very first thing you'd compute here, just one step? "
        "Show me only that first step."
    ),
    "prerequisite_review": (
        "Before we go further, let's check the underlying skill. "
        "Can you remind me how this kind of operation works in a simpler case? "
        "We'll come back to the main problem once that piece is solid."
    ),
    "hint": (
        "Here's a small hint to nudge you forward. "
        "Think about what relationship connects the quantities in the problem. "
        "What does that suggest about the next step?"
    ),
    "simpler_explanation": (
        "Let me explain this in a simpler way. "
        "The idea here is just that two things are being combined or compared. "
        "Once you see which one, the rest follows naturally."
    ),
    "conceptual_analogy": (
        "Think of it like sharing slices of a pizza or filling a container. "
        "The same logic that applies in that everyday situation applies here. "
        "Does that picture help you see the next step?"
    ),
}

def template_for(action: str) -> str:
    return ACTION_TEMPLATES[action]