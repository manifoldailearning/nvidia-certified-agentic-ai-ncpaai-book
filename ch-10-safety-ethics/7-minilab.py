from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
import re

# -----------------------------------------
# 1. Define State
# -----------------------------------------
class GuardrailState(TypedDict, total=False):
    response: str                    # The text to be checked / cleaned
    guardrail_status: str            # "passed" | "blocked"
    blocked_reason: Optional[str]    # Why it was blocked (if blocked)
    pii_found: bool                  # Was any PII detected?
    toxicity_flag: bool              # Was toxic content detected?


# -----------------------------------------
# 2. Config: Restricted Terms & Patterns
# -----------------------------------------
RESTRICTED_TERMS = {
    "hate",
    "violence",
    "discrimination",
}

PHONE_REGEX = r"\b\d{10}\b"
EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"


# -----------------------------------------
# 3. Guardrail Node
# -----------------------------------------
def guardrail_node(state: GuardrailState) -> GuardrailState:
    raw_response = state["response"]

    # Flags for audit
    toxicity = False
    pii = False

    # --- 3.1 Check for restricted / toxic content ---
    if any(term in raw_response.lower() for term in RESTRICTED_TERMS):
        toxicity = True
        return {
            "response": "Policy violation detected. Content blocked by guardrails.",
            "guardrail_status": "blocked",
            "blocked_reason": "restricted_terms",
            "pii_found": False,
            "toxicity_flag": True,
        }

    # --- 3.2 Mask PII (phone, email) ---
    masked = raw_response

    if re.search(PHONE_REGEX, masked):
        pii = True
        masked = re.sub(PHONE_REGEX, "[PHONE]", masked)

    if re.search(EMAIL_REGEX, masked):
        pii = True
        masked = re.sub(EMAIL_REGEX, "[EMAIL]", masked)

    # --- 3.3 Return enriched state ---
    return {
        "response": masked,
        "guardrail_status": "passed",
        "blocked_reason": None,
        "pii_found": pii,
        "toxicity_flag": toxicity,
    }


# -----------------------------------------
# 4. Build & Compile Graph
# -----------------------------------------
graph = StateGraph(GuardrailState)

graph.add_node("guardrails", guardrail_node)
graph.add_edge(START, "guardrails")
graph.add_edge("guardrails", END)

app = graph.compile()

# -----------------------------------------
# 5. Test Harness
# -----------------------------------------
test_inputs = [
    "This contains hate and my email is abc@example.com",
    "You can call me at 9876543210 for details.",
    "Normal text, nothing sensitive here.",
]

for text in test_inputs:
    result = app.invoke({"response": text})
    print("INPUT :", text)
    print("OUTPUT:", result["response"])
    print("STATUS:", result.get("guardrail_status"))
    print("PII   :", result.get("pii_found"))
    print("TOXIC :", result.get("toxicity_flag"))
    print("REASON:", result.get("blocked_reason"))
    print("-" * 60)
