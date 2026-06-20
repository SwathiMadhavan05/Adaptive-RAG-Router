"""
Evaluation set: queries paired with their correct route AND, where
applicable, a ground-truth answer (or key facts that must appear in
the answer) so we can measure both:
  1. Router accuracy (did it pick the right strategy?)
  2. Answer quality (did the chosen strategy actually produce a correct answer?)

This is intentionally a different/held-out set from data/training_data.py
-- evaluating on your training examples would inflate accuracy numbers
and isn't meaningful.
"""

EVAL_SET = [
    # ---- PARAMETRIC ----
    {"query": "What is the capital of Japan?", "expected_route": "PARAMETRIC",
     "must_contain": ["tokyo"]},
    {"query": "What is 9 times 8?", "expected_route": "PARAMETRIC",
     "must_contain": ["72"]},
    {"query": "What is a linked list?", "expected_route": "PARAMETRIC",
     "must_contain": ["node"]},
    {"query": "Who painted the Mona Lisa?", "expected_route": "PARAMETRIC",
     "must_contain": ["vinci"]},
    {"query": "What does CPU stand for?", "expected_route": "PARAMETRIC",
     "must_contain": ["central", "processing"]},
    {"query": "Explain what an API is.", "expected_route": "PARAMETRIC",
     "must_contain": ["interface"]},
    {"query": "What's the freezing point of water in Fahrenheit?", "expected_route": "PARAMETRIC",
     "must_contain": ["32"]},
    {"query": "What is object-oriented programming?", "expected_route": "PARAMETRIC",
     "must_contain": ["object"]},

    # ---- SIMPLE_RAG (answerable from sample_documents.py) ----
    {"query": "How long is the refund window for the Premium plan?", "expected_route": "SIMPLE_RAG",
     "must_contain": ["30 days"]},
    {"query": "What is the rate limit for Basic plan API requests per minute?", "expected_route": "SIMPLE_RAG",
     "must_contain": ["100"]},
    {"query": "What is the maximum API file upload size for the Enterprise plan?", "expected_route": "SIMPLE_RAG",
     "must_contain": ["500"]},
    {"query": "How many days notice must Enterprise customers give to cancel?", "expected_route": "SIMPLE_RAG",
     "must_contain": ["30"]},
    {"query": "What service credit do customers get if uptime drops below 95%?", "expected_route": "SIMPLE_RAG",
     "must_contain": ["50"]},
    {"query": "What is the initial response time for a P1 support ticket?", "expected_route": "SIMPLE_RAG",
     "must_contain": ["1 hour"]},
    {"query": "How long is data retained after account deletion?", "expected_route": "SIMPLE_RAG",
     "must_contain": ["90 days"]},
    {"query": "How many days of onboarding do full-time employees get?", "expected_route": "SIMPLE_RAG",
     "must_contain": ["5"]},

    # ---- MULTI_HOP (requires combining 2+ documents) ----
    {"query": "If a Premium customer cancels mid-cycle, do they get a refund for the remaining days, and why or why not?",
     "expected_route": "MULTI_HOP", "must_contain": ["no", "prorated"]},
    {"query": "How does a P1 ticket relate to the SLA uptime guarantee?",
     "expected_route": "MULTI_HOP", "must_contain": ["uptime", "p1"]},
    {"query": "Compare onboarding access timelines for full-time employees vs contractors.",
     "expected_route": "MULTI_HOP", "must_contain": ["5", "1"]},
    {"query": "Do EU and US customers have the same data deletion guarantees?",
     "expected_route": "MULTI_HOP", "must_contain": ["gdpr", "30"]},
    {"query": "How does the cancellation policy interact with the refund policy for a mid-cycle cancellation?",
     "expected_route": "MULTI_HOP", "must_contain": ["cancel", "refund"]},

    # ---- WEB_SEARCH ----
    {"query": "What is the current price of Bitcoin?", "expected_route": "WEB_SEARCH", "must_contain": []},
    {"query": "What's the latest version of the Python programming language released?", "expected_route": "WEB_SEARCH", "must_contain": []},
    {"query": "Is there any recent news about OpenAI today?", "expected_route": "WEB_SEARCH", "must_contain": []},
    {"query": "What's the weather like in Delhi right now?", "expected_route": "WEB_SEARCH", "must_contain": []},
]
