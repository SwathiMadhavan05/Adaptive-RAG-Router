"""
Labeled training data for the query router classifier.

4 classes:
  0 = PARAMETRIC   -> model already knows this, no retrieval needed
  1 = SIMPLE_RAG   -> single retrieval pass answers it
  2 = MULTI_HOP    -> needs retrieve -> reason -> retrieve again
  3 = WEB_SEARCH   -> needs current/external info not in the corpus

NOTE: Swap/expand these examples to match YOUR actual document corpus.
The "SIMPLE_RAG" and "MULTI_HOP" examples below assume a corpus of
company policy / product documentation. Replace with real examples
from whatever docs you index.
"""

LABELS = {
    0: "PARAMETRIC",
    1: "SIMPLE_RAG",
    2: "MULTI_HOP",
    3: "WEB_SEARCH",
}

TRAINING_DATA = [
    # ---------------- PARAMETRIC (0) ----------------
    ("What is the capital of France?", 0),
    ("What's 15 times 12?", 0),
    ("Who wrote Romeo and Juliet?", 0),
    ("Explain what a binary search tree is.", 0),
    ("What's the difference between a list and a tuple in Python?", 0),
    ("How many continents are there?", 0),
    ("What is the boiling point of water in Celsius?", 0),
    ("Define photosynthesis.", 0),
    ("What does HTTP stand for?", 0),
    ("Hi, how are you?", 0),
    ("Can you tell me a joke?", 0),
    ("What is the time complexity of quicksort?", 0),
    ("Translate 'good morning' to Spanish.", 0),
    ("What year did World War 2 end?", 0),
    ("Explain the difference between TCP and UDP.", 0),
    ("What is recursion in programming?", 0),
    ("Who is the author of '1984'?", 0),
    ("What's the formula for the area of a circle?", 0),
    ("What is machine learning?", 0),
    ("How does a hash table work?", 0),
    ("What's the chemical symbol for gold?", 0),
    ("Summarize the plot of Hamlet.", 0),
    ("What is REST API?", 0),
    ("Convert 100 Fahrenheit to Celsius.", 0),
    ("What's the largest planet in our solar system?", 0),
    ("How does TCP's three-way handshake work?", 0),
    ("What is the difference between supervised and unsupervised learning?", 0),
    ("Who developed the theory of relativity?", 0),
    ("What is a stack data structure used for?", 0),
    ("How do you reverse a string in Python?", 0),
    ("What's the difference between SQL and NoSQL databases?", 0),
    ("Explain what Big O notation means.", 0),
    ("What is the speed of light?", 0),
    ("How many bits are in a byte?", 0),
    ("What's the difference between a process and a thread?", 0),
    ("Explain how a neural network's backpropagation works.", 0),
    ("What is the Pythagorean theorem?", 0),
    ("Who is considered the father of computer science?", 0),
    ("What's the difference between == and === in JavaScript?", 0),
    ("Define what an algorithm is.", 0),
    ("What's a good joke about programmers?", 0),
    ("Thanks, that helps!", 0),
    ("What is Newton's second law of motion?", 0),
    ("How do you calculate compound interest?", 0),
    ("What's the difference between Git and GitHub?", 0),

    # ---------------- SIMPLE_RAG (1) ----------------
    ("What is our company's refund policy?", 1),
    ("How many days of sick leave do employees get?", 1),
    ("What does section 4.2 of the contract say about termination?", 1),
    ("What is the maximum file upload size for the API?", 1),
    ("What are the steps to reset a user's password according to our docs?", 1),
    ("What's the warranty period for product X?", 1),
    ("What is the rate limit for the /search endpoint?", 1),
    ("What benefits are included in the premium plan?", 1),
    ("How do I request reimbursement for travel expenses?", 1),
    ("What is the process for submitting a bug report?", 1),
    ("What's the minimum age requirement mentioned in the terms of service?", 1),
    ("What is the late fee policy for overdue invoices?", 1),
    ("How long is the trial period for the subscription?", 1),
    ("What does the onboarding checklist include?", 1),
    ("What is the escalation contact for P1 incidents?", 1),
    ("What's the data retention period stated in the privacy policy?", 1),
    ("What are the supported file formats for upload?", 1),
    ("What is the SLA uptime guarantee?", 1),
    ("How do I cancel my subscription according to the help docs?", 1),
    ("What's listed as the eligibility criteria for the loyalty program?", 1),
    ("What is the password complexity requirement in the security policy?", 1),
    ("What's the return window for purchased items?", 1),
    ("What does the document say about remote work eligibility?", 1),
    ("What is the API authentication method described in the docs?", 1),
    ("How many requests per day can a Premium plan make?", 1),
    ("What's the credit percentage if uptime falls between 95% and 99%?", 1),
    ("What is the initial response time for a P2 support ticket?", 1),
    ("How long are backups retained according to the privacy policy?", 1),
    ("What's the onboarding length for contractors?", 1),
    ("What is the maximum upload size for the Basic plan?", 1),
    ("According to the SLA, what uptime percentage is guaranteed?", 1),
    ("What happens to access when a contractor's contract ends?", 1),
    ("What does the refund policy say about Enterprise contracts?", 1),
    ("How is a P1 incident defined in the escalation policy?", 1),
    ("What's the rate limit per minute for Enterprise customers?", 1),
    ("Within how many days must GDPR deletion requests be fulfilled?", 1),
    ("What does the cancellation policy say about Enterprise notice periods?", 1),
    ("What training must contractors complete before getting system access?", 1),

    # ---------------- MULTI_HOP (2) ----------------
    ("Compare the refund policies in the basic plan vs the premium plan.", 2),
    ("How does our termination clause differ between the 2022 and 2023 contracts?", 2),
    ("What changed between version 1 and version 2 of the API rate limits?", 2),
    ("Which plan offers better support response times, and how does that compare to its pricing?", 2),
    ("Summarize the differences between our data retention policy and our deletion policy, and explain how they interact.", 2),
    ("If an employee is on probation, do the sick leave and termination policies both apply the same way?", 2),
    ("Compare the security requirements in the API docs against what's stated in the compliance policy.", 2),
    ("What's the relationship between the escalation policy and the SLA guarantees?", 2),
    ("How does the refund policy interact with the cancellation policy if a user cancels mid-cycle?", 2),
    ("Compare onboarding steps for contractors vs full-time employees and note any conflicts.", 2),
    ("Does the warranty policy contradict anything in the return policy?", 2),
    ("Trace how a support ticket escalates from P3 to P1 according to both the escalation doc and SLA doc.", 2),
    ("Compare pricing tiers across the basic, pro, and enterprise plans and identify what's missing in basic.", 2),
    ("How do the data privacy terms differ between the EU and US sections of the policy?", 2),
    ("What's the combined effect of the late fee policy and the grace period policy on an overdue account?", 2),
    ("How does the Basic plan's lack of an SLA interact with its refund policy if there's an outage?", 2),
    ("If a P1 ticket breaches the SLA, what credit does the customer get and how is escalation handled?", 2),
    ("Compare data retention timelines between active accounts and deleted accounts.", 2),
    ("How does contractor offboarding relate to the onboarding access rules?", 2),
    ("What's the interaction between rate limits and file upload size across plan tiers?", 2),
    ("If an Enterprise customer cancels, how do the notice period and refund terms both apply?", 2),
    ("Compare how P1 vs P3 tickets are handled and how that maps to SLA credit tiers.", 2),
    ("Does the EU data storage requirement conflict with the standard backup retention policy?", 2),
    ("How do the Basic and Premium plans differ in both rate limits and refund eligibility?", 2),
    ("What's the full chain of escalation from a P3 ticket up to SLA credit eligibility?", 2),
    ("Compare the security training requirement against the system access timeline for new hires.", 2),

    # ---------------- WEB_SEARCH (3) ----------------
    ("What's the latest news on the Federal Reserve's interest rate decision?", 3),
    ("Who won the most recent F1 race?", 3),
    ("What's the current stock price of NVIDIA?", 3),
    ("Are there any recent outages reported for AWS today?", 3),
    ("What's the weather forecast for tomorrow in Mumbai?", 3),
    ("What's the latest version of Python released this year?", 3),
    ("Did our competitor just release a new product?", 3),
    ("What are people saying about the new iPhone release on social media?", 3),
    ("What's the current exchange rate between USD and INR?", 3),
    ("Is there a new CVE reported for this library recently?", 3),
    ("What's trending in AI news this week?", 3),
    ("Has there been a recent update to GDPR regulations?", 3),
    ("What's the latest score in the ongoing match?", 3),
    ("Did the company announce earnings today?", 3),
    ("What's the most recent release notes for LangChain?", 3),
    ("Has the Federal Reserve announced anything today?", 3),
    ("What's the current price of gold per ounce?", 3),
    ("Did any major tech company have layoffs this week?", 3),
    ("What's the latest iPhone model available right now?", 3),
    ("Is there a new version of PyTorch released recently?", 3),
    ("What happened in the news today?", 3),
    ("Who's currently leading the Premier League table?", 3),
    ("What's the latest update on the war in Ukraine?", 3),
    ("Are there any new AI model releases this month?", 3),
    ("What's the current unemployment rate in the US?", 3),
    ("Has there been a security breach reported recently for any major company?", 3),
    ("What's the latest Tesla stock price?", 3),
    ("What new features did the latest ChatGPT update add?", 3),
]

if __name__ == "__main__":
    from collections import Counter
    counts = Counter(label for _, label in TRAINING_DATA)
    print("Class distribution:")
    for label_id, count in sorted(counts.items()):
        print(f"  {LABELS[label_id]}: {count}")
    print(f"Total examples: {len(TRAINING_DATA)}")
