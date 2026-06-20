REFUND_POLICY = """
Refund Policy

Basic Plan: Customers on the Basic plan may request a full refund within
14 days of purchase, provided usage is under 100 API calls. After 14 days,
no refunds are issued for the Basic plan.

Premium Plan: Customers on the Premium plan may request a full refund
within 30 days of purchase, regardless of usage. Refund requests after
30 days are evaluated on a case-by-case basis by the billing team and
may be granted as account credit instead of a cash refund.

Enterprise Plan: Refunds for Enterprise contracts are governed by the
terms negotiated in the individual service agreement and are not subject
to the standard 14/30 day windows above.

All refund requests must be submitted through the billing portal. Refunds
are processed within 5-7 business days of approval.
"""

CANCELLATION_POLICY = """
Cancellation Policy

Customers may cancel their subscription at any time from the account
settings page. Cancellation takes effect at the end of the current
billing cycle; customers retain access to paid features until that date.

If a customer cancels mid-cycle, no prorated refund is issued for the
remaining days in that cycle -- the cancellation simply prevents the
next billing cycle from starting. This is separate from the Refund
Policy, which governs refunds for a payment already made, whereas
cancellation governs whether future payments occur.

Enterprise customers must provide 30 days written notice to cancel,
per their service agreement.
"""

SLA_POLICY = """
Service Level Agreement (SLA)

We guarantee 99.9% uptime measured monthly for all Premium and Enterprise
customers. Basic plan customers receive best-effort uptime with no formal
SLA guarantee.

If uptime falls below 99.9% in a given month for an eligible customer,
the customer is entitled to service credits as follows:
- 99.0% - 99.9% uptime: 10% credit on that month's bill
- 95.0% - 99.0% uptime: 25% credit on that month's bill
- Below 95.0% uptime: 50% credit on that month's bill

Service credits must be requested within 30 days of the end of the
affected billing period and are applied to the next invoice.
"""

ESCALATION_POLICY = """
Support Escalation Policy

Support tickets are categorized into three priority levels:

P1 (Critical): Complete service outage or data loss. Initial response
within 1 hour, 24/7. P1 tickets are automatically escalated to the
on-call engineering lead if not acknowledged within 15 minutes.

P2 (High): Significant feature degradation affecting multiple users.
Initial response within 4 business hours. Escalates to team lead if
unresolved after 8 business hours.

P3 (Standard): Minor bugs or general questions. Initial response within
1 business day. No automatic escalation; customer may request escalation
to P2 if business impact increases.

P1 incidents are directly tied to our SLA uptime guarantees -- a P1
caused by an outage on our end counts toward the monthly uptime
calculation described in the SLA Policy.
"""

API_DOCS = """
API Documentation - Rate Limits and Authentication

All API requests must include a Bearer token in the Authorization header,
obtained from the developer dashboard.

Rate limits by plan:
- Basic: 100 requests/minute, 10,000 requests/day
- Premium: 1,000 requests/minute, 200,000 requests/day
- Enterprise: Custom limits negotiated per contract

Exceeding the rate limit returns a 429 status code with a Retry-After
header indicating when the next request will be accepted.

The maximum file upload size via the API is 25 MB for Basic and Premium
plans, and 500 MB for Enterprise plans with the bulk-upload add-on.
"""

PRIVACY_POLICY = """
Data Privacy Policy

User data is retained for 90 days after account deletion, after which
it is permanently purged from production systems. Backups containing
user data are retained for an additional 30 days beyond that (120 days
total) before being purged from backup storage.

For customers in the EU, data is processed in accordance with GDPR.
EU customer data is stored exclusively in EU-based data centers. EU
customers may request full data export or deletion at any time via the
privacy portal, and such requests must be fulfilled within 30 days per
GDPR Article 17.

For customers in the US, data may be stored in US or EU data centers
depending on account configuration at signup. US customers do not have
the GDPR-specific 30-day deletion guarantee, though standard deletion
requests are typically fulfilled within 14 days as a matter of practice,
not legal requirement.
"""

ONBOARDING_POLICY = """
Employee Onboarding Policy

Full-time employees complete a 5-day onboarding program covering company
systems, security training, and team introductions. Full-time employees
receive immediate access to all internal tools upon their start date.

Contractors complete a condensed 1-day onboarding covering only security
training and the specific systems relevant to their project. Contractors
receive access only to systems explicitly approved by their hiring
manager, and access is automatically revoked at the end of the contract
period unless explicitly renewed.

Both full-time employees and contractors must complete security training
before receiving any system access, with no exceptions.
"""

DOCUMENTS = {
    "refund_policy.txt": REFUND_POLICY,
    "cancellation_policy.txt": CANCELLATION_POLICY,
    "sla_policy.txt": SLA_POLICY,
    "escalation_policy.txt": ESCALATION_POLICY,
    "api_docs.txt": API_DOCS,
    "privacy_policy.txt": PRIVACY_POLICY,
    "onboarding_policy.txt": ONBOARDING_POLICY,
}
