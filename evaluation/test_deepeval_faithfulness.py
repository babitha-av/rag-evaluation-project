from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

model = OllamaModel(
    model="llama3.2",
    base_url="http://localhost:11434",
)

metric = FaithfulnessMetric(
    threshold=0.5,
    model=model,
    include_reason=True,
)

test_case = LLMTestCase(
    input="How long is the refund period?",
    actual_output="The refund period is 14 days from the initial purchase date.",
    retrieval_context=[
        """# Refund Policy

Customers can request a refund within 14 days of their initial purchase.

Refund requests must be submitted through the support portal.

Refunds are processed within 5-7 business days after approval.

Subscription cancellations do not automatically trigger a refund for previous payments.

Annual subscription purchases are eligible for a refund only if the request is submitted within the 14-day refund period.""",

        """# Subscription Management

Users can upgrade or downgrade their subscription from the account
settings page.

When upgrading from Monthly to Annual, the new subscription becomes
effective after the upgrade is completed.

Users can cancel their subscription from account settings.

Cancellation prevents future renewal but does not automatically
provide a refund for previous payments.""",

        """# Security

AcmeAI encrypts customer data during transmission using HTTPS.

Customer data stored by the platform is encrypted at rest.

Users can enable multi-factor authentication from their account
security settings.

AcmeAI performs regular security reviews of its infrastructure."""
    ],
)

metric.measure(test_case)

print("Score:", metric.score)
print("Reason:", metric.reason)
