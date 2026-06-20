"""
Provider-agnostic LLM interface.

WHY THIS EXISTS: the project is built free-tier-first with Groq, but the
target deployment skill set includes AWS Bedrock. Rather than hardcoding
Groq calls throughout the codebase, every caller goes through generate()
below. Swapping to Bedrock in production means implementing one function
(_generate_bedrock) and changing one line in get_llm_provider() -- no
business logic elsewhere needs to change.

This mirrors a real production pattern: providers change (cost, latency,
model availability), business logic shouldn't have to.
"""
from app.config import GROQ_API_KEY, GROQ_MODEL


def _generate_groq(prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 1024) -> str:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _generate_bedrock(prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """
    Placeholder for production swap-in. Example implementation:

        import boto3, json
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        })
        resp = client.invoke_model(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", body=body)
        return json.loads(resp["body"].read())["content"][0]["text"]
    """
    raise NotImplementedError("Bedrock provider not wired up in this free-tier build. See docstring for the swap-in.")


PROVIDERS = {
    "groq": _generate_groq,
    "bedrock": _generate_bedrock,
}

ACTIVE_PROVIDER = "groq"


def generate(prompt: str, system: str = "", temperature: float = 0.2, max_tokens: int = 1024) -> str:
    fn = PROVIDERS[ACTIVE_PROVIDER]
    return fn(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
