This session is one STaRK retrieval benchmark task. Use the provided retrieval tools to gather evidence.
Your final assistant message is consumed directly by a strict JSON parser. Output exactly one JSON object
with the single key "node_ids" whose value is a ranked list of at most 100 unique integer benchmark IDs.
Do not include explanations, Markdown fences, citations, headings, or text before or after the object.
If no answers are supported, output {"node_ids": []} directly. The entire final text must be the JSON object.
