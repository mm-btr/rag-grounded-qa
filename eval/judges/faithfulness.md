You measure the Faithfulness of a generated answer: the share of its factual statements that are supported by the provided context. The context and the answer are in Russian.

Break the answer into atomic, self-contained factual statements (resolve pronouns to their subjects), verdict each against the context — 1 supported, 0 unsupported or contradicted, with one line of reasoning each — and return faithfulness = supported / total.

Rules:
1. Only claims about the world are statements. Openers, hedges and the standard refusal formula («Не могу ответить…») assert nothing. An answer with no factual statements at all (a bare refusal) scores 1.0: nothing asserted, nothing unfaithful.
2. A claim of absence — a statement that the documents lack certain information — is supported when the context indeed lacks it, and unsupported when the context does contain it.
3. Numbers: a value differing from the context only by rounding or by an equivalent unit/scale conversion is supported; arithmetic correctly derived from context numbers (a difference, a growth rate) is supported.

Context: {{context}}
Answer: {{answer}}
