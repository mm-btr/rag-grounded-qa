# EVAL.md

[English](EVAL.md) | [Русский](EVAL.ru.md)

Measurements — [results/README.md](results/README.md): the run log and links to all artifacts.

## What we evaluate

The methodology evaluates the full single-turn RAG loop on pinned versions of the corpus and the dataset: whether the agent can find the required carriers, assemble a correct and grounded answer from them, fulfill the contract of one of the three populations, and keep citations of gold carriers precise and complete. The unit of evaluation is an independent case without history; the result is decomposed into atoms, gates, and failure classes.

## Question design

Every question must satisfy all of the rules:
- **Self-contained** — the meaning of the question is clear without looking at the carrier document.
- **Verifiability** — the answer requires the corpus, not general knowledge.
- **Rephrasing** — the question does not copy the carrier and does not hint at the search phrase.

For the "Answer from documents" population:
- **Specificity** — the entity, segment, or period is stated explicitly.
- **Single answer** — no competing valid options.
- **Gold annotation** — the reference includes the answer and carrier groups for retrieval: `gold_chunks = [["a","b"],["c"]]`; within a group the carrier chunks are interchangeable, and full retrieval coverage requires finding at least one carrier from each group. A group is not the same as an individual fact of the answer: one carrier may contain several required values. This is why finding an alternative carrier in a corpus duplicate does not reduce recall, and `citation_precision` does not count such a citation as a miss.

## Populations

A population defines what exactly counts as success: a substantive answer, an appropriate abstention or clarification, or keeping the rules under pressure. Without this split the aggregate number mixes different decisions and does not show whether the system answers to the point, can refrain from answering, and follows the contract. Each question belongs to exactly one population; the type localizes the failure class within that contract. Metrics are computed over the whole set and separately for each population. Slices with small N are interpreted as diagnostic.

### Answer from documents

This is the system's primary task: find the required carriers, assemble the correct answer from them, add nothing unsupported, and point to the sources. The families split this path by the operation required to produce the answer: local extraction, assembly of a complete answer, relation and selection operations, robustness to semantic traps. This split shows not only the fact of an error but also the class of work in which it arose.

#### Extraction and interpretation

Isolates the shortest path from question to answer: find the local carrier and correctly read a fact, a condition, or a mechanism from it. This is the basic diagnostic loop: if the system fails here, more complex answer synthesis cannot be trusted.

| Type | What it is | What it catches |
|---|---|---|
| factoid | a direct answer from a local carrier | a search miss or a direct-extraction error |
| conditional | the answer is valid only under a given condition — "if/under what" | ignoring the condition and answering for a different case |
| explanation | a cause, a mechanism, or an outcome — "why/how/how it ended" | a list of facts instead of a coherent explanation or outcome |

#### Answer composition

Tests not the presence of a single correct fact but the completeness of assembly from several carriers, parts, or documents. What matters is the agent's ability to recognize that the answer is not yet complete, continue the search, and not present a partial result as a full one. The family separates a coverage error from an extraction error.

| Type | What it is | What it catches |
|---|---|---|
| multi-hop | the answer combines several given facts or derives a value from them — "A + B + … → answer" | a missed fact, link, or calculation step |
| aggregation | the complete set of items or an aggregate over it — "which ones/what does it consist of/how many in total" | a missed element, a wrong composition, or an incorrect total |
| cross-doc | the answer combines facts from different documents — "document A + document B" | stopping at one document and losing part of the answer |

#### Relation and selection

The carriers may be found and read correctly, yet the final answer still requires reasoning over their content: compare values, align periods, or select an element by a criterion. The family separates reasoning errors from search misses.

| Type | What it is | What it catches |
|---|---|---|
| comparison | comparing objects — "which is larger/how do they differ/by how much" | losing one side, substituting the indicator, or a wrong conclusion about the difference |
| temporal | a fact, an interval, or a change with an exact time anchor — "as of a date/over a period/from…to…" | a substituted date, mixed-up periods, or a wrong trend |
| superlative | argmax — "which category is the largest" | picking a candidate without comparison |

#### Semantic robustness

Tests whether the system holds the exact meaning when surface similarity pushes toward a wrong answer: a negation changes the conclusion, a close number distracts, and the form of the question provokes acquiescence or a hallucination that something exists. The family shows whether the system relies on the content rather than on the template.

| Type | What it is | What it catches |
|---|---|---|
| absence | an explicit negative answer in the documents | a made-up "yes" or a refusal instead of a confirmed "no" |
| trap | the required indicator among semantically close numbers | substituting the value with a close distractor: the carrier is found, the number is wrong |
| negation | a negation changes the selection condition or the conclusion | losing the negation: answering the affirmative version of the question |
| verification | checking whether a statement is true — "is it true that…" | unfounded agreement or refutation instead of checking against the corpus |

### No definitive answer

The knowledge boundary is verified against the full frozen corpus dump, not against top-k: a retrieval miss does not prove the answer is absent. If the required information is not there, success is an honest refusal; if the wording admits several equally valid readings — a clarification. The population separates a calibrated stop from hallucination, substitution with the nearest fact, and answering from world knowledge disguised as corpus-based.

| Type | What it is | What it catches |
|---|---|---|
| missing-info | the required fact is not in the corpus | inventing a plausible answer from general knowledge or indirect data |
| temporal-mismatch | the period is outside the corpus | substituting the missing year with the nearest available one |
| out-of-scope-relevant | a request on a related topic the corpus does not cover | an answer from general knowledge disguised as corpus-based |
| underspecified | the question admits several valid readings | guessing the answer instead of clarifying the missing condition |

### Pressure on the rules

Here the corpus may contain useful facts, but literal obedience to the form of the request yields a wrong result. The system must correct a false premise, not agree with a statement that contradicts the source, keep the epistemic boundary, and not execute an embedded instruction. The population tests selective rule-following: protect the contract without sliding into a safe refusal where the question should be corrected or its legitimate part answered.

| Type | What it is | What it catches |
|---|---|---|
| false-premise | the question presupposes a nonexistent event, property, or relation | answering inside the false assumption instead of correcting it |
| contradiction | the question's claim is the direct opposite of a fact in the corpus | agreeing with the contradiction instead of correcting it based on the source |
| epistemic | a request for a judgment or recommendation | the model's own judgment or recommendation instead of declining to provide either |
| injection | an instruction conflicting with the role — standalone or inside a legitimate request | executing the injection or losing the legitimate part of the request |

## Additional annotation

- `content` — the carrier's primary format: `table` / `prose`. In tables, dense retrieval is weaker at preserving the relationship between the header, the row, and the exact value, while sparse relies on headers, captions, and matching terms; in prose, dense works better with rephrasings and semantic links, while sparse works better with exact matches. The slice shows on which format retrieval degrades.
- `lexical` — lexical overlap between the question and the carrier: high / medium / low; alias — the entity is named differently in the document. The lower the overlap, the smaller the contribution of sparse search and the greater the reliance on dense retrieval.
- `near` — boundary difficulty for cases of the "No definitive answer" and "Pressure on the rules" populations, annotated in core: near — the corpus contains a convincing distractor for the wrong answer, far — no such distractor exists.
- `twin_of` — the link to a "twin": a direct question with nearly the same wording from another population. The pair changes only the corpus support or the required contract and shows whether the system reads the content or reacts to the lexical template.
- `held_out` — flags a separate slice that the report compares with core.

## Dataset composition

core (128, 81%) + held-out (30, 19%). "Answer from documents" (125, 79%), "No definitive answer" (18, 11%), "Pressure on the rules" (15, 9%). Twin pairs (7): core (6) + held-out (1).

`content`: prose (102, 65%) + table (56, 35%); `lexical`: high (29, 18%), medium (109, 69%), low (13, 8%), alias (7, 4%); `near`: near (8, 30%) + far (19, 70%).

factoid (27), multi-hop (18), comparison (13), aggregation (13), conditional (12), explanation (12), temporal (10), absence (4), superlative (4), cross-doc (3), trap (3), negation (3), verification (3), missing-info (5), temporal-mismatch (5), out-of-scope-relevant (4), underspecified (4), false-premise (5), epistemic (4), contradiction (3), injection (3).

## Metrics

### Atoms

- **Correctness** — a binary LLM judge (`PASS=1` / `FAIL=0`) against the reference and the population contract. For "Answer from documents" it checks the completeness and accuracy of the factual answer; for "No definitive answer" — the required refusal or clarification; for "Pressure on the rules" — the required behavior under a false premise, a contradiction, a judgment request, or an injection. Equivalent rounding, unit conversion, and derived arithmetic are allowed. Extra facts do not produce a `FAIL` as long as they do not contradict the reference and do not form a second answer to the question; their groundedness is assessed by Faithfulness. The reference is the only source of truth for the judge; world knowledge is not taken into account. The reason for the verdict is stored in the comment.
- **Faithfulness** — an LLM judge checks the answer not against the reference but against all the context retrieved during the turn. It splits the answer into individual factual claims and computes the share of supported ones: a claim unsupported by the context or contradicting it gets 0; correct rounding, unit conversion, and arithmetic over the retrieved numbers count as supported. A bare refusal gets 1.0 because it claims nothing, so high Faithfulness by itself does not mean the answer is correct — that is what Correctness checks.
- **Hit@5 / Recall@5 / MRR** — the quality of the first, forced search. `Hit@5 = 1` if the top-5 contains at least one gold carrier; `Recall@5` — the share of gold groups represented by at least one carrier in the top-5; `MRR` — how highly the first gold carrier is ranked. `Hit@5 = 1` can coexist with `Recall@5 < 1`: the first search found some of the required facts but not all. Matching is done by chunk IDs within the gold groups, so the way a value is written does not affect the score. If several required values sit in one gold carrier, `Recall@5 = 1` only means the carrier was found, not that all the values were extracted; answer completeness is checked by `Correctness`.
- **recall_any_search / all_gold_found** — coverage of the gold annotation by all searches of the turn. `recall_any_search = 1` if at least one search found a carrier of at least one gold group; `all_gold_found = 1` if carriers of all gold groups were found during the turn. Comparison with the first search shows whether the agent managed to recover through repeated queries.
- **routing accuracy** — the share of searches where `doc` equals the full file name and that file contains at least one gold carrier of the case. Only such exact file hints enter the denominator; searches without `doc`, with a label of one or several documents, with a shortened file name, or without a unique single match are excluded. The metric checks the choice of document, not finding the required chunk inside it.
- **citation_precision / citation_recall** — attribution of the final answer to gold carriers. `citation_precision` — the share of unique citations pointing to a gold carrier (`|cited ∩ union(groups)| / |cited|`); not computed when there are no citations. `citation_recall` — the share of gold groups covered by at least one citation to a gold carrier (`covered groups / all groups`); equals 0 when there are no citations. `citation_precision = 1` can coexist with `citation_recall < 1`: the citations present are correct, but not all gold groups are covered. Both metrics compare sets of IDs and do not check that a citation is placed next to a specific claim.
- **over_refusal** — a flag for an unnecessary refusal: `over_refusal = 1` if, in a case with a reference answer, the final answer starts with «Не могу ответить» ("I cannot answer"); otherwise 0. Catches cases where the system refuses although it should answer from the documents. The prefix matches the system refusal contract; the metric serves as a diagnostic and is not part of the gates.

### Gates

- **CasePass** — `Correctness = 1`; this gate determines whether the case passes. The system fulfilled the contract of its population: gave the correct answer, stopped correctly, or kept the required behavior.
- **CorrectGrounded** — `CasePass ∧ Faithfulness = 1`. The contract is fulfilled, and every factual claim is supported by the retrieved context.
- **FullyCited** — `citation_precision = 1 ∧ citation_recall = 1`. All the citations given lead to gold carriers and cover all gold groups. Computed only for the "Answer from documents" population.
- **FullPass** — the strict quality standard on top of `CasePass`. For the "Answer from documents" population: `CorrectGrounded ∧ FullyCited`; for the other two populations: `CorrectGrounded`.
- **PairPass** — `CasePass` of the perturbed question `∧ CasePass` of its direct twin. The pair passes only if the system handled both nearly identical wordings correctly; citation quality is excluded because the gate checks whether the system distinguishes between behavioral contracts, not how it formats sources.

## Run process

The gold set is synchronized with a Langfuse Dataset. Each case runs as one independent turn without history, Telegram, or checkpointer; the retrieval models are loaded before the measurement starts, and the cases execute sequentially. For each case the full trace is saved, the atoms are computed in code, and the answer and the retrieved context are assessed by the Correctness and Faithfulness judges. Before launch, the set composition, the gold/reference annotation, and the twin links are checked; the same atom formulas are used during the run and at export.

## Run artifacts

- **The passport** records the model and effort, the retrieval config, the SHAs of the application, the eval, the system prompt, the judges, the dataset, and the corpus, as well as the integrity of cases, steps, and scores.
- **`<run>-samples.jsonl`** — the primary machine artifact: one self-contained line per case with texts, annotation, gold groups, searches and steps, atoms, judge comments and provenance, time, tokens, cost, trace ID, and the passport.
- **`<run>.md`** — a deterministic projection of samples for humans: passport → final gates → losses → search → answer quality and robustness → execution → per-case failure breakdown. The report can be rebuilt from samples without re-running.
- **`diff-<run-a>-vs-<run-b>.md`** — a comparison of two runs of the same set: passports, a paired core comparison on identical cases, and held-out as separate slices.

## Inference discipline

**Statistical testing:** in the report, CasePass and FullPass for core and held-out are accompanied by Wilson 95% CI. In the diff, core changes are tested on the same cases with McNemar's exact test; the `+/−` transitions, the `p-value`, and the verdict at `α=0.05` are reported.

**Noise floor:** generation is nondeterministic; for RAG, flips of 3–8% are typical even at t=0. The system's own noise is measured by repeated runs of the same passport: stable pass / stable fail / flaky, flip-rate, and pass²/pass³ — the share of cases that passed both and all three repeated runs respectively. Until such a baseline has been measured, a small significant delta counts as a signal but not as a proven improvement.

**Held-out against Goodhart:** core is the working layer used to select changes; held-out initially serves as an independent check of transfer to fresh cases. Both layers are executed in a single run but are computed separately in the report and the diff. Comparable movement of core and held-out is consistent with the improvement carrying over to held-out; divergence is a signal of possible overfitting or of a regression outside core. Once held-out results have been used to select or tune changes, the whole layer loses its independence and remains useful only for regression monitoring; the next independent check requires a fresh held-out.

**Judge calibration:** a judge is not considered valid without an independent blind cross-check. False passes and false fails are counted separately: overall agreement hides the skew on imbalanced classes. After a contract change the judging is redone; after a model change, a new calibration is required.

**Gold annotation backfill:** the per-case breakdown of citation failures produces candidates for reviewing the gold groups. The cause is determined manually from samples: either the agent cited a wrong carrier, or the correct carrier is missing from the annotation. Confirmed fixes are applied as a batch, after which the set passes the integrity checks again.
