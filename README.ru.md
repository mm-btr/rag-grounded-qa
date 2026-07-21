# RAG — grounded Q&A по документам

[English](README.md) | [Русский](README.ru.md)

[![tests](https://github.com/mm-btr/rag-grounded-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/mm-btr/rag-grounded-qa/actions/workflows/tests.yml)

RAG-сервис для вопросов по PDF, DOCX и TXT. Ответы снабжаются проверенными цитатами с указанием документа и, при наличии, раздела и страницы; если корпус не подтверждает ответ, агент отказывает или просит уточнить вопрос. Telegram принимает документы, текстовые и голосовые вопросы.

Живые ответы по текущему корпусу:

<p>
  <img src="assets/demo-segments.png" width="45.6%" alt="Сравнение сегментов таблицей">
  <img src="assets/demo-sandwich.png" width="52.4%" alt="Оценочный вопрос — факты с цитатами, без рекомендаций">
</p>

<details>
<summary>Ещё два примера: кросс-документные ответы</summary>

<p>
  <img src="assets/demo-board.png" width="46.1%" alt="Кросс-док ответ: устав + МСФО">
  <img src="assets/demo-auditor.png" width="51.9%" alt="Кросс-док ответ: устав + отчёт эмитента">
</p>

</details>

## Измеренное качество

`v2-158` — 158 single-turn кейсов на зафиксированных версиях корпуса, датасета, системного промпта и судей. Контекст замера — русскоязычный финансово-юридический корпус: три документа, 758 чанков. Набор проверяет три разных контракта: ответ по документам, отсутствие однозначного ответа и соблюдение правил под давлением.

`CasePass` означает, что контракт кейса выполнен. `FullPass` — более строгий стандарт: ответ дополнительно обоснован найденным контекстом, а для ответов по документам каждая gold-группа покрыта цитатой и лишних ссылок нет.

| Слой | N | CasePass | FullPass |
|---|---:|---:|---:|
| Core | 128 | **117/128 — 91%** (95% CI 85–95%) | **104/128 — 81%** (95% CI 74–87%) |
| Held-out | 30 | **26/30 — 87%** (95% CI 70–95%) | **19/30 — 63%** (95% CI 46–78%) |

Retrieval по всем 125 кейсам с gold-разметкой: Hit@5 **0.77** · Recall@5 **0.73** · MRR **0.65** · все gold-группы найдены за ход в **118/125** кейсах.

Исполнение: p50 **30с** · p95 **112с** · около **86%** времени занимает CPU-поиск. Агент израсходовал **$1.24** на 158 кейсов — примерно **$0.0079/кейс**; стоимость судей считается отдельно.

Первичный артефакт прогона — [samples](results/samples/v2-158-samples.jsonl) с полными ответами, поисками, оценками по отдельным метрикам, комментариями судей и параметрами их запусков, токенами, стоимостью и trace ID. Из samples детерминированно строится [отчёт](results/reports/v2-158.md); [сравнение](results/diffs/diff-v1-158-vs-v2-158.md) — из samples обоих прогонов. `Correctness` прошёл полную [калибровку](results/calibration/calibration-v2-158.md), `Faithfulness` сверен выборочно. Полная методика — в [EVAL](EVAL.ru.md).

## Stack

| Контур | Компонент | Роль |
|---|---|---|
| Ingest | Docling HybridChunker | парсинг документов и структурный чанкинг |
| Ingest | Llama Prompt Guard 2 86M | детекция prompt injection |
| Retrieval | BGE-M3 | dense+sparse embeddings |
| Retrieval | Qdrant | hybrid vector store с фильтрами |
| Retrieval | bge-reranker-v2-m3 | cross-encoder reranking |
| Agent | LangChain + LangGraph | agent runtime и middleware |
| Generation | OpenAI GPT-5.4 mini | выбор инструментов и генерация ответа |
| State | PostgreSQL | доступ, реестр документов, дедуп и checkpoints |
| Channel | aiogram | Telegram-адаптер |
| Channel | ElevenLabs Scribe v2 | голосовые вопросы → текст |
| Observability | Langfuse | трассировка ходов, эксперименты и LLM-судьи |

## Architecture

```text
Telegram [text · voice]
  → aiogram [dedup · allowlist · tenant gate · busy gate]
       └─ voice → ElevenLabs Scribe v2 → text
  → LangGraph agent [GPT-5.4 mini · forced search · budget 5]
       ├─ memory [Postgres checkpointer · ≈25 turns]
       ├─ search(query, doc?) [doc resolver · corpus fallback] ⇄ Hybrid retrieval
       └─ grounding [artifact citation allowlist · retry/refusal]
  → fetch_locators [Qdrant payload]
  → Telegram [answer · numbered sources]

Hybrid retrieval
  → BGE-M3 [dense + sparse]
  → Qdrant [tenant_id · ingest_ready · source?]
       dense prefetch + sparse prefetch → RRF top-10
  → bge-reranker-v2-m3 → top-5

Admin upload [PDF · DOCX · TXT · ≤20 MB]
  → aiogram [admin gate · tenant ingest gate]
  → Postgres documents [pending → processing]
  → ingest [global lock]
       → Docling HybridChunker [section/page provenance · Markdown tables]
       → sanitize_text + Prompt Guard
       → Qdrant [delete source]
       → BGE-M3 [dense + sparse]
       → Qdrant batch upsert [ingest_ready=false]
       → Qdrant source activation [ingest_ready=true]
  ├─ success → Postgres documents [ready]
  └─ error → Postgres documents [failed]
```

## Ограничения

- Развёртывание рассчитано на single-host pilot.
- Ingest принимает текстовые PDF, DOCX и TXT размером до 20 МБ; OCR для сканов отсутствует.
- Результаты эвала относятся к зафиксированному русскоязычному финансово-юридическому корпусу, а не к production-распределению запросов.
