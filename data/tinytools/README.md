# TinyTools: writing pretraining documents for Eve

TinyTools is a TinyStories-style corpus: short, simple documents about what people ask assistants
for and what tools do. Eve is pretrained on it (mixed with web text) so that a ~30M model learns
that "20 plus 10" and "Add two numbers" mean the same thing. Code draws a prompt for each
document; a generator writes the document; code checks it.

## Files

- `nouns.txt`, `verbs.txt`, `adjectives.txt`: the words a prompt asks a document to use.
- `domains.txt`: topics. `concept-only` marks the held-out tools' domains (arithmetic, units,
  currency, flights, hotels): documents may talk about them, but never invent a tool for them.
- `names.txt`: about 300 first names from many languages; each scene prompt names its person.
- `<run>/prompts.jsonl`: written by `uv run eve corpus prompts`, one prompt per line.
- `<run>/documents.jsonl`: written by the generator, one document per line.
- `<run>/tinytools.parquet` and `<run>/report.json`: written by `uv run eve corpus check`.

The pilot runs are `data/tinytools/pilot` (2,000 prompts) and `data/tinytools/pilot-v2` (the same
2,000 draws with the reworded prompts).

## Writing a run

```bash
uv run eve corpus prompts --n 20000 --seed 1 --out data/tinytools/<run>
uv run eve corpus generate data/tinytools/<run> --workers 8
```

`eve corpus generate` runs headless Claude Code (`claude -p`, no tools, in an empty directory) on
batches of 50 prompts and checks each document as it comes back. A rejected document goes back
with the checker's reasons, up to `--rounds` calls per batch (default 3). Each batch is saved to
`<run>/batches/batch-NNNN.jsonl` beside its prompts and appended to `<run>/documents.jsonl` as soon
as it finishes, so running the command again resumes. It ends with `eve corpus check`.

- `--limit 200`: write only that many (a cheap look before a big run).
- `--rewrite`: also send back documents the checker rejects, near-duplicates included.
- `--model sonnet`: the generator's model (Claude Code's default if unset).
- It stops when a batch gets nothing back but errors (no login, rate limit); run it again later.

The cost so far is printed per batch and at the end. The rules the generator gets are the ones
below (`SYSTEM` in `src/eve/data/generate/headless.py`).

## For the generator

1. Read `<run>/prompts.jsonl`. Each line has an `id`, a `kind` and a `prompt`, plus what the
   prompt asks for (`domain`, `noun`, `verb`, `adjective`, `feature`, `speaker`).
2. For each line, write the document the `prompt` asks for. Follow it exactly: the length, the
   three words (any form: "bought" for "buy"), the feature, and the format for tool cards and
   dialogues.
3. Append one JSON object per document to `<run>/documents.jsonl`:
   ```json
   {"id": "tt-000042", "text": "Mia asked the kitchen speaker ..."}
   ```
   Keep line breaks inside `text` as `\n`. Order doesn't matter; ids must match the prompts.
4. Check: `uv run eve corpus check <run>`. Rewrite every rejected document (its reasons are in
   `<run>/report.json`) and check again.

Rules that matter more than they look:

- Write only the document: no "Sure, here is...", no title, no notes afterwards.
- Spell the three words correctly; a misspelt word doesn't count as used.
- In cards and ways lists, the three words may go outside the requests (in `does:`, `inputs:` or
  the first line). Every request must be something a real person would say, so don't bend one
  to fit a word.
- Simple, everyday English. Vary the people, places, numbers and sentence shapes from one document
  to the next; a near-copy of an earlier document is rejected.
- Never write angle brackets (`<`, `>`): Eve's special tokens look like `<tool>`.
- Never name a tool from `data/tools/heldout` (`add`, `subtract`, `multiply`, `divide`,
  `convert_units`, `search_flights`, `book_hotel`, `get_flight_status`, `convert_currency`)
  as a tool: no `convert_units`, no `add(`, no `name: add`. Saying "add the two numbers" in a story
  is fine.

## The kinds

| kind | share | what it is |
|---|---|---|
| `scene` | 40% | a tiny story: someone asks an assistant for something in their own words, and the story says plainly what it did |
| `explain` | 20% | how something works in everyday life, and the words people use to ask for changes |
| `ways` | 15% | one plain line saying what someone wants, then 6 to 10 different ways to ask for it |
| `card` | 15% | an invented tool: `name:`, `does:`, `inputs:`, `people say:` and 4 to 6 requests |
| `dialogue` | 10% | `Person:` / `Assistant:` turns: a missing detail the assistant asks for, or a polite refusal |
