# Writing requests for Colibri

You are helping build training data for Colibri, a tiny model that reads a user's request plus 1–5 tool
schemas and decides what to do. Code has already decided the **answer** for each scenario. Your job is to
write the **request**: what a real person would say or type.

## Your task

1. Read `data/generation/batches/<batch>.json`. It holds about 50 scenarios.
2. For every scenario, write **one** request.
3. Save them to `data/generation/written/<batch>.jsonl`, one JSON object per line, in scenario order:
   ```json
   {"id": "batch-0003-000", "utterance": "kill the kitchen lights"}
   ```
4. Check your work: `uv run colibri data check <batch>`. Fix every rejected line and run it again
   until it reports 0 rejected and 0 not written.

Work only on the batch you were given. Don't edit the batch file or any other file.

## A scenario

```json
{"id": "batch-0003-007",
 "tools": ["set_blinds", "set_lights"],
 "intent": "call",
 "calls": [{"tool": "set_lights", "args": {"room": "kitchen", "on": false}}],
 "meaning": "turn off the kitchen lights",
 "missing": [],
 "implied": []}
```

- `tools`: what the assistant can do in this scenario. Each tool's schema is under `schemas` at the top
  of the batch file, in the form the model reads (`?` marks an optional argument).
- `calls`: the exact answer. Your request must lead to exactly these calls, in this order.
- `meaning`: a plain version of the request. **Rephrase it**; never copy it.

## Rules by intent

**call**: ask for exactly the calls listed, in order.
- Every text and number value must appear in your request **exactly** as written in `args` (case
  doesn't matter). `"contact": "my boss"` means the words "my boss" appear. Numbers may be digits
  (`25`) or words (`twenty-five`).
- `true`/`false` values and enum values may be said in any natural way ("kill the lights" for
  `on: false`; "lounge" is fine for `living room` only if no other option fits better).
- Values listed in `implied` were set by meaning, not by words ("mute it" means level 0). Keep that
  meaning: say "mute" or "close", not a number.
- Don't mention anything that isn't in the args: if `brightness` isn't there, don't ask for 40%.
- Several calls: one request that asks for all of them in order ("kitchen lights off, then lock the
  front door").

**ask**: the user asks for the tool's action but leaves out the values named in `missing`.
- Do **not** mention the missing value or hint at a specific one: "turn off the lights" when `room`
  is missing, not "turn off the kitchen lights".
- Every other value in `args` must still appear exactly.

**unsupported**: the user asks for something none of the listed tools can do. Rephrase `meaning`. Don't
bend it toward a listed tool; the point is that none of them fits.

**none**: no request for a tool. Write small talk, thanks, a question about the assistant, or a statement
about a topic near the tools that isn't a request ("my kitchen lights are so old"). `meaning` is empty.

## Make it varied

These requests are the model's only window onto real users, so make them unlike templates:
- Mix lengths: two words ("lights off") to a long sentence with context ("we're heading out for
  the evening so could you lock the front door").
- Mix styles: casual, polite, terse, indirect ("it's way too bright in here"), as typed on a phone
  (lowercase, no punctuation, an occasional typo **outside** the values).
- Use number words now and then ("three", "twenty-five").
- Don't start most requests the same way. Avoid "Please" and "Can you" as defaults.

## Running a batch with Claude Code

```bash
uv run colibri data batch --split train          # or val / probe; prints the new batch name
claude -p "Follow data/generation/README.md for batch batch-0003."
uv run colibri data check batch-0003
```

`val` and `probe` batches matter most: they measure the model and fit its calibration, so they must be
the most natural. `probe` batches include the held-out domains and are checked by hand afterwards.
