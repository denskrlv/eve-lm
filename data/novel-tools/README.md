# The novel-tools benchmark

Made-up tools that no training split has, with requests written by hand. It measures what
matters most for an SDK user: calling a tool the model has never seen.

- `tools/*.json`: 41 tools in 12 domains, in the format of `data/tools` (no templates).
- `requests.jsonl`: 243 requests (168 CALL, 25 ASK, 30 UNSUPPORTED, 20 NONE), each
  `{"utterance", "intent", "calls"}`.

`eve data build` writes `data/evals/novel-tools.jsonl`. It gives each request its gold tools
plus distractors from other domains (1 to 5 tools, seeded), and each CALL and ASK row a `group`:
`shared word` if the request has a word of a gold tool's name (`check_uv_index`: check, uv,
index; endings aside), else `no shared word`. Earlier models matched words, not meaning, so the
second group is the real test. `eve evaluate` scores the groups separately.

Rules when adding requests:
- Tool names must not be in `data/tools` or the generated tools; the build refuses them.
- Values follow `data/generation/README.md`: every text and number value appears in the request
  as written; ASK leaves the `"?"` values out.
- Keep a tool's capability distinct from every other tool here, so a distractor never fits.
- Don't reuse a phrasing from training data or TinyTools: the build removes duplicates from
  `train` and `val`, and TinyTools drops any request in `data/evals`. Rebuild both after changes.
