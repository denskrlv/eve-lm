import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from colibri.data.draw import pick, shuffled
from colibri.data.generate.library import ToolSpec
from colibri.data.generate.sampler import MAX_PROMPT_TOOLS, ToolBank, prompt_size
from colibri.data.generate.templates import MAX_CALLS
from colibri.data.generate.validate import check_example, mentions, value_forms
from colibri.data.jsonl import read_jsonl, write_jsonl
from colibri.data.render import Call, TaskExample
from colibri.data.schema import MAX_ARGS, Arg, SchemaError, Tool

# Topics of the held-out domains (math, travel). An external example whose tools or request touch
# them is dropped, or the held-out test would no longer measure unseen tools.
HELD_OUT = re.compile(
    r"\b(flights?|airports?|airlines?|hotels?|travel\w*|trips?|currenc\w*|exchange rates?|"
    r"convert\w*|conversion|calculat\w*|arithmetic|math\w*|sum|addition|subtract\w*|"
    r"multipl(y|ies|ying|ication)|divide[sd]?|dividing|division|factorial|fibonacci|primes?|"
    r"average|median|gcd|lcm|equations?|"
    r"integral|derivatives?|matrix|matrices)\b",
    re.IGNORECASE,
)
UNSUPPORTED_SHARE = 0.1  # extra UNSUPPORTED copies, made by removing the correct tools
VARIANT_TRIES = 30  # random tools tried for an UNSUPPORTED variant
MIN_SLOT_COUNT, MIN_SLOT_SHARE = 5, 0.03  # how often a MASSIVE slot must appear to be an arg


def snake(name: str) -> str:
    """Any identifier as snake_case: getUserInfo / get-user.info -> get_user_info."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    s = re.sub(r"_+", "_", s)
    return "n_" + s if s[:1].isdigit() else s


def short(text: str, limit: int) -> str:
    """The first sentence, cut at a word boundary to at most ``limit`` characters."""
    text = re.sub(r"[<>\s]+", " ", text or "").strip()
    text = re.split(r"(?<=[.!?])\s", text)[0].rstrip(".")
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0]
    return text


def _coerce(arg: Arg, value: Any) -> Any:
    """A value converted to the argument's type, or None if it doesn't fit."""
    if isinstance(value, bool):
        return value if arg.type == "bool" else None
    if arg.type == "bool":
        return {"true": True, "false": False}.get(str(value).lower())
    if arg.type == "str":
        return str(value) if isinstance(value, str | int | float) and str(value).strip() else None
    if isinstance(value, str) and re.fullmatch(r"-?\d+(\.\d+)?", value.strip()):
        value = float(value)
    if not isinstance(value, int | float):
        return None
    if arg.type == "int":
        return int(value) if float(value).is_integer() else None
    return float(value)


# ---- xLAM ------------------------------------------------------------------------------------

_XLAM_TYPES = {"str": "str", "int": "int", "float": "float", "bool": "bool"}


def xlam_tool(d: dict[str, Any]) -> tuple[Tool, dict[str, str], dict[str, Any]] | None:
    """An xLAM tool as a Tool, with its parameter name map and defaults. None if any parameter
    isn't a simple type (lists, dicts, tuples...)."""
    args: list[Arg] = []
    names: dict[str, str] = {}
    defaults: dict[str, Any] = {}
    for pname, p in (d.get("parameters") or {}).items():
        parts = [x.strip() for x in str(p.get("type", "")).split(",")]
        if parts[0] not in _XLAM_TYPES:
            return None
        optional = any(x.startswith(("optional", "default")) for x in parts[1:])
        optional = optional or p.get("default") not in (None, "")
        name = snake(pname)
        if not name or name in names.values():
            return None
        names[pname] = name
        defaults[name] = p.get("default")
        desc = short(p.get("description", ""), 60)
        args.append(Arg(name, _XLAM_TYPES[parts[0]], required=not optional, description=desc))
    if len(args) > MAX_ARGS:
        return None
    try:
        description = short(d.get("description", ""), 120) or d["name"].replace("_", " ")
        return Tool(snake(d["name"]), description, tuple(args), "xlam"), names, defaults
    except SchemaError:
        return None


XlamTool = tuple[Tool, dict[str, str], dict[str, Any]]  # see xlam_tool


def _held_out_xlam(query: str, raw_tools: list[dict[str, Any]]) -> bool:
    """The request or one of the offered tools touches a held-out topic."""
    return bool(HELD_OUT.search(query)) or any(
        HELD_OUT.search(t["name"].replace("_", " ") + " " + t.get("description", ""))
        for t in raw_tools
    )


def _xlam_tools(
    raw_tools: list[dict[str, Any]], gold_names: list[str]
) -> tuple[dict[str, XlamTool] | None, str]:
    """The offered tools that convert, by xLAM name, or None and why the example is dropped."""
    converted = {}
    for t in raw_tools:
        c = xlam_tool(t)
        if c is None and t["name"] in gold_names:
            return None, "list, dict or other complex argument"
        if c is not None:
            converted[t["name"]] = c
    if len({c[0].name for c in converted.values()}) != len(converted):
        return None, "two tools with the same name"
    if any(n not in converted for n in gold_names):
        return None, "call to a tool not in the list"
    return converted, "ok"


def _absent(arg: Arg, value: Any, default: Any, query: str) -> bool:
    """An optional value the user never said: the default applies, so the call leaves it out."""
    said = arg.type == "bool" or any(mentions(query, f) for f in value_forms(arg, value))
    return value == default or not said


def _xlam_call(
    answer: dict[str, Any], converted: dict[str, XlamTool], query: str
) -> tuple[Call | None, str]:
    """One xLAM answer as a call, or None and why the example is dropped."""
    tool, names, defaults = converted[answer["name"]]
    args = {}
    for key, raw in (answer.get("arguments") or {}).items():
        if key not in names:
            return None, "unknown argument"
        arg = tool.arg(names[key])
        value = _coerce(arg, raw)
        if value is None or arg.check(value) is not None:
            return None, "value of the wrong type"
        if not arg.required and _absent(arg, value, defaults[arg.name], query):
            continue
        args[arg.name] = value
    return Call(tool.name, args), "ok"


def convert_xlam(row: dict[str, Any], rng: np.random.Generator) -> tuple[TaskExample | None, str]:
    """One xLAM example in our format, or None and the reason it was dropped."""
    query = row["query"]
    raw_tools, answers = json.loads(row["tools"]), json.loads(row["answers"])
    if _held_out_xlam(query, raw_tools):
        return None, "held-out topic"
    if not 1 <= len(answers) <= MAX_CALLS:
        return None, f"more than {MAX_CALLS} calls"
    gold_names = list(dict.fromkeys(a["name"] for a in answers))
    converted, reason = _xlam_tools(raw_tools, gold_names)
    if converted is None:
        return None, reason
    calls = []
    for answer in answers:
        call, reason = _xlam_call(answer, converted, query)
        if call is None:
            return None, reason
        calls.append(call)

    gold = [converted[n][0] for n in gold_names]
    others = [c[0] for n, c in converted.items() if n not in gold_names]
    others = shuffled(rng, others)[: max(0, MAX_PROMPT_TOOLS - len(gold))]
    if len(gold) > MAX_PROMPT_TOOLS:
        return None, f"more than {MAX_PROMPT_TOOLS} tools needed"
    tools = shuffled(rng, gold + others)
    try:
        example = TaskExample(query, tuple(tools), "call", tuple(calls))
    except SchemaError:
        return None, "invalid call"
    if check_example(example):
        return None, "value not in the request"
    return example, "ok"


def unsupported_variant(
    example: TaskExample, pool: list[Tool], rng: np.random.Generator
) -> TaskExample | None:
    """The same request with random tools from other examples that share no name word with
    the correct ones, labelled UNSUPPORTED."""
    words = {w for c in example.calls for w in c.tool.split("_") if len(w) > 3}
    tools: dict[str, Tool] = {}
    want = prompt_size(rng)
    for _ in range(VARIANT_TRIES):
        if len(tools) == want:
            break
        t = pick(rng, pool)
        if not words & set(t.name.split("_")) and t.name not in {c.tool for c in example.calls}:
            tools[t.name] = t
    if not tools:
        return None
    return TaskExample(example.utterance, tuple(tools.values()), "unsupported")


# ---- MASSIVE ---------------------------------------------------------------------------------

# Intent -> the tool a person would have written for it. Left out: transport_*, qa_maths,
# qa_currency and datetime_convert (held-out travel and math), and intents with < 30 examples.
MASSIVE_TOOLS = {
    "alarm_query": ("list_alarms", "List the alarms that are set"),
    "alarm_remove": ("remove_alarm", "Delete an alarm"),
    "alarm_set": ("create_alarm", "Set an alarm"),
    "audio_volume_down": ("volume_down", "Make the speaker quieter"),
    "audio_volume_mute": ("mute_audio", "Mute or stop the speaker"),
    "audio_volume_up": ("volume_up", "Make the speaker louder"),
    "calendar_query": ("check_calendar", "Look up events in the calendar"),
    "calendar_remove": ("delete_calendar_event", "Remove an event from the calendar"),
    "calendar_set": ("add_calendar_event", "Add an event or reminder to the calendar"),
    "cooking_recipe": ("find_recipe", "Find a recipe or cooking instructions"),
    "datetime_query": ("get_date_time", "Tell the current date or time"),
    "email_addcontact": ("add_email_contact", "Save a contact's email address"),
    "email_query": ("check_email", "Check the inbox for emails"),
    "email_querycontact": ("get_contact_info", "Look up a contact's details"),
    "email_sendemail": ("send_email", "Write and send an email"),
    "iot_cleaning": ("start_cleaning", "Start the robot vacuum or another cleaning device"),
    "iot_coffee": ("make_coffee", "Make coffee with the coffee machine"),
    "iot_hue_lightchange": ("change_light_color", "Change the color of the lights"),
    "iot_hue_lightdim": ("dim_lights", "Make the lights dimmer"),
    "iot_hue_lightoff": ("lights_off", "Turn the lights off"),
    "iot_hue_lighton": ("lights_on", "Turn the lights on"),
    "iot_hue_lightup": ("brighten_lights", "Make the lights brighter"),
    "iot_wemo_off": ("smart_plug_off", "Turn off a device plugged into a smart plug"),
    "iot_wemo_on": ("smart_plug_on", "Turn on a device plugged into a smart plug"),
    "lists_createoradd": ("add_list_item", "Create a list or add an item to it"),
    "lists_query": ("read_list_items", "Read out a list"),
    "lists_remove": ("remove_list_item", "Remove an item from a list, or a whole list"),
    "music_likeness": ("save_music_preference", "Remember music the user likes"),
    "music_query": ("identify_music", "Tell which song or artist is playing"),
    "music_settings": ("music_player_settings", "Change player settings like shuffle or repeat"),
    "news_query": ("get_news", "Read the latest news"),
    "play_audiobook": ("play_audiobook", "Play or resume an audiobook"),
    "play_game": ("start_game", "Start a game"),
    "play_music": ("play_songs", "Play music"),
    "play_podcasts": ("play_podcast_episode", "Play a podcast"),
    "play_radio": ("play_radio", "Play a radio station"),
    "qa_definition": ("define_word", "Give the definition of a word"),
    "qa_factoid": ("answer_question", "Answer a factual question"),
    "qa_stock": ("get_stock_price", "Get a company's stock price"),
    "recommendation_events": ("suggest_events", "Suggest events to go to"),
    "recommendation_locations": ("suggest_places", "Suggest places like restaurants or shops"),
    "recommendation_movies": ("suggest_movies", "Suggest movies to watch"),
    "social_post": ("post_on_social_media", "Post a message or complaint on social media"),
    "social_query": ("check_social_media", "Check updates on social media"),
    "takeaway_order": ("order_takeaway", "Order food for delivery or pickup"),
    "takeaway_query": ("check_takeaway", "Ask about takeaway food or an order"),
    "weather_query": ("check_weather", "Get the weather"),
}
MASSIVE_NONE = ("general_greet", "general_joke", "general_quirky")
SLOT_NOTES = {
    "house_place": "room",
    "change_amount": "how much",
    "player_setting": "e.g. shuffle, pause, resume",
    "timeofday": "e.g. morning",
    "general_frequency": "e.g. every day",
    "relation": "e.g. mom, boss",
    "media_type": "e.g. twitter, podcast",
    "order_type": "delivery or pickup",
    "personal_info": "e.g. phone number",
    "weather_descriptor": "e.g. rain, sunny",
}
_SLOT = re.compile(r"\[(\w+) : ([^\]]+)\]")


def massive_tools(rows: list[dict[str, Any]]) -> dict[str, Tool]:
    """One tool per intent. Its arguments are the slot types seen in at least 3% (and 5) of
    the intent's train examples, all optional strings."""
    counts: dict[str, Counter[str]] = {}
    sizes: Counter[str] = Counter()
    for r in rows:
        if r["partition"] == "train" and r["intent"] in MASSIVE_TOOLS:
            sizes[r["intent"]] += 1
            counts.setdefault(r["intent"], Counter()).update(
                s for s, _ in _SLOT.findall(r["annot_utt"])
            )
    tools = {}
    for intent, (name, description) in MASSIVE_TOOLS.items():
        common = [
            s
            for s, n in counts.get(intent, Counter()).most_common(MAX_ARGS)
            if n >= max(MIN_SLOT_COUNT, MIN_SLOT_SHARE * sizes[intent])
        ]
        args = tuple(
            Arg(s, "str", required=False, description=SLOT_NOTES.get(s, "")) for s in common
        )
        tools[intent] = Tool(name, description, args, "massive-" + intent.split("_")[0])
    return tools


def _massive_bank(tools: dict[str, Tool]) -> ToolBank:
    """The MASSIVE tools as a bank. Tools of one scenario overlap: none of them is offered for
    an UNSUPPORTED copy of another's request."""
    by_scenario: dict[str, list[str]] = {}
    for t in tools.values():
        by_scenario.setdefault(t.domain, []).append(t.name)
    specs = [
        ToolSpec(t, (), capability=t.name, overlaps=tuple(by_scenario[t.domain]))
        for t in tools.values()
    ]
    return ToolBank(specs)


def _massive_row(
    row: dict[str, Any], tools: dict[str, Tool], bank: ToolBank, rng: np.random.Generator
) -> tuple[list[TaskExample], str | None]:
    """The examples one MASSIVE row gives, or none and why it was dropped."""
    intent, text = row["intent"], row["utt"]
    if HELD_OUT.search(text):  # e.g. "can you convert my this money into", in any intent
        return [], "held-out topic in the request"
    if intent in MASSIVE_NONE:
        k = prompt_size(rng)
        prompt = [bank.specs[i].tool for i in bank.distractors([], k, rng, hard=False)]
        return [TaskExample(text, tuple(prompt), "none")], None
    if intent not in tools:
        return [], "intent left out (held-out topic or too rare)"
    slots = _SLOT.findall(row["annot_utt"])
    if len({s for s, _ in slots}) != len(slots):
        return [], "the same slot twice"
    tool = tools[intent]
    names = {a.name for a in tool.args}
    args = {s: v.strip() for s, v in slots if s in names}
    g = bank.index[tool.name]
    k = prompt_size(rng)
    indices = [g] + bank.distractors([g], k - 1, rng)
    prompt = [bank.specs[i].tool for i in shuffled(rng, indices)]
    example = TaskExample(text, tuple(prompt), "call", (Call(tool.name, args),))
    if check_example(example):
        return [], "value not in the request"
    examples = [example]
    if rng.random() < UNSUPPORTED_SHARE:
        k = prompt_size(rng)
        prompt = [bank.specs[i].tool for i in bank.distractors([g], k, rng, unsupported=True)]
        if prompt:
            examples.append(TaskExample(text, tuple(prompt), "unsupported"))
    return examples, None


def convert_massive(
    rows: list[dict[str, Any]], rng: np.random.Generator
) -> tuple[dict[str, list[TaskExample]], Counter[str]]:
    """MASSIVE rows by partition (train/dev/test) as our examples, plus why rows were dropped.

    Distractors are other MASSIVE intents' tools, mostly from the same scenario. A share of
    the requests is copied once more as UNSUPPORTED, with the whole scenario's tools removed.
    """
    tools = massive_tools(rows)
    bank = _massive_bank(tools)
    out: dict[str, list[TaskExample]] = {"train": [], "dev": [], "test": []}
    dropped: Counter[str] = Counter()
    for row in rows:
        examples, reason = _massive_row(row, tools, bank, rng)
        if reason is not None:
            dropped[reason] += 1
        if examples:
            out[row["partition"]] += examples
    return out, dropped


# ---- both ------------------------------------------------------------------------------------


def convert_external(
    raw_dir: str | Path = "data/raw", out_dir: str | Path = "data/external", seed: int = 0
) -> dict[str, Any]:
    """Convert xLAM and MASSIVE (when downloaded) into ``out_dir/*.jsonl``; return a report."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {}

    xlam_path = raw_dir / "xlam" / "xlam_function_calling_60k.json"
    if xlam_path.exists():
        reasons: Counter[str] = Counter()
        examples: list[TaskExample] = []
        for row in json.loads(xlam_path.read_text()):
            example, reason = convert_xlam(row, rng)
            reasons[reason] += 1
            if example is not None:
                examples.append(example)
        pool = list({t.name: t for ex in examples for t in ex.tools}.values())
        variants = [
            v
            for ex in examples
            if rng.random() < UNSUPPORTED_SHARE and (v := unsupported_variant(ex, pool, rng))
        ]
        rows = [{**ex.to_json(inline=True), "source": "xlam"} for ex in examples + variants]
        write_jsonl(out_dir / "xlam.jsonl", rows)
        report["xlam"] = {
            "kept": len(examples),
            "unsupported_variants": len(variants),
            "reasons": dict(reasons),
        }

    massive_path = raw_dir / "massive" / "1.1" / "data" / "en-US.jsonl"
    if massive_path.exists():
        rows = read_jsonl(massive_path)
        parts, dropped = convert_massive(rows, rng)
        for name, examples in parts.items():
            out = [{**ex.to_json(inline=True), "source": "massive"} for ex in examples]
            write_jsonl(out_dir / f"massive-{name}.jsonl", out)
        report["massive"] = {
            "examples": {k: len(v) for k, v in parts.items()},
            "intent": dict(Counter(ex.intent for v in parts.values() for ex in v)),
            "dropped": dict(dropped),
        }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report
