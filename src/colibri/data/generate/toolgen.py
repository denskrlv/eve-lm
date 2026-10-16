from dataclasses import dataclass

import numpy as np

from colibri.data.generate.library import ToolSpec
from colibri.data.schema import Arg, Tool

# Generated tools teach the model to read schemas it has never seen (users will bring their
# own @tool functions). Each capability gets several tools that differ in name, argument names
# and wording. Topics of the held-out domains (math, travel) are avoided on purpose.

ROOMS = ("kitchen", "bedroom", "living room", "bathroom", "office", "garage", "basement", "nursery")


@dataclass(frozen=True)
class Level:
    attr: str  # "fan speed"
    lo: int
    hi: int
    unit: str = ""  # said after the number: "degrees"


@dataclass(frozen=True)
class Device:
    noun: str
    levels: tuple[Level, ...] = ()
    modes: tuple[str, ...] = ()
    power: bool = True
    rooms: bool = False


DEVICES = (
    Device("fan", (Level("speed", 1, 5),), ("normal", "sleep", "turbo"), rooms=True),
    Device("space heater", (Level("heat level", 1, 5),), ("eco", "comfort"), rooms=True),
    Device("humidifier", (Level("humidity", 30, 70, "percent"),), ("auto", "sleep"), rooms=True),
    Device("air purifier", (Level("fan speed", 1, 3),), ("auto", "sleep", "turbo"), rooms=True),
    Device("coffee machine", (Level("strength", 1, 5),), ("espresso", "latte", "americano")),
    Device("kettle", (Level("temperature", 60, 100, "degrees"),)),
    Device("oven", (Level("temperature", 50, 250, "degrees"),), ("bake", "grill", "fan")),
    Device("dishwasher", (), ("eco", "quick", "intensive")),
    Device(
        "washing machine",
        (Level("temperature", 20, 90, "degrees"),),
        ("cotton", "delicates", "quick wash"),
    ),
    Device("dryer", (), ("low heat", "medium heat", "high heat")),
    Device("sprinkler", (Level("duration", 5, 60, "minutes"),)),
    Device("pool pump", (Level("speed", 1, 3),)),
    Device("tv", (Level("channel", 1, 999),), ("hdmi one", "hdmi two", "antenna")),
    Device("projector", (Level("brightness", 1, 10),)),
    Device("electric blanket", (Level("heat level", 1, 10),)),
    Device("fireplace", (Level("flame height", 1, 5),)),
    Device("aquarium heater", (Level("temperature", 22, 30, "degrees"),)),
    Device("lawn mower", (Level("cutting height", 20, 60, "millimeters"),), ("eco", "normal")),
    Device("water heater", (Level("temperature", 40, 70, "degrees"),), ("eco", "boost")),
    Device("dehumidifier", (Level("humidity", 30, 70, "percent"),), rooms=True),
    Device("heated floor", (Level("warmth", 1, 5),), rooms=True),
    Device("printer", (), ("color", "black and white", "draft")),
    Device("rice cooker", (), ("white rice", "brown rice", "porridge")),
    Device("slow cooker", (Level("cooking time", 1, 12, "hours"),), ("low", "high", "keep warm")),
    Device("bread maker", (), ("basic", "whole wheat", "quick")),
    Device("security camera", (), ("motion only", "always on")),
    Device("aroma diffuser", (Level("intensity", 1, 3),), rooms=True),
    Device("massage chair", (Level("intensity", 1, 5),), ("relax", "deep tissue", "stretch")),
    Device("baby monitor", (Level("sensitivity", 1, 5),), power=True),
)


@dataclass(frozen=True)
class Record:
    entity: str
    titles: tuple[str, ...]
    statuses: tuple[str, ...] = ()
    extra: tuple = ()  # ("enum", name, options, phrase) or ("int", name, lo, hi, phrase)

    @property
    def article(self) -> str:
        return "an" if self.entity[0] in "aeiou" else "a"


RECORDS = (
    Record(
        "ticket",
        ("login bug", "printer jam", "slow wifi", "billing error", "broken checkout"),
        ("open", "in progress", "closed"),
        ("enum", "priority", ("low", "medium", "high"), "with <x> priority"),
    ),
    Record(
        "invoice",
        ("Acme Corp", "the Smith project", "March consulting", "web design work", "Globex"),
        ("paid", "unpaid", "overdue"),
        ("int", "amount", 10, 5000, "for <x> dollars"),
    ),
    Record(
        "expense",
        ("team dinner", "office chairs", "taxi to the client", "software license", "printer paper"),
        (),
        ("int", "amount", 1, 500, "of <x> dollars"),
    ),
    Record(
        "bookmark",
        (
            "the React docs",
            "the pasta recipe",
            "the tax form",
            "the news article",
            "the gym schedule",
        ),
    ),
    Record(
        "recipe",
        ("banana bread", "chicken curry", "pancakes", "lentil soup", "lasagna"),
        (),
        ("int", "servings", 1, 12, "for <x> servings"),
    ),
    Record(
        "project",
        ("website redesign", "garden shed", "mobile app", "kitchen remodel", "thesis"),
        ("active", "paused", "done"),
    ),
    Record(
        "issue",
        (
            "crash on startup",
            "typo on the homepage",
            "slow search",
            "dark mode glitch",
            "memory leak",
        ),
        ("open", "closed"),
        ("enum", "severity", ("minor", "major", "critical"), "marked <x>"),
    ),
    Record(
        "document",
        ("budget plan", "meeting minutes", "lease agreement", "resume", "onboarding guide"),
        ("draft", "final"),
    ),
    Record(
        "deck",
        ("Spanish verbs", "capital cities", "chemistry terms", "French greetings", "anatomy"),
    ),
    Record(
        "plant",
        ("monstera", "basil", "cactus", "fern", "orchid"),
        (),
        ("int", "water_every", 1, 14, "watered every <x> days"),
    ),
    Record(
        "habit",
        ("meditation", "reading", "stretching", "journaling", "flossing"),
        ("done", "skipped"),
        ("int", "times_per_week", 1, 7, "<x> times a week"),
    ),
    Record(
        "goal",
        ("run a marathon", "save for a car", "learn Spanish", "read twenty books"),
        ("active", "achieved", "abandoned"),
    ),
    Record(
        "photo album",
        ("summer vacation", "the wedding", "Max's birthday", "hiking trip", "graduation"),
    ),
    Record(
        "playbook", ("customer onboarding", "incident response", "sales demo", "release checklist")
    ),
)


@dataclass(frozen=True)
class Lookup:
    thing: str  # "air quality"
    key: str  # argument: "city"
    keys: tuple[str, ...]
    phrasings: tuple[str, ...]  # <k> = the key slot


LOOKUPS = (
    Lookup(
        "stock price",
        "symbol",
        ("AAPL", "TSLA", "NVDA", "MSFT", "AMZN"),
        (
            "what's the price of <k>",
            "how is <k> stock doing",
            "<k> share price",
            "check the stock price[ of <k>]",
        ),
    ),
    Lookup(
        "air quality",
        "city",
        ("Berlin", "Delhi", "Los Angeles", "Beijing", "Mexico City"),
        (
            "what's the air quality in <k>",
            "is the air clean in <k> today",
            "air quality[ in <k>]",
        ),
    ),
    Lookup(
        "sunrise time",
        "city",
        ("Berlin", "Oslo", "Cape Town", "Seattle", "Singapore"),
        ("when is sunrise in <k>", "what time does the sun rise in <k>"),
    ),
    Lookup(
        "definition",
        "word",
        ("serendipity", "ephemeral", "ubiquitous", "quixotic", "laconic"),
        ("define <k>", "what does <k> mean", "look up the word <k>"),
    ),
    Lookup(
        "game score",
        "team",
        ("Lakers", "Arsenal", "Yankees", "Real Madrid", "Packers"),
        (
            "what's the score of the <k> game",
            "how did the <k> do last night",
            "did the <k> win",
        ),
    ),
    Lookup(
        "package status",
        "tracking_number",
        ("1Z999AA1", "TBA123456", "JD014600", "LX55821"),
        ("track package <k>", "where is my package <k>", "status of package[ <k>]"),
    ),
    Lookup(
        "showtimes",
        "movie",
        ("Dune", "Inside Out 2", "Oppenheimer", "The Batman", "Wicked"),
        ("when is <k> playing", "showtimes for <k>", "is <k> on tonight"),
    ),
    Lookup(
        "pollen count",
        "city",
        ("Atlanta", "Vienna", "Tokyo", "Dallas", "Munich"),
        ("pollen count in <k>", "how bad is the pollen in <k>"),
    ),
    Lookup(
        "tide times",
        "beach",
        ("Venice Beach", "Bondi", "Brighton", "Malibu", "Cannon Beach"),
        ("when is high tide at <k>", "tide times for <k>"),
    ),
    Lookup(
        "horoscope",
        "sign",
        ("Aries", "Leo", "Virgo", "Pisces", "Gemini"),
        (
            "what's the horoscope for <k>",
            "today's horoscope for <k>",
            "read me the <k> horoscope",
        ),
    ),
    Lookup(
        "bus times",
        "stop",
        ("Oak Street", "the university", "Market Square", "Pine Avenue"),
        ("when is the next bus from <k>", "bus times at <k>", "next bus[ from <k>]"),
    ),
    Lookup(
        "parking availability",
        "garage",
        ("City Center garage", "Mall parking", "Station garage"),
        ("are there free spots at <k>", "check parking at <k>", "how full is <k>"),
    ),
)


def _key(text: str) -> str:
    return text.replace(" ", "_").replace("'", "")


def _sub(text: str, **values: str) -> str:
    """Fill <name> placeholders: _sub("turn {<x>:on|off} the <noun>", x="power", noun="fan")."""
    for name, value in values.items():
        text = text.replace(f"<{name}>", value)
    return text


def _pick(rng: np.random.Generator, options):
    return options[rng.integers(len(options))]


def _variants(rng, names, make, n=2) -> list[ToolSpec]:
    """``n`` tools for one capability, each with a different name."""
    chosen = rng.choice(len(names), size=min(n, len(names)), replace=False)
    return [make(names[i]) for i in chosen]


def _rooms(device: Device, rng, templates, everywhere) -> tuple[list[Arg], tuple[str, ...]]:
    """A room argument for a device that lives in rooms, and the templates that use it.

    ``<r>`` in a template marks where "in the {room}" goes; ``everywhere`` templates ask for
    every room. A required room is named in every template, and "everywhere" fans out to one
    call per room. An optional room is simply left out for "everywhere" (sometimes with the
    description saying so), which teaches the model that the schema decides, not the words.
    """
    if not device.rooms:
        return [], tuple(t.replace("<r>", "") for t in templates)
    name = _pick(rng, ["room", "location"])
    rooms = tuple(rng.choice(ROOMS, size=int(rng.integers(3, 6)), replace=False).tolist())
    required = bool(rng.random() < 0.4)
    if required:
        templates = [t if "<r>" in t else t + "<r>" for t in templates]
        templates += [e + "{" + name + "=*}" for e in everywhere]
        description = ""
    else:
        templates = list(templates) + list(everywhere)
        description = _pick(rng, ["", "all rooms if not given"])
    segment = "[ in the {" + name + "}]"
    arg = Arg(name, "enum", required=required, options=rooms, description=description)
    return [arg], tuple(t.replace("<r>", segment) for t in templates)


def _device_specs(device: Device, rng, weight: float) -> list[ToolSpec]:
    k, noun = _key(device.noun), device.noun
    specs = []

    if device.power:

        def power(name):
            x = _pick(rng, ["on", "power", "enabled", "turn_on"])
            desc = _pick(rng, ["Turn the <noun> on or off", "Switch the <noun> on or off"])
            templates = [
                "turn {<x>:on|off} the <noun><r>",
                "turn the <noun> {<x>:on|off}<r>",
                "switch {<x>:on|off} the <noun><r>",
                "<noun> {<x>:on|off}",
                "can you switch the <noun> {<x>:on|off}",
            ]
            everywhere = [
                "turn {<x>:on|off} the <noun> everywhere",
                "switch {<x>:on|off} every <noun>",
            ]
            templates = [_sub(t, x=x, noun=noun) for t in templates]
            everywhere = [_sub(t, x=x, noun=noun) for t in everywhere]
            rooms, templates = _rooms(device, rng, templates, everywhere)
            tool = Tool(name, _sub(desc, noun=noun), (Arg(x, "bool"), *rooms), "generated-devices")
            return ToolSpec(tool, templates, capability=f"{k}.power", weight=weight)

        names = [
            f"set_{k}_power",
            f"toggle_{k}",
            f"switch_{k}",
            f"{k}_power",
            f"turn_{k}_on_or_off",
        ]
        specs += _variants(rng, names, power)

    for level in device.levels:
        a = _key(level.attr)
        unit = " " + level.unit if level.unit else ""

        def set_level(name, level=level, a=a, unit=unit):
            v = _pick(rng, [a, "level", "value"])
            desc = _pick(rng, ["Set the <noun> <attr>", "Change the <attr> of the <noun>"])
            templates = [
                "set the <noun> <attr> to {<v>}<unit><r>",
                "change the <noun> <attr> to {<v>}<unit>",
                "put the <noun> <attr> at {<v>}<unit>",
                "<noun> <attr> {<v>}<unit>",
                "set the <noun> <attr>[ to {<v>}<unit>]",
            ]
            subs = {"noun": noun, "attr": level.attr, "v": v, "unit": unit}
            everywhere = [_sub("set the <noun> <attr> to {<v>}<unit> in every room", **subs)]
            templates = [_sub(t, **subs) for t in templates]
            rooms, templates = _rooms(device, rng, templates, everywhere)
            arg = Arg(v, "int", min=level.lo, max=level.hi, description=level.unit)
            tool = Tool(name, _sub(desc, **subs), (arg, *rooms), "generated-devices")
            return ToolSpec(tool, templates, capability=f"{k}.{a}", weight=weight)

        names = [f"set_{k}_{a}", f"adjust_{k}_{a}", f"change_{k}_{a}", f"{k}_set_{a}"]
        specs += _variants(rng, names, set_level)

    if device.modes:

        def set_mode(name):
            m = _pick(rng, ["mode", "setting", "program"])
            desc = _pick(rng, ["Set the <noun> mode", "Choose a program for the <noun>"])
            templates = [
                "set the <noun> to {<m>}[ mode]<r>",
                "switch the <noun> to {<m>}",
                "put the <noun> on {<m>}",
                "use {<m>} on the <noun>",
                "change the <noun> mode[ to {<m>}]",
            ]
            everywhere = [_sub("set every <noun> to {<m>}", noun=noun, m=m)]
            templates = [_sub(t, noun=noun, m=m) for t in templates]
            rooms, templates = _rooms(device, rng, templates, everywhere)
            arg = Arg(m, "enum", options=device.modes)
            tool = Tool(name, _sub(desc, noun=noun), (arg, *rooms), "generated-devices")
            return ToolSpec(tool, templates, capability=f"{k}.mode", weight=weight)

        names = [f"set_{k}_mode", f"change_{k}_mode", f"{k}_mode", f"select_{k}_program"]
        specs += _variants(rng, names, set_mode)
    return specs


def _record_specs(record: Record, rng, weight: float) -> list[ToolSpec]:
    k, e = _key(record.entity), record.entity
    subs = {"e": e, "art": record.article}
    titles = record.titles
    specs = []

    def create(name):
        t = _pick(rng, ["title", "name"])
        args = [Arg(t, "str")]
        extra = ""
        if record.extra:
            kind, x = record.extra[0], record.extra[1]
            if kind == "enum":
                args.append(Arg(x, "enum", required=False, options=record.extra[2]))
            else:
                args.append(Arg(x, "int", required=False, min=record.extra[2], max=record.extra[3]))
            extra = "[ " + _sub(record.extra[-1], x="{" + x + "}") + "]"
        desc = _pick(rng, ["Create a new <e>", "Add <art> <e>"])
        templates = [
            "create <art> <e> called {<t>}<extra>",
            "add a new <e>[ named {<t>}]",
            "new <e>: {<t>}",
            "make <art> <e> for {<t>}<extra>",
        ]
        templates = tuple(_sub(x, t=t, extra=extra, **subs) for x in templates)
        tool = Tool(name, _sub(desc, **subs), tuple(args), "generated-records")
        return ToolSpec(tool, templates, {t: titles}, f"{k}.create", weight=weight)

    def delete(name):
        t = _pick(rng, ["title", "name"])
        templates = [
            "delete the {<t>} <e>",
            "remove the <e> called {<t>}",
            "get rid of the {<t>} <e>",
            "delete the <e>[ {<t>}]",
        ]
        templates = tuple(_sub(x, t=t, **subs) for x in templates)
        desc = _sub(_pick(rng, ["Delete a <e>", "Remove a <e>"]), **subs)
        tool = Tool(name, desc, (Arg(t, "str"),), "generated-records")
        return ToolSpec(tool, templates, {t: titles}, f"{k}.delete", weight=weight)

    def find(name):
        q = _pick(rng, ["query", "text", "keyword"])
        templates = [
            "search my <e>s for {<q>}",
            "find <e>s about {<q>}",
            "do I have any <e>s about {<q>}",
        ]
        templates = tuple(_sub(x, q=q, **subs) for x in templates)
        desc = _sub(_pick(rng, ["Search <e>s", "Find <e>s matching a text"]), **subs)
        tool = Tool(name, desc, (Arg(q, "str"),), "generated-records")
        return ToolSpec(tool, templates, {q: titles}, f"{k}.find", weight=weight)

    specs += _variants(rng, [f"create_{k}", f"add_{k}", f"new_{k}"], create)
    specs += _variants(rng, [f"delete_{k}", f"remove_{k}"], delete)
    specs += _variants(rng, [f"search_{k}s", f"find_{k}", f"lookup_{k}"], find)

    if record.statuses:

        def status(name):
            t = _pick(rng, ["title", "name"])
            s = _pick(rng, ["status", "state"])
            templates = [
                "mark the {<t>} <e> as {<s>}",
                "change the status of {<t>} to {<s>}",
                "the {<t>} <e> is {<s>} now",
                "set the <e>[ {<t>}] to {<s>}",
            ]
            templates = tuple(_sub(x, t=t, s=s, **subs) for x in templates)
            args = (Arg(t, "str"), Arg(s, "enum", options=record.statuses))
            tool = Tool(name, f"Change the status of a {e}", args, "generated-records")
            return ToolSpec(tool, templates, {t: titles}, f"{k}.status", weight=weight)

        specs += _variants(rng, [f"set_{k}_status", f"update_{k}_status", f"mark_{k}"], status)
    return specs


def _lookup_specs(lookup: Lookup, rng, weight: float) -> list[ToolSpec]:
    k, thing, key = _key(lookup.thing), lookup.thing, lookup.key

    def get(name):
        desc = _pick(rng, ["Get the <thing>", "Look up the <thing>", "Check the <thing>"])
        templates = tuple(_sub(p, k="{" + key + "}") for p in lookup.phrasings)
        tool = Tool(name, _sub(desc, thing=thing), (Arg(key, "str"),), "generated-lookups")
        return ToolSpec(tool, templates, {key: lookup.keys}, f"lookup.{k}", weight=weight)

    return _variants(rng, [f"get_{k}", f"check_{k}", f"lookup_{k}", f"fetch_{k}"], get)


def generated_specs(seed: int = 0, weight: float = 0.2) -> list[ToolSpec]:
    """All generated tools. The same seed always gives the same tools."""
    rng = np.random.default_rng(seed)
    specs = []
    for device in DEVICES:
        specs += _device_specs(device, rng, weight)
    for record in RECORDS:
        specs += _record_specs(record, rng, weight)
    for lookup in LOOKUPS:
        specs += _lookup_specs(lookup, rng, weight)
    return specs
