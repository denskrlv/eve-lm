from collections.abc import Callable, Sequence
from functools import partial

import numpy as np

from colibri.data.draw import pick
from colibri.data.generate.catalog import (
    DEVICES,
    LOOKUPS,
    MORE_DEVICES,
    MORE_LOOKUPS,
    MORE_RECORDS,
    OPERATIONS,
    RECORDS,
    ROOMS,
    Device,
    Level,
    Lookup,
    Operation,
    Record,
)
from colibri.data.generate.library import ToolSpec
from colibri.data.schema import Arg, Tool

# Each builder below makes one tool for a capability. Its templates use <name> placeholders
# (filled by _sub) and <r> where "in the {room}" goes (filled by _rooms).
POWER_TEMPLATES = (
    "turn {<x>:on|off} the <noun><r>",
    "turn the <noun> {<x>:on|off}<r>",
    "switch {<x>:on|off} the <noun><r>",
    "<noun> {<x>:on|off}",
    "can you switch the <noun> {<x>:on|off}",
)
POWER_EVERYWHERE = (
    "turn {<x>:on|off} the <noun> everywhere",
    "switch {<x>:on|off} every <noun>",
)
LEVEL_TEMPLATES = (
    "set the <noun> <attr> to {<v>}<unit><r>",
    "change the <noun> <attr> to {<v>}<unit>",
    "put the <noun> <attr> at {<v>}<unit>",
    "<noun> <attr> {<v>}<unit>",
    "set the <noun> <attr>[ to {<v>}<unit>]",
)
MODE_TEMPLATES = (
    "set the <noun> to {<m>}[ mode]<r>",
    "switch the <noun> to {<m>}",
    "put the <noun> on {<m>}",
    "use {<m>} on the <noun>",
    "change the <noun> mode[ to {<m>}]",
)
# Templates for a create tool whose extra field is required hold both required values in
# optional segments, so an ASK example can say one and leave out the other.
CREATE_TEMPLATES_BOTH_REQUIRED = (
    "create <art> <e> called {<t>}<extra>",
    "add a new <e>[ named {<t>}]<extra>",
    "new <e>[: {<t>}]<extra>",
    "make <art> <e>[ for {<t>}]<extra>",
)
CREATE_TEMPLATES = (
    "create <art> <e> called {<t>}<extra>",
    "add a new <e>[ named {<t>}]",
    "new <e>: {<t>}",
    "make <art> <e> for {<t>}<extra>",
)
DELETE_TEMPLATES = (
    "delete the {<t>} <e>",
    "remove the <e> called {<t>}",
    "get rid of the {<t>} <e>",
    "delete the <e>[ {<t>}]",
)
FIND_TEMPLATES = (
    "search my <e>s for {<q>}",
    "find <e>s about {<q>}",
    "do I have any <e>s about {<q>}",
)
GENERATED_WEIGHT = 0.08  # sampling weight per generated tool (hand-written tools have 1.0)
OPERATION_WEIGHT_FACTOR = 3  # operations are few, so each counts this many times more
MIN_ROOMS, MAX_ROOMS = 3, 5  # rooms a device with a room argument can be in
REQUIRED_ROOM_SHARE = 0.4  # devices whose room argument is required
REQUIRED_EXTRA_SHARE = 0.5  # create tools whose extra field is required
STATUS_TEMPLATES = (
    "mark the {<t>} <e> as {<s>}",
    "change the status of {<t>} to {<s>}",
    "the {<t>} <e> is {<s>} now",
    "set the <e>[ {<t>}] to {<s>}",
)


def _key(text: str) -> str:
    return text.replace(" ", "_").replace("'", "")


def _sub(text: str, **values: str) -> str:
    """Fill <name> placeholders: _sub("turn {<x>:on|off} the <noun>", x="power", noun="fan")."""
    for name, value in values.items():
        text = text.replace(f"<{name}>", value)
    return text


def _variants(
    rng: np.random.Generator, names: Sequence[str], make: Callable[[str], ToolSpec], count: int = 2
) -> list[ToolSpec]:
    """``count`` tools for one capability, each with a different name."""
    chosen = rng.choice(len(names), size=min(count, len(names)), replace=False)
    return [make(names[i]) for i in chosen]


def _rooms(
    device: Device, rng: np.random.Generator, templates: Sequence[str], everywhere: Sequence[str]
) -> tuple[list[Arg], tuple[str, ...]]:
    """A room argument for a device that lives in rooms, and the templates that use it.

    ``<r>`` in a template marks where "in the {room}" goes; ``everywhere`` templates ask for
    every room. A required room is named in every template, and "everywhere" fans out to one
    call per room. An optional room is simply left out for "everywhere" (sometimes with the
    description saying so), which teaches the model that the schema decides, not the words.
    """
    if not device.rooms:
        return [], tuple(t.replace("<r>", "") for t in templates)
    name = pick(rng, ["room", "location"])
    rooms = tuple(
        rng.choice(ROOMS, size=int(rng.integers(MIN_ROOMS, MAX_ROOMS + 1)), replace=False).tolist()
    )
    required = bool(rng.random() < REQUIRED_ROOM_SHARE)
    if required:
        templates = [t if "<r>" in t else t + "<r>" for t in templates]
        templates += [e + "{" + name + "=*}" for e in everywhere]
        description = ""
    else:
        templates = list(templates) + list(everywhere)
        description = pick(rng, ["", "all rooms if not given"])
    segment = "[ in the {" + name + "}]"
    arg = Arg(name, "enum", required=required, options=rooms, description=description)
    return [arg], tuple(t.replace("<r>", segment) for t in templates)


# ---- devices --------------------------------------------------------------------------------


def _power_tool(name: str, device: Device, rng: np.random.Generator, weight: float) -> ToolSpec:
    noun = device.noun
    power_arg = pick(rng, ["on", "power", "enabled", "turn_on"])
    desc = pick(rng, ["Turn the <noun> on or off", "Switch the <noun> on or off"])
    templates = [_sub(t, x=power_arg, noun=noun) for t in POWER_TEMPLATES]
    everywhere = [_sub(t, x=power_arg, noun=noun) for t in POWER_EVERYWHERE]
    rooms, room_templates = _rooms(device, rng, templates, everywhere)
    tool = Tool(name, _sub(desc, noun=noun), (Arg(power_arg, "bool"), *rooms), "generated-devices")
    return ToolSpec(tool, room_templates, capability=f"{_key(noun)}.power", weight=weight)


def _level_tool(
    name: str, device: Device, level: Level, rng: np.random.Generator, weight: float
) -> ToolSpec:
    attr_key = _key(level.attr)
    unit = " " + level.unit if level.unit else ""
    value_arg = pick(rng, [attr_key, "level", "value"])
    desc = pick(rng, ["Set the <noun> <attr>", "Change the <attr> of the <noun>"])
    subs = {"noun": device.noun, "attr": level.attr, "v": value_arg, "unit": unit}
    everywhere = [_sub("set the <noun> <attr> to {<v>}<unit> in every room", **subs)]
    templates = [_sub(t, **subs) for t in LEVEL_TEMPLATES]
    rooms, room_templates = _rooms(device, rng, templates, everywhere)
    arg = Arg(value_arg, "int", min=level.lo, max=level.hi, description=level.unit)
    tool = Tool(name, _sub(desc, **subs), (arg, *rooms), "generated-devices")
    capability = f"{_key(device.noun)}.{attr_key}"
    return ToolSpec(tool, room_templates, capability=capability, weight=weight)


def _mode_tool(name: str, device: Device, rng: np.random.Generator, weight: float) -> ToolSpec:
    noun = device.noun
    mode_arg = pick(rng, ["mode", "setting", "program"])
    desc = pick(rng, ["Set the <noun> mode", "Choose a program for the <noun>"])
    everywhere = [_sub("set every <noun> to {<m>}", noun=noun, m=mode_arg)]
    templates = [_sub(t, noun=noun, m=mode_arg) for t in MODE_TEMPLATES]
    rooms, room_templates = _rooms(device, rng, templates, everywhere)
    arg = Arg(mode_arg, "enum", options=device.modes)
    tool = Tool(name, _sub(desc, noun=noun), (arg, *rooms), "generated-devices")
    return ToolSpec(tool, room_templates, capability=f"{_key(noun)}.mode", weight=weight)


def _device_specs(device: Device, rng: np.random.Generator, weight: float) -> list[ToolSpec]:
    """Power, level and mode tools for a device, two differently named tools for each."""
    noun_key = _key(device.noun)
    specs = []
    if device.power:
        names = [
            f"set_{noun_key}_power",
            f"toggle_{noun_key}",
            f"switch_{noun_key}",
            f"{noun_key}_power",
            f"turn_{noun_key}_on_or_off",
        ]
        specs += _variants(rng, names, partial(_power_tool, device=device, rng=rng, weight=weight))
    for level in device.levels:
        attr_key = _key(level.attr)
        names = [
            f"set_{noun_key}_{attr_key}",
            f"adjust_{noun_key}_{attr_key}",
            f"change_{noun_key}_{attr_key}",
            f"{noun_key}_set_{attr_key}",
        ]
        make = partial(_level_tool, device=device, level=level, rng=rng, weight=weight)
        specs += _variants(rng, names, make)
    if device.modes:
        names = [
            f"set_{noun_key}_mode",
            f"change_{noun_key}_mode",
            f"{noun_key}_mode",
            f"select_{noun_key}_program",
        ]
        specs += _variants(rng, names, partial(_mode_tool, device=device, rng=rng, weight=weight))
    return specs


# ---- records --------------------------------------------------------------------------------


def _record_subs(record: Record) -> dict[str, str]:
    return {"e": record.entity, "art": record.article}


def _extra_arg(record: Record, required: bool) -> tuple[Arg, str]:
    """The record's extra field as an argument, and its optional template segment."""
    kind, name = record.extra[0], record.extra[1]
    if kind == "enum":
        arg = Arg(name, "enum", required=required, options=record.extra[2])
    else:
        lo, hi = record.extra[2], record.extra[3]
        arg = Arg(name, "int", required=required, min=lo, max=hi)
    return arg, "[ " + _sub(record.extra[-1], x="{" + name + "}") + "]"


def _create_tool(name: str, record: Record, rng: np.random.Generator, weight: float) -> ToolSpec:
    title_arg = pick(rng, ["title", "name"])
    args = [Arg(title_arg, "str")]
    extra = ""
    both = bool(record.extra) and rng.random() < REQUIRED_EXTRA_SHARE
    if record.extra:
        extra_arg, extra = _extra_arg(record, required=both)
        args.append(extra_arg)
    desc = pick(rng, ["Create a new <e>", "Add <art> <e>"])
    subs = _record_subs(record)
    chosen = CREATE_TEMPLATES_BOTH_REQUIRED if both else CREATE_TEMPLATES
    templates = tuple(_sub(x, t=title_arg, extra=extra, **subs) for x in chosen)
    tool = Tool(name, _sub(desc, **subs), tuple(args), "generated-records")
    capability = f"{_key(record.entity)}.create"
    return ToolSpec(tool, templates, {title_arg: record.titles}, capability, weight=weight)


def _delete_tool(name: str, record: Record, rng: np.random.Generator, weight: float) -> ToolSpec:
    title_arg = pick(rng, ["title", "name"])
    subs = _record_subs(record)
    templates = tuple(_sub(x, t=title_arg, **subs) for x in DELETE_TEMPLATES)
    desc = _sub(pick(rng, ["Delete a <e>", "Remove a <e>"]), **subs)
    tool = Tool(name, desc, (Arg(title_arg, "str"),), "generated-records")
    capability = f"{_key(record.entity)}.delete"
    return ToolSpec(tool, templates, {title_arg: record.titles}, capability, weight=weight)


def _find_tool(name: str, record: Record, rng: np.random.Generator, weight: float) -> ToolSpec:
    query_arg = pick(rng, ["query", "text", "keyword"])
    subs = _record_subs(record)
    templates = tuple(_sub(x, q=query_arg, **subs) for x in FIND_TEMPLATES)
    desc = _sub(pick(rng, ["Search <e>s", "Find <e>s matching a text"]), **subs)
    tool = Tool(name, desc, (Arg(query_arg, "str"),), "generated-records")
    capability = f"{_key(record.entity)}.find"
    return ToolSpec(tool, templates, {query_arg: record.titles}, capability, weight=weight)


def _status_tool(name: str, record: Record, rng: np.random.Generator, weight: float) -> ToolSpec:
    title_arg = pick(rng, ["title", "name"])
    status_arg = pick(rng, ["status", "state"])
    templates = tuple(
        _sub(x, t=title_arg, s=status_arg, **_record_subs(record)) for x in STATUS_TEMPLATES
    )
    args = (Arg(title_arg, "str"), Arg(status_arg, "enum", options=record.statuses))
    tool = Tool(name, f"Change the status of a {record.entity}", args, "generated-records")
    capability = f"{_key(record.entity)}.status"
    return ToolSpec(tool, templates, {title_arg: record.titles}, capability, weight=weight)


def _record_specs(record: Record, rng: np.random.Generator, weight: float) -> list[ToolSpec]:
    """Create, delete, find and (with statuses) status tools for a kind of record."""
    entity_key = _key(record.entity)
    builders: list[tuple[list[str], Callable[..., ToolSpec]]] = [
        ([f"create_{entity_key}", f"add_{entity_key}", f"new_{entity_key}"], _create_tool),
        ([f"delete_{entity_key}", f"remove_{entity_key}"], _delete_tool),
        ([f"search_{entity_key}s", f"find_{entity_key}", f"lookup_{entity_key}"], _find_tool),
    ]
    if record.statuses:
        builders.append(
            (
                [f"set_{entity_key}_status", f"update_{entity_key}_status", f"mark_{entity_key}"],
                _status_tool,
            )
        )
    specs = []
    for names, build in builders:
        specs += _variants(rng, names, partial(build, record=record, rng=rng, weight=weight))
    return specs


# ---- lookups and operations -----------------------------------------------------------------


def _lookup_tool(name: str, lookup: Lookup, rng: np.random.Generator, weight: float) -> ToolSpec:
    thing_key, key = _key(lookup.thing), lookup.key
    desc = pick(rng, ["Get the <thing>", "Look up the <thing>", "Check the <thing>"])
    templates = tuple(_sub(p, k="{" + key + "}") for p in lookup.phrasings)
    tool = Tool(name, _sub(desc, thing=lookup.thing), (Arg(key, "str"),), "generated-lookups")
    return ToolSpec(tool, templates, {key: lookup.keys}, f"lookup.{thing_key}", weight=weight)


def _lookup_specs(lookup: Lookup, rng: np.random.Generator, weight: float) -> list[ToolSpec]:
    thing_key = _key(lookup.thing)
    names = [f"get_{thing_key}", f"check_{thing_key}", f"lookup_{thing_key}", f"fetch_{thing_key}"]
    return _variants(rng, names, partial(_lookup_tool, lookup=lookup, rng=rng, weight=weight))


def _operation_tool(name: str, op: Operation, rng: np.random.Generator, weight: float) -> ToolSpec:
    tool = Tool(name, pick(rng, op.descriptions), op.args, "generated-operations")
    capability = "op." + op.names[0]
    return ToolSpec(tool, op.templates, op.values, capability, weight=weight)


def _operation_specs(op: Operation, rng: np.random.Generator, weight: float) -> list[ToolSpec]:
    make = partial(_operation_tool, op=op, rng=rng, weight=weight)
    return _variants(rng, list(op.names), make)


def generated_specs(seed: int = 0, weight: float = GENERATED_WEIGHT) -> list[ToolSpec]:
    """All generated tools. The same seed always gives the same tools.

    ``weight`` is per tool: ~1,000 of them at 0.08 against 47 hand-written tools at 1.0 make
    about 63% of the template examples generated, each tool the gold one ~50 times in 80k.
    """
    rng = np.random.default_rng(seed)
    specs = []
    for device in DEVICES + MORE_DEVICES:
        specs += _device_specs(device, rng, weight)
    for record in RECORDS + MORE_RECORDS:
        specs += _record_specs(record, rng, weight)
    for lookup in LOOKUPS + MORE_LOOKUPS:
        specs += _lookup_specs(lookup, rng, weight)
    for op in OPERATIONS:  # few tools, so each counts more
        specs += _operation_specs(op, rng, OPERATION_WEIGHT_FACTOR * weight)
    return specs
