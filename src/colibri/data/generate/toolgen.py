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


# More objects (2026-10-17). With only the 55 above, a model can pass every generated example by
# memorizing which noun goes with which capability; with many more it has to compare the request
# with each schema. They stay clear of what the hand-written tools do (lights, temperature,
# locks, blinds, vacuums, media, workouts, weather, time, notes, lists, messages, cars, events)
# and of the held-out topics.

MORE_DEVICES = (
    Device("toaster", (Level("browning", 1, 6),), ("bagel", "defrost", "reheat")),
    Device("microwave", (Level("power level", 1, 10),), ("defrost", "popcorn", "reheat")),
    Device("blender", (Level("speed", 1, 10),), ("pulse", "smoothie", "ice crush")),
    Device("air fryer", (Level("temperature", 80, 200, "degrees"),), ("fries", "chicken", "bake")),
    Device("stand mixer", (Level("speed", 1, 10),)),
    Device("juicer", ()),
    Device("food processor", (Level("speed", 1, 3),), ("chop", "puree", "knead")),
    Device("ice maker", (), ("small cubes", "large cubes", "crushed ice")),
    Device("wine fridge", (Level("temperature", 5, 18, "degrees"),)),
    Device("freezer", (Level("cold setting", 1, 7),), ("fast freeze", "holiday")),
    Device("range hood", (Level("fan speed", 1, 4),)),
    Device("garbage disposal", ()),
    Device("sous vide", (Level("temperature", 40, 90, "degrees"),)),
    Device(
        "pressure cooker", (Level("cooking time", 1, 120, "minutes"),), ("soup", "stew", "beans")
    ),
    Device("coffee grinder", (Level("grind size", 1, 20),), ("espresso", "drip", "french press")),
    Device("popcorn maker", ()),
    Device("waffle iron", (Level("browning", 1, 5),)),
    Device("soda maker", (Level("fizz level", 1, 3),)),
    Device("exhaust fan", (Level("speed", 1, 3),), rooms=True),
    Device("towel warmer", (Level("heat level", 1, 3),)),
    Device("hair dryer", (Level("heat", 1, 3),), ("cool shot", "diffuser", "turbo")),
    Device("electric toothbrush", (), ("clean", "whiten", "sensitive")),
    Device("smart mirror", (Level("brightness", 1, 10),), ("makeup", "news", "standby")),
    Device("sauna", (Level("temperature", 60, 100, "degrees"),)),
    Device("hot tub", (Level("temperature", 30, 40, "degrees"),), ("jets", "economy", "filter")),
    Device("steam shower", (Level("steam level", 1, 5),)),
    Device("ceiling fan", (Level("speed", 1, 6),), ("breeze", "reverse"), rooms=True),
    Device("pool heater", (Level("temperature", 20, 35, "degrees"),)),
    Device("fountain", (Level("flow", 1, 5),), ("gentle", "jet", "cascade")),
    Device("pond pump", (Level("flow", 1, 5),)),
    Device("ventilation system", (Level("airflow", 1, 4),), ("boost", "away", "home")),
    Device("electric bike", (Level("assist level", 1, 5),), ("eco", "tour", "sport")),
    Device(
        "electric scooter", (Level("speed limit", 10, 25, "kilometers per hour"),), ("eco", "sport")
    ),
    Device("kiln", (Level("temperature", 600, 1300, "degrees"),), ("bisque", "glaze")),
    Device("label printer", ()),
    Device("paper shredder", (), ("auto", "reverse")),
    Device("laminator", (Level("heat level", 1, 3),)),
    Device("monitor", (Level("brightness", 0, 100, "percent"),), ("game", "movie", "reading")),
    Device("game console", (), ("performance", "quiet")),
    Device("router", (), ("normal", "gaming", "guest only")),
    Device("pet feeder", (Level("portion size", 1, 5),)),
    Device("litter box", (), ("auto clean", "manual")),
    Device("pet fountain", (Level("flow", 1, 3),)),
    Device("grill", (Level("temperature", 100, 300, "degrees"),), ("smoke", "sear")),
    Device("smoker", (Level("temperature", 90, 150, "degrees"),)),
    Device("pizza oven", (Level("temperature", 200, 500, "degrees"),)),
    Device("patio heater", (Level("heat level", 1, 3),)),
    Device("snow blower", ()),
    Device("leaf blower", (Level("speed", 1, 3),)),
    Device("pressure washer", (Level("pressure", 1, 5),), ("rinse", "soap")),
    Device("hedge trimmer", ()),
    Device("greenhouse fan", (Level("speed", 1, 3),)),
    Device("garage heater", (Level("heat level", 1, 5),)),
    Device("dust collector", ()),
    Device("air compressor", (Level("pressure", 60, 150, "psi"),)),
    Device(
        "white noise machine",
        (Level("intensity", 1, 10),),
        ("rain", "ocean", "fan sound"),
        rooms=True,
    ),
    Device("crib mobile", (), ("lullaby", "nature sounds")),
    Device("bottle warmer", (Level("temperature", 30, 45, "degrees"),)),
    Device("baby swing", (Level("speed", 1, 5),)),
    Device("clothes steamer", (Level("steam level", 1, 3),)),
    Device("sewing machine", (Level("speed", 1, 10),)),
    Device("foot spa", (Level("temperature", 30, 45, "degrees"),), ("massage", "bubbles")),
    Device("massage gun", (Level("speed", 1, 5),)),
    Device("standing desk", (Level("height", 60, 125, "centimeters"),), power=False),
    Device(
        "smart bed", (Level("firmness", 1, 10),), ("snore", "zero gravity", "flat"), power=False
    ),
    Device("photo frame", (Level("brightness", 1, 10),), ("slideshow", "clock")),
    Device("smart plug", (), rooms=True),
    Device("heated driveway", ()),
    Device("awning", (Level("extension", 0, 100, "percent"),), power=False),
    Device("trash compactor", ()),
    Device("beverage cooler", (Level("temperature", 2, 12, "degrees"),)),
    Device("yogurt maker", (Level("fermentation time", 4, 12, "hours"),)),
    Device("food dehydrator", (Level("temperature", 35, 75, "degrees"),)),
    Device("egg cooker", (), ("soft", "medium", "hard")),
    Device("milk frother", (), ("hot foam", "cold foam", "warm milk")),
    Device("deep fryer", (Level("temperature", 150, 190, "degrees"),)),
    Device("induction hob", (Level("power level", 1, 9),)),
    Device("warming drawer", (Level("temperature", 30, 80, "degrees"),)),
    Device("water dispenser", (), ("hot", "cold", "room temperature")),
    Device("aquarium pump", (Level("flow", 1, 5),)),
    Device("egg incubator", (Level("temperature", 35, 39, "degrees"),)),
    Device("home battery", (), ("backup", "self consumption", "time of use"), power=False),
    Device("generator", ()),
    Device("sump pump", ()),
)

MORE_RECORDS = (
    Record(
        "budget",
        ("groceries", "holiday fund", "home repairs", "eating out"),
        ("on track", "over budget"),
        ("int", "amount", 50, 5000, "of <x> dollars"),
    ),
    Record(
        "subscription",
        ("Netflix", "the gym membership", "Spotify", "cloud storage"),
        ("active", "cancelled"),
        ("int", "price", 1, 100, "at <x> dollars a month"),
    ),
    Record(
        "warranty", ("the dishwasher", "my laptop", "the sofa", "my phone"), ("valid", "expired")
    ),
    Record(
        "lead",
        ("Initech", "Hooli", "Umbrella Inc", "Stark Industries"),
        ("new", "contacted", "qualified"),
        ("enum", "source", ("website", "referral", "trade show"), "from <x>"),
    ),
    Record(
        "deal",
        ("the Q3 renewal", "the Globex upgrade", "the pilot program", "the school contract"),
        ("open", "won", "lost"),
        ("int", "value", 100, 90000, "worth <x> dollars"),
    ),
    Record(
        "candidate",
        ("Maria Lopez", "James Chen", "Aisha Khan", "Tom Becker"),
        ("applied", "interviewing", "hired", "rejected"),
    ),
    Record(
        "job posting",
        ("backend engineer", "product designer", "sales lead", "office manager"),
        ("open", "closed"),
    ),
    Record(
        "survey",
        ("customer feedback", "team pulse", "event satisfaction", "onboarding check"),
        ("draft", "live", "closed"),
    ),
    Record(
        "poll", ("lunch spot", "team name", "offsite date", "office snacks"), ("open", "closed")
    ),
    Record(
        "book",
        ("Dune", "Educated", "Atomic Habits", "The Hobbit", "Beloved"),
        ("to read", "reading", "finished"),
        ("int", "rating", 1, 5, "rated <x> stars"),
    ),
    Record(
        "course",
        ("intro to Python", "watercolor basics", "public speaking", "beginner guitar"),
        ("enrolled", "completed", "dropped"),
    ),
    Record(
        "assignment",
        ("history essay", "lab report", "book review", "chemistry worksheet"),
        ("todo", "submitted", "graded"),
    ),
    Record("client", ("Dr. Patel", "the Lee family", "Northwind", "Bluebird Cafe")),
    Record("vendor", ("Northwind Traders", "Contoso", "Fabrikam", "Wingtip Supplies")),
    Record(
        "purchase order",
        ("office laptops", "printer toner", "standing desks", "safety gloves"),
        ("pending", "approved", "received"),
        ("int", "quantity", 1, 500, "for <x> units"),
    ),
    Record(
        "asset",
        ("laptop 42", "projector B", "the spare monitor", "van 3"),
        ("in use", "in storage", "retired"),
    ),
    Record(
        "wiki page",
        ("onboarding", "API guide", "holiday policy", "coding standards"),
        ("draft", "published"),
    ),
    Record(
        "blog post",
        ("launch announcement", "hiring update", "year in review", "customer story"),
        ("draft", "scheduled", "published"),
    ),
    Record(
        "newsletter",
        ("October issue", "product update", "holiday edition", "welcome series"),
        ("draft", "sent"),
    ),
    Record(
        "campaign",
        ("spring sale", "back to school", "referral push", "summer launch"),
        ("planned", "running", "ended"),
        ("int", "budget", 100, 50000, "with a budget of <x> dollars"),
    ),
    Record(
        "coupon",
        ("WELCOME10", "SPRING25", "FREESHIP", "VIP50"),
        ("active", "expired"),
        ("int", "discount", 5, 90, "for <x> percent off"),
    ),
    Record(
        "listing",
        ("vintage lamp", "mountain bike", "old textbooks", "baby stroller"),
        ("active", "sold"),
        ("int", "price", 1, 2000, "for <x> dollars"),
    ),
    Record(
        "pet",
        ("Max", "Luna", "Bella", "Charlie"),
        (),
        ("enum", "species", ("dog", "cat", "rabbit", "parrot"), "who is a <x>"),
    ),
    Record(
        "supplement",
        ("vitamin D", "magnesium", "fish oil", "zinc"),
        (),
        ("int", "dose", 1, 5, "<x> pills a day"),
    ),
    Record("journal entry", ("first day at work", "rainy sunday", "big news", "lessons learned")),
    Record(
        "gift idea",
        ("mom's birthday", "secret santa", "our anniversary", "the teacher"),
        ("idea", "bought", "wrapped"),
    ),
    Record(
        "chore",
        ("take out the trash", "water the lawn", "clean the gutters", "mop the floors"),
        ("todo", "done"),
        ("int", "times_per_week", 1, 7, "<x> times a week"),
    ),
    Record(
        "repair",
        ("leaky faucet", "cracked tile", "broken fence", "loose railing"),
        ("reported", "scheduled", "fixed"),
    ),
    Record("garden bed", ("tomatoes", "herbs", "strawberries", "pumpkins")),
    Record("meal plan", ("monday dinner", "sunday brunch", "work lunches", "party snacks")),
    Record(
        "volunteer shift",
        ("food bank", "animal shelter", "park cleanup", "book fair"),
        ("open", "filled"),
    ),
    Record("folder", ("taxes", "receipts", "school stuff", "house papers")),
    Record(
        "domain name",
        ("myshop.com", "colibri.dev", "bestpizza.net", "lucyart.io"),
        ("active", "expiring"),
    ),
    Record(
        "server",
        ("web one", "db primary", "staging box", "build agent"),
        ("running", "stopped"),
    ),
    Record(
        "backup",
        ("photos", "work laptop", "the database", "family videos"),
        ("scheduled", "completed", "failed"),
    ),
    Record(
        "experiment",
        ("new onboarding flow", "blue button", "price test", "shorter signup"),
        ("draft", "running", "stopped"),
    ),
    Record(
        "feature flag",
        ("dark mode", "new checkout", "beta search", "smart replies"),
        ("enabled", "disabled"),
    ),
    Record(
        "sprint",
        ("sprint 12", "sprint 13", "the holiday sprint", "the cleanup sprint"),
        ("planned", "active", "closed"),
    ),
    Record(
        "milestone",
        ("beta launch", "version two", "first customer", "public release"),
        ("open", "reached"),
    ),
    Record(
        "quote",
        ("roof repair", "kitchen tiles", "website build", "fence painting"),
        ("sent", "accepted", "declined"),
        ("int", "amount", 50, 20000, "for <x> dollars"),
    ),
    Record(
        "contract",
        ("office lease", "cleaning service", "phone plan", "internet plan"),
        ("draft", "signed", "expired"),
    ),
    Record(
        "work shift",
        ("monday morning", "weekend night", "friday late", "holiday cover"),
        ("open", "covered"),
    ),
)


def _lookup(thing: str, key: str, keys: tuple[str, ...], *extra: str) -> Lookup:
    """A lookup with phrasings that name the thing, plus any ``extra`` ones."""
    phrasings = (
        f"what's the {thing} for <k>",
        f"check the {thing} of <k>",
        f"{thing} for <k>",
        f"look up the {thing}[ for <k>]",
        *extra,
    )
    return Lookup(thing, key, keys, phrasings)


MORE_LOOKUPS = (
    _lookup("calorie count", "food", ("a banana", "a bagel", "an avocado", "greek yogurt")),
    _lookup("synonyms", "word", ("happy", "fast", "big", "smart"), "another word for <k>"),
    _lookup("rhymes", "word", ("orange", "cat", "light", "dream"), "what rhymes with <k>"),
    _lookup(
        "population", "country", ("Canada", "Kenya", "Japan", "Peru"), "how many people live in <k>"
    ),
    _lookup(
        "opening hours",
        "place",
        ("the library", "City Museum", "the post office", "Green Grocer"),
        "when does <k> open",
        "is <k> open now",
    ),
    _lookup(
        "wait time",
        "restaurant",
        ("Joe's Diner", "Sakura Sushi", "The Burger Barn"),
        "how long is the wait at <k>",
    ),
    _lookup(
        "menu", "restaurant", ("Luigi's", "Taco Town", "Golden Wok"), "what's on the menu at <k>"
    ),
    _lookup(
        "book availability",
        "title",
        ("Dune", "Beloved", "Circe", "Educated"),
        "is <k> available at the library",
    ),
    _lookup(
        "lyrics",
        "song",
        ("Yesterday", "Hey Jude", "Bohemian Rhapsody"),
        "show me the lyrics to <k>",
    ),
    _lookup(
        "plant care",
        "plant",
        ("monstera", "fiddle leaf fig", "snake plant", "aloe"),
        "how do I take care of a <k>",
    ),
    _lookup(
        "breed info", "breed", ("beagle", "corgi", "husky", "poodle"), "tell me about the <k> breed"
    ),
    _lookup(
        "standings",
        "league",
        ("Premier League", "NBA", "Bundesliga", "NHL"),
        "who is top of the <k>",
    ),
    _lookup(
        "player stats",
        "player",
        ("LeBron James", "Erling Haaland", "Serena Williams"),
        "how is <k> playing this season",
    ),
    _lookup("water level", "river", ("the Thames", "the Rhine", "the Danube")),
    _lookup(
        "sunset time", "city", ("Lisbon", "Reykjavik", "Denver", "Nairobi"), "when is sunset in <k>"
    ),
    _lookup(
        "moon phase",
        "date",
        ("tonight", "Friday", "next Saturday"),
        "what will the moon look like <k>",
    ),
    _lookup("earnings date", "symbol", ("KO", "JNJ", "PG", "PEP"), "when does <k> report earnings"),
    _lookup(
        "repository stars",
        "repository",
        ("react", "pytorch", "linux"),
        "how many stars does <k> have",
    ),
    _lookup("website status", "website", ("example.com", "github.com", "my blog"), "is <k> down"),
    _lookup(
        "domain availability",
        "domain",
        ("colibri.dev", "bestpizza.com", "myband.io"),
        "is <k> taken",
    ),
    _lookup("return policy", "store", ("IKEA", "Target", "Best Buy"), "can I return things to <k>"),
    _lookup(
        "product reviews",
        "product",
        ("the Kindle", "AirPods Pro", "the Instant Pot"),
        "are <k> any good",
    ),
    _lookup("price history", "product", ("the Kindle", "a PS5", "the iPad Air")),
    _lookup(
        "lunch menu",
        "school",
        ("Lincoln Elementary", "Oak Ridge High", "Maple Middle School"),
        "what's for lunch at <k>",
    ),
    _lookup(
        "recycling schedule",
        "street",
        ("Maple Street", "5th Avenue", "Elm Road"),
        "when is recycling picked up on <k>",
    ),
    _lookup(
        "power outage map",
        "neighborhood",
        ("Downtown", "Riverside", "Hillcrest"),
        "is there a power cut in <k>",
    ),
    _lookup("movie rating", "movie", ("Dune", "Barbie", "Up", "Arrival"), "is <k> any good"),
    _lookup(
        "public holidays", "country", ("Germany", "Japan", "Brazil"), "what are the holidays in <k>"
    ),
    _lookup(
        "salary range", "job", ("nurse", "data analyst", "electrician"), "how much does a <k> earn"
    ),
    _lookup("nutrition label", "product", ("oat milk", "peanut butter", "granola")),
    _lookup(
        "museum exhibits",
        "museum",
        ("the Louvre", "MoMA", "the Science Museum"),
        "what's on at <k>",
    ),
    _lookup(
        "trail conditions",
        "trail",
        ("Eagle Peak", "River Loop", "Cedar Ridge"),
        "is the <k> trail muddy",
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


def generated_specs(seed: int = 0, weight: float = 0.08) -> list[ToolSpec]:
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
    return specs
