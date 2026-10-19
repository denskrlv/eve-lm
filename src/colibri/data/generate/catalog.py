from dataclasses import dataclass, field

from colibri.data.schema import Arg

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


@dataclass(frozen=True)
class Operation:
    """A tool that does something to values instead of to a named object: "roll two dice",
    "how do you say thank you in French". Many requests use other words than the tool's name
    and description, so the model can't just match a noun. Several have 2-3 required
    arguments, each in an optional template segment: ASK examples then give one value and
    leave out another ("reserve a table at Luigi's for 4", no time).
    """

    names: tuple[str, ...]
    descriptions: tuple[str, ...]
    args: tuple[Arg, ...]
    templates: tuple[str, ...]
    values: dict[str, tuple[str, ...]] = field(default_factory=dict)


WORDS = ("necessary", "banana", "listen", "colibri", "rhythm", "pineapple", "garden", "silent")
DATES = ("May 3", "December 25", "July 4", "March 15", "October 31", "February 29")
RESTAURANTS = ("Luigi's", "Sakura Sushi", "Golden Wok", "The Burger Barn", "Casa Maria")
STORES = ("Target", "Best Buy", "IKEA", "Walmart", "the corner shop")

OPERATIONS = (
    Operation(
        ("roll_dice", "dice_roll", "throw_dice"),
        ("Roll dice", "Throw one or more dice"),
        (Arg("count", "int", min=1, max=10), Arg("sides", "int", required=False, min=4, max=20)),
        (
            "roll {count} dice",
            "throw {count} dice[ with {sides} sides]",
            "roll[ {count}] dice for me",
            "I need {count} random dice rolls",
        ),
    ),
    Operation(
        ("flip_coin", "coin_toss", "toss_coin"),
        ("Flip a coin", "Toss a coin and say heads or tails"),
        (Arg("times", "int", required=False, min=1, max=10),),
        ("flip a coin[ {times} times]", "heads or tails", "toss a coin[ {times} times]"),
    ),
    Operation(
        ("random_number", "pick_number", "generate_random_number"),
        ("Pick a random whole number in a range", "Generate a random number"),
        (Arg("low", "int", min=0, max=50), Arg("high", "int", min=51, max=1000)),
        (
            "pick a number between {low} and {high}",
            "give me a random number from {low} to {high}",
            "choose a number[ from {low}][ up to {high}]",
            "random number[ between {low} and {high}]",
        ),
    ),
    Operation(
        ("days_until", "countdown_to", "time_left_until"),
        ("Count the days until a date or holiday", "How long until an occasion"),
        (Arg("occasion", "str"),),
        (
            "how many days until {occasion}",
            "how long is it until {occasion}",
            "count down to[ {occasion}]",
            "how many sleeps until {occasion}",
        ),
        {
            "occasion": (
                "Christmas",
                "Halloween",
                "New Year's Day",
                "my birthday",
                "Valentine's Day",
                "the summer holidays",
            )
        },
    ),
    Operation(
        ("day_of_week", "weekday_for_date", "which_weekday"),
        ("Tell which day of the week a date falls on", "Find the weekday of a date"),
        (Arg("date", "str"),),
        (
            "what day of the week is {date}",
            "which day is {date} this year",
            "is {date} a weekend",
            "what weekday does[ {date}] fall on",
        ),
        {"date": DATES},
    ),
    Operation(
        ("reverse_text", "reverse_word", "mirror_text"),
        ("Write a text backwards", "Reverse the letters of a word"),
        (Arg("text", "str"),),
        ("what is {text} backwards", "write {text} back to front", "reverse[ the word {text}]"),
        {"text": WORDS},
    ),
    Operation(
        ("count_letters", "letter_count", "word_length"),
        ("Count the letters in a word", "Tell how long a word is"),
        (Arg("word", "str"),),
        (
            "how many letters are in {word}",
            "how long is the word {word}",
            "count the letters[ in {word}]",
        ),
        {"word": WORDS},
    ),
    Operation(
        ("find_anagrams", "anagram_finder", "make_anagram"),
        ("Find anagrams of a word", "Rearrange a word's letters into other words"),
        (Arg("word", "str"),),
        (
            "find an anagram of {word}",
            "what words can I make from the letters in {word}",
            "scramble[ {word}] into another word",
        ),
        {"word": WORDS},
    ),
    Operation(
        ("to_morse_code", "morse_encode", "write_in_morse"),
        ("Write a text in Morse code", "Encode text as dots and dashes"),
        (Arg("text", "str"),),
        (
            "how do you write {text} in Morse code",
            "{text} in dots and dashes",
            "tap out[ {text}] in Morse",
        ),
        {"text": ("SOS", "hello", "help me", "good night", "yes")},
    ),
    Operation(
        ("roman_numerals", "to_roman", "write_roman_numeral"),
        ("Write a number as Roman numerals", "Give the Roman numeral for a number"),
        (Arg("number", "int", min=1, max=3999),),
        (
            "what is {number} in Roman numerals",
            "how do the Romans write {number}",
            "write[ {number}] as a Roman numeral",
        ),
    ),
    Operation(
        ("is_leap_year", "check_leap_year", "leap_year"),
        ("Tell if a year is a leap year", "Check whether a year has a February 29"),
        (Arg("year", "int", min=1900, max=2100),),
        ("is {year} a leap year", "does {year} have a February 29", "was[ {year}] a leap year"),
    ),
    Operation(
        ("zodiac_sign", "star_sign_for", "get_star_sign"),
        ("Give the star sign for a birthday", "Find the zodiac sign of a date"),
        (Arg("birthday", "str"),),
        (
            "what star sign is someone born on {birthday}",
            "if my birthday is {birthday} what's my sign",
            "what's my star sign[ if I was born on {birthday}]",
        ),
        {"birthday": DATES},
    ),
    Operation(
        ("shorten_link", "short_url", "make_short_link"),
        ("Make a long link shorter", "Create a short URL"),
        (Arg("url", "str"),),
        ("shorten {url}", "make {url} shorter", "give me a short link[ for {url}]"),
        {
            "url": (
                "example.com/very/long/path",
                "github.com/denskrlv/colibri-lm",
                "shop.io/item/42",
            )
        },
    ),
    Operation(
        ("make_qr_code", "qr_code", "create_qr"),
        ("Create a QR code", "Make a scannable code for a text or link"),
        (Arg("content", "str"),),
        (
            "make a QR code for {content}",
            "I want a code people can scan for {content}",
            "generate a QR code[ with {content}]",
        ),
        {"content": ("example.com", "my wifi password", "my contact card", "the party invite")},
    ),
    Operation(
        ("generate_password", "new_password", "strong_password"),
        ("Generate a random password", "Create a strong password"),
        (Arg("length", "int", min=8, max=32), Arg("symbols", "bool", required=False)),
        (
            "make me a {length} character password",
            "I need a new password[ that is {length} characters long]",
            "come up with a {length} letter password[ {symbols:with symbols|without symbols}]",
        ),
    ),
    Operation(
        ("translate_text", "translate", "say_in_language"),
        ("Translate a phrase into another language", "Translate text"),
        (
            Arg("phrase", "str"),
            Arg("language", "enum", options=("Spanish", "French", "German", "Italian", "Japanese")),
        ),
        (
            "how do you say {phrase} in {language}",
            "translate {phrase} to {language}",
            "what's {phrase} in {language}",
            "translate[ {phrase}][ into {language}]",
        ),
        {
            "phrase": (
                "good morning",
                "thank you very much",
                "where is the station",
                "I love you",
                "see you tomorrow",
            )
        },
    ),
    Operation(
        ("find_emoji", "emoji_for", "suggest_emoji"),
        ("Find an emoji for a word", "Suggest a matching emoji"),
        (Arg("word", "str"),),
        ("what emoji means {word}", "give me an emoji for {word}", "which emoji[ fits {word}]"),
        {"word": ("pizza", "celebration", "sad", "rocket", "love", "sleepy")},
    ),
    Operation(
        ("reserve_table", "book_table", "make_reservation"),
        ("Reserve a table at a restaurant", "Book a restaurant table"),
        (
            Arg("restaurant", "str"),
            Arg("people", "int", min=1, max=12),
            Arg("time", "str"),
        ),
        (
            "reserve a table at {restaurant} for {people} at {time}",
            "get us a table for {people} at {restaurant} at {time}",
            "book {restaurant}[ for {people} people][ at {time}]",
            "can we eat at {restaurant} tonight[ at {time}], we are {people}",
        ),
        {"restaurant": RESTAURANTS, "time": ("7 pm", "8:30", "noon", "half past six")},
    ),
    Operation(
        ("order_food", "order_delivery", "get_takeout"),
        ("Order food from a restaurant", "Order a dish for delivery"),
        (Arg("dish", "str"), Arg("restaurant", "str")),
        (
            "order {dish} from {restaurant}",
            "I'm hungry, get me {dish} from {restaurant}",
            "get[ {dish}] delivered[ from {restaurant}]",
        ),
        {
            "dish": ("pad thai", "a margherita pizza", "two burritos", "sushi rolls", "dumplings"),
            "restaurant": RESTAURANTS,
        },
    ),
    Operation(
        ("check_stock", "in_stock", "store_availability"),
        ("Check if a store has a product in stock", "See whether a shop has an item"),
        (Arg("product", "str"), Arg("store", "str")),
        (
            "does {store} have {product} in stock",
            "can I get {product} at {store} today",
            "is[ {product}] available[ at {store}]",
        ),
        {
            "product": ("AirPods", "a yoga mat", "printer ink", "running shoes", "a desk lamp"),
            "store": STORES,
        },
    ),
    Operation(
        ("rate_place", "leave_rating", "review_place"),
        ("Give a place a star rating", "Rate a restaurant or shop"),
        (Arg("place", "str"), Arg("stars", "int", min=1, max=5)),
        (
            "give {place} {stars} stars",
            "rate {place} {stars} out of 5",
            "{place} deserves {stars} stars",
            "rate[ {place}][ {stars} stars]",
        ),
        {"place": RESTAURANTS + STORES},
    ),
    Operation(
        ("fun_fact", "random_fact", "tell_fact"),
        ("Tell a fun fact", "Share an interesting fact"),
        (
            Arg(
                "topic",
                "enum",
                required=False,
                options=("space", "animals", "history", "the ocean", "food"),
            ),
        ),
        ("tell me a fun fact[ about {topic}]", "I want to learn something new[ about {topic}]"),
    ),
)
