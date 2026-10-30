# Where the data and artifacts live, relative to the repository root.
PARQUET_GLOB = "data/raw/fineweb-edu/*.parquet"
RAW_DIR = "data/raw"
TOOLS_DIR = "data/tools"
COMMANDS_FILE = "data/tokenizer/commands.txt"
GENERATION_DIR = "data/generation"
TINYTOOLS_DIR = "data/tinytools"  # word lists and domains for the TinyTools corpus
TINYTOOLS_RUN = f"{TINYTOOLS_DIR}/pilot"  # prompts, documents and checked output of one run
EXTERNAL_DIR = "data/external"
TASK_DIR = "data/task"
BIN_DIR = "data/bin"
TOKENIZER_FILE = "artifacts/tokenizer/eve-8k.json"
CONFIG_DIR = "configs"  # experiments/*.toml and the components' variant files
RUNS_DIR = "artifacts/runs"
PRETRAIN_RUN = f"{RUNS_DIR}/pretrain"  # where a run without --out writes
FINETUNE_RUN = f"{RUNS_DIR}/finetune"
PRETRAINED_MODEL = f"{RUNS_DIR}/base-900m/model.pt"  # the best ones so far
FINETUNED_MODEL = f"{RUNS_DIR}/joint-base/model.pt"

# Defaults shared by the command line and the library.
TOKENIZER_WEIGHT = 50  # how many times tool schemas and commands count in tokenizer training
MAX_NEW_TOKENS = 24  # most tokens fill_in writes
TEMPLATE_EXAMPLES = 80_000  # template examples in the train split
HELDOUT_EXAMPLES = 2_000  # template examples whose correct tool is held out
