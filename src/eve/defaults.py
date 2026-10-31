# Where the data and artifacts live, relative to the repository root.
PARQUET_GLOB = "data/raw/fineweb-edu/*.parquet"
RAW_DIR = "data/raw"
TOOLS_DIR = "data/tools"
COMMANDS_FILE = "data/tokenizer/commands.txt"
GENERATION_DIR = "data/generation"
EXTERNAL_DIR = "data/external"
TASK_DIR = "data/task"  # what fine-tuning trains on: train.jsonl
EVALS_DIR = "data/evals"  # every set a model is scored on, never trained on
NOVEL_TOOLS_DIR = "data/novel-tools"  # the novel-tools benchmark's tools and requests
BIN_DIR = "data/bin"
TINYTOOLS_DIR = "data/tinytools"  # one parquet file per round-1 source
VOCAB_DIR = "data/vocab"  # everyday nouns and first names
TOKENIZER_FILE = "artifacts/tokenizer/eve-8k.json"
CONFIG_DIR = "configs"  # experiments/*.toml and the components' variant files
RUNS_DIR = "artifacts/runs"
PRETRAIN_RUN = f"{RUNS_DIR}/pretrain"  # where a run without --out writes
FINETUNE_RUN = f"{RUNS_DIR}/finetune"
PRETRAINED_MODEL = f"{RUNS_DIR}/eve-32m-pretrain/model.pt"
FINETUNED_MODEL = f"{RUNS_DIR}/eve-32m/model.pt"

# Defaults shared by the command line and the library.
TOKENIZER_WEIGHT = 50  # how many times tool schemas and commands count in tokenizer training
MAX_NEW_TOKENS = 24  # most tokens fill_in writes
TEMPLATE_EXAMPLES = 80_000  # template examples in the train split
HELDOUT_EXAMPLES = 2_000  # template examples whose correct tool is held out
