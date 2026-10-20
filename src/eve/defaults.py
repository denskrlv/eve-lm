# Where the data and artifacts live, relative to the repository root.
PARQUET_GLOB = "data/raw/fineweb-edu/*.parquet"
RAW_DIR = "data/raw"
TOOLS_DIR = "data/tools"
COMMANDS_FILE = "data/tokenizer/commands.txt"
GENERATION_DIR = "data/generation"
EXTERNAL_DIR = "data/external"
TASK_DIR = "data/task"
BIN_DIR = "data/bin"
TOKENIZER_FILE = "artifacts/tokenizer/eve-8k.json"
PRETRAIN_RUN = "artifacts/runs/pretrain-small"
PRETRAINED_MODEL = f"{PRETRAIN_RUN}/model.pt"
FINETUNE_RUN = "artifacts/runs/questions"
FINETUNED_MODEL = "artifacts/runs/questions-full/model.pt"

# Defaults shared by the command line and the library.
TOKENIZER_WEIGHT = 50  # how many times tool schemas and commands count in tokenizer training
MAX_NEW_TOKENS = 24  # most tokens fill_in writes
TEMPLATE_EXAMPLES = 80_000  # template examples in the train split
HELDOUT_EXAMPLES = 2_000  # template examples whose correct tool is held out
