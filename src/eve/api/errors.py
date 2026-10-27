class EveError(Exception):
    """Something the library can't do with what it was given."""


class ToolError(EveError, ValueError):
    """A tool Eve can't read: an unsupported argument type, a missing description, too many
    tools for one request."""


class CheckpointError(EveError, ValueError):
    """A model file that doesn't exist, can't be downloaded, or isn't a fine-tuned Eve."""
