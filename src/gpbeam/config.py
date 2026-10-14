import os
import tomllib

from platformdirs import user_config_dir
from pathlib import Path

config_dir = Path(user_config_dir())
config_file = config_dir / "guinea"

def load_config():
    if not config_file.exists():
        return {"WORKDIR": "/dev/shm/", "EXECUTABLE_GUINEA_PLUSPLUS": "/opt/SimulationCodes/guinea-pig/bin/guinea"}
    return tomllib.loads(config_file.read_text())

config = load_config()

# key runtime variables
try:
    WORKDIR = os.path.normpath(config["WORKDIR"])
except KeyError:
    print("invalid configuration for WORKDIR")

if not os.path.isdir(WORKDIR):
    raise IOError("not a valid working directory: " + WORKDIR)

def EXECUTABLE_GUINEA_PLUSPLUS():
    try:
        return config["EXECUTABLE_GUINEA_PLUSPLUS"]
    except KeyError:
        raise ValueError("invalid configuration for EXECUTABLE_GUINEA_PLUSPLUS")

def EXECUTABLE_GUINEA_PIG2():
    try:
        return config["EXECUTABLE_GUINEA_PIG2"]
    except KeyError:
        raise ValueError("invalid configuration for EXECUTABLE_GUINEA_PIG2 "
                         "(add it to " + str(config_file) + " to use guinea-pig2)")


##########################
### program selection
##########################
GUINEA_PLUSPLUS = "guinea-pig++"
GUINEA_PIG2 = "guinea-pig2"

_PROGRAM_ALIASES = {
    GUINEA_PLUSPLUS: GUINEA_PLUSPLUS, "guineapig++": GUINEA_PLUSPLUS, "gp++": GUINEA_PLUSPLUS, "g++": GUINEA_PLUSPLUS,
    GUINEA_PIG2: GUINEA_PIG2, "guineapig2": GUINEA_PIG2, "gp2": GUINEA_PIG2, "g2": GUINEA_PIG2,
}

def normalize_program(program: str) -> str:
    """Map a user supplied program name (e.g. "g++", "guinea-pig2") to its canonical name."""
    try:
        return _PROGRAM_ALIASES[str(program).strip().lower()]
    except KeyError:
        raise ValueError(f"unknown program '{program}', choose '{GUINEA_PLUSPLUS}' or '{GUINEA_PIG2}'")

def executable(program: str) -> str:
    """Path to the executable of the selected program."""
    if normalize_program(program) == GUINEA_PIG2:
        return EXECUTABLE_GUINEA_PIG2()
    return EXECUTABLE_GUINEA_PLUSPLUS()


##########################
### debug
##########################
if __name__ == "__main__":
    print("config directory:", config_dir)
    print("config file:", config_file)
    print("\nconfig file parsed:")
    print(config)
    print()
    print("WORKDIR: ", WORKDIR)
    print("EXECUTABLE_GUINEA_PLUSPLUS: ", EXECUTABLE_GUINEA_PLUSPLUS())
    print("EXECUTABLE_GUINEA_PIG2: ", config.get("EXECUTABLE_GUINEA_PIG2", "<not configured>"))
