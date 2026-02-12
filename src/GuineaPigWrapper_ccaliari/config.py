import os
import tomllib

from platformdirs import user_config_dir
from pathlib import Path

config_dir = Path(user_config_dir("guineapig"))
config_file = config_dir / "config.toml"

def load_config():
    if not config_file.exists():
        return {"WORKDIR": "/dev/shm/", "EXECUTABLE_GUINEA_PLUSPLUS": "/opt/SimulationCodes/guinea-pig/bin/guinea", "EXECUTABLE_GUINEA_LEGACY": "guinea"}
    return tomllib.loads(config_file.read_text())

config = load_config()

# key runtime variables
try:
    WORKDIR = os.path.dirname(config["WORKDIR"])
except KeyError:
    print("invalid configuration for WORKDIR")

if not os.path.isdir(WORKDIR):
    raise IOError("not a valid working directory: " + WORKDIR)

def EXECUTABLE_GUINEA_PLUSPLUS():
    try:
        return config["EXECUTABLE_GUINEA_PLUSPLUS"]
    except KeyError:
        raise ValueError("invalid configuration for EXECUTABLE_GUINEA_PLUSPLUS")

def EXECUTABLE_GUINEA_LEGACY():
    try:
        return config["EXECUTABLE_GUINEA_LEGACY"]
    except KeyError:
        raise ValueError("invalid configuration for EXECUTABLE_GUINEA_LEGACY")


##########################
### debug
##########################
if __name__ == "__main__":
    print("WORKDIR: ", WORKDIR)
    print("EXECUTABLE_GUINEA_PLUSPLUS: ", EXECUTABLE_GUINEA_PLUSPLUS())
    print("EXECUTABLE_GUINEA_LEGACY: ", EXECUTABLE_GUINEA_LEGACY())