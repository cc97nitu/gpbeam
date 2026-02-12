# Guinea-Pig Beam-Beam Simulation

Python wrapper for the [Guinea-Pig](https://gitlab.cern.ch/clic-software/guinea-pig) code simulating collisions of charged particle beams.

## Configuration

Before using this Python wrapper, you need to create a configuration file to specify the path to the Guinea-Pig executable and a working directory for intermediate files.

```
~/.config/guineapig/config.toml
```

Example configuration:
```
# Path to the working directory for intermediate files
WORKDIR = "/dev/shm/"

# Path to the Guinea-Pig executable
EXECUTABLE_GUINEA_PLUSPLUS = "/opt/SimulationCodes/guinea-pig/bin/guinea"
```
