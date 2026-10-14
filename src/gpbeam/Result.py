import re
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, Optional

from .config import GUINEA_PLUSPLUS, GUINEA_PIG2, normalize_program


@dataclass
class BeamResult:
    tracked_macroparticles: int
    energy_loss: float
    avg_photon_energy: float
    photons_per_particle: float

    def as_dict(self):
        return asdict(self)


@dataclass
class GeneralResults:
    lumi_fine: float
    lumi_ee: float
    lumi_ee_high: float
    lumi_pp: float
    lumi_eg: float
    lumi_ge: float
    lumi_gg: float
    lumi_gg_high: float
    upsmax: float
    E_cm: float
    E_cm_var: float

    def as_dict(self):
        return asdict(self)


@dataclass
class GuineaPigOutput:
    beam1: BeamResult
    beam2: BeamResult
    general: GeneralResults

    # additional scalar key=value style results
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


# -------------------------------------------------
# Parse GuineaPig output file
# -------------------------------------------------
def parse_guinea_output(file_path: str, program: Optional[str] = None) -> GuineaPigOutput:
    """
    Parse an output file of guinea-pig++ or guinea-pig2.

    program: "guinea-pig++" or "guinea-pig2" (aliases see config.normalize_program).
             If None, the format is detected from the file content.
    """
    with open(file_path, "r") as f:
        content = f.read()

    program = detect_program(content) if program is None else normalize_program(program)
    if program == GUINEA_PIG2:
        return _parse_guinea_pig2_output(content)

    # ---------------- guinea-pig++ ----------------
    # Remove everything after step table
    content = content.split("step   lumi total   lumi peak")[0]

    # -------------------------------------------------
    # Extract beam results (compact key=value section)
    # -------------------------------------------------
    kv_pairs = dict(
        # re.findall(r"([A-Za-z0-9_.]+)\s*=\s*([-+0-9.eE]+)", content)
        re.findall(r"([A-Za-z0-9_.-]+)\s*=\s*([-+0-9.eE]+)", content)
    )

    def get_float(key, default=0.0):
        return float(kv_pairs.get(key, default))

    def get_int_from_line(pattern):
        m = re.search(pattern, content)
        return int(m.group(1)) if m else 0

    def get_int_multiple_line(pattern):
        m = re.search(pattern, content, re.DOTALL)
        return int(m.group(1)) if m else 0

    # -------------------------------------------------
    # Beam 1
    # -------------------------------------------------
    beam1 = BeamResult(
        tracked_macroparticles=get_int_multiple_line(
            r"beam1.*?number of tracked macroparticles\s*:\s*(\d+)"
        ),
        energy_loss=get_float("de1"),
        avg_photon_energy=get_float("phot-e1"),
        photons_per_particle=get_float("n_phot1"),
    )

    # -------------------------------------------------
    # Beam 2
    # -------------------------------------------------
    beam2 = BeamResult(
        tracked_macroparticles=get_int_multiple_line(
            r"beam2.*?number of tracked macroparticles\s*:\s*(\d+)"
        ),
        energy_loss=get_float("de2"),
        avg_photon_energy=get_float("phot-e2"),
        photons_per_particle=get_float("n_phot2"),
    )

    # -------------------------------------------------
    # General results
    # -------------------------------------------------
    def extract_float(pattern):
        m = re.search(pattern, content)
        return float(m.group(1)) if m else 0.0

    general = GeneralResults(
        lumi_fine=extract_float(r"lumi_fine\s*=\s*([-+0-9.eE]+)"),
        lumi_ee=extract_float(r"lumi_ee\s*=\s*([-+0-9.eE]+)"),
        lumi_ee_high=extract_float(r"lumi_ee_high\s*=\s*([-+0-9.eE]+)"),
        lumi_pp=extract_float(r"lumi_pp\s*=\s*([-+0-9.eE]+)"),
        lumi_eg=extract_float(r"lumi_eg\s*=\s*([-+0-9.eE]+)"),
        lumi_ge=extract_float(r"lumi_ge\s*=\s*([-+0-9.eE]+)"),
        lumi_gg=extract_float(r"lumi_gg\s*=\s*([-+0-9.eE]+)"),
        lumi_gg_high=extract_float(r"lumi_gg_high\s*=\s*([-+0-9.eE]+)"),
        upsmax=extract_float(r"upsmax=\s*([-+0-9.eE]+)"),
        E_cm=extract_float(r"E_cm\s*=\s*([-+0-9.eE]+)"),
        E_cm_var=extract_float(r"E_cm_var\s*=\s*([-+0-9.eE]+)"),
    )

    # -------------------------------------------------
    # Extra numeric key=value pairs (angles, hadrons, etc.)
    # -------------------------------------------------
    extra = {
        k: float(v)
        for k, v in kv_pairs.items()
        if k not in {
            "phot-e1", "phot-e2",
            "n_phot1", "n_phot2",
            "de1", "de2",
            "lumi_ee", "lumi_ee_high"
        }
    }

    return GuineaPigOutput(
        beam1=beam1,
        beam2=beam2,
        general=general,
        extra=extra,
    )


# -------------------------------------------------
# Format detection
# -------------------------------------------------
def detect_program(content: str) -> str:
    """
    guinea-pig2 writes '{key=value;...}' blocks followed by '$HISTOGRAM::' sections,
    guinea-pig++ writes a human readable report with a '****' banner.
    """
    stripped = content.lstrip()
    if stripped.startswith("{") or "$HISTOGRAM::" in content:
        return GUINEA_PIG2
    return GUINEA_PLUSPLUS


# -------------------------------------------------
# Parse guinea-pig2 output file
# -------------------------------------------------
def _parse_guinea_pig2_output(content: str) -> GuineaPigOutput:
    """
    guinea-pig2 layout:
        {accelerator echo}  {parameter echo}  {results}
        phot-e-1 : <energy> <number>   (one line per stage, last = final)
        $HISTOGRAM:: ...               (ignored)
    """
    # Remove histograms (analogous to the step table of guinea-pig++)
    content = content.split("$HISTOGRAM::")[0]

    kv_pairs = dict(
        re.findall(r"([A-Za-z0-9_.-]+)\s*=\s*([-+0-9.eE]+)", content)
    )

    def get_float(key, default=0.0):
        return float(kv_pairs.get(key, default))

    # photon statistics: "phot-e-N : <avg photon energy> <photons per macroparticle>"
    phot = {}
    for beam, energy, number in re.findall(
        r"phot-e-(\d)\s*:\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)", content
    ):
        phot[beam] = (float(energy), float(number))  # keep last (= final) entry

    def beam_result(i):
        e_last, n_last = phot.get(str(i), (0.0, 0.0))
        return BeamResult(
            tracked_macroparticles=int(get_float(f"n_m.{i}")),
            energy_loss=get_float(f"de{i}"),
            avg_photon_energy=get_float(f"e_phot.{i}", e_last),
            photons_per_particle=n_last,
        )

    beam1 = beam_result(1)
    beam2 = beam_result(2)

    # key names of the results block coincide with the GeneralResults fields
    general = GeneralResults(**{f.name: get_float(f.name) for f in fields(GeneralResults)})

    extra = {
        k: float(v)
        for k, v in kv_pairs.items()
        if k not in {
            "e_phot.1", "e_phot.2",
            "de1", "de2",
            "lumi_ee", "lumi_ee_high"
        }
    }
    # guinea-pig++ reports a single "miss" value; provide the same key (worst of both beams)
    extra["miss"] = max(get_float("miss.1"), get_float("miss.2"))

    return GuineaPigOutput(
        beam1=beam1,
        beam2=beam2,
        general=general,
        extra=extra,
    )
