"""Study constants (protocol_go_function.md; design_go_function.md §6.4, §13.2, §15)."""
from __future__ import annotations

# ---------------------------------------------------------------- analysis (§13.2)
FLOOR = 0.01
ALPHA = 0.05
UNSTABLE_SHARE = 0.05
ANALYSIS_SEED = 20261011

# ---------------------------------------------------------------- withholding (§4.10)
RETENTION_SEED = 20261011
RETENTION_SEED_2 = 20261012
SEEDS = (13, 21, 42, 87, 123)
KS = (0, 1, 5, 10, 25, "all")

# ---------------------------------------------------------------- GO
ASPECTS = ("bp", "mf")
NAMESPACES = {"bp": "biological_process", "mf": "molecular_function",
              "cc": "cellular_component"}
ROOTS = {"bp": "GO:0008150", "mf": "GO:0003674", "cc": "GO:0005575"}
PROTEIN_BINDING = "GO:0005515"
BINDING = "GO:0005488"

#: DGZ utils.EXP_CODES (deepgozero/utils.py).
EXP_CODES = frozenset(["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "TAS", "IC",
                       "HTP", "HDA", "HMP", "HGI", "HEP"])
EXP_CODES_NO_IPI = EXP_CODES - {"IPI"}

#: go.norm relation IRIs (after DGZ's replace('_', ':')) -> OBO relationship names.
RELATION_IRIS = {
    "BFO:0000050": "part_of",
    "BFO:0000051": "has_part",
    "BFO:0000066": "occurs_in",
    "RO:0002211": "regulates",
    "RO:0002212": "negatively_regulates",
    "RO:0002213": "positively_regulates",
    "RO:0002092": "happens_during",
    "RO:0002093": "ends_during",
}
OWL_NOTHING = "owl:Nothing"

# ---------------------------------------------------------------- design vocabulary
ARMS = ("V0", "V1", "V2", "V3", "V3c", "V4", "V5", "V6")
RULES = ("W2", "W1", "WDGZrel", "W2drop")
REGIMES = ("TP", "DGZ", "TPnoIPI")

# ---------------------------------------------------------------- exit codes (§15)
EXIT_OK = 0
EXIT_FATAL = 2
EXIT_STOP = 3
EXIT_REFUSED = 4

#: Pin keys (protocol §13).
PIN_KEYS = ("G1-labels", "G1-norm", "G2", "G2b", "G3", "G4", "G5", "G6", "G7", "G8")
