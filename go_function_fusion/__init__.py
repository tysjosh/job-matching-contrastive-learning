"""Inference-time GO-structure fusion for protein-function prediction (S1, S2 add-on).

Preregistration: ``.agents/tasks/ontology-withholding-feasibility/protocol_go_function_fusion.md``.
Protected packages (``go_function``, ``go_ppi_fusion``, ``trials_domain``) are imported,
never edited.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_REL = ".agents/tasks/ontology-withholding-feasibility/protocol_go_function_fusion.md"
RESULTS_REL = "results/go_function_fusion"
DEV_RESULTS_REL = "results/go_function_fusion_dev"
CONFIG_REL = "config/go_function_fusion"
