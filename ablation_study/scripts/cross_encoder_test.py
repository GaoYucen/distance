"""Audit R1 entry point. Historical implementation remains at commit c4fb479.
Pass --data data/audit/protocol_r1/splits.npz and --family cross.
Validation selects checkpoints; test is evaluated only after restoring that checkpoint.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.audit_cross_encoder import CrossEncoder

if __name__ == '__main__':
    from scripts.audit_pilot import main
    main(default_family='cross')
