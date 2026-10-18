# Construction Pipeline

This directory contains the anonymized construction pipeline.
API keys are intentionally replaced with `xxxxx`; set the real keys locally before running.

Run order:

1. `sample_generation.py`
2. `DEC_Check.py`
3. `EAC_Check.py`
4. `MC_Check.py`
5. `evidence_localization.py`
6. `evidence_verfication.py`
7. `output.py`

Model choices are preserved from the original scripts:

- `sample_generation.py`: `qwen3.5-plus`
- Check and verification scripts: `gpt-5`

Stage outputs:

1. `01_sample_generation.json`
2. `02_DEC_Check.json`
3. `03_EAC_Check.json`
4. `04_MC_Check.json`
5. `05_evidence_localization.json`
6. `06_evidence_verification.json`
7. `07_output.json`
