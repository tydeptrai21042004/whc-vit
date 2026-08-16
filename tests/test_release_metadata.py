from __future__ import annotations
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def test_release_metadata_and_environment_are_version_aligned():
    version=(ROOT/'VERSION').read_text().strip()
    cff=yaml.safe_load((ROOT/'CITATION.cff').read_text())
    codemeta=json.loads((ROOT/'codemeta.json').read_text())
    zenodo=json.loads((ROOT/'.zenodo.json').read_text())
    env=yaml.safe_load((ROOT/'environment.yml').read_text())
    assert cff['version']==codemeta['version']==zenodo['version']==version
    assert cff['repository-code']=='https://github.com/tydeptrai21042004/whc-vit'
    assert env['name'].endswith(version)
    for rel in ['proposal_spec.json','proposal_contract.py','proposal_fingerprint.py','check_environment.py','validate_release.sh']:
        assert (ROOT/rel).is_file(), rel
