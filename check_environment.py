#!/usr/bin/env python3
from __future__ import annotations
import json, platform, sys
from importlib.metadata import PackageNotFoundError, version

REQUIRED = ["torch","torchvision","numpy","scipy","scikit-learn","pandas","Pillow","PyYAML","timm","fvcore","iopath","yacs","simplejson","termcolor","tabulate","tqdm","ml-collections"]
report={"python":sys.version,"platform":platform.platform(),"packages":{},"missing":[]}
for name in REQUIRED:
    try: report["packages"][name]=version(name)
    except PackageNotFoundError: report["missing"].append(name)
print(json.dumps(report,indent=2))
raise SystemExit(1 if report["missing"] else 0)
