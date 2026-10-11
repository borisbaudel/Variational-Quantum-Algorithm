"""
merge_ws_log.py -- fold any system sizes still missing from
results/model1_warmstart.json in from the run log.

run_model1_warmstart.py writes its JSON once per L, after the whole layer
ladder for that L has finished.  The L = 16 ladder is by far the most expensive
computation in this project, so if it is cut short at some P the completed
rungs are in the log but not yet in the JSON.  This script parses those lines
back out, so that no measured point is thrown away and no point is invented:
every number it writes was printed by the run itself.
"""

import json
import re
import sys

LOG = "results/model1_ws.log"
JSON = "results/model1_warmstart.json"

head = re.compile(r"^=== L=(\d+)\s+E0\(ED\) = ([-+0-9.eE]+)")
rung = re.compile(r"^\s+P=(\d+)\s+E=([-+0-9.eE]+)\s+dE=([-+0-9.eE]+)")


def main():
    with open(JSON) as f:
        data = json.load(f)

    parsed, L, E0 = {}, None, None
    for line in open(LOG):
        m = head.match(line)
        if m:
            L, E0 = int(m.group(1)), float(m.group(2))
            parsed[L] = {"E0": E0, "ladder": {}}
            continue
        m = rung.match(line)
        if m and L is not None:
            P, E = int(m.group(1)), float(m.group(2))
            parsed[L]["ladder"][str(P)] = {
                "E": E, "abs_err": E - E0, "rel_err": abs((E - E0) / E0),
                "n_params": 2 * P, "source": "run log",
            }

    added = []
    for L, rec in parsed.items():
        if str(L) not in data["systems"] and rec["ladder"]:
            data["systems"][str(L)] = rec
            added.append((L, max(map(int, rec["ladder"]))))

    with open(JSON, "w") as f:
        json.dump(data, f, indent=2)
    if added:
        for L, Pmax in added:
            print(f"  merged L={L} from the log (ladder complete up to P={Pmax})")
    else:
        print("  nothing to merge; the JSON is already complete")


if __name__ == "__main__":
    sys.exit(main())
