# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Grade all three published graphs using an independent ordered task checklist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Operation, subject and destination, in the order requested in the pinned instructions.
# @names bind identity on first occurrence. Subsequent occurrences must match.
CASES = {
    1171: "align can;inside can @tray;inside bottle @tray;on cup coaster",
    1172: "open @drawer;land apple @plate;land orange @plate;closed @drawer",
    1173: "on cup @pad;pour pot cup;restore pot;press bell",
    1174: "on @cup @pad;press button;inside @cup tray",
    1175: "open @drawer;vertical can;inside can @drawer;align bottle;inside bottle @drawer;closed @drawer",
    1176: "open @drawer;inside block @drawer;inside cube @drawer;closed @drawer;press button",
    1177: "coarse knob;open @tray;on bowl mat;closed @tray",
    1178: "on cup @coaster;on bottle @bottlemat;carry @tray @storage",
    1179: "inside bowl @tray;inside cup @tray;pour kettle bowl;restore kettle",
    1180: "on cup coaster;open @drawer;inside spoon @drawer;closed @drawer",
    1181: "on cup @serving;right_left bottle;on bottle @bottlemat;right_left spoon;on spoon @utensilmat",
    1182: "pour @measuringcup @bowl;restore @measuringcup;inside @bowl tray;on @measuringcup coaster",
    1183: "on bowl @pad;on spoon @pad;coarse knob;press button",
    1184: "press button;coarse knob;inside bottle @tray;inside cup @tray",
    1185: "open @tray;inside @bowl @serving;right_of cup @bowl;closed @tray",
    1186: "right_left apple;land apple @plate;right_left orange;land orange @plate;carry @plate @serving",
    1187: "on @cup coaster;pour kettle @cup;restore kettle;inside @cup tray",
    1188: "on @cup @prep;pour @bottle @cup;restore @bottle;on @bottle @tray;left_right @cup;on @cup @serving",
    1189: "align @cup;on @cup @pad;press button;inside @cup tray",
    1190: "inside block tray;open @drawer;left_right bottle;inside bottle @drawer;closed @drawer",
}
ALIASES = {
    "can": ("can", "罐"),
    "cup": ("cup", "杯"),
    "measuringcup": ("cup", "杯"),
    "bottle": ("bottle", "瓶"),
    "tray": ("tray", "托盘", "盘"),
    "drawer": ("drawer", "抽屉"),
    "plate": ("plate", "dish", "bowl", "tray", "盘"),
    "serving": ("serv", "mat", "垫", "托盘"),
    "bottlemat": ("bottle", "瓶"),
    "utensilmat": ("utensil", "spoon", "cutlery", "餐具"),
    "storage": ("stor", "tray", "收纳"),
    "coaster": ("coaster", "mouse", "pad", "mat", "垫"),
    "pot": ("pot", "壶"),
    "kettle": ("kettle", "pot", "壶"),
    "button": ("button", "按钮"),
    "bell": ("bell", "铃"),
    "block": ("block", "木块"),
    "cube": ("cube", "魔方"),
    "bowl": ("bowl", "碗"),
    "apple": ("apple", "苹果"),
    "orange": ("orange", "橙"),
    "fork": ("fork", "叉"),
    "spoon": ("spoon", "勺"),
    "knob": ("knob", "dial", "旋钮"),
    "mat": ("mat", "pad", "垫"),
    "pad": ("mat", "pad", "垫"),
    "prep": ("prep", "mat", "备料"),
}


def events(folder, route):
    if route == "template":
        data = json.loads((folder / "template.json").read_text())
        graph = data["semantic_task_graph"]
        annotations = data["semantic_graph_validation"]["source"]["annotations"]
        bindings = {}
    else:
        graph = json.loads((folder / route / "semantic_task_graph.json").read_text())
        annotations = {}
        path = folder / route / "pending_operation_bindings.json"
        bindings = json.loads(path.read_text()) if path.exists() else {}
    out = []
    for n in graph["nodes"]:
        c = n["call"]
        cid = c.get("call_id", c["kind"])
        args = c.get("arguments", c)
        res = c["resources"]
        ann = annotations.get(n["id"], {})
        b = bindings.get(n["id"], {})
        from embodichain.gen_sim.task_spec import canonical_hash

        if b.get("call") != canonical_hash(c):
            b = {}
        obj = ann.get("object") or args.get("object") or args.get("handle", "")
        dest = ann.get("reference") or args.get("reference") or args.get("target", "")
        intent = ann.get("intent", {})
        typ = cid
        if cid == "simulation.axis_align":
            typ = (
                "vertical"
                if b.get("orientation", intent.get("orientation_goal")) == "vertical"
                else "align"
            )
        elif cid == "simulation.articulation_link_slide":
            typ = b.get("state", intent.get("target_state", "unbound"))
        elif cid == "simulation.twist":
            typ = b.get("mode", intent.get("twist_goal_mode", "unbound"))
        elif cid == "simulation.press":
            typ = "press"
        elif cid == "simulation.pour":
            typ = "pour"
            dest = b.get("reference", dest)
        elif cid == "hand_over":
            typ = res.get("source", "") + "_" + res.get("destination", "")
        elif cid == "place":
            typ = next((r for r in ("on", "inside", "at") if args.get(r)), "unknown")
            dest = dest or args.get(typ, "")
            if b.get("kind") == "return_to_origin" or intent.get("return_to_origin"):
                typ = "restore"
        elif cid == "simulation.place_relative":
            typ = args["relation"]
        elif cid == "simulation.coordinated_transport" and res == {
            "left": "left",
            "right": "right",
        }:
            typ = "carry"
        out.append((typ, str(obj), str(dest)))
        if cid == "hand_over":
            # A named receiver must retain this object until releasing it.
            if not any(
                next_node["call"].get("resources", {}).get("primary")
                == res.get("destination")
                for next_node in graph["nodes"][graph["nodes"].index(n) + 1 :]
                if next_node["call"].get("kind") == "place"
                and next_node["call"].get("object") == args["object"]
            ):
                out.append(("wrong_receiver", str(obj), ""))
    return out


def audit(case, stream, *, checklist=None):
    identities = {}
    errors = []
    idx = 0
    matched = set()
    for term in (CASES[case] if checklist is None else checklist).split(";"):
        op, *args = term.split()
        for i in range(idx, len(stream)):
            kind, *actual = stream[i]
            if not (kind == op or op == "land" and kind in {"on", "inside"}):
                continue
            proposed = dict(identities)
            for key, val in zip(args, actual):
                if not any(a in val.lower() for a in ALIASES[key.lstrip("@")]):
                    break
                if key.startswith("@") and (
                    proposed.setdefault(key, val) != val
                    or list(proposed.values()).count(val) > 1
                ):
                    break
            else:
                identities = proposed
                matched.add(i)
                idx = i + 1
                break
        else:
            errors.append(term)
    for i, (kind, obj, _) in enumerate(stream):
        if i not in matched and kind in {
            "open",
            "closed",
            "coarse",
            "setting",
            "press",
            "pour",
            "carry",
            "left_right",
            "right_left",
            "wrong_receiver",
        }:
            errors.append(f"Unexpected operation: {kind} {obj}")
    return errors


def quoted_obligations(folder, route, case):
    """Match independent annotated instruction fragments to actual producing calls."""
    oracle = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "tests/gen_sim/task_engine/data/image_task_expectations.json"
        ).read_text()
    )["cases"][f"task{case}"]
    quotes = []
    if route == "template":
        result = json.loads((folder / "template.json").read_text())
        graph = result["semantic_task_graph"]
        quotes = [
            (r["quote"], r["node_ids"])
            for r in result["semantic_graph_validation"]["source"]["requirements"]
        ]
    else:

        def read(k):
            return json.loads((folder / route / (k + ".json")).read_text())

        graph = read("semantic_task_graph")
        producers = read("coverage")["producers"]
        for c in read("task_spec_v1")["interpretation"]["clauses"]:
            n = producers.get(c["target"])
            if n:
                quotes.append((c["quote"], [n[0] if isinstance(n, list) else n]))
    nodes = {
        n["id"]: n["call"].get("call_id", n["call"]["kind"]) for n in graph["nodes"]
    }
    instruction = graph["instruction"]
    missing = []
    for chk in oracle:
        fragments = chk["quote_fragments"]
        for fragment in fragments:
            # A short noun may occur earlier in a disambiguation clause.
            # Any actual occurrence of the independently annotated fragment
            # may witness its correctly typed producer.
            starts = [
                i
                for i in range(len(instruction))
                if instruction.startswith(fragment, i)
            ]
            if not any(
                nodes.get(node) in chk["call_alternatives"]
                for q, ids in quotes
                for start in starts
                if min(instruction.index(q) + len(q), start + len(fragment))
                > max(instruction.index(q), start)
                for node in ids
            ):
                missing.append(chk["label"] + " / " + fragment)
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("controls", type=Path)
    parser.add_argument("hybrid", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    rows = []
    for case in CASES:
        inputs = set()
        for route in ("template", "goal", "hybrid"):
            root = (a.hybrid if route == "hybrid" else a.controls) / f"task{case}"
            r = json.loads((root / (route + "-trial.json")).read_text())
            if (
                r["case"] != f"task{case}"
                or r["route"] != route
                or len(r["calls"]) > r["limit"]
            ):
                raise ValueError("Mismatched case, route or call budget")
            inputs.add(
                (
                    r["instruction"],
                    r["image_sha256"],
                    tuple(sorted(r["model"].items())),
                    r["limit"],
                )
            )
            errors = (
                audit(case, events(root, route)) + quoted_obligations(root, route, case)
                if r["status"] in {"covered", "uncovered"}
                else []
            )
            rows.append(
                dict(
                    case=case,
                    route=route,
                    calls=len(r["calls"]),
                    status=r["status"],
                    error=r.get("reason"),
                    missing=errors,
                    passed=r["status"] == "covered" and not errors,
                    model=r["model"],
                    image_sha256=r["image_sha256"],
                    budget=r["limit"],
                )
            )
        if len(inputs) != 1:
            raise ValueError(f"Unpaired trial inputs: {case}")
    a.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
