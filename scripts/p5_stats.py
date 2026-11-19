
pass

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e5_gen_pools import load_classes, DEFAULT_N
from e5_probe import load_test, encode_pool, mean_protos, _norm
from p4_b1_feas import CUBL_ALIAS, paths_for

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, "results")
DATASETS = ("aircraft", "euro_sat", "dtd", "cub")
ARMS = ("onto", "magent", "omni")




def audit_one(ds):
    P = paths_for(ds)
    classes = load_classes(ds)
    cls_set = set(classes)
    n_desc = DEFAULT_N[ds]
    out = {"n_classes": len(classes), "n_desc_expected": n_desc}


    pools = {}
    for arm in ("attr",) + ARMS:
        key = "attr_pool" if arm == "attr" else f"{arm}_pool"
        path = P[key]
        if not os.path.exists(path):
            out.setdefault("missing_pool", []).append(arm)
            continue
        pool = json.load(open(path, encoding="utf-8"))
        pools[arm] = pool
        bad_keys = [k for k in pool if _norm(k) not in
                    {_norm(c) for c in cls_set}]
        lens = sorted({len(v) for v in pool.values()})
        out[f"pool_{arm}"] = {
            "n_classes": len(pool), "complete": len(pool) == len(classes),
            "desc_counts": lens, "n_desc_ok": lens == [n_desc],
            "bad_keys": bad_keys[:3],
        }


    for arm in ("magent", "omni"):
        raw_path = P[f"{arm}_raw"]
        if not os.path.exists(raw_path):
            continue
        raw = json.load(open(raw_path, encoding="utf-8"))
        fb = [k for k, v in raw.items() if v.get("fallback")]
        out[f"fallback_{arm}"] = {"n": len(fb), "classes": fb[:5]}


    if os.path.exists(P["onto"]):
        onto = json.load(open(P["onto"], encoding="utf-8"))
        bad = []
        for cls, o in onto.items():
            for s in o.get("confusable_siblings", []):
                if s not in cls_set:
                    bad.append(f"{cls}:sib:{s}")
            if not (o.get("family") or o.get("lineage")):
                bad.append(f"{cls}:lineage")
            if len(o.get("family_visual_attributes", [])) < 3:
                bad.append(f"{cls}:fam_attrs<3")
            if len(o.get("distinguishing_visual_attributes", [])) < 2:
                bad.append(f"{cls}:dist_attrs<2")
        out["ontology"] = {"n": len(onto), "complete": len(onto) == len(classes),
                           "n_issues": len(bad), "issues": bad[:10]}


    if os.path.exists(P["feas"]):
        feas = json.load(open(P["feas"], encoding="utf-8"))
        drifts = {k: v for k, v in feas.get("sanity", {}).items()}
        out["feas_sanity"] = {"all_zero": all(v == 0 for v in drifts.values()),
                              "drifts": drifts}
    return out


def stage_audit():
    res = {}
    for ds in DATASETS:
        res[ds] = audit_one(ds)
        r = res[ds]
        line = [f"{ds}:"]
        for arm in ("attr",) + ARMS:
            p = r.get(f"pool_{arm}")
            line.append(f"{arm}={p['n_classes']}/{r['n_classes']}"
                        f"({'OK' if p['complete'] and p['n_desc_ok'] else 'CHECK'})"
                        if p else f"{arm}=MISS")
        ont = r.get("ontology", {})
        if ont:
            line.append(f"onto_issues={ont['n_issues']}")
        for arm in ("magent", "omni"):
            fb = r.get(f"fallback_{arm}")
            if fb:
                line.append(f"fb_{arm}={fb['n']}")
        san = r.get("feas_sanity", {})
        if san:
            line.append(f"sanity_zero={san['all_zero']}")
        print("  ".join(line))
    out_path = os.path.join(RESULTS_DIR, "p5_audit.json")
    json.dump(res, open(out_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"-> {out_path}")




def mcnemar_exact(n01, n10):
    pass
    n = n01 + n10
    if n == 0:
        return 1.0
    m = min(n01, n10)
    logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            for k in range(m + 1)]
    mx = max(logs)
    log_sum = mx + math.log(sum(math.exp(l - mx) for l in logs))
    return min(1.0, 2.0 * math.exp(log_sum - n * math.log(2)))


def direct_predict(feats, protos):
    return (feats @ protos.t()).argmax(1)


def stage_mcnemar():
    import torch
    from vendor.clip import clip
    sys.path.insert(0, os.path.join(REPO, "vendor", "clip"))

    model, _ = clip.load("ViT-B/32", device="cpu",
                         download_root=os.path.expanduser(r"~\.cache\clip"))
    model.eval()
    res = {}

    for ds in DATASETS:
        P = paths_for(ds)
        feas_path = P["feas"]
        if not os.path.exists(feas_path):
            print(f"[{ds}] 缺 feas，跳过")
            continue
        feats, labels, classes = load_test(ds)
        name2idx = {_norm(c): i for i, c in enumerate(classes)}


        cupl_descs = json.load(open(os.path.join(
            REPO, "cupl_descriptions", f"{ds}.json"), encoding="utf-8"))
        cupl_keys = list(cupl_descs.keys())
        cupl_feats = torch.load(os.path.join(REPO, "cache", f"{ds}_desc_vitb32.pt"),
                                map_location="cpu", weights_only=False)["feats"].float()
        cupl_spans, s = [], 0
        for k in cupl_keys:
            n = len(cupl_descs[k])
            cupl_spans.append((s, s + n))
            s += n
        cupl_remap = torch.tensor(
            [name2idx[_norm(CUBL_ALIAS.get(k, k))] for k in cupl_keys])
        cupl_pred = direct_predict(
            feats, mean_protos(cupl_feats, cupl_spans, cupl_remap))

        res[ds] = {}

        preds = {"cupl": cupl_pred}
        accs = {"cupl": round((cupl_pred == labels).float().mean().item() * 100, 2)}
        if os.path.exists(P["attr_pool"]):
            df, spans, remap = encode_pool(model, ds, "regen_attr_gpt54",
                                           P["attr_pool"], classes)
            preds["attr"] = direct_predict(feats, mean_protos(df, spans, remap))
            accs["attr"] = round((preds["attr"] == labels).float().mean().item() * 100, 2)

        for arm in ARMS:
            pool_path = P[f"{arm}_pool"]
            if not os.path.exists(pool_path):
                continue
            df, spans, remap = encode_pool(model, ds, f"regen_{arm}",
                                           pool_path, classes)
            if len(spans) != len(classes):
                print(f"[{ds}] {arm} 池不齐，跳过")
                continue
            arm_pred = direct_predict(feats, mean_protos(df, spans, remap))
            accs[arm] = round((arm_pred == labels).float().mean().item() * 100, 2)
            res[ds][arm] = {}
            for ctrl in ("cupl", "attr"):
                if ctrl not in preds:
                    continue
                cp = preds[ctrl]
                n01 = int(((cp == labels) & (arm_pred != labels)).sum())
                n10 = int(((cp != labels) & (arm_pred == labels)).sum())
                res[ds][arm][f"vs_{ctrl}"] = {
                    "n01_ctrl_right_arm_wrong": n01,
                    "n10_ctrl_wrong_arm_right": n10,
                    "p_exact": round(mcnemar_exact(n01, n10), 6),
                    "delta_pt": round(accs[arm] - accs[ctrl], 2),
                }
        res[ds]["acc"] = accs

        parts = [f"{ds}: " + "  ".join(f"{k}={v}" for k, v in accs.items())]
        for arm in ARMS:
            if arm in res[ds]:
                for ctrl in ("cupl", "attr"):
                    key = f"vs_{ctrl}"
                    if key in res[ds][arm]:
                        m = res[ds][arm][key]
                        parts.append(f"  {arm} {key}: {m['delta_pt']:+.2f} "
                                     f"(n01={m['n01_ctrl_right_arm_wrong']}, "
                                     f"n10={m['n10_ctrl_wrong_arm_right']}, "
                                     f"p={m['p_exact']})")
        print("\n".join(parts))

    out_path = os.path.join(RESULTS_DIR, "p5_mcnemar.json")
    json.dump(res, open(out_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "mcnemar"], required=True)
    args = ap.parse_args()
    if args.stage == "audit":
        stage_audit()
    else:
        stage_mcnemar()


if __name__ == "__main__":
    main()
