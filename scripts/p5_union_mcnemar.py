
pass

import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e5_gen_pools import load_classes
from e5_probe import load_test, encode_pool, mean_protos, _norm
from p4_b1_feas import CUBL_ALIAS, paths_for

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, "results")
DESC_DIR = os.path.join(REPO, "cupl_descriptions")
DATASETS = ("aircraft", "euro_sat", "dtd", "cub")
ARMS = ("attr", "onto", "magent", "omni", "cupl_re")
INTERVENTION = ("onto", "magent", "omni")


EXPECTED = {
    "aircraft": {"cupl": 20.94, "attr": 20.46, "onto": 20.85,
                 "magent": 21.09, "omni": 21.18, "cupl_re": 20.07},
    "euro_sat": {"cupl": 51.30, "attr": 51.25, "onto": 51.40,
                "magent": 51.01, "omni": 50.99, "cupl_re": 49.54},
    "dtd": {"cupl": 50.11, "attr": 51.54, "onto": 51.97,
            "magent": 52.18, "omni": 50.90, "cupl_re": 46.97},
    "cub": {"cupl": 52.93, "attr": 53.54, "onto": 54.00,
            "magent": 54.37, "omni": 54.02, "cupl_re": 48.74},
}


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


def bh_adj(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, min(1.0, ps[i] * m / rank))
        adj[i] = prev
    return adj


def union_protos(pool_feats, n_classes):
    pass
    sums = [None] * n_classes
    cnts = [0] * n_classes
    for feats, spans, remap in pool_feats:
        for (s, e), r in zip(spans, remap.tolist()):
            block = feats[s:e]
            if sums[r] is None:
                sums[r] = block.sum(0)
            else:
                sums[r] += block.sum(0)
            cnts[r] += e - s
    protos = []
    for i in range(n_classes):
        p = sums[i] / cnts[i]
        protos.append(p / p.norm())
    return torch.stack(protos)


def main():
    from vendor.clip import clip
    sys.path.insert(0, os.path.join(REPO, "vendor", "clip"))
    model, _ = clip.load("ViT-B/32", device="cpu",
                         download_root=os.path.expanduser(r"~\.cache\clip"))
    model.eval()

    res = {}
    fam_raw = []
    for ds in DATASETS:
        P = paths_for(ds)
        feats, labels, classes = load_test(ds)
        n_cls = len(classes)
        name2idx = {_norm(c): i for i, c in enumerate(classes)}


        cupl_descs = json.load(open(os.path.join(
            DESC_DIR, f"{ds}.json"), encoding="utf-8"))
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
        cupl_pred = (feats @ mean_protos(cupl_feats, cupl_spans, cupl_remap).t()).argmax(1)
        cupl_acc = round((cupl_pred == labels).float().mean().item() * 100, 2)
        assert abs(cupl_acc - EXPECTED[ds]["cupl"]) < 0.005, (ds, cupl_acc)

        res[ds] = {"cupl_acc": cupl_acc, "arms": {}}
        line = [f"[{ds}] cupl={cupl_acc:.2f}"]
        for arm in ARMS:
            if arm == "cupl_re":
                pool_path = os.path.join(DESC_DIR, f"{ds}_regen_cupl_gpt54.json")
                tag = "regen_cupl_gpt54"
            else:
                pool_path = P["attr_pool"] if arm == "attr" else P[f"{arm}_pool"]
                tag = "regen_attr_gpt54" if arm == "attr" else f"regen_{arm}"
            df, spans, remap = encode_pool(model, ds, tag, pool_path, classes)
            upred = (feats @ union_protos(
                [(cupl_feats, cupl_spans, cupl_remap), (df, spans, remap)],
                n_cls).t()).argmax(1)
            uacc = round((upred == labels).float().mean().item() * 100, 2)
            assert abs(uacc - EXPECTED[ds][arm]) < 0.005, (ds, arm, uacc)
            n01 = int(((cupl_pred == labels) & (upred != labels)).sum())
            n10 = int(((cupl_pred != labels) & (upred == labels)).sum())
            p = round(mcnemar_exact(n01, n10), 6)
            delta = round(uacc - cupl_acc, 2)
            rec = {"union_acc": uacc, "delta_vs_cupl": delta,
                   "n01_cupl_right_union_wrong": n01,
                   "n10_cupl_wrong_union_right": n10, "p_exact": p}
            if arm in INTERVENTION:
                fam_raw.append((ds, arm, p))
            res[ds]["arms"][arm] = rec
            line.append(f"{arm}={uacc:.2f}({delta:+.2f},p={p})")
        print("  ".join(line))


    fam_ps = [p for _, _, p in fam_raw]
    fam_bh = bh_adj(fam_ps)
    for (ds, arm, _), adj in zip(fam_raw, fam_bh):
        res[ds]["arms"][arm]["p_bh"] = round(adj, 6)
        res[ds]["arms"][arm]["bh_sig_05"] = adj < 0.05

    out_path = os.path.join(RESULTS_DIR, "p5_union_mcnemar.json")
    json.dump(res, open(out_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n-> {out_path}")

    print("\n==== 干预臂并集显著性（BH校正后）====")
    for ds in DATASETS:
        for arm in INTERVENTION:
            r = res[ds]["arms"][arm]
            star = "**" if r["p_bh"] < 0.05 else ("*" if r["p_exact"] < 0.05 else "  ")
            print(f"{star} {ds:9s} {arm:7s} Δ={r['delta_vs_cupl']:+.2f}  "
                  f"p={r['p_exact']:.4f}  BH={r['p_bh']:.4f}")


if __name__ == "__main__":
    main()
