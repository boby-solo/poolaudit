
pass

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p5_stats import mcnemar_exact, direct_predict, DATASETS
from e5_gen_pools import load_classes
from e5_probe import load_test, encode_pool, mean_protos, _norm
from p4_b1_feas import CUBL_ALIAS, paths_for, DESC_DIR, RESULTS_DIR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMP_MARKERS = ("than", "unlike", "compared to", "rather than", "instead of", "whereas")


def cupl_pool_path(ds):
    return os.path.join(DESC_DIR, f"{ds}_regen_cupl_gpt54.json")




def comp_sib_rates(ds):
    pool = json.load(open(cupl_pool_path(ds), encoding="utf-8"))
    keys_norm = [_norm(k) for k in pool]
    total, comp_hits, sib_hits = 0, 0, 0
    for cls, sents in pool.items():
        own = _norm(cls)
        for s in sents:
            total += 1
            low = s.lower()
            if any(m in low for m in COMP_MARKERS):
                comp_hits += 1
            low_n = _norm(low)
            for kn in keys_norm:
                if kn != own and kn in low_n:
                    sib_hits += 1
                    break
    return round(comp_hits / total * 100, 1), round(sib_hits / total * 100, 1)


def stage_comparative():
    res = {"method_note": "Same protocol as p5_comparative.json method_note "
                          "(Comp markers; Sib case-insensitive, hyphens/underscores "
                          "as spaces, substring vs pool keys, sentence credited to "
                          "own class). Computed for the cupl_gpt54 arm only; "
                          "five-arm values live in p5_comparative.json (unchanged)."}
    for ds in DATASETS:
        c, s = comp_sib_rates(ds)
        res[ds] = {"cupl_gpt54": [c, s]}
        print(f"{ds}: cupl_gpt54 comp={c}% sib={s}%")

    ref = json.load(open(os.path.join(RESULTS_DIR, "p5_comparative.json"),
                         encoding="utf-8"))
    for ds in DATASETS:
        res[ds]["refs"] = ref.get(ds, {})
    out = os.path.join(RESULTS_DIR, "p5_comparative_cupl.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"-> {out}")




def get_pred(model, ds, pool_name, pool_path, classes, feats):
    df, spans, remap = encode_pool(model, ds, pool_name, pool_path, classes)
    if len(spans) != len(classes):
        sys.exit(f"[{ds}] {pool_name} 池不齐 {len(spans)}/{len(classes)}")
    return direct_predict(feats, mean_protos(df, spans, remap))


def incumbent_pred(ds, feats, labels, classes, name2idx):
    import torch
    descs = json.load(open(os.path.join(REPO, "cupl_descriptions",
                                        f"{ds}.json"), encoding="utf-8"))
    keys = list(descs.keys())
    feats_d = torch.load(os.path.join(REPO, "cache", f"{ds}_desc_vitb32.pt"),
                          map_location="cpu", weights_only=False)["feats"].float()
    spans, s = [], 0
    for k in keys:
        n = len(descs[k])
        spans.append((s, s + n))
        s += n
    remap = torch.tensor([name2idx[_norm(CUBL_ALIAS.get(k, k))] for k in keys])
    return direct_predict(feats, mean_protos(feats_d, spans, remap))


def stage_mcnemar():
    import torch
    sys.path.insert(0, os.path.join(REPO, "vendor", "clip"))
    from vendor.clip import clip
    model, _ = clip.load("ViT-B/32", device="cpu",
                         download_root=os.path.expanduser(r"~\.cache\clip"))
    model.eval()

    res = {}
    for ds in DATASETS:
        P = paths_for(ds)
        feats, labels, classes = load_test(ds)
        name2idx = {_norm(c): i for i, c in enumerate(classes)}

        cupl_pred = get_pred(model, ds, "regen_cupl_gpt54",
                             cupl_pool_path(ds), classes, feats)
        ctrl_preds = {
            "attr": get_pred(model, ds, "regen_attr_gpt54",
                             P["attr_pool"], classes, feats),
            "omni": get_pred(model, ds, "regen_omni",
                             P["omni_pool"], classes, feats),
            "incumbent": incumbent_pred(ds, feats, labels, classes, name2idx),
        }
        accs = {"cupl_gpt54": round((cupl_pred == labels).float().mean().item() * 100, 2)}
        res[ds] = {}
        for cname, cp in ctrl_preds.items():
            accs[cname] = round((cp == labels).float().mean().item() * 100, 2)
            n01 = int(((cp == labels) & (cupl_pred != labels)).sum())
            n10 = int(((cp != labels) & (cupl_pred == labels)).sum())
            res[ds][f"vs_{cname}"] = {
                "n01_ctrl_right_cupl_wrong": n01,
                "n10_ctrl_wrong_cupl_right": n10,
                "p_exact": round(mcnemar_exact(n01, n10), 6),
                "delta_pt": round(accs["cupl_gpt54"] - accs[cname], 2),
            }
        res[ds]["acc"] = accs
        print(f"{ds}: " + "  ".join(f"{k}={v}" for k, v in accs.items()))
        for cname in ctrl_preds:
            m = res[ds][f"vs_{cname}"]
            print(f"  cupl_gpt54 vs {cname}: {m['delta_pt']:+.2f} "
                  f"(n01={m['n01_ctrl_right_cupl_wrong']}, "
                  f"n10={m['n10_ctrl_wrong_cupl_right']}, p={m['p_exact']})")

    out = os.path.join(RESULTS_DIR, "p5_mcnemar_cupl.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"-> {out}")


def main():
    print("== Comp/Sib 率（口径与 p5_comparative.json 一致）==")
    stage_comparative()
    print("\n== McNemar 三族 ==")
    stage_mcnemar()


if __name__ == "__main__":
    main()
