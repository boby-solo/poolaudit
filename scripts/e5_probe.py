pass

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finegrained_run import encode_texts, _norm, proto_of, greedy_oracle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vendor.clip import clip

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "cache")


REFS = {
    "cars": {"direct": 60.81, "k1": 64.07, "greedy": 62.28, "source": "d3_cars.json"},
    "aircraft": {"direct": 20.94, "k1": 26.07, "greedy": 23.49, "source": "d3_aircraft.json"},
    "dtd": {"direct": 50.11, "k1": 59.57, "greedy": 52.13, "source": "d2g_dtd_oracle.json"},
    "cub": {"direct": 52.93, "k1": 55.68, "greedy": 56.44, "source": "d1_baseline.json / d1_oracle.json"},
}

CUB_ROOT = os.environ.get("CUB_ROOT", os.path.join(REPO, "data", "dataset", "cub", "CUB_200_2011"))


def build_pools(ds):
    pass
    pools = {
        "cupl": os.path.join(REPO, "cupl_descriptions", f"{ds}.json"),
        "waffle_random": os.path.join(REPO, "cupl_descriptions", f"{ds}_waffle_random.json"),
    }
    for path in sorted(glob.glob(os.path.join(REPO, "cupl_descriptions", f"{ds}_regen_attr*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        pools[stem[len(ds) + 1:]] = path
    return pools


def load_test(ds):
    blob = torch.load(os.path.join(CACHE, f"{ds}_test_vitb32.pt"), map_location="cpu", weights_only=False)
    if "classes" in blob:
        classes = blob["classes"]
    else:

        with open(os.path.join(CUB_ROOT, "classes.txt"), encoding="utf-8") as f:
            raw = [l.split()[1].strip() for l in f if l.strip()]
        classes = [r.split(".", 1)[1].replace("_", " ") for r in raw]
    return blob["feats"].float(), torch.tensor(blob["labels"]), classes


def encode_pool(model, ds, pool_name, desc_path, classes):
    pass
    if pool_name == "cupl":
        cache_path = os.path.join(CACHE, f"{ds}_desc_vitb32.pt")
    else:
        cache_path = os.path.join(CACHE, f"{ds}_{pool_name}_desc_vitb32.pt")
    descs = json.load(open(desc_path, encoding="utf-8"))
    content_sha = hashlib.sha256(open(desc_path, "rb").read()).hexdigest()
    json_keys = list(descs.keys())
    name2idx = {_norm(c): i for i, c in enumerate(classes)}
    remap = torch.tensor([name2idx[_norm(jk)] for jk in json_keys])
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        spans = [tuple(x) for x in blob["spans"]]
        lens_ok = [e - s for s, e in spans] == [len(descs[k]) for k in json_keys]
        keys_ok = blob.get("keys", json_keys) == json_keys
        sha_ok = blob.get("content_sha", content_sha) == content_sha
        if keys_ok and lens_ok and sha_ok:
            print(f"  [{pool_name}] desc feats from cache ({blob['feats'].shape[0]} texts)")
            return blob["feats"].float(), spans, remap
    print(f"  [{pool_name}] encoding {sum(len(descs[k]) for k in json_keys)} texts (one-off)...")
    feats, spans = [], []
    for jk in json_keys:
        f = encode_texts(model, descs[jk])
        start = spans[-1][1] if spans else 0
        spans.append((start, start + f.shape[0]))
        feats.append(f)
    desc_feats = torch.cat(feats)
    torch.save({"feats": desc_feats, "spans": spans, "keys": json_keys,
                "content_sha": content_sha}, cache_path)
    return desc_feats, spans, remap


def mean_protos(desc_feats, spans, remap):
    C = len(spans)
    protos = torch.zeros(C, desc_feats.shape[1])
    for i, (s, e) in enumerate(spans):
        protos[remap[i]] = proto_of(desc_feats[s:e])
    return protos


def k1_joint_fast(image_feats, labels, desc_feats, spans, remap):
    pass
    C, N = len(spans), image_feats.shape[0]
    protos_full = mean_protos(desc_feats, spans, remap)
    sim = image_feats @ protos_full.t()
    best_cols = torch.zeros(C, dtype=torch.long)
    for i, (s, e) in enumerate(spans):
        c = remap[i].item()
        sim_other = sim.clone()
        sim_other[:, c] = float("-inf")
        m_other, a_other = sim_other.max(dim=1)
        correct_other = a_other == labels
        feats_i = desc_feats[s:e]
        v = image_feats @ feats_i.t()
        le = v <= m_other.unsqueeze(1)
        term1 = (correct_other.unsqueeze(1) & le).sum(dim=0)
        term2 = ((v > m_other.unsqueeze(1)) & (labels == c).unsqueeze(1)).sum(dim=0)
        acc = (term1 + term2).float() / N
        best_cols[i] = acc.argmax()
    protos_k1 = torch.zeros(C, desc_feats.shape[1])
    for i, (s, e) in enumerate(spans):
        protos_k1[remap[i]] = desc_feats[s:e][best_cols[i]]
    pred = (image_feats @ protos_k1.t()).argmax(dim=1)
    return (pred == labels).float().mean().item()


def evaluate(image_feats, labels, protos):
    return ((image_feats @ protos.t()).argmax(dim=1) == labels).float().mean().item()


def sample_dump(pool_name, desc_path, k=3):
    pass
    descs = json.load(open(desc_path, encoding="utf-8"))
    for cls in list(descs.keys())[:2]:
        print(f"    [{pool_name}] {cls} 样例:")
        for s in descs[cls][:k]:
            print(f"      - {s[:110]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cars", "aircraft", "dtd", "cub"], required=True)
    ap.add_argument("--pool", action="append", help="可重复；缺省=全部已存在的池")
    ap.add_argument("--skip-greedy", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    pools = build_pools(ds)

    if args.pool:
        names = ["cupl"] + [p for p in args.pool if p != "cupl"]
    else:
        names = [p for p, path in pools.items() if os.path.exists(path)]
    print(f"dataset: {ds} | pools: {names}   (greedy {'skipped' if args.skip_greedy else 'on'})")

    t0 = time.time()
    image_feats, labels, classes = load_test(ds)
    print(f"test feats {tuple(image_feats.shape)}, {len(classes)} classes")

    model, _ = clip.load("ViT-B/32", device="cpu", download_root=os.path.expanduser(r"~\.cache\clip"))
    model.eval()

    results = {"dataset": ds, "n_test": int(image_feats.shape[0]), "n_classes": len(classes),
               "skip_greedy": args.skip_greedy, "pools": {}, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    for name in names:
        print(f"\n===== pool: {name} =====")
        desc_feats, spans, remap = encode_pool(model, ds, name, pools[name], classes)
        n_desc = [e - s for s, e in spans]
        protos = mean_protos(desc_feats, spans, remap)
        acc_direct = evaluate(image_feats, labels, protos)
        print(f"  direct (CuPL bare-mean): {acc_direct*100:.2f}%   desc/class min/mean/max = "
              f"{min(n_desc)}/{sum(n_desc)/len(n_desc):.1f}/{max(n_desc)}")
        sample_dump(name, pools[name])

        acc_k1 = k1_joint_fast(image_feats, labels, desc_feats, spans, remap)
        print(f"  k1-joint ceiling      : {acc_k1*100:.2f}%")

        acc_gr = None
        if not args.skip_greedy:
            print("  greedy ceiling (test leakage, ceiling only)...")
            acc_gr, sizes = greedy_oracle(image_feats, labels, desc_feats, spans, remap)
            print(f"  greedy ceiling        : {acc_gr*100:.2f}%  (kept mean {sum(sizes)/len(sizes):.1f}/{max(n_desc)})")

        results["pools"][name] = {
            "direct": round(acc_direct * 100, 2),
            "k1": round(acc_k1 * 100, 2),
            "greedy": round(acc_gr * 100, 2) if acc_gr is not None else None,
            "ceiling": round(max(acc_k1, acc_gr or 0) * 100, 2),
            "desc_per_class": {"min": min(n_desc), "mean": round(sum(n_desc) / len(n_desc), 1), "max": max(n_desc)},
        }


    if "cupl" in results["pools"]:
        ref = REFS[ds]
        drift = abs(results["pools"]["cupl"]["direct"] - ref["direct"])
        drift_k1 = abs(results["pools"]["cupl"]["k1"] - ref["k1"])
        print(f"\n[sanity] cupl direct {results['pools']['cupl']['direct']} vs reference {ref['direct']} (|Δ|={drift:.2f}, {ref['source']})")
        print(f"[sanity] cupl k1     {results['pools']['cupl']['k1']} vs reference {ref['k1']} (|Δ|={drift_k1:.2f})")
        results["sanity"] = {"drift_direct": drift, "drift_k1": drift_k1,
                             "pass": drift < 0.3 and drift_k1 < 0.3}
        if not results["sanity"]["pass"]:
            print("WARNING: cupl 复现漂移 >0.3pt，k1 快速实现可能有 bug，停止判读前先核查！")


    regen_names = [n for n in results["pools"] if n.startswith("regen")]
    if regen_names and "cupl" in results["pools"]:
        results["regen_vs_cupl"] = {}
        verdicts = []
        cupl_mean_n = results["pools"]["cupl"]["desc_per_class"]["mean"]
        for n in sorted(regen_names):
            p = results["pools"][n]
            d_direct = p["direct"] - results["pools"]["cupl"]["direct"]
            d_ceiling = p["ceiling"] - results["pools"]["cupl"]["ceiling"]
            if d_direct >= 0.8 or d_ceiling >= 1.5:
                v = "GO"
            elif d_direct < 0.3 and d_ceiling < 0.5:
                v = "NO-GO"
            else:
                v = "GRAY"
            verdicts.append(v)
            results["regen_vs_cupl"][n] = {"delta_direct": round(d_direct, 2),
                                           "delta_ceiling": round(d_ceiling, 2),
                                           "verdict": v}
            print(f"{n} vs cupl: Δdirect = {d_direct:+.2f}pt | Δceiling = {d_ceiling:+.2f}pt -> {v}")

            if abs(p["desc_per_class"]["mean"] - cupl_mean_n) / cupl_mean_n > 0.2:
                print(f"  NOTE: {n} 条数 {p['desc_per_class']['mean']}/类 vs cupl {cupl_mean_n}/类，"
                      f"条数差>20%，k1 天花板对比含候选数混淆，判读以直评为主。")
        if "GO" in verdicts:
            results["verdict"] = "GO"
        elif all(v == "NO-GO" for v in verdicts):
            results["verdict"] = "NO-GAIN (text-side saturation: the ceiling does not move with pool content)"
        else:
            results["verdict"] = "GRAY (灰区：建议提前接入 Phase 2 图像接地校验判据再判)"
        print(f"VERDICT: {results['verdict']}")

    if "waffle_random" in results["pools"] and "cupl" in results["pools"]:
        w_vs_c = results["pools"]["waffle_random"]["direct"] - results["pools"]["cupl"]["direct"]
        results["waffle_vs_cupl_direct"] = round(w_vs_c, 2)
        note = "随机词池直评 ≈/≥ CuPL 原池 → 原池被证伪得更彻底（WaffleCLIP 式诊断证据）" \
            if w_vs_c >= -0.2 else "随机词池明显低于原池（原池仍含一定视觉信号）"
        print(f"waffle vs cupl direct: {w_vs_c:+.2f}pt -> {note}")

    results_path = os.path.join(REPO, "results", f"p4_e5_{ds}_probe.json")
    json.dump(results, open(results_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nresults -> {results_path}  ({time.time()-t0:.0f}s)")

    print(f"""
=== 判读标准（E5-v2 Phase 1b, {ds}）===
1. GO        : regen 直评 ≥ cupl+0.8pt 或 regen 天花板 ≥ cupl 天花板+1.5pt
               → 稀释端点可救，生成侧路线成立，Phase 2 图像接地校验 + 扩 CUB/比较式/本体臂
2. NO-GAIN    : regen direct < cupl+0.3 and ceiling < cupl+0.5
               -> no measurable headroom at this endpoint
3. GRAY       : 两者之间 → 提前接 Phase 2 无标签接地校验再判
4. 独立证据   : waffle 随机词池 vs 原池 —— 本数据集上描述语义内容的边际贡献
5. 混淆提示   : 重生成池与原池条数差 >20% 时，k1 天花板对比含候选数混淆（脚本会自动标注）
""")


if __name__ == "__main__":
    main()
