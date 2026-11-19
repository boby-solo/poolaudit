
pass

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p4_b1_feas import (
    call_retry, gen_max_tokens, paths_for, DATASETS, DESC_DIR,
    RESULTS_DIR, CUBL_ALIAS,
)
from e5_gen_pools import load_classes, DEFAULT_N, parse_desc_array


CUPL_PROMPT = """What are useful visual features to distinguish a {cls}?

Write {n} short descriptions of the useful visual features that distinguish this class. Each description is one full sentence in plain English. There are no other constraints on content or wording.

Output: ONLY a JSON array of {n} strings, nothing else."""


def pool_paths(ds):
    return {
        "pool": os.path.join(DESC_DIR, f"{ds}_regen_cupl_gpt54.json"),
        "raw": os.path.join(RESULTS_DIR, f"p5_cupl_{ds}_raw.json"),
        "out": os.path.join(RESULTS_DIR, f"p5_cupl_{ds}.json"),
    }


def stage_gen(args, classes):
    n_desc = DEFAULT_N[args.dataset]
    sp = pool_paths(args.dataset)
    pool, raw_log = {}, {}
    if os.path.exists(sp["pool"]):
        pool = json.load(open(sp["pool"], encoding="utf-8"))
        print(f"[resume] cupl 池已有 {len(pool)}/{len(classes)} 类")
    if os.path.exists(sp["raw"]):
        raw_log = json.load(open(sp["raw"], encoding="utf-8"))
    todo = [c for c in classes if c not in pool]
    print(f"[cupl] 生成：{len(todo)} 类 | n_desc={n_desc} | "
          f"model={args.model} T={args.temperature}")
    print("[cupl] prompt 预览：\n" + CUPL_PROMPT.format(cls="X", n=n_desc))

    def one_gen(cls):
        raw = call_retry(args.api_base, args.api_key, args.model,
                         CUPL_PROMPT.format(cls=cls, n=n_desc),
                         args.temperature, gen_max_tokens(n_desc),
                         f"cupl:{cls}")
        descs, dropped = parse_desc_array(raw, n_desc)
        if len(descs) < max(5, n_desc // 3):
            raise ValueError(f"{cls}: parsed only {len(descs)}")
        return cls, descs, raw, dropped

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_gen, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                c2, descs, raw, dropped = fut.result()
                pool[c2] = descs
                raw_log[c2] = {"raw": raw, "n_parsed": len(descs),
                               "n_dropped": dropped}
            except Exception as e:
                failed.append(str(e))
            if i % 20 == 0:
                json.dump(pool, open(sp["pool"], "w", encoding="utf-8"),
                          indent=2, ensure_ascii=False)
                print(f"  cupl 池 {i}/{len(todo)}（失败 {len(failed)}）", flush=True)
    json.dump(pool, open(sp["pool"], "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump(raw_log, open(sp["raw"], "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"-> {sp['pool']}（{len(pool)}/{len(classes)}，失败 {len(failed)}）")
    for f in failed[:5]:
        print("  FAIL:", f)


def stage_eval(args, classes):
    import torch
    from e5_probe import load_test, encode_pool, mean_protos, k1_joint_fast, _norm
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "vendor", "clip"))
    from vendor.clip import clip

    sp = pool_paths(args.dataset)
    if not os.path.exists(sp["pool"]):
        print("[eval] 池缺失，先 --stage gen")
        sys.exit(1)
    feats, labels, cls_list = load_test(args.dataset)
    name2idx = {_norm(c): i for i, c in enumerate(cls_list)}
    model, _ = clip.load("ViT-B/32", device="cpu",
                         download_root=os.path.expanduser(r"~\.cache\clip"))
    model.eval()


    cupl_descs = json.load(open(os.path.join(DESC_DIR, f"{args.dataset}.json"),
                                encoding="utf-8"))
    cupl_keys = list(cupl_descs.keys())
    cache = os.path.join(os.path.dirname(DESC_DIR), "cache")
    cupl_feats = torch.load(os.path.join(cache, f"{args.dataset}_desc_vitb32.pt"),
                            map_location="cpu", weights_only=False)["feats"].float()
    cupl_spans, s0 = [], 0
    for k in cupl_keys:
        n = len(cupl_descs[k])
        cupl_spans.append((s0, s0 + n))
        s0 += n
    try:
        cupl_remap = torch.tensor([name2idx[_norm(k)] for k in cupl_keys])
    except KeyError:
        cupl_remap = torch.tensor(
            [name2idx[_norm(CUBL_ALIAS.get(k, k))] for k in cupl_keys])

    def union_with_incumbent(df, spans, remap):
        span_by_label = {int(remap[i]): spans[i] for i in range(len(spans))}
        uni, u_spans, pos = [], [], 0
        for i in range(len(cupl_keys)):
            li = int(cupl_remap[i])
            a = cupl_feats[cupl_spans[i][0]:cupl_spans[i][1]]
            b = df[span_by_label[li][0]:span_by_label[li][1]]
            seg = torch.cat([a, b])
            uni.append(seg)
            u_spans.append((pos, pos + seg.shape[0]))
            pos += seg.shape[0]
        up = mean_protos(torch.cat(uni), u_spans, cupl_remap)
        return ((feats @ up.t()).argmax(1) == labels).float().mean().item() * 100


    df, spans, remap = encode_pool(model, args.dataset, "regen_cupl_gpt54",
                                   sp["pool"], cls_list)
    if len(spans) != len(cls_list):
        print(f"[eval] 池不完整 {len(spans)}/{len(cls_list)}")
        sys.exit(1)
    protos = mean_protos(df, spans, remap)
    direct = ((feats @ protos.t()).argmax(1) == labels).float().mean().item() * 100
    k1 = k1_joint_fast(feats, labels, df, spans, remap) * 100
    uni = union_with_incumbent(df, spans, remap)


    feas = json.load(open(paths_for(args.dataset)["feas"], encoding="utf-8"))
    attr = feas["baseline"]["attr"]
    refs = {
        "incumbent": feas["baseline"].get("cupl"),
        "attr": attr,
        "onto": {kk: feas["onto"][kk] for kk in ("direct", "k1", "union_direct")},
        "magent": {kk: feas["magent"][kk] for kk in ("direct", "k1", "union_direct")},
        "omni": {kk: feas["omni"][kk] for kk in ("direct", "k1", "union_direct")},
    }
    cupl_vals = {"direct": round(direct, 2), "k1": round(k1, 2),
                 "union_direct": round(uni, 2)}
    deltas = {}
    for arm in ("attr", "onto", "magent", "omni"):
        deltas[f"{arm}_minus_cupl"] = {
            kk: round(refs[arm][kk] - cupl_vals[kk], 2)
            for kk in ("direct", "k1") if kk in refs[arm]}

    out = {
        "dataset": args.dataset, "model": args.model,
        "temperature": args.temperature,
        "n_desc_per_class": DEFAULT_N[args.dataset],
        "cupl_gpt54": cupl_vals,
        "refs": refs,
        "deltas_vs_cupl": deltas,
        "note": ("cupl_gpt54 = CuPL 协议 + GPT-5.4 + T=0.7 + 条数对齐五臂，"
                 "single draw；incumbent 为现役参照（早期模型，非公平对比单列）。"
                 "判读：omni_minus_cupl 大 = record 结构必要；"
                 "attr_minus_cupl 小 = 属性基线不弱；"
                 "粗粒度 cupl 反超 = 印证避免结构约束。"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json.dump(out, open(sp["out"], "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"-> {sp['out']}")
    print("cupl_gpt54:", json.dumps(cupl_vals, ensure_ascii=False))
    print("deltas_vs_cupl:", json.dumps(deltas, indent=1, ensure_ascii=False))


def stage_ping(args):
    from e5_gen_pools import chat_once
    import traceback
    t0 = time.time()
    try:
        out = chat_once(args.api_base, args.api_key, args.model,
                        "Reply with the single word: pong", 0.7, 16, timeout=60)
        print(f"[ping] OK  {time.time()-t0:.1f}s  response: {out[:60]!r}")
    except Exception:
        print(f"[ping] FAILED  {time.time()-t0:.1f}s")
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description="CuPL 协议复现臂（横向方法对比）")
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--stage", default="all", choices=["ping", "gen", "eval", "all"])
    ap.add_argument("--api-base",
                    default=os.environ.get("E5_API_BASE",
                                           "https://az.gptplus5.com/v1"))
    ap.add_argument("--model", default=os.environ.get("E5_MODEL", "gpt-5.4"))
    ap.add_argument("--api-key", default=os.environ.get("E5_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not args.api_key:
        print("[main] 缺 API key（--api-key 或环境变量 E5_API_KEY）")
        sys.exit(1)
    classes = load_classes(args.dataset)
    if args.stage == "ping":
        stage_ping(args)
        return
    if args.stage in ("gen", "all"):
        stage_gen(args, classes)
    if args.stage in ("eval", "all"):
        stage_eval(args, classes)


if __name__ == "__main__":
    main()
