
pass
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p4_b1_feas import (
    call_retry, gen_max_tokens, build_suite, paths_for,
    DATASETS, DESC_DIR, RESULTS_DIR, CUBL_ALIAS,
)
from e5_gen_pools import load_classes, DEFAULT_N, parse_desc_array


def seed_paths(ds, s):
    return {
        "pool": os.path.join(DESC_DIR, f"{ds}_regen_onto_seed{s}.json"),
        "raw": os.path.join(RESULTS_DIR, f"p5_{ds}_onto_seed{s}_raw.json"),
        "summary": os.path.join(RESULTS_DIR, f"p5_{ds}_seeds.json"),
    }




def stage_gen(args, classes, P, suite):
    onto = json.load(open(P["onto"], encoding="utf-8"))
    missing = [c for c in classes if c not in onto]
    if missing:
        print(f"[gen] 本体缺 {len(missing)} 类，先跑 p4_b1_feas --stage onto")
        sys.exit(1)
    n_desc = DEFAULT_N[args.dataset]

    for s in range(1, args.seeds + 1):
        sp = seed_paths(args.dataset, s)
        pool, raw_log = {}, {}
        if os.path.exists(sp["pool"]):
            pool = json.load(open(sp["pool"], encoding="utf-8"))
            print(f"[resume] seed{s} 池已有 {len(pool)}/{len(classes)} 类")
        if os.path.exists(sp["raw"]):
            raw_log = json.load(open(sp["raw"], encoding="utf-8"))
        todo = [c for c in classes if c not in pool]
        if not todo:
            print(f"[seed {s}] 已完成，跳过")
            continue
        print(f"[seed {s}] 本体引导生成：{len(todo)} 类 | "
              f"model={args.model} T={args.temperature}")

        def one_gen(cls):

            p = suite["onto_gen"].format(cls=cls, n=n_desc, **suite["dk"],
                                         **suite["onto_fields"](onto[cls]))
            raw = call_retry(args.api_base, args.api_key, args.model, p,
                             args.temperature, gen_max_tokens(n_desc),
                             f"seed{s}:gen:{cls}")
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
                    print(f"  seed{s} 池 {i}/{len(todo)}（失败 {len(failed)}）",
                          flush=True)
        json.dump(pool, open(sp["pool"], "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        json.dump(raw_log, open(sp["raw"], "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        print(f"[seed {s}] -> {sp['pool']}（{len(pool)}/{len(classes)}，"
              f"失败 {len(failed)}）")
        for f in failed[:5]:
            print("  FAIL:", f)




def stage_eval(args, classes, P):
    import torch
    from e5_probe import load_test, encode_pool, mean_protos, k1_joint_fast, _norm
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "vendor", "clip"))
    from vendor.clip import clip

    feats, labels, cls_list = load_test(args.dataset)
    assert cls_list == classes or set(cls_list) == set(classes), "类表不一致"
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
    assert cupl_feats.shape[0] == s0, "CuPL 缓存条数不符（缓存过期？）"
    try:
        cupl_remap = torch.tensor([name2idx[_norm(k)] for k in cupl_keys])
    except KeyError:
        cupl_remap = torch.tensor(
            [name2idx[_norm(CUBL_ALIAS.get(k, k))] for k in cupl_keys])
        print("[eval] CuPL 键名经别名表对齐")

    def union_with_cupl(df, spans, remap):
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

    def eval_pool_file(pool_name, pool_path):
        df, spans, remap = encode_pool(model, args.dataset, pool_name,
                                       pool_path, cls_list)
        if len(spans) != len(cls_list):
            print(f"[eval] {pool_name} 池不完整（{len(spans)}/{len(cls_list)} 类）")
            return None
        protos = mean_protos(df, spans, remap)
        direct = ((feats @ protos.t()).argmax(1) == labels
                  ).float().mean().item() * 100
        k1 = k1_joint_fast(feats, labels, df, spans, remap) * 100
        return {"direct": round(direct, 2), "k1": round(k1, 2),
                "union_direct": round(union_with_cupl(df, spans, remap), 2)}


    feas = json.load(open(P["feas"], encoding="utf-8"))
    attr = feas["baseline"]["attr"]
    draws = {"orig": {"direct": feas["onto"]["direct"], "k1": feas["onto"]["k1"],
                      "union_direct": feas["onto"]["union_direct"]}}


    present = []
    for s in range(1, args.seeds + 1):
        sp = seed_paths(args.dataset, s)
        if not os.path.exists(sp["pool"]):
            print(f"[eval] seed{s} 池缺失，跳过（先 --stage gen）")
            continue
        r = eval_pool_file(f"regen_onto_seed{s}", sp["pool"])
        if r is not None:
            draws[f"seed{s}"] = r
            present.append(f"seed{s}")


    ua = json.load(open(os.path.join(RESULTS_DIR, "p5_union_attr.json"),
                        encoding="utf-8"))

    def stats(key, ref):
        vals = [round(d[key] - ref, 2) for d in draws.values() if key in d]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 \
            if len(vals) > 1 else 0.0
        same_sign = sum(1 for v in vals if (v > 0) == (vals[0] > 0))
        return {"values": vals, "mean": round(mean, 2), "std": round(std, 2),
                "min": round(min(vals), 2), "max": round(max(vals), 2),
                "same_sign_as_first": f"{same_sign}/{len(vals)}"}

    out = {
        "dataset": args.dataset, "model": args.model,
        "temperature": args.temperature, "n_draws": len(draws),
        "attr": attr, "draws": draws,
        "structure_effect": {
            "direct_vs_attr": stats("direct", attr["direct"]),
            "k1_vs_attr": stats("k1", attr["k1"]),
            "union_vs_attr": stats("union_direct", ua[args.dataset]),
        },
        "note": ("onto 生成重复采样（本体固定复用）；draw orig=p4_b1_feas 原始"
                 "onto 池；结构效应=onto-attr（direct/k1）或 union-union(attr)"
                 "（参照 p5_union_attr.json）；same_sign_as_first=各 draw 与首个"
                 "draw 同号的比例"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = os.path.join(RESULTS_DIR, f"p5_{args.dataset}_seeds.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"-> {out_path}")
    print(json.dumps(out["structure_effect"], indent=1, ensure_ascii=False))


def stage_ping(args):
    pass
    import traceback
    from e5_gen_pools import chat_once
    t0 = time.time()
    try:
        out = chat_once(args.api_base, args.api_key, args.model,
                        "Reply with the single word: pong", 0.7, 16,
                        timeout=60)
        print(f"[ping] OK  {time.time()-t0:.1f}s  model={args.model}")
        print(f"[ping] response: {out[:100]!r}")
        print("[ping] API 链路正常，可直接跑 --stage gen")
    except Exception as e:
        print(f"[ping] FAILED  {time.time()-t0:.1f}s")
        traceback.print_exc()
        print(f"\n[ping] 请求地址: {args.api_base.rstrip('/')}/chat/completions")
        print(f"[ping] model={args.model}  key前缀={args.api_key[:8]}...")
        print("[ping] 常见原因: key 无效(401) / 模型名不对(404/400) / "
              "代理或防火墙(timeout/URLError)")




def main():
    ap = argparse.ArgumentParser(description="seed stability of the structure effect")
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--stage", default="all",
                    choices=["ping", "gen", "eval", "all"])
    ap.add_argument("--seeds", type=int, default=2,
                    help="新增重采样次数（默认 2，连同 orig 共 3 个 draw）")

    ap.add_argument("--api-base",
                    default=os.environ.get("E5_API_BASE",
                                           "https://az.gptplus5.com/v1"))
    ap.add_argument("--model",
                    default=os.environ.get("E5_MODEL", "gpt-5.4"))
    ap.add_argument("--api-key",
                    default=os.environ.get("E5_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.api_key:
        print("[main] 缺 API key：请传 --api-key sk-xxx，"
              "或设环境变量 $env:E5_API_KEY='sk-xxx'（与 p4_b1_feas 一致）")
        sys.exit(1)

    if args.stage == "ping":
        stage_ping(args)
        return

    classes = load_classes(args.dataset)
    P = paths_for(args.dataset)
    suite = build_suite(args.dataset)

    if args.stage in ("gen", "all"):
        stage_gen(args, classes, P, suite)
    if args.stage in ("eval", "all"):
        stage_eval(args, classes, P)


if __name__ == "__main__":
    main()
