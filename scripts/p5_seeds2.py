
pass
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p4_b1_feas import (
    call_retry, gen_max_tokens, build_suite, build_omni_suite, paths_for,
    parse_json_object, DATASETS, DESC_DIR, RESULTS_DIR, CUBL_ALIAS,
)
from e5_gen_pools import load_classes, DEFAULT_N, parse_desc_array, PROMPTS


def arm_paths(ds, arm, s):
    return {
        "pool": os.path.join(DESC_DIR, f"{ds}_regen_{arm}_seed{s}.json"),
        "raw": os.path.join(RESULTS_DIR, f"p5_{ds}_{arm}_seed{s}_raw.json"),
        "out": os.path.join(RESULTS_DIR, f"p5_{ds}_{arm}_seeds2.json"),
    }




def stage_gen(args, classes, P, suite, omni_suite):
    onto = json.load(open(P["onto"], encoding="utf-8"))
    missing = [c for c in classes if c not in onto]
    if missing:
        print(f"[gen] 本体缺 {len(missing)} 类，先跑 p4_b1_feas --stage onto")
        sys.exit(1)
    n_desc = DEFAULT_N[args.dataset]
    all_classes = ", ".join(classes)
    gen_tpl = PROMPTS[args.dataset]

    def one_onto(cls):
        p = suite["onto_gen"].format(cls=cls, n=n_desc, **suite["dk"],
                                     **suite["onto_fields"](onto[cls]))
        raw = call_retry(args.api_base, args.api_key, args.model, p,
                         args.temperature, gen_max_tokens(n_desc),
                         f"seed:gen:{cls}")
        descs, dropped = parse_desc_array(raw, n_desc)
        if len(descs) < max(5, n_desc // 3):
            raise ValueError(f"{cls}: parsed only {len(descs)}")
        return cls, descs, {"raw": raw, "n_parsed": len(descs), "n_dropped": dropped}

    def one_magent(cls):
        raw_g = call_retry(args.api_base, args.api_key, args.model,
                           gen_tpl.format(cls=cls, n=n_desc),
                           args.temperature, gen_max_tokens(n_desc), f"gen:{cls}")
        drafts, _ = parse_desc_array(raw_g, n_desc)
        if len(drafts) < max(5, n_desc // 3):
            raise ValueError(f"{cls}: drafts {len(drafts)}")
        numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(drafts))
        raw_c = call_retry(args.api_base, args.api_key, args.model,
                           suite["critic"].format(all_classes=all_classes,
                                                  cls=cls, drafts=numbered,
                                                  **suite["dk"]),
                           args.temperature, 1800, f"critic:{cls}")
        critique = parse_json_object(raw_c) or {}
        raw_r = call_retry(args.api_base, args.api_key, args.model,
                           suite["refiner"].format(
                               cls=cls,
                               confusable=", ".join(critique.get("confusable", [])[:4]),
                               drafts=numbered,
                               too_generic=critique.get("too_generic", []) or "none",
                               likely_wrong=critique.get("likely_wrong", []) or "none",
                               missing="; ".join(critique.get("missing", [])[:4]) or "none",
                               n=n_desc, **suite["dk"]),
                           args.temperature, gen_max_tokens(n_desc), f"refine:{cls}")
        descs, dropped = parse_desc_array(raw_r, n_desc)
        if len(descs) < max(5, n_desc // 3):
            descs = drafts
        return cls, descs, {"gen": raw_g, "critique": raw_c, "refined": raw_r,
                            "n_final": len(descs), "fallback": descs is drafts}

    def one_omni(cls):
        f = suite["onto_fields"](onto[cls])
        raw_g = call_retry(args.api_base, args.api_key, args.model,
                           suite["onto_gen"].format(cls=cls, n=n_desc,
                                                    **suite["dk"], **f),
                           args.temperature, gen_max_tokens(n_desc), f"gen:{cls}")
        drafts, _ = parse_desc_array(raw_g, n_desc)
        if len(drafts) < max(5, n_desc // 3):
            raise ValueError(f"{cls}: drafts {len(drafts)}")
        numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(drafts))
        raw_c = call_retry(args.api_base, args.api_key, args.model,
                           omni_suite["critic_omni"].format(
                               all_classes=all_classes, cls=cls,
                               drafts=numbered, **omni_suite["dk"], **f),
                           args.temperature, 1800, f"critic:{cls}")
        critique = parse_json_object(raw_c) or {}
        raw_r = call_retry(args.api_base, args.api_key, args.model,
                           omni_suite["refiner_omni"].format(
                               cls=cls, drafts=numbered,
                               too_generic=critique.get("too_generic", []) or "none",
                               likely_wrong=critique.get("likely_wrong", []) or "none",
                               missing="; ".join(critique.get("missing", [])[:4]) or "none",
                               n=n_desc, **omni_suite["dk"], **f),
                           args.temperature, gen_max_tokens(n_desc), f"refine:{cls}")
        descs, dropped = parse_desc_array(raw_r, n_desc)
        if len(descs) < max(5, n_desc // 3):
            descs = drafts
        return cls, descs, {"gen": raw_g, "critique": raw_c, "refined": raw_r,
                            "n_final": len(descs), "fallback": descs is drafts}

    def one_attr(cls):
        raw = call_retry(args.api_base, args.api_key, args.model,
                         gen_tpl.format(cls=cls, n=n_desc),
                         args.temperature, gen_max_tokens(n_desc), f"gen:{cls}")
        descs, dropped = parse_desc_array(raw, n_desc)
        if len(descs) < max(5, n_desc // 3):
            raise ValueError(f"{cls}: parsed only {len(descs)}")
        return cls, descs, {"raw": raw, "n_parsed": len(descs), "n_dropped": dropped}

    one_fn = {"onto": one_onto, "magent": one_magent, "omni": one_omni,
              "attr": one_attr}[args.arm]

    for s in range(1, args.seeds + 1):
        sp = arm_paths(args.dataset, args.arm, s)
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
        print(f"[seed {s}] {args.arm} 生成：{len(todo)} 类 | "
              f"model={args.model} T={args.temperature}")
        failed = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(one_fn, c): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    c2, descs, raw = fut.result()
                    pool[c2] = descs
                    raw_log[c2] = raw
                except Exception as e:
                    failed.append(str(e))
                if i % 20 == 0:
                    json.dump(pool, open(sp["pool"], "w", encoding="utf-8"),
                              indent=2, ensure_ascii=False)
                    print(f"  seed{s} {i}/{len(todo)}（失败 {len(failed)}）", flush=True)
        json.dump(pool, open(sp["pool"], "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        json.dump(raw_log, open(sp["raw"], "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        print(f"[seed {s}] -> {sp['pool']}（{len(pool)}/{len(classes)}，失败 {len(failed)}）")
        for f in failed[:5]:
            print("  FAIL:", f)




def mcnemar_exact(n01, n10):
    n = n01 + n10
    if n == 0:
        return 1.0
    m = min(n01, n10)
    logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            for k in range(m + 1)]
    mx = max(logs)
    log_sum = mx + math.log(sum(math.exp(l - mx) for l in logs))
    return min(1.0, 2.0 * math.exp(log_sum - n * math.log(2)))


def stage_eval(args, classes, P):
    import torch
    from e5_probe import load_test, encode_pool, mean_protos, k1_joint_fast, _norm
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "vendor", "clip"))
    from vendor.clip import clip

    feats, labels, cls_list = load_test(args.dataset)
    name2idx = {_norm(c): i for i, c in enumerate(cls_list)}
    model, _ = clip.load("ViT-B/32", device="cpu",
                         download_root=os.path.expanduser(r"~\.cache\clip"))
    model.eval()


    cupl_descs = json.load(open(os.path.join(DESC_DIR, f"{args.dataset}.json"),
                                 encoding="utf-8"))
    cupl_keys = list(cupl_descs.keys())
    cupl_feats = torch.load(os.path.join(os.path.dirname(DESC_DIR), "cache",
                                         f"{args.dataset}_desc_vitb32.pt"),
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
        print("[eval] CuPL 键名经别名表对齐")
    cupl_pred = (feats @ mean_protos(cupl_feats, cupl_spans, cupl_remap).t()).argmax(1)
    cupl_acc = (cupl_pred == labels).float().mean().item() * 100


    df_a, spans_a, remap_a = encode_pool(model, args.dataset, "regen_attr_gpt54",
                                         P["attr_pool"], cls_list)
    attr_pred = (feats @ mean_protos(df_a, spans_a, remap_a).t()).argmax(1)
    attr_acc = (attr_pred == labels).float().mean().item() * 100

    feas = json.load(open(P["feas"], encoding="utf-8"))
    print(f"[{args.dataset}/{args.arm}] cupl={cupl_acc:.2f} attr={attr_acc:.2f} "
          f"(feas: {feas['baseline']['attr']['direct']}/{feas['baseline']['cupl']['direct']})")

    def union_pred_of(df, spans, remap):
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
        return (feats @ up.t()).argmax(1)

    def paired(pred_a, pred_b):
        n01 = int(((pred_a == labels) & (pred_b != labels)).sum())
        n10 = int(((pred_a != labels) & (pred_b == labels)).sum())
        return n01, n10

    def eval_pool_file(pool_name, pool_path):
        df, spans, remap = encode_pool(model, args.dataset, pool_name,
                                       pool_path, cls_list)
        if len(spans) != len(cls_list):
            print(f"[eval] {pool_name} 池不完整（{len(spans)}/{len(cls_list)} 类）")
            return None
        pred = (feats @ mean_protos(df, spans, remap).t()).argmax(1)
        acc = (pred == labels).float().mean().item() * 100
        k1 = k1_joint_fast(feats, labels, df, spans, remap) * 100
        upred = union_pred_of(df, spans, remap)
        uacc = (upred == labels).float().mean().item() * 100
        rec = {"direct": round(acc, 2), "k1": round(k1, 2),
               "union_direct": round(uacc, 2),
               "d_direct_vs_attr": round(acc - attr_acc, 2),
               "d_direct_vs_cupl": round(acc - cupl_acc, 2),
               "d_union_vs_cupl": round(uacc - cupl_acc, 2)}
        n01, n10 = paired(attr_pred, pred)
        rec["p_direct_vs_attr"] = round(mcnemar_exact(n01, n10), 6)
        n01, n10 = paired(cupl_pred, pred)
        rec["p_direct_vs_cupl"] = round(mcnemar_exact(n01, n10), 6)
        n01, n10 = paired(cupl_pred, upred)
        rec["p_union_vs_cupl"] = round(mcnemar_exact(n01, n10), 6)
        return rec

    draws = {}
    orig_pool = P[f"{args.arm}_pool"]
    r = eval_pool_file(f"regen_{args.arm}", orig_pool)
    if r is not None:
        draws["orig"] = r
    for s in range(1, args.seeds + 1):
        sp = arm_paths(args.dataset, args.arm, s)
        if not os.path.exists(sp["pool"]):
            continue
        pool_n = len(json.load(open(sp["pool"], encoding="utf-8")))
        if pool_n == 0:
            print(f"[eval] regen_{args.arm}_seed{s} 池为空，跳过")
            continue
        r = eval_pool_file(f"regen_{args.arm}_seed{s}", sp["pool"])
        if r is not None:
            draws[f"seed{s}"] = r

    def stats(key):
        vals = [d[key] for d in draws.values()]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 \
            if len(vals) > 1 else 0.0
        pos = sum(1 for v in vals if v > 0)
        return {"values": vals, "mean": round(mean, 2), "std": round(std, 2),
                "n_positive": f"{pos}/{len(vals)}"}

    out = {
        "dataset": args.dataset, "arm": args.arm,
        "model": args.model, "temperature": args.temperature,
        "n_draws": len(draws),
        "cupl_direct": round(cupl_acc, 2), "attr_direct": round(attr_acc, 2),
        "draws": draws,
        "summary": {k: stats(k) for k in
                    ("d_direct_vs_attr", "d_direct_vs_cupl", "d_union_vs_cupl")},
        "note": ("多臂种子重采样（本体固定复用）；orig=原始臂池；每draw含配对精确"
                 "McNemar（vs attr / vs 现役 / union vs 现役）"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    sp_out = arm_paths(args.dataset, args.arm, 1)
    json.dump(out, open(sp_out["out"], "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"-> {sp_out['out']}")
    for name, r in draws.items():
        print(f"  {name:6s} direct={r['direct']:.2f} vs_attr={r['d_direct_vs_attr']:+.2f}"
              f"(p={r['p_direct_vs_attr']}) vs_cupl={r['d_direct_vs_cupl']:+.2f}"
              f"(p={r['p_direct_vs_cupl']}) union_vs_cupl={r['d_union_vs_cupl']:+.2f}"
              f"(p={r['p_union_vs_cupl']})")
    print("  summary:", json.dumps(out["summary"], ensure_ascii=False))




def stage_ping(args):
    import traceback
    from e5_gen_pools import chat_once
    t0 = time.time()
    try:
        out = chat_once(args.api_base, args.api_key, args.model,
                        "Reply with the single word: pong", 0.7, 16, timeout=60)
        print(f"[ping] OK {time.time()-t0:.1f}s model={args.model} -> {out[:60]!r}")
    except Exception:
        print(f"[ping] FAILED {time.time()-t0:.1f}s")
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(description="multi-arm seed stability (level 1)")
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--arm", required=True,
                    choices=["onto", "magent", "omni", "attr"])
    ap.add_argument("--stage", default="all", choices=["ping", "gen", "eval", "all"])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--api-base",
                    default=os.environ.get("E5_API_BASE",
                                           "https://az.gptplus5.com/v1"))
    ap.add_argument("--model", default=os.environ.get("E5_MODEL", "gpt-5.4"))
    ap.add_argument("--api-key", default=os.environ.get("E5_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.stage == "ping":
        if not args.api_key:
            print("[ping] 缺 --api-key / $env:E5_API_KEY")
            sys.exit(1)
        stage_ping(args)
        return

    classes = load_classes(args.dataset)
    P = paths_for(args.dataset)
    if args.stage in ("gen", "all"):
        if not args.api_key:
            print("[gen] 缺 API key：--api-key sk-xxx 或 $env:E5_API_KEY")
            sys.exit(1)
        suite = build_suite(args.dataset)
        omni_suite = build_omni_suite(args.dataset)
        stage_gen(args, classes, P, suite, omni_suite)
    if args.stage in ("eval", "all"):
        stage_eval(args, classes, P)


if __name__ == "__main__":
    main()
