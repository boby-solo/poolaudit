
pass

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from e5_gen_pools import (
    load_classes, DEFAULT_N, PROMPTS, chat_once, parse_desc_array,
)
from p4_b1_feas import gen_max_tokens, CUBL_ALIAS, paths_for

DESC_DIR = os.path.join(REPO, "cupl_descriptions")
RESULTS_DIR = os.path.join(REPO, "results")
CACHE = os.path.join(REPO, "cache")
DATASETS = ("aircraft", "euro_sat", "dtd", "cub")


CUPL_PROMPT = """What are useful visual features to distinguish a {cls}?

Write {n} short descriptions of the useful visual features that distinguish this class. Each description is one full sentence in plain English. There are no other constraints on content or wording.

Output: ONLY a JSON array of {n} strings, nothing else."""

KIND = {"aircraft": "aircraft variant", "euro_sat": "land-use",
        "dtd": "texture", "cub": "bird species"}


ATTR_FILL = """You are helping build a zero-shot image classifier (CLIP) that matches images to {kind} classes by comparing images with text descriptions.

{kind} class: "{cls}"

These descriptions have already been written:
{existing}

Write exactly {k} MORE short descriptions of this class. Each description:
- ONE full sentence, starting with the class name.
- Describes ONE visually verifiable attribute that is NOT covered by the existing descriptions.
- Must not repeat, paraphrase, or restate any existing description.

STRICTLY FORBIDDEN: performance or technical specs, marketing or subjective wording, scene or context content, photography or meta language, non-visual facts.

Output: ONLY a JSON array of {k} strings, nothing else."""

CUPL_FILL = """What are useful visual features to distinguish a {cls}?

These descriptions have already been written:
{existing}

Write exactly {k} MORE short descriptions of useful visual features that distinguish this class. Each description is one full sentence in plain English, must NOT repeat or paraphrase any existing description, and there are no other constraints on content or wording.

Output: ONLY a JSON array of {k} strings, nothing else."""


def pool_paths(ds, tag):
    return {
        "attr": os.path.join(DESC_DIR, f"{ds}_regen_attr_{tag}.json"),
        "attr_raw": os.path.join(RESULTS_DIR, f"p4_e5_{ds}_attr_raw_{tag}.json"),
        "cupl": os.path.join(DESC_DIR, f"{ds}_regen_cupl_{tag}.json"),
        "cupl_raw": os.path.join(RESULTS_DIR, f"p5_model2_{ds}_cupl_raw_{tag}.json"),
    }


def _call_parse(prompt, n, args, what):
    pass
    last = None
    for attempt in range(3):
        try:
            raw = chat_once(args.api_base, args.api_key, args.model, prompt,
                            args.temperature,
                            args.max_tokens or gen_max_tokens(n))
            descs, dropped = parse_desc_array(raw, n)
            if len(descs) < max(5, n // 3):
                raise ValueError(f"parsed {len(descs)}/{n}")
            return descs, raw, dropped
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{what}: {last}")


def gen_pool(ds, arm, tag, args, classes):
    n = DEFAULT_N[ds]
    pp = pool_paths(ds, tag)
    out_path, raw_path = pp[arm], pp[f"{arm}_raw"]
    pool, raw_log = {}, {}
    if os.path.exists(out_path):
        pool = json.load(open(out_path, encoding="utf-8"))
        print(f"[resume] {ds}/{arm} 已有 {len(pool)}/{len(classes)} 类")
    if os.path.exists(raw_path):
        raw_log = json.load(open(raw_path, encoding="utf-8"))
    todo = [c for c in classes if c not in pool]
    tpl = PROMPTS[ds] if arm == "attr" else CUPL_PROMPT
    print(f"[gen] {ds}/{arm}: {len(todo)} 类待生成 | n={n} | "
          f"model={args.model} T={args.temperature} workers={args.workers}")

    def one(cls):
        prompt = tpl.format(cls=cls, n=n)
        descs, raw, dropped = _call_parse(prompt, n, args, f"{arm}:{cls}")
        return cls, descs, raw, dropped

    failed, t0, done = [], time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, c): c for c in todo}
        for fut in as_completed(futs):
            try:
                cls, descs, raw, dropped = fut.result()
                pool[cls] = descs
                raw_log[cls] = {"raw": raw, "n_parsed": len(descs),
                                "n_dropped": dropped}
                done += 1
            except Exception as e:
                failed.append(str(e))
            if done and done % 20 == 0:
                json.dump(pool, open(out_path, "w", encoding="utf-8"),
                          indent=2, ensure_ascii=False)
                json.dump(raw_log, open(raw_path, "w", encoding="utf-8"),
                          indent=1, ensure_ascii=False)
                rate = done / max(time.time() - t0, 1e-9)
                left = (len(todo) - done) / max(rate, 1e-9) / 60
                print(f"  {ds}/{arm} {done}/{len(todo)} "
                      f"({rate:.2f} 类/s, 剩余 ~{left:.0f} 分钟)", flush=True)

    json.dump(pool, open(out_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump(raw_log, open(raw_path, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    cnts = [len(v) for v in pool.values()]
    print(f"-> {out_path} | {len(pool)}/{len(classes)} 类, 失败 {len(failed)}")
    if cnts:
        print(f"   条数 min/mean/max = {min(cnts)}/{sum(cnts)/len(cnts):.1f}/{max(cnts)}")
    for f in failed[:5]:
        print("   FAIL:", f)
    return len(pool), len(failed)


def _fill_call(prompt, k, args, what):
    last = None
    for attempt in range(3):
        try:
            raw = chat_once(args.api_base, args.api_key, args.model, prompt,
                            args.temperature,
                            args.max_tokens or max(800, 60 * k))
            descs, dropped = parse_desc_array(raw, k)
            if descs:
                return descs, raw, dropped
            raise ValueError("empty parse")
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{what}: {last}")


def fill_pool(ds, arm, tag, args, classes):
    pass
    n = DEFAULT_N[ds]
    pp = pool_paths(ds, tag)
    pool = json.load(open(pp[arm], encoding="utf-8"))
    if len(pool) < len(classes):
        print(f"[fill] {ds}/{arm} 池不完整 {len(pool)}/{len(classes)}，先跑 gen")
        return
    raw_path = os.path.join(RESULTS_DIR,
                            f"p5_model2_{ds}_{arm}_fill_raw_{tag}.json")
    fill_raw = {}
    if os.path.exists(raw_path):
        fill_raw = json.load(open(raw_path, encoding="utf-8"))
    todo = [c for c in classes if len(pool.get(c, [])) < n]
    print(f"[fill] {ds}/{arm}: {len(todo)} 类短于 {n} 条，需补差额")
    tpl = ATTR_FILL if arm == "attr" else CUPL_FILL

    def one(cls):
        existing = list(pool.get(cls, []))
        rounds = []
        for _ in range(3):
            need = n - len(existing)
            if need <= 0:
                break
            prompt = tpl.format(kind=KIND[ds], cls=cls, k=need,
                                existing=json.dumps(existing, ensure_ascii=False))
            descs, raw, dropped = _fill_call(
                prompt, need, args, f"fill:{arm}:{cls}")
            seen = {s.casefold() for s in existing}
            added = []
            for s in descs:
                if s.casefold() in seen:
                    continue
                existing.append(s)
                seen.add(s.casefold())
                added.append(s)
                if len(existing) >= n:
                    break
            rounds.append({"need": need, "n_added": len(added),
                           "n_dropped": dropped, "raw": raw})
            if not added:
                break
        pool[cls] = existing[:n]
        return cls, rounds

    failed, t0, done = [], time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, c): c for c in todo}
        for fut in as_completed(futs):
            try:
                cls, rounds = fut.result()
                fill_raw[cls] = rounds
                done += 1
            except Exception as e:
                failed.append(str(e))
            if done and done % 20 == 0:
                json.dump(dict(pool), open(pp[arm], "w", encoding="utf-8"),
                          indent=2, ensure_ascii=False)
                json.dump(fill_raw, open(raw_path, "w", encoding="utf-8"),
                          indent=1, ensure_ascii=False)
                rate = done / max(time.time() - t0, 1e-9)
                left = (len(todo) - done) / max(rate, 1e-9) / 60
                print(f"  {ds}/{arm} fill {done}/{len(todo)} "
                      f"({rate:.2f} 类/s, 剩余 ~{left:.0f} 分钟)", flush=True)


    for cls in list(pool):
        pool[cls] = pool[cls][:n]
    json.dump(dict(pool), open(pp[arm], "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    json.dump(fill_raw, open(raw_path, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    cnts = [len(v) for v in pool.values()]
    short = sum(x < n for x in cnts)
    print(f"-> {pp[arm]} | 仍短于 {n}: {short} 类 | 失败 {len(failed)}")
    print(f"   条数 min/mean/max = {min(cnts)}/{sum(cnts)/len(cnts):.1f}/{max(cnts)}")
    for f in failed[:5]:
        print("   FAIL:", f)




def eval_dataset(ds, tag, model_name):
    import torch
    from e5_probe import load_test, encode_pool, mean_protos, k1_joint_fast, _norm
    from p5_stats import mcnemar_exact
    sys.path.insert(0, os.path.join(REPO, "vendor", "clip"))
    from vendor.clip import clip

    pp = pool_paths(ds, tag)
    gen_classes = load_classes(ds)
    for arm in ("attr", "cupl"):
        if not os.path.exists(pp[arm]):
            raise FileNotFoundError(pp[arm])
        got = len(json.load(open(pp[arm], encoding="utf-8")))
        if got != len(gen_classes):
            raise RuntimeError(f"{ds}/{arm} 池不完整 {got}/{len(gen_classes)}；"
                               f"先补 --stage gen")

    feats, labels, cls_list = load_test(ds)
    name2idx = {_norm(c): i for i, c in enumerate(cls_list)}
    model, _ = clip.load("ViT-B/32", device="cpu",
                         download_root=os.path.expanduser(r"~\.cache\clip"))
    model.eval()


    cupl_descs = json.load(open(os.path.join(DESC_DIR, f"{ds}.json"),
                                encoding="utf-8"))
    cupl_keys = list(cupl_descs.keys())
    cupl_feats = torch.load(os.path.join(CACHE, f"{ds}_desc_vitb32.pt"),
                            map_location="cpu", weights_only=False)["feats"].float()
    cupl_spans, s0 = [], 0
    for k in cupl_keys:
        nk = len(cupl_descs[k])
        cupl_spans.append((s0, s0 + nk))
        s0 += nk
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

    def metrics(df, spans, remap, with_union=True):
        protos = mean_protos(df, spans, remap)
        pred = (feats @ protos.t()).argmax(1)
        direct = (pred == labels).float().mean().item() * 100
        k1 = k1_joint_fast(feats, labels, df, spans, remap) * 100
        uni = union_with_incumbent(df, spans, remap) if with_union else None
        counts = [e - s for s, e in spans]
        return {
            "pred": pred, "direct": round(direct, 2), "k1": round(k1, 2),
            "union_direct": round(uni, 2) if uni is not None else None,
            "desc_per_class": {"min": min(counts),
                               "mean": round(sum(counts) / len(counts), 1),
                               "max": max(counts)},
        }


    m2 = {}
    for arm in ("attr", "cupl"):
        df, spans, remap = encode_pool(model, ds, f"regen_{arm}_{tag}",
                                       pp[arm], cls_list)
        if len(spans) != len(cls_list):
            raise RuntimeError(f"{ds}/{arm} spans {len(spans)} != classes {len(cls_list)}")
        m2[arm] = metrics(df, spans, remap)


    feas = json.load(open(paths_for(ds)["feas"], encoding="utf-8"))
    attr_gpt = feas["baseline"]["attr"]
    cupl_json = json.load(open(os.path.join(RESULTS_DIR, f"p5_cupl_{ds}.json"),
                               encoding="utf-8"))
    cupl_gpt = cupl_json["cupl_gpt54"]
    incumbent = feas["baseline"]["cupl"]
    bare = json.load(open(os.path.join(RESULTS_DIR, "p5_bare_template.json"),
                          encoding="utf-8"))[ds]["top1"]

    gpt = {}
    for arm, path, pname in (
            ("attr", paths_for(ds)["attr_pool"], "regen_attr_gpt54"),
            ("cupl", os.path.join(DESC_DIR, f"{ds}_regen_cupl_gpt54.json"),
             "regen_cupl_gpt54")):
        df, spans, remap = encode_pool(model, ds, pname, path, cls_list)
        pred = (feats @ mean_protos(df, spans, remap).t()).argmax(1)
        direct = (pred == labels).float().mean().item() * 100
        gpt[arm] = {"pred": pred, "direct": round(direct, 2)}
        stored = attr_gpt["direct"] if arm == "attr" else cupl_gpt["direct"]
        if abs(direct - stored) > 0.3:
            print(f"  [WARN] {ds}/{arm} 直评复算 {direct:.2f} != 存档 {stored:.2f}")

    def mc(a_correct, b_correct):
        a = a_correct.bool()
        b = b_correct.bool()
        n01 = int((a & ~b).sum())
        n10 = int((~a & b).sum())
        return {"n01": n01, "n10": n10,
                "p_exact": round(mcnemar_exact(n01, n10), 6),
                "delta_pt": round((b.float().mean() - a.float().mean()).item() * 100, 2)}

    ca = m2["attr"]["pred"] == labels
    cc = m2["cupl"]["pred"] == labels
    ga = gpt["attr"]["pred"] == labels
    gc = gpt["cupl"]["pred"] == labels
    tests = {
        "attr_m2_vs_cupl_m2": mc(ca, cc),
        "attr_gpt54_vs_cupl_gpt54": mc(ga, gc),
        "attr_m2_vs_attr_gpt54": mc(ga, ca),
        "cupl_m2_vs_cupl_gpt54": mc(gc, cc),
    }

    gaps = {
        "protocol_m2": round(m2["attr"]["direct"] - m2["cupl"]["direct"], 2),
        "protocol_gpt54": round(attr_gpt["direct"] - cupl_gpt["direct"], 2),
        "model_attr": round(m2["attr"]["direct"] - attr_gpt["direct"], 2),
        "model_cupl": round(m2["cupl"]["direct"] - cupl_gpt["direct"], 2),
        "attr_m2_minus_incumbent": round(m2["attr"]["direct"] - incumbent["direct"], 2),
        "cupl_m2_minus_incumbent": round(m2["cupl"]["direct"] - incumbent["direct"], 2),
        "cupl_m2_minus_bare": round(m2["cupl"]["direct"] - bare, 2),
    }

    return {
        "dataset": ds, "tag": tag, "model": model_name,
        "temperature": 0.7, "n_desc_per_class": DEFAULT_N[ds],
        "n_classes": len(cls_list), "n_test": int(labels.numel()),
        "arms": {
            "attr": {k: m2["attr"][k] for k in
                     ("direct", "k1", "union_direct", "desc_per_class")},
            "cupl": {k: m2["cupl"][k] for k in
                     ("direct", "k1", "union_direct", "desc_per_class")},
        },
        "refs": {
            "incumbent": incumbent,
            "attr_gpt54": attr_gpt,
            "cupl_gpt54": cupl_gpt,
            "bare_template": bare,
        },
        "gaps": gaps,
        "mcnemar": tests,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def summarize(results, tag, model_name, expected):
    rows = []
    for ds in DATASETS:
        r = results.get(ds)
        if not r:
            continue
        rows.append({
            "ds": ds,
            "attr": r["arms"]["attr"]["direct"],
            "cupl": r["arms"]["cupl"]["direct"],
            "protocol": r["gaps"]["protocol_m2"],
            "attr_gpt": r["refs"]["attr_gpt54"]["direct"],
            "cupl_gpt": r["refs"]["cupl_gpt54"]["direct"],
            "model_attr": r["gaps"]["model_attr"],
            "model_cupl": r["gaps"]["model_cupl"],
            "p_prot": r["mcnemar"]["attr_m2_vs_cupl_m2"]["p_exact"],
            "p_ma": r["mcnemar"]["attr_m2_vs_attr_gpt54"]["p_exact"],
            "p_mc": r["mcnemar"]["cupl_m2_vs_cupl_gpt54"]["p_exact"],
            "cupl_below_bare": r["gaps"]["cupl_m2_minus_bare"] < 0,
        })

    pos = [x["ds"] for x in rows if x["protocol"] > 0]
    dom = [x["ds"] for x in rows if abs(x["protocol"]) >
           max(abs(x["model_attr"]), abs(x["model_cupl"]))]
    below = [x["ds"] for x in rows if x["cupl_below_bare"]]
    n = len(rows)
    if n and len(pos) == n and len(dom) >= max(3, n - 1) and len(below) >= 3:
        verdict = "protocol_dominates (strong)"
    elif n and len(pos) >= 3 and len(dom) >= 3:
        verdict = "protocol_dominates (partial)"
    else:
        verdict = "mixed / do_not_overclaim"

    summary = {
        "tag": tag, "model": model_name,
        "n_datasets_expected": len(expected), "n_datasets_done": n,
        "missing": [d for d in expected if d not in results],
        "protocol_positive": pos,
        "protocol_dominates_model": dom,
        "cupl_below_bare": below,
        "verdict": verdict,
        "rows": rows,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = os.path.join(RESULTS_DIR, f"p5_model2_summary_{tag}.json")
    json.dump(summary, open(out, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    print("\n===== 第二模型复现汇总 =====")
    print(f"model={model_name} tag={tag}  {n}/{len(expected)} 数据集")
    hdr = ("数据集", "attr_M2", "cupl_M2", "协议差", "attr_GPT", "cupl_GPT",
           "模型差(attr)", "模型差(cupl)", "p协议", "p模型attr", "p模型cupl", "cupl<裸模板")
    print(("{:<9}{:>8}{:>8}{:>8}{:>9}{:>9}{:>11}{:>11}"
           "{:>9}{:>11}{:>11}{:>11}").format(*hdr))
    for x in rows:
        print(("{:<9}{:>8.2f}{:>8.2f}{:>+8.2f}{:>9.2f}{:>9.2f}"
               "{:>+11.2f}{:>+11.2f}{:>9.4f}{:>11.4f}{:>11.4f}{:>11}"
               ).format(x["ds"], x["attr"], x["cupl"], x["protocol"],
                        x["attr_gpt"], x["cupl_gpt"], x["model_attr"],
                        x["model_cupl"], x["p_prot"], x["p_ma"], x["p_mc"],
                        "是" if x["cupl_below_bare"] else "否"))
    print(f"\n协议为正: {len(pos)}/{n} {pos}")
    print(f"协议差>模型差: {len(dom)}/{n} {dom}")
    print(f"cupl 低于裸模板: {len(below)}/{n} {below}")
    print(f"判读: {verdict}")
    print(f"-> {out}")
    return summary


def print_plan(dss, arms, tag, model_name, limit):
    total_calls = total_sent = 0
    print(f"计划：model={model_name or '<未指定>'} tag={tag} arms={arms}")
    for ds in dss:
        classes = load_classes(ds)
        k = min(len(classes), limit) if limit else len(classes)
        n = DEFAULT_N[ds]
        calls = k * len(arms)
        sents = calls * n
        total_calls += calls
        total_sent += sents
        pp = pool_paths(ds, tag)
        print(f"  {ds:<9} {k:>3} 类 × {n:>2} 句 × {len(arms)} 臂 = "
              f"{calls:>3} 次调用, {sents:>5} 句")
        for arm in arms:
            print(f"      {arm:<4} -> {pp[arm]}")
    print(f"合计：{total_calls} 次调用，{total_sent} 句。"
          f"（8 并发、单次 5 秒约 {total_calls/8*5/60:.0f} 分钟；单次 20 秒约 "
          f"{total_calls/8*20/60:.0f} 分钟；不含 CLIP 编码与 k1）")
    if limit:
        print("警告：--limit-classes 只用于冒烟，池不完整，--stage eval 会拒绝。")


def main():
    ap = argparse.ArgumentParser(description="second-generator reproduction (attr + cupl-re)")
    ap.add_argument("--dataset", default="all", choices=["all"] + list(DATASETS))
    ap.add_argument("--stage", default="all",
                    choices=["plan", "gen", "fill", "eval", "all"])
    ap.add_argument("--tag", required=True,
                    help="输出后缀，如 qwen3_32b；不得为 gpt54")
    ap.add_argument("--model", default=os.environ.get("E5_MODEL", ""),
                    help="API 模型 ID（生成阶段必填）")
    ap.add_argument("--api-base", default=os.environ.get(
        "E5_API_BASE", "https://az.gptplus5.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get(
        "E5_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0=按条数自动（max(2500, 60*n)）；接口上限较小时可手动调低")
    ap.add_argument("--arms", default="attr,cupl",
                    help="逗号分隔：attr,cupl")
    ap.add_argument("--limit-classes", type=int, default=0,
                    help="冒烟用：每数据集只跑前 N 类；0=全部")
    args = ap.parse_args()

    if args.tag == "gpt54" or args.tag.startswith("gpt54_"):
        sys.exit("tag must not be gpt54: it would overwrite the existing pools. Use another tag.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        sys.exit("tag 只能含字母、数字、下划线、点、连字符")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ("attr", "cupl")]
    if bad:
        sys.exit(f"未知 arm: {bad}（只支持 attr,cupl）")
    dss = list(DATASETS) if args.dataset == "all" else [args.dataset]

    if args.stage == "plan":
        print_plan(dss, arms, args.tag, args.model, args.limit_classes)
        return

    if args.stage in ("gen", "fill", "all"):
        if not args.model:
            sys.exit("生成/补齐阶段缺 --model（或设 E5_MODEL）")
        if not args.api_key:
            sys.exit("生成/补齐阶段缺 API key（--api-key 或 E5_API_KEY）")

    if args.stage in ("gen", "all"):
        for ds in dss:
            classes = load_classes(ds)
            if args.limit_classes:
                classes = classes[:args.limit_classes]
            for arm in arms:
                gen_pool(ds, arm, args.tag, args, classes)

    if args.stage in ("fill", "all"):
        for ds in dss:
            classes = load_classes(ds)
            if args.limit_classes:
                classes = classes[:args.limit_classes]
            for arm in arms:
                fill_pool(ds, arm, args.tag, args, classes)

    if args.stage in ("eval", "all"):
        results, expected = {}, []
        for ds in dss:
            expected.append(ds)
            pp = pool_paths(ds, args.tag)
            if not all(os.path.exists(pp[a]) for a in ("attr", "cupl")):
                print(f"[skip] {ds}: 池缺失")
                continue
            try:
                r = eval_dataset(ds, args.tag, args.model or args.tag)
            except Exception as e:
                print(f"[skip] {ds}: {e}")
                continue
            out = os.path.join(RESULTS_DIR,
                               f"p5_model2_{ds}_{args.tag}.json")
            json.dump(r, open(out, "w", encoding="utf-8"),
                      indent=1, ensure_ascii=False)
            results[ds] = r
            print(f"-> {out}")
            print(f"  attr {r['arms']['attr']['direct']:.2f} | "
                  f"cupl {r['arms']['cupl']['direct']:.2f} | "
                  f"协议差 {r['gaps']['protocol_m2']:+.2f} | "
                  f"模型差 attr {r['gaps']['model_attr']:+.2f}, "
                  f"cupl {r['gaps']['model_cupl']:+.2f}")
        if results:
            summarize(results, args.tag, args.model or args.tag, expected)
        if len(results) < len(expected):
            sys.exit(1)


if __name__ == "__main__":
    main()
