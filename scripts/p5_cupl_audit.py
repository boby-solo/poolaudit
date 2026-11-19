
pass

import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e5_probe import load_test, _norm
from p4_b1_feas import DESC_DIR, RESULTS_DIR, DATASETS

POOL_FILES = {
    "incumbent": "{ds}.json",
    "attr": "{ds}_regen_attr_gpt54.json",
    "instr": "{ds}_regen_attrrel.json",
    "onto": "{ds}_regen_onto.json",
    "magent": "{ds}_regen_magent.json",
    "omni": "{ds}_regen_omni.json",
    "cupl_re": "{ds}_regen_cupl_gpt54.json",
}

BARE_TEMPLATES = {
    "aircraft": "a photo of a {}, a type of aircraft.",
    "euro_sat": "a photo of a {}, a type of satellite image.",
    "dtd": "a photo of a {} texture.",
    "cub": "a photo of a {}, a type of bird.",
}


D1_CUB_REF = 52.78


def name_rates(ds):
    out = {}
    for pool_name, tpl in POOL_FILES.items():
        path = os.path.join(DESC_DIR, tpl.format(ds=ds))
        if not os.path.exists(path):
            print(f"[skip] {pool_name}: {path} 不存在")
            continue
        pool = json.load(open(path, encoding="utf-8"))
        keys_norm = [_norm(k) for k in pool]
        total = self_hits = any_hits = 0
        for cls, sents in pool.items():
            own = _norm(cls)
            for s in sents:
                total += 1
                low = _norm(s)
                if own and own in low:
                    self_hits += 1
                    any_hits += 1
                    continue
                if any(kn and kn != own and kn in low for kn in keys_norm):
                    any_hits += 1
        out[pool_name] = {
            "n_sentences": total,
            "self_rate": round(self_hits / total * 100, 1),
            "any_name_rate": round(any_hits / total * 100, 1),
            "no_name_rate": round((total - any_hits) / total * 100, 1),
        }
    return out


def stage_text():
    res = {"method_note": "类名对句子做 _norm(小写、去连字符/下划线) 子串匹配；"
                          "self=含自身类名；any=含池键名中任一类名；"
                          "no_name=不含任何类名。口径同 p5_comparative.json。",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    for ds in DATASETS:
        res[ds] = name_rates(ds)
        print(f"{ds}:")
        for p, r in res[ds].items():
            print(f"  {p:10s} self={r['self_rate']:5.1f}%  any={r['any_name_rate']:5.1f}%"
                  f"  no_name={r['no_name_rate']:5.1f}%  (n={r['n_sentences']})")
    out = os.path.join(RESULTS_DIR, "p5_cupl_audit.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"-> {out}")


def stage_bare():
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "vendor", "clip"))
    from vendor.clip import clip
    from finegrained_run import encode_texts

    model, _ = clip.load("ViT-B/32", device="cpu",
                         download_root=os.path.expanduser(r"~\.cache\clip"))
    model.eval()

    res = {"protocol": "单一手写模板；CLIP ViT-B/32 冻结；L2 归一文本特征即类原型；"
                       "缓存图像特征直评。与 d1_baseline.json（CUB 52.78）同口径。",
           "templates": BARE_TEMPLATES,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    for ds in DATASETS:
        feats, labels, cls_list = load_test(ds)
        texts = [BARE_TEMPLATES[ds].format(c) for c in cls_list]
        tf = encode_texts(model, texts)
        pred = (feats @ tf.t()).argmax(1)
        top1 = round((pred == labels).float().mean().item() * 100, 2)
        res[ds] = {"top1": top1, "n_classes": len(cls_list)}
        print(f"{ds}: bare template top1 = {top1:.2f}  ({len(cls_list)} classes)")
        if ds == "cub":
            print(f"  [check] d1_baseline 参照 {D1_CUB_REF} -> "
                  f"{'OK' if abs(top1 - D1_CUB_REF) < 0.05 else 'MISMATCH'}")
    out = os.path.join(RESULTS_DIR, "p5_bare_template.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"-> {out}")


def main():
    print("== 类名自含/无类名句比例（纯文本，秒级）==")
    stage_text()
    print("\n== 裸类名模板基线（CLIP 缓存，分钟级）==")
    stage_bare()


if __name__ == "__main__":
    main()
