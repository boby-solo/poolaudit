pass

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vendor.clip import clip

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROAPO_DATA = os.environ.get("PROAPO_DATA", os.path.join(REPO, "data", "dataset"))

CFG = {
    "aircraft": {
        "data_root": os.path.join(PROAPO_DATA, "fgvc_aircraft"),
        "hand_template": "a photo of a {}, a type of aircraft.",
        "domain_word": "aircraft",
        "templates": [
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a {}, a type of aircraft.",
            "a close-up photo of a {}.",
            "a photo of one {}.",
            "the {} aircraft.",
            "an image of a {} aircraft.",
        ],
    },
    "cars": {
        "data_root": os.path.join(PROAPO_DATA, "stanford_cars"),
        "hand_template": "a photo of a {}, a type of car.",
        "domain_word": "car",
        "templates": [
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a {}, a type of car.",
            "a close-up photo of a {}.",
            "a photo of one {}.",
            "the {} car.",
            "an image of a {} car.",
        ],
    },
}


def _norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())




def load_aircraft_test():
    data = os.path.join(CFG["aircraft"]["data_root"], "fgvc-aircraft", "data")
    with open(os.path.join(data, "variants.txt"), encoding="utf-8") as f:
        classes = [l.strip() for l in f if l.strip()]
    c2i = {v: i for i, v in enumerate(classes)}
    with open(os.path.join(data, "images_variant_test.txt"), encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    paths, labels = [], []
    for line in lines:
        img_id, variant = line.split(None, 1)
        paths.append(os.path.join(data, "images", img_id + ".jpg"))
        labels.append(c2i[variant.strip()])
    return paths, labels, classes


def load_cars_test():
    root = os.path.join(CFG["cars"]["data_root"], "stanford_cars")
    split = json.load(open(os.path.join(CFG["cars"]["data_root"], "split_zhou_StanfordCars.json"), encoding="utf-8"))
    label2name = {}
    for part in ["train", "val", "test"]:
        for rel, lab, name in split[part]:
            label2name[int(lab)] = name
    labels_sorted = sorted(label2name)
    assert labels_sorted == list(range(len(labels_sorted))), "labels not contiguous 0..C-1"
    classes = [label2name[i] for i in labels_sorted]
    paths = [os.path.join(root, rel) for rel, lab, name in split["test"]]
    labels = [int(lab) for rel, lab, name in split["test"]]
    return paths, labels, classes




@torch.no_grad()
def encode_images(model, preprocess, paths, batch_size=32):
    feats = []
    t0 = time.time()
    for s in range(0, len(paths), batch_size):
        batch = [preprocess(Image.open(p).convert("RGB")) for p in paths[s : s + batch_size]]
        feats.append(model.encode_image(torch.stack(batch)).float())
        if (s // batch_size) % 25 == 0:
            print(f"  images {s + len(batch)}/{len(paths)}  ({time.time()-t0:.0f}s)", flush=True)
    f = torch.cat(feats)
    return f / f.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_texts(model, texts, batch=256):
    outs = []
    for s in range(0, len(texts), batch):
        tok = clip.tokenize(texts[s : s + batch], truncate=True)
        f = model.encode_text(tok).float()
        outs.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(outs)


def ens_protos(model, phrases, templates):
    texts = [t.format(p) for p in phrases for t in templates]
    f = encode_texts(model, texts)
    return f.reshape(len(phrases), len(templates), -1).mean(1)


def proto_of(f):
    p = f.mean(0)
    return p / p.norm()




def build_protos(desc_feats, spans, remap, masks=None, weights=None, agg="mean", trim=0.1):
    C = len(spans)
    protos = torch.zeros(C, desc_feats.shape[1])
    for i, (s, e) in enumerate(spans):
        f = desc_feats[s:e]
        if masks is not None:
            f = f[masks[i]]
        if weights is not None:
            p = (f * weights[i].view(-1, 1)).sum(0)
        elif agg == "mean":
            p = f.mean(0)
        elif agg == "median":
            p = f.median(dim=0).values
        elif agg == "trim":
            n = f.shape[0]
            k = max(1, int(round(n * trim)))
            if 2 * k < n:
                center = f.mean(0)
                d = (f - center).norm(dim=1)
                keep = d.argsort()[:(n - 2 * k)]
                p = f[keep].mean(0)
            else:
                p = f.mean(0)
        protos[remap[i]] = p / p.norm()
    return protos


def evaluate(image_feats, labels, protos):
    pred = (image_feats @ protos.t()).argmax(dim=1)
    return (pred == labels).float().mean().item()


def greedy_oracle(image_feats, labels, desc_feats, spans, remap, max_rounds=3):
    C = len(spans)
    masks = [torch.ones(e - s, dtype=torch.bool) for s, e in spans]
    protos = torch.zeros(C, image_feats.shape[1])
    for i, (s, e) in enumerate(spans):
        protos[remap[i]] = proto_of(desc_feats[s:e][masks[i]])
    sim = image_feats @ protos.t()
    acc = evaluate_from_sim(sim, labels)
    for rnd in range(1, max_rounds + 1):
        improved = 0
        t0 = time.time()
        for i in range(C):
            c = remap[i].item()
            feats_i = desc_feats[spans[i][0]:spans[i][1]]
            mask = masks[i]
            best_acc, best_action = acc, None
            for j in range(feats_i.shape[0]):
                if mask[j]:
                    if mask.sum() == 1:
                        continue
                    new_mask = mask.clone(); new_mask[j] = False
                else:
                    new_mask = mask.clone(); new_mask[j] = True
                new_col = image_feats @ proto_of(feats_i[new_mask])
                old_col = sim[:, c].clone()
                sim[:, c] = new_col
                a = evaluate_from_sim(sim, labels)
                sim[:, c] = old_col
                if a > best_acc + 1e-6:
                    best_acc, best_action = a, (j, new_mask, new_col)
            if best_action is not None:
                _, new_mask, new_col = best_action
                masks[i] = new_mask
                sim[:, c] = new_col
                acc = best_acc
                improved += 1
        print(f"    oracle round {rnd}: +{improved} -> {acc*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
        if improved == 0:
            break
    sizes = [int(m.sum()) for m in masks]
    return acc, sizes


def evaluate_from_sim(sim, labels):
    return (sim.argmax(dim=1) == labels).float().mean().item()


def k1_joint(image_feats, labels, desc_feats, spans, remap):
    C = len(spans)
    protos_full = torch.zeros(C, image_feats.shape[1])
    for i, (s, e) in enumerate(spans):
        protos_full[remap[i]] = proto_of(desc_feats[s:e])
    best_cols = {}
    for i in range(C):
        c = remap[i].item()
        feats_i = desc_feats[spans[i][0]:spans[i][1]]
        best_acc, best_j = -1, 0
        for j in range(feats_i.shape[0]):
            protos_full[c] = feats_i[j]
            a = evaluate(image_feats, labels, protos_full)
            if a > best_acc:
                best_acc, best_j = a, j
        best_cols[i] = best_j
        protos_full[c] = proto_of(desc_feats[spans[i][0]:spans[i][1]])
    protos_k1 = torch.zeros(C, image_feats.shape[1])
    for i in range(C):
        protos_k1[remap[i]] = desc_feats[spans[i][0]:spans[i][1]][best_cols[i]]
    return evaluate(image_feats, labels, protos_k1)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["aircraft", "cars"], required=True)
    args = parser.parse_args()
    ds = args.dataset
    cfg = CFG[ds]
    t0 = time.time()

    test_cache = os.path.join(REPO, "cache", f"{ds}_test_vitb32.pt")
    desc_cache = os.path.join(REPO, "cache", f"{ds}_desc_vitb32.pt")
    desc_path = os.path.join(REPO, "cupl_descriptions", f"{ds}.json")
    results_path = os.path.join(REPO, "results", f"d3_{ds}.json")


    if not os.path.exists(desc_path):
        src = os.path.join(cfg["data_root"], "category_description", "CuPL_full.json")
        json.dump(json.load(open(src, encoding="utf-8")), open(desc_path, "w", encoding="utf-8"), indent=2)
        print("desc copied ->", desc_path)

    model, preprocess = clip.load("ViT-B/32", device="cpu", download_root=os.path.expanduser(r"~\.cache\clip"))
    model.eval()


    if os.path.exists(test_cache):
        blob = torch.load(test_cache, map_location="cpu", weights_only=False)
        image_feats, labels = blob["feats"].float(), torch.tensor(blob["labels"])
        classes = blob["classes"]
        print(f"[cache] {ds} test {tuple(image_feats.shape)}")
    else:
        if ds == "aircraft":
            paths, labels, classes = load_aircraft_test()
        else:
            paths, labels, classes = load_cars_test()
        print(f"{ds} official test: {len(paths)} images, {len(classes)} classes")
        print("encoding test images (one-off)...")
        image_feats = encode_images(model, preprocess, paths)
        torch.save({"feats": image_feats, "labels": labels, "classes": classes}, test_cache)
        labels = torch.tensor(labels)
        print(f"cached -> {test_cache}")


    descs = json.load(open(desc_path, encoding="utf-8"))
    json_keys = list(descs.keys())
    name2idx = {_norm(c): i for i, c in enumerate(classes)}
    missing = [jk for jk in json_keys if _norm(jk) not in name2idx]
    if missing:
        print("WARNING unmatched keys:", missing[:5])
        raise SystemExit("class-name alignment failed")
    remap = torch.tensor([name2idx[_norm(jk)] for jk in json_keys])

    if os.path.exists(desc_cache):
        blob = torch.load(desc_cache, map_location="cpu", weights_only=False)
        desc_feats, spans = blob["feats"], [tuple(x) for x in blob["spans"]]
    else:
        print("encoding descriptions (one-off)...")
        feats, spans = [], []
        for jk in json_keys:
            f = encode_texts(model, descs[jk])
            start = spans[-1][1] if spans else 0
            spans.append((start, start + f.shape[0]))
            feats.append(f)
        desc_feats = torch.cat(feats)
        torch.save({"feats": desc_feats, "spans": spans}, desc_cache)
    print(f"desc total {desc_feats.shape[0]}, avg {desc_feats.shape[0]/len(spans):.1f}/class")

    results = {"dataset": ds, "n_test": int(image_feats.shape[0]), "n_classes": len(classes), "runs": []}

    def log(mode, strategy, acc):
        results["runs"].append({"mode": mode, "strategy": strategy, "acc": round(acc * 100, 2)})
        print(f"[{mode:10s}] {strategy:26s}: {acc*100:.2f}%")


    acc_base = evaluate(image_feats, labels, build_protos(desc_feats, spans, remap))
    results["baseline"] = round(acc_base * 100, 2)
    log("mean", "baseline (CuPL)", acc_base)
    hand_texts = [cfg["hand_template"].format(c) for c in classes]
    hand_feats = encode_texts(model, hand_texts)
    log("template", "hand-crafted", evaluate(image_feats, labels, hand_feats))


    cn_protos = ens_protos(model, classes, cfg["templates"])
    own_name = [desc_feats[s:e] @ cn_protos[remap[i]] for i, (s, e) in enumerate(spans)]

    for m in [1, 2, 3, 5]:
        masks = []
        for i in range(len(spans)):
            mk = torch.ones(spans[i][1] - spans[i][0], dtype=torch.bool)
            if m < mk.shape[0]:
                mk[own_name[i].topk(m, largest=False).indices] = False
            masks.append(mk)
        log("own-name", f"drop-{m}", evaluate(image_feats, labels, build_protos(desc_feats, spans, remap, masks=masks)))

    own_w = {}
    for beta in [5, 10, 20]:
        own_w[beta] = [torch.softmax(beta * own_name[i], dim=0) for i in range(len(spans))]
        log("own-name", f"softmax-beta={beta}", evaluate(image_feats, labels, build_protos(desc_feats, spans, remap, weights=own_w[beta])))

    log("median", "all-desc", evaluate(image_feats, labels, build_protos(desc_feats, spans, remap, agg="median")))
    for trim in [0.10, 0.20]:
        log("trim", f"trim={trim}", evaluate(image_feats, labels, build_protos(desc_feats, spans, remap, agg="trim", trim=trim)))


    combo = torch.zeros(len(spans), desc_feats.shape[1])
    for i, (s, e) in enumerate(spans):
        f = desc_feats[s:e]
        w = own_w[5][i]
        center = (f * w.view(-1, 1)).sum(0)
        d = (f - center).norm(dim=1)
        n = f.shape[0]
        k = max(1, int(round(n * 0.1)))
        keep = d.argsort()[:(n - k)]
        p = (f[keep] * w[keep].view(-1, 1)).sum(0)
        combo[remap[i]] = p / p.norm()
    log("combo", "softmax-b5-trim0.1", evaluate(image_feats, labels, combo))


    print("\n  [oracle] joint k=1...")
    acc_k1 = k1_joint(image_feats, labels, desc_feats, spans, remap)
    log("oracle", "k1-joint", acc_k1)
    print("  [oracle] greedy (test leakage, ceiling only)...")
    acc_gr, sizes = greedy_oracle(image_feats, labels, desc_feats, spans, remap)
    log("oracle", "greedy", acc_gr)
    results["oracle_sizes"] = {"min": min(sizes), "max": max(sizes), "mean": round(sum(sizes) / len(sizes), 1)}

    best = max((r for r in results["runs"] if not r["mode"].startswith("oracle")), key=lambda r: r["acc"])
    results["best_engine"] = best
    print(f"\n=== {ds.upper()} SUMMARY ===")
    print(f"baseline {results['baseline']}% | oracle {acc_gr*100:.2f}% (+{(acc_gr-acc_base)*100:.2f}pt) | k1 {acc_k1*100:.2f}%")
    print(f"best engine: {best['mode']} / {best['strategy']} = {best['acc']:.2f}%  (+{best['acc']-results['baseline']:.2f}pt)")
    json.dump(results, open(results_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"results -> {results_path}  ({time.time()-t0:.0f}s)")
    print(f"""
=== 判读标准（{ds}）===
1. combo 相对基线 >= +1pt → CUB 模式复现成功，主表三件套成立
2. oracle 增益 >= +2pt 且 best engine 拿到其中 >=40% → 机制+引擎双成立
3. oracle 增益大但引擎增益 <0.5pt → 机制在、信号失效（类名锚问题，如 aircraft 密码类名"707-320"）
4. oracle 增益本身 <1pt → 该数据集描述池本来就干净，机制非普遍（主表叙事收缩）
""")


if __name__ == "__main__":
    main()
