pass

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESC_DIR = os.path.join(REPO, "cupl_descriptions")
RESULTS_DIR = os.path.join(REPO, "results")



DEFAULT_N = {"cars": 60, "aircraft": 20, "dtd": 60, "cub": 39,
             "flo": 20, "oxford_pets": 20, "euro_sat": 20, "ucf101": 50}

CUB_ROOT = os.environ.get("CUB_ROOT", os.path.join(REPO, "data", "dataset", "cub", "CUB_200_2011"))


DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO, "data", "dataset"))
BREADTH_SPLITS = {
    "flo": os.path.join(DATA_ROOT, "flo", "split_zhou_OxfordFlowers.json"),
    "oxford_pets": os.path.join(DATA_ROOT, "oxford_pets", "split_zhou_OxfordPets.json"),
    "euro_sat": os.path.join(DATA_ROOT, "euro_sat", "split_zhou_EuroSAT.json"),
    "ucf101": os.path.join(DATA_ROOT, "ucf101", "split_zhou_UCF101.json"),
}

PROMPTS = {
    "cars": """You are helping build a zero-shot image classifier (CLIP) that matches photos to car classes by comparing images with text descriptions.

Car class: "{cls}"

Write exactly {n} short descriptions of this car class. Each description:
- ONE full sentence, starting with the full class name (including the year).
- Describes ONE visually verifiable attribute: body style and segment, overall shape and proportions, roofline, hood, grille, headlights, taillights, bumpers, door count and door arrangement, windows, wheels, mirrors, spoiler, exhausts, bed or trunk shape (for trucks), materials, typical exterior colors, and distinctive styling cues of this specific model.

STRICTLY FORBIDDEN (these hurt vision models):
- marketing or subjective wording (reliable, stylish, luxury, sporty, iconic, popular, best-selling, comfortable)
- scene or context content (on the road, in a showroom, parked, driving, with a driver)
- photography or meta language (photo, image, angle, can be identified, one way to identify, look for, distinguishable by)
- text-based identifiers readable only up close: VIN numbers, manufacturer codes, model numbers, badges with text
- non-visual facts: release year as its own sentence, price, target audience, brand reputation, horsepower, top speed, fuel economy
- near-duplicate sentences (each of the {n} descriptions must cover a different attribute)

Plain English, no markdown, no numbering.

Output: ONLY a JSON array of {n} strings, nothing else.""",
    "aircraft": """You are helping build a zero-shot image classifier (CLIP) that matches photos to aircraft variant classes by comparing images with text descriptions.

Aircraft class: "{cls}"

Write exactly {n} short descriptions of this aircraft variant. Each description:
- ONE full sentence, starting with the class name (the variant designator).
- Describes ONE visually verifiable attribute: fuselage length and shape, wing position (high or low mounted) and sweep angle, number of engines and their placement (underwing, wing-mounted, rear fuselage, tail-mounted), engine type (jet vs propeller, propeller blade count), tail design (T-tail, conventional, twin tail), nose and cockpit window shape, door and emergency-exit layout, overall size category, landing gear configuration.

STRICTLY FORBIDDEN (these hurt vision models):
- performance or technical specs (range, speed, fuel, seating capacity, payload)
- airline names, liveries, or operator information
- production history, development facts, price, target market
- text-based identifiers: registration numbers, serial numbers, fleet codes, badges with text
- photography or meta language (photo, image, angle, can be identified, one way to identify, look for)
- near-duplicate sentences (each of the {n} descriptions must cover a different attribute)

Plain English, no markdown, no numbering.

Output: ONLY a JSON array of {n} strings, nothing else.""",
    "dtd": """You are helping build a zero-shot image classifier (CLIP) that matches photos to texture classes by comparing images with text descriptions.

Texture class: "{cls}"

Write exactly {n} short descriptions of this texture class. Each description:
- ONE full sentence, starting with the class name.
- Describes ONE visually verifiable aspect of the texture's appearance: the repeating elements and their shape, the arrangement (regular, irregular, layered, interlaced, scattered), orientation and directionality, scale and granularity, color and contrast pattern, surface relief (raised, flat, indented, porous), and what the pattern visually resembles.

STRICTLY FORBIDDEN (these hurt vision models):
- scene or location context (where the texture is typically found: kitchen, forest, wall, ground)
- photography or meta language (photo, image, close-up, macro, can be identified, look for)
- marketing or subjective wording (beautiful, popular, classic, elegant)
- near-duplicate sentences (each of the {n} descriptions must cover a different aspect)

Plain English, no markdown, no numbering.

Output: ONLY a JSON array of {n} strings, nothing else.""",
    "cub": """You are helping build a zero-shot image classifier (CLIP) that matches photos to bird species by comparing images with text descriptions.

Bird class: "{cls}"

Write exactly {n} short descriptions of this bird species. Each description:
- ONE full sentence, starting with the full species name.
- Describes ONE visually verifiable attribute: plumage colors and patterns by body part (head, back, breast, belly, wings, tail), bill or beak shape and color, leg color, wing shape and markings, tail shape, overall size and body proportions, eye pattern, and distinctive field marks that distinguish this species from similar species.

STRICTLY FORBIDDEN (these hurt vision models):
- habitat, geographic range, or migration facts (found near coasts, native to North America)
- behavior, song, call, diet, or nesting facts (non-visual)
- photography or meta language (photo, image, can be identified, one way to identify, look for)
- marketing or subjective wording (beautiful, popular, iconic)
- near-duplicate sentences (each of the {n} descriptions must cover a different attribute)

Plain English, no markdown, no numbering.

Output: ONLY a JSON array of {n} strings, nothing else.""",
    "flo": """You are helping build a zero-shot image classifier (CLIP) that matches photos to flower species by comparing images with text descriptions.

Flower class: "{cls}"

Write exactly {n} short descriptions of this flower species. Each description:
- ONE full sentence, starting with the full class name.
- Describes ONE visually verifiable attribute: petal color, petal count and shape, flower form (daisy-like, bell-shaped, spiked, tubular), inflorescence arrangement, stem and leaf characteristics, overall plant size and growth form.

STRICTLY FORBIDDEN (these hurt vision models):
- habitat, season, or growing facts (blooms in spring, found in gardens, native to...)
- fragrance or other non-visual properties
- cultural or symbolic meanings
- photography or meta language (photo, image, can be identified, look for)
- near-duplicate sentences (each of the {n} descriptions must cover a different attribute)

Plain English, no markdown, no numbering.

Output: ONLY a JSON array of {n} strings, nothing else.""",
    "oxford_pets": """You are helping build a zero-shot image classifier (CLIP) that matches photos to pet breeds by comparing images with text descriptions.

Pet breed class: "{cls}"

Write exactly {n} short descriptions of this pet breed. Each description:
- ONE full sentence, starting with the full class name.
- Describes ONE visually verifiable attribute: coat color and pattern, coat length and texture, body size and proportions, ear shape and carriage (floppy, pointed, folded), face and muzzle structure, tail characteristics, and distinctive breed features.

STRICTLY FORBIDDEN (these hurt vision models):
- temperament or behavior (friendly, energetic, intelligent, loyal)
- breed history, origin, or registration facts
- popularity or marketing wording
- photography or meta language (photo, image, can be identified, look for)
- near-duplicate sentences (each of the {n} descriptions must cover a different attribute)

Plain English, no markdown, no numbering.

Output: ONLY a JSON array of {n} strings, nothing else.""",
    "euro_sat": """You are helping build a zero-shot image classifier (CLIP) that matches satellite images to land-use classes by comparing images with text descriptions.

Satellite image class: "{cls}"

Write exactly {n} short descriptions of this land-use class as seen from above. Each description:
- ONE full sentence, starting with the full class name.
- Describes ONE visually verifiable attribute from an aerial or satellite view: land cover composition, geometric patterns of fields or plots, road and street network layout, vegetation texture and color, presence of water bodies, building density and arrangement, overall color palette of the scene.

STRICTLY FORBIDDEN (these hurt vision models):
- ground-level scene content (people walking, cars driving, close-up objects)
- non-visual facts (climate, region, agricultural practice details not visible from above)
- photography or meta language (photo, image, can be identified, look for)
- near-duplicate sentences (each of the {n} descriptions must cover a different attribute)

Plain English, no markdown, no numbering.

Output: ONLY a JSON array of {n} strings, nothing else.""",
    "ucf101": """You are helping build a zero-shot image classifier (CLIP) that matches photos to action classes by comparing images with text descriptions.

Action class: "{cls}"

Write exactly {n} short descriptions of this action class. Each description:
- ONE full sentence, starting with the full class name.
- Describes ONE visually verifiable attribute as seen in a SINGLE still frame of this action: the objects and tools involved, the body pose and hand position, the visible setting or location type, and the typical scene composition.

STRICTLY FORBIDDEN (these hurt vision models):
- motion-only phrasing that cannot be verified in one still frame (describe what is visible, not just what is moving)
- video or camera language (video, footage, camera, clip)
- photography or meta language (can be identified, look for)
- near-duplicate sentences (each of the {n} descriptions must cover a different attribute)

Plain English, no markdown, no numbering.

Output: ONLY a JSON array of {n} strings, nothing else.""",
}


WAFFLE_WORDS = list(dict.fromkeys("""waffle spoon mountain river cloud feather marble velvet ribbon
anchor basket bottle candle carpet chair chimney compass curtain diamond drum engine fossil garden
hammer helmet island jacket kettle ladder lantern lemon lizard magnet meadow needle onion orange
paddle parcel pencil pepper pillow planet plough pocket pudding pyramid quartz radiator rainbow rust
saddle sandpaper satellite scarf seashell shadow shovel silk slate snail solar sparkle spear spider
spinach spiral sponge stamp stone sugar sunset sweater sword thimble thunder ticket tiger toffee
tomato tornado trellis trumpet tulip tunnel turtle umbrella vanilla velvet vine volcano voucher
walnut waterfall whisper window winter wolf zebra zinc amber apples autumn bacon bamboo banner barley
basalt beacon beetle birch biscuit blossom brass brick bronze butter cactus camel canvas carbon cedar
cement cherry citrus clay cobalt copper coral cotton crystal cyclone daisy dandelion diesel dune eagle
eclipse elm emerald falcon fern frost glacier granite gravel hazel helium holly indigo ivory jet jasper
juniper kelp kiwi lagoon larch lava lilac linen lotus lynx mahogany maize mango maple marsh mint moss
neon nickel oak oats olive opal orchid osprey pewter pine platinum poppy quill radish raven reef ridge
rosemary rowan rye saffron sage salmon sandal sapphire sequoia soy spruce sumac syrup tangerine
terracotta thistle thyme timber topaz umber vermouth violet wheat willow yarrow yew yucca""".split()))


def load_classes(ds):
    pass
    cache_path = os.path.join(REPO, "cache", f"{ds}_test_vitb32.pt")
    if os.path.exists(cache_path):
        import torch
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        if "classes" in blob:
            return blob["classes"]
        with open(os.path.join(CUB_ROOT, "classes.txt"), encoding="utf-8") as f:
            raw = [l.split()[1].strip() for l in f if l.strip()]
        return [r.split(".", 1)[1].replace("_", " ") for r in raw]

    if ds in BREADTH_SPLITS:
        d = json.load(open(BREADTH_SPLITS[ds], encoding="utf-8"))
        label2name = {}
        for part in ["train", "val", "test"]:
            for rel, lab, name in d[part]:
                label2name[int(lab)] = name
        ids = sorted(label2name)
        assert ids == list(range(len(ids))), f"{ds}: labels not contiguous"

        return [label2name[i].replace("_", " ") for i in ids]
    raise FileNotFoundError(f"{ds}: 既无测试缓存也不在 BREADTH_SPLITS，无法取类名")




def gen_waffle(ds, classes, n_desc):
    rng = random.Random(42)
    pool = {}
    for cls in classes:
        words = rng.sample(WAFFLE_WORDS, n_desc)
        pool[cls] = [f"{cls}, which has {w}." for w in words]
    out = os.path.join(DESC_DIR, f"{ds}_waffle_random.json")
    json.dump(pool, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"waffle pool -> {out}  ({len(pool)} classes x {n_desc})")




def chat_once(api_base, api_key, model, prompt, temperature, max_tokens, timeout=180):
    pass
    url = api_base.rstrip("/") + "/chat/completions"
    base_body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    payloads = [
        {**base_body, "temperature": temperature, "max_tokens": max_tokens},
        base_body,
    ]
    last_err = None
    for body in payloads:
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST", headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"].get("content")
            if not content:
                raise ValueError("empty content in response")
            return content
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(last_err)


def parse_desc_array(raw, n_expect):
    pass
    text = raw.strip()
    text = re.sub(r"^```(json)?\s*|```\s*$", "", text, flags=re.IGNORECASE).strip()
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        return [], n_expect
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [], n_expect
    out, dropped = [], 0
    for item in arr:
        if isinstance(item, str):
            s = item.strip()
            if 20 <= len(s) <= 400 and s not in out:
                out.append(s)
            else:
                dropped += 1
        else:
            dropped += 1
    return out, dropped


def gen_attr(ds, classes, args, api_key):

    suffix = f"_{args.tag}" if args.tag else ""
    attr_out = os.path.join(DESC_DIR, f"{ds}_regen_attr{suffix}.json")
    raw_out = os.path.join(RESULTS_DIR, f"p4_e5_{ds}_attr_raw{suffix}.json")


    pool, raw_log = {}, {}
    if args.resume and os.path.exists(attr_out):
        pool = json.load(open(attr_out, encoding="utf-8"))
        print(f"[resume] 已有 {len(pool)}/{len(classes)} 类，跳过")
    if args.resume and os.path.exists(raw_out):
        raw_log = json.load(open(raw_out, encoding="utf-8"))

    prompt_tpl = PROMPTS[ds]
    todo = [c for c in classes if c not in pool]
    print(f"attr 臂（{ds}）：{len(todo)} 类待生成 | model={args.model} | base={args.api_base}")

    def one(cls):
        prompt = prompt_tpl.format(cls=cls, n=args.n_desc)
        last_err = None
        for attempt in range(3):
            try:
                raw = chat_once(args.api_base, api_key, args.model, prompt,
                                args.temperature, args.max_tokens)
                descs, dropped = parse_desc_array(raw, args.n_desc)
                if len(descs) < max(5, args.n_desc // 3):
                    raise ValueError(f"parsed only {len(descs)}/{args.n_desc}")
                return cls, descs, raw, dropped
            except Exception as e:
                last_err = e
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"{cls}: {last_err}")

    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, c): c for c in todo}
        for fut in as_completed(futs):
            try:
                c2, descs, raw, dropped = fut.result()
                pool[c2] = descs
                raw_log[c2] = {"raw": raw, "n_parsed": len(descs), "n_dropped": dropped}
                done += 1
            except Exception as e:
                failed.append(str(e))
            if done % 20 == 0 and done:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(todo)}  ({rate:.1f} 类/s, 剩余 ~{(len(todo)-done)/max(rate,1e-9)/60:.0f} 分钟)", flush=True)


    json.dump(pool, open(attr_out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(raw_log, open(raw_out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    cnts = [len(v) for v in pool.values()]
    print(f"\nattr pool -> {attr_out}")
    print(f"  classes ok: {len(pool)}/{len(classes)} | failed: {len(failed)}")
    if cnts:
        print(f"  desc/class min/mean/max: {min(cnts)}/{sum(cnts)/len(cnts):.1f}/{max(cnts)}")
    for f in failed[:10]:
        print("  FAIL:", f)
    if failed:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    choices=["cars", "aircraft", "dtd", "cub", "flo", "oxford_pets", "euro_sat", "ucf101"],
                    required=True)
    ap.add_argument("--arm", choices=["attr", "waffle", "both"], default="both")
    ap.add_argument("--n-desc", type=int, default=None,
                    help="缺省按数据集定：aircraft=20/dtd=60 匹配原池，cars=60 沿用 Phase 1 口径")
    ap.add_argument("--api-base", default=os.environ.get("E5_API_BASE", "https://api.openai.com/v1"))
    ap.add_argument("--model", default=os.environ.get("E5_MODEL", "gpt-4o-mini"))
    ap.add_argument("--api-key", default=os.environ.get("E5_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", default="", help="输出文件名后缀（如 gpt54）；多生成模型对比时避免覆盖")
    args = ap.parse_args()

    if args.n_desc is None:
        args.n_desc = DEFAULT_N[args.dataset]

    os.makedirs(DESC_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    classes = load_classes(args.dataset)
    print(f"{args.dataset}: {len(classes)} classes | n_desc = {args.n_desc}")

    if args.arm in ("waffle", "both"):
        gen_waffle(args.dataset, classes, args.n_desc)
    if args.arm in ("attr", "both"):
        if not args.api_key:
            sys.exit("缺少 API key：传 --api-key 或设 E5_API_KEY / OPENAI_API_KEY")
        gen_attr(args.dataset, classes, args, args.api_key)

    print(f"\n下一步：python e5_probe.py --dataset {args.dataset}   （直评 + k1/greedy 天花板 + 判读）")


if __name__ == "__main__":
    main()
