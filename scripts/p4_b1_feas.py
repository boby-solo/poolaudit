
pass

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e5_gen_pools import chat_once, parse_desc_array, load_classes, DEFAULT_N

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESC_DIR = os.path.join(REPO, "cupl_descriptions")
RESULTS_DIR = os.path.join(REPO, "results")
DATASETS = ("aircraft", "euro_sat", "dtd", "cub")



CUBL_ALIAS = {
    "brushland or shrubland": "Herbaceous Vegetation Land",
    "industrial buildings or commercial buildings": "Industrial Buildings",
    "residential buildings or homes or apartments": "Residential Buildings",
    "lake or sea": "Sea or Lake",
}


SANITY_REFS = {
    "aircraft": {"cupl": (20.94, 26.07), "attr": (19.20, 21.84)},
    "euro_sat": {"cupl": (51.30, 53.64), "attr": (None, None)},
    "dtd": {"cupl": (50.11, 59.57), "attr": (51.65, 59.57)},
    "cub": {"cupl": (52.93, 55.68), "attr": (51.73, 57.77)},
}


def paths_for(ds):
    return {
        "onto": os.path.join(RESULTS_DIR, f"p4_b1_{ds}_ontology.json"),
        "onto_raw": os.path.join(RESULTS_DIR, f"p4_b1_{ds}_onto_raw.json"),
        "onto_gen_raw": os.path.join(RESULTS_DIR, f"p4_b1_{ds}_onto_gen_raw.json"),
        "magent_raw": os.path.join(RESULTS_DIR, f"p4_b1_{ds}_magent_raw.json"),
        "omni_raw": os.path.join(RESULTS_DIR, f"p4_b1_{ds}_omni_raw.json"),
        "onto_pool": os.path.join(DESC_DIR, f"{ds}_regen_onto.json"),
        "magent_pool": os.path.join(DESC_DIR, f"{ds}_regen_magent.json"),
        "omni_pool": os.path.join(DESC_DIR, f"{ds}_regen_omni.json"),
        "attr_pool": os.path.join(DESC_DIR, f"{ds}_regen_attr_gpt54.json"),
        "feas": os.path.join(RESULTS_DIR, f"p4_b1_{ds}_feas.json"),
    }








ONTOLOGY_PROMPT_AIRCRAFT = """You are building a type ontology (knowledge base) of aircraft variants for a zero-shot image classifier.

The complete list of variant classes in this task is:
{all_classes}

Target variant: "{cls}"

Based on your knowledge of aircraft, output a JSON object with exactly these fields:
{{
  "manufacturer": "the manufacturer",
  "family": "the aircraft family this variant belongs to (e.g., 'Boeing 737 narrow-body airliner family')",
  "role": "role category (e.g., narrow-body airliner, regional turboprop, fighter jet, military cargo transport, business jet)",
  "era": "approximate service-entry decade, e.g., '1990s'",
  "confusable_siblings": [2 to 4 variant names FROM THE LIST ABOVE that look most similar to this variant],
  "family_visual_attributes": [4 to 6 visually verifiable attributes shared across the whole family, useful for telling this family apart from OTHER families],
  "distinguishing_visual_attributes": [3 to 5 visually verifiable attributes that distinguish THIS variant from its confusable siblings]
}}

Rules:
- confusable_siblings MUST be chosen from the list above and spelled exactly as in the list.
- All attributes must be visually verifiable from a photo (shape, proportions, engine count and placement, wing configuration, tail design, size category). No performance specs, no prices, no airline names.
- If unsure, still give your best guess (the ontology will be verified separately).

Output: ONLY the JSON object, nothing else."""

ONTO_GEN_PROMPT_AIRCRAFT = """You are helping build a zero-shot image classifier (CLIP) that matches photos to aircraft variant classes by comparing images with text descriptions.

Aircraft class: "{cls}"

Verified type-ontology record for this variant:
- Family: {family} (manufacturer: {manufacturer})
- Role: {role}; era: {era}
- Family-level visual attributes (shared across the family): {fam_attrs}
- Most confusable sibling variants in this task: {siblings}
- What visually distinguishes this variant from those siblings: {dist_attrs}

Write exactly {n} short descriptions of this aircraft variant. Each description:
- ONE full sentence, starting with the class name (the variant designator).
- Describes ONE visually verifiable attribute: fuselage length and proportions, wing position and sweep, number of engines and their placement, engine type, tail design, nose and cockpit shape, door layout, size category, landing gear, or a family-identifying visual cue that separates this family from other families.
- When useful, phrase the attribute relative to the confusable siblings (e.g., "shorter fuselage than the 737-800"), because the classifier must tell these variants apart.

STRICTLY FORBIDDEN:
- performance specs (range, speed, seating capacity, payload), prices, airline or livery names, registration or serial numbers
- photography or meta language (photo, image, angle, can be identified, look for)
- near-duplicate sentences (each of the {n} descriptions must cover a different attribute)

Plain English, no markdown, no numbering.
Output: ONLY a JSON array of {n} strings, nothing else."""

CRITIC_PROMPT_AIRCRAFT = """You are a strict reviewer for a zero-shot aircraft-variant classifier (CLIP).

The complete list of variant classes is:
{all_classes}

Target class: "{cls}"

Draft descriptions (numbered):
{drafts}

Your job:
1. From the class list, pick the 2-4 variants most visually similar to the target (easiest to confuse).
2. Judge each draft: could it also apply to one of those confusable variants (too generic)? Is any visual claim likely wrong for this variant?
3. Note what distinguishing visual information is missing.

Output a JSON object with exactly these fields:
{{
  "confusable": ["2-4 variant names from the list"],
  "too_generic": [draft numbers that are not discriminative],
  "likely_wrong": [draft numbers with doubtful visual claims],
  "missing": ["what visual distinction should be added"]
}}
Output ONLY the JSON object, nothing else."""

REFINER_PROMPT_AIRCRAFT = """You are refining descriptions for a zero-shot aircraft-variant classifier (CLIP).

Target class: "{cls}"
Confusable variants that must be told apart: {confusable}

Current drafts (numbered):
{drafts}

Reviewer critique:
- Not discriminative (too generic): drafts {too_generic}
- Doubtful visual claims: drafts {likely_wrong}
- Missing distinctions: {missing}

Rewrite into exactly {n} short descriptions. Each:
- ONE full sentence, starting with the class name.
- ONE visually verifiable attribute; when useful, phrase it relative to the confusable variants (e.g., "longer fuselage than the 737-700").
- Fix or drop the flagged drafts; cover the missing distinctions.

STRICTLY FORBIDDEN: performance specs, airline or livery names, registration or serial numbers, photography or meta language, near-duplicate sentences.
Plain English. Output: ONLY a JSON array of {n} strings, nothing else."""




ONTOLOGY_PROMPT_TPL = """You are building a type ontology (knowledge base) of {domain} classes for a zero-shot image classifier.

The complete list of classes in this task is:
{all_classes}

Target class: "{cls}"

Based on your knowledge, output a JSON object with exactly these fields:
{{
  "lineage": "{lineage_hint}",
  "position": "{position_hint}",
  "confusable_siblings": [2 to 4 class names FROM THE LIST ABOVE that look most similar to this class],
  "family_visual_attributes": [4 to 6 visually verifiable attributes shared across the whole lineage, useful for telling this lineage apart from OTHER lineages],
  "distinguishing_visual_attributes": [3 to 5 visually verifiable attributes that distinguish THIS class from its confusable siblings]
}}

Rules:
- confusable_siblings MUST be chosen from the list above and spelled exactly as in the list.
- All attributes must be visually verifiable {verifiable_scope}. {domain_forbidden}
- If unsure, still give your best guess (the ontology will be verified separately).

Output: ONLY the JSON object, nothing else."""

ONTO_GEN_PROMPT_TPL = """You are helping build a zero-shot image classifier (CLIP) that matches {domain_phrase} by comparing images with text descriptions.

Class: "{cls}"

Verified type-ontology record for this class:
- Lineage: {family} ({position})
- Family-level visual attributes (shared across the lineage): {fam_attrs}
- Most confusable sibling classes in this task: {siblings}
- What visually distinguishes this class from those siblings: {dist_attrs}

Write exactly {n} short descriptions of this class. Each description:
- ONE full sentence, starting with the full class name.
- Describes ONE visually verifiable attribute: {checklist}
- When useful, phrase the attribute relative to the confusable siblings, because the classifier must tell these classes apart.

STRICTLY FORBIDDEN:
{forbidden_block}
- near-duplicate sentences (each of the {n} descriptions must cover a different attribute)

Plain English, no markdown, no numbering.
Output: ONLY a JSON array of {n} strings, nothing else."""

CRITIC_PROMPT_TPL = """You are a strict reviewer for a zero-shot {domain} classifier (CLIP).

The complete list of classes is:
{all_classes}

Target class: "{cls}"

Draft descriptions (numbered):
{drafts}

Your job:
1. From the class list, pick the 2-4 classes most visually similar to the target (easiest to confuse).
2. Judge each draft: could it also apply to one of those confusable classes (too generic)? Is any visual claim likely wrong for this class?
3. Note what distinguishing visual information is missing.

Output a JSON object with exactly these fields:
{{
  "confusable": ["2-4 class names from the list"],
  "too_generic": [draft numbers that are not discriminative],
  "likely_wrong": [draft numbers with doubtful visual claims],
  "missing": ["what visual distinction should be added"]
}}
Output ONLY the JSON object, nothing else."""

REFINER_PROMPT_TPL = """You are refining descriptions for a zero-shot {domain} classifier (CLIP).

Target class: "{cls}"
Confusable classes that must be told apart: {confusable}

Current drafts (numbered):
{drafts}

Reviewer critique:
- Not discriminative (too generic): drafts {too_generic}
- Doubtful visual claims: drafts {likely_wrong}
- Missing distinctions: {missing}

Rewrite into exactly {n} short descriptions. Each:
- ONE full sentence, starting with the full class name.
- ONE visually verifiable attribute; when useful, phrase it relative to the confusable classes, because the classifier must tell these classes apart.
- Fix or drop the flagged drafts; cover the missing distinctions.

STRICTLY FORBIDDEN: {forbidden_inline}
Plain English. Output: ONLY a JSON array of {n} strings, nothing else."""


CRITIC_OMNI_PROMPT_TPL = """You are a strict reviewer for a zero-shot {domain} classifier (CLIP).

The complete list of classes is:
{all_classes}

Target class: "{cls}"

Verified type-ontology record for this class:
- Lineage: {family}
- Most confusable siblings (from the ontology): {siblings}
- Family-level visual attributes: {fam_attrs}
- What visually distinguishes this class from those siblings: {dist_attrs}

Draft descriptions (numbered):
{drafts}

Your job:
1. The ontology record above already lists the confusable siblings — use them directly; add more from the class list ONLY if something obvious is missing.
2. Judge each draft: could it also apply to one of the confusable siblings (too generic)? Is any visual claim likely wrong for this class, or contradicting the ontology record?
3. Note which distinguishing attributes from the ontology record are missing from the drafts.

Output a JSON object with exactly these fields:
{{
  "too_generic": [draft numbers that are not discriminative],
  "likely_wrong": [draft numbers with doubtful or contradicted visual claims],
  "missing": ["which ontology distinctions should be added"]
}}
Output ONLY the JSON object, nothing else."""

REFINER_OMNI_PROMPT_TPL = """You are refining descriptions for a zero-shot {domain} classifier (CLIP).

Target class: "{cls}"

Verified type-ontology record:
- Lineage: {family}
- Confusable siblings that must be told apart: {siblings}
- Distinguishing visual attributes (from the ontology): {dist_attrs}

Current drafts (numbered):
{drafts}

Reviewer critique:
- Not discriminative (too generic): drafts {too_generic}
- Doubtful visual claims: drafts {likely_wrong}
- Missing distinctions: {missing}

Rewrite into exactly {n} short descriptions. Each:
- ONE full sentence, starting with the full class name.
- ONE visually verifiable attribute; when useful, phrase it relative to the confusable siblings, because the classifier must tell these classes apart.
- Fix or drop the flagged drafts; the missing distinctions MUST be covered, drawing on the ontology's distinguishing attributes.

STRICTLY FORBIDDEN: {forbidden_inline}
Plain English. Output: ONLY a JSON array of {n} strings, nothing else."""




DOMAIN_CFG = {
    "euro_sat": {
        "domain": "satellite land-use/land-cover",
        "domain_phrase": "satellite images to land-use classes",
        "lineage_hint": "the broader land-cover category this class belongs to (e.g., 'agricultural land', 'urban built-up', 'water bodies')",
        "position_hint": "this class's specific position within that category",
        "verifiable_scope": "from a top-down satellite or aerial image (patch shape and size, geometric regularity, colors, texture, spatial patterns, linear features, vegetation density, building arrangement)",
        "domain_forbidden": "No climate, no country or region names, no ground-level details.",
        "checklist": "patch shape and size, geometric regularity of field or street patterns, dominant colors, vegetation texture and density, linear features (roads, rivers), building arrangement and density, presence of water.",
        "forbidden_block": """- ground-level scene content (people walking, cars driving, close-up objects)
- non-visual facts (climate, region, agricultural practice details not visible from above)
- photography or meta language (photo, image, can be identified, look for)""",
        "forbidden_inline": "ground-level scene content, non-visual facts (climate, region, farming practice), photography or meta language, near-duplicate sentences",
    },
    "dtd": {
        "domain": "texture",
        "domain_phrase": "photos to texture classes",
        "lineage_hint": "the broader texture family this class belongs to (e.g., 'line-based textures', 'regular repeating patterns', 'random disorder textures')",
        "position_hint": "this texture's specific position within that family",
        "verifiable_scope": "from the texture image itself (element shape, arrangement, orientation, scale, regularity, color and contrast, surface relief)",
        "domain_forbidden": "No material identity claims that cannot be seen, no scene or location context.",
        "checklist": "the repeating elements and their shape, arrangement (regular, irregular, layered, interlaced, scattered), orientation, scale and granularity, color and contrast pattern, surface relief, and what the pattern visually resembles.",
        "forbidden_block": """- scene or location context (where the texture is typically found: kitchen, forest, wall, ground)
- photography or meta language (photo, image, close-up, macro, can be identified, look for)
- marketing or subjective wording (beautiful, popular, classic, elegant)""",
        "forbidden_inline": "scene or location context, photography or meta language, subjective wording, near-duplicate sentences",
    },
    "cub": {
        "domain": "bird-species",
        "domain_phrase": "photos to bird species",
        "lineage_hint": "the taxonomic family or species group this bird belongs to (e.g., 'albatross family (Diomedeidae)')",
        "position_hint": "this species' position within the group (e.g., 'large North Pacific albatross')",
        "verifiable_scope": "from a photo of the bird (plumage colors and patterns by body part, bill shape and size, wing shape, tail pattern, overall size and proportions, eye markings)",
        "domain_forbidden": "No song, no habitat, no geographic range, no behavior.",
        "checklist": "plumage colors and patterns by body part (head, back, breast, belly, wings, tail), bill shape and color, leg color, wing shape and markings, tail shape, overall size and proportions, eye pattern, distinctive field marks.",
        "forbidden_block": """- habitat, geographic range, or migration facts (found near coasts, native to North America)
- behavior, song, call, diet, or nesting facts (non-visual)
- photography or meta language (photo, image, can be identified, one way to identify, look for)
- marketing or subjective wording (beautiful, popular, iconic)""",
        "forbidden_inline": "habitat, range, or migration facts; behavior, song, or diet; photography or meta language; near-duplicate sentences",
    },
}


def build_suite(ds):
    pass
    if ds == "aircraft":
        def onto_fields(o):
            return dict(
                family=o["family"], manufacturer=o.get("manufacturer", ""),
                role=o.get("role", ""), era=o.get("era", ""),
                fam_attrs="; ".join(o["family_visual_attributes"]),
                siblings=", ".join(o["confusable_siblings"]),
                dist_attrs="; ".join(o["distinguishing_visual_attributes"]),
                position="")
        return {
            "ontology": ONTOLOGY_PROMPT_AIRCRAFT,
            "onto_gen": ONTO_GEN_PROMPT_AIRCRAFT,
            "critic": CRITIC_PROMPT_AIRCRAFT,
            "refiner": REFINER_PROMPT_AIRCRAFT,
            "onto_fields": onto_fields,
            "dk": {},
        }
    cfg = DOMAIN_CFG[ds]

    def onto_fields(o):
        return dict(
            family=o.get("lineage", ""), position=o.get("position", ""),
            fam_attrs="; ".join(o.get("family_visual_attributes", [])),
            siblings=", ".join(o.get("confusable_siblings", [])),
            dist_attrs="; ".join(o.get("distinguishing_visual_attributes", [])),
            manufacturer="", role="", era="")
    return {
        "ontology": ONTOLOGY_PROMPT_TPL,
        "onto_gen": ONTO_GEN_PROMPT_TPL,
        "critic": CRITIC_PROMPT_TPL,
        "refiner": REFINER_PROMPT_TPL,
        "onto_fields": onto_fields,
        "dk": dict(cfg),
    }


def build_omni_suite(ds):
    pass
    if ds == "aircraft":
        dk = {"domain": "aircraft-variant",
              "forbidden_inline": (
                  "performance specs, airline or livery names, registration or "
                  "serial numbers, photography or meta language, near-duplicate sentences")}
    else:
        dk = dict(DOMAIN_CFG[ds])
    return {
        "critic_omni": CRITIC_OMNI_PROMPT_TPL,
        "refiner_omni": REFINER_OMNI_PROMPT_TPL,
        "dk": dk,
    }




def parse_json_object(raw):
    text = re.sub(r"^```(json)?\s*|```\s*$", "", raw.strip(), flags=re.IGNORECASE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call_retry(api_base, key, model, prompt, temperature, max_tokens, what):
    last = None
    for attempt in range(3):
        try:
            return chat_once(api_base, key, model, prompt, temperature, max_tokens)
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{what}: {last}")


def gen_max_tokens(n_desc):
    return max(2500, 60 * n_desc)




def stage_onto(args, classes, P, suite):
    onto, onto_raw = {}, {}
    if os.path.exists(P["onto"]):
        onto = json.load(open(P["onto"], encoding="utf-8"))
        print(f"[resume] 本体已有 {len(onto)}/{len(classes)} 类")
    if os.path.exists(P["onto_raw"]):
        onto_raw = json.load(open(P["onto_raw"], encoding="utf-8"))

    all_classes = ", ".join(classes)
    todo = [c for c in classes if c not in onto]
    print(f"[stage onto/1] 本体构建：{len(todo)} 类 | model={args.model}")

    def one_onto(cls):
        p = suite["ontology"].format(all_classes=all_classes, cls=cls,
                                     **suite["dk"])
        raw = call_retry(args.api_base, args.api_key, args.model, p,
                         args.temperature, 1800, f"onto:{cls}")
        obj = parse_json_object(raw)
        if not obj or not all(k in obj for k in
                              ("confusable_siblings", "family_visual_attributes",
                               "distinguishing_visual_attributes")) \
                or not (obj.get("family") or obj.get("lineage")):
            raise ValueError(f"{cls}: ontology parse failed")
        return cls, obj, raw

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_onto, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                c2, obj, raw = fut.result()
                onto[c2] = obj
                onto_raw[c2] = {"raw": raw}
            except Exception as e:
                print("  ONTO FAIL:", e)
            if i % 20 == 0:
                json.dump(onto, open(P["onto"], "w", encoding="utf-8"),
                          indent=1, ensure_ascii=False)
                json.dump(onto_raw, open(P["onto_raw"], "w", encoding="utf-8"),
                          indent=1, ensure_ascii=False)
                print(f"  本体 {i}/{len(todo)}", flush=True)
    json.dump(onto, open(P["onto"], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(onto_raw, open(P["onto_raw"], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"本体 -> {P['onto']}（{len(onto)}/{len(classes)} 类）")
    if len(onto) < len(classes):
        sys.exit(1)


    pool, raw_log = {}, {}
    if os.path.exists(P["onto_pool"]):
        pool = json.load(open(P["onto_pool"], encoding="utf-8"))
        print(f"[resume] onto池已有 {len(pool)}/{len(classes)} 类")
    if os.path.exists(P["onto_gen_raw"]):
        raw_log = json.load(open(P["onto_gen_raw"], encoding="utf-8"))
    todo = [c for c in classes if c not in pool]
    print(f"[stage onto/2] 本体引导生成：{len(todo)} 类")

    n_desc = DEFAULT_N[args.dataset]

    def one_gen(cls):
        p = suite["onto_gen"].format(cls=cls, n=n_desc, **suite["dk"],
                                     **suite["onto_fields"](onto[cls]))
        raw = call_retry(args.api_base, args.api_key, args.model, p,
                         args.temperature, gen_max_tokens(n_desc), f"gen:{cls}")
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
                raw_log[c2] = {"raw": raw, "n_parsed": len(descs), "n_dropped": dropped}
            except Exception as e:
                failed.append(str(e))
            if i % 20 == 0:
                json.dump(pool, open(P["onto_pool"], "w", encoding="utf-8"),
                          indent=2, ensure_ascii=False)
                print(f"  onto池 {i}/{len(todo)}", flush=True)
    json.dump(pool, open(P["onto_pool"], "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(raw_log, open(P["onto_gen_raw"], "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print(f"onto 池 -> {P['onto_pool']}（{len(pool)}/{len(classes)}，失败 {len(failed)}）")
    for f in failed[:5]:
        print("  FAIL:", f)




def stage_check(classes, P):
    onto = json.load(open(P["onto"], encoding="utf-8"))
    cls_set = set(classes)
    bad = []
    for cls, o in onto.items():
        for s in o.get("confusable_siblings", []):
            if s not in cls_set:
                bad.append(f"{cls}: 兄弟 '{s}' 不在类表（幻觉或拼写）")
        fam = str(o.get("family") or o.get("lineage") or "")
        if not fam:
            bad.append(f"{cls}: 缺谱系字段")
        if len(o.get("family_visual_attributes", [])) < 3:
            bad.append(f"{cls}: 家族属性不足3条")
        if len(o.get("distinguishing_visual_attributes", [])) < 2:
            bad.append(f"{cls}: 区分属性不足2条")
        if "manufacturer" in o and not o.get("manufacturer"):
            bad.append(f"{cls}: 缺制造商")

        if "manufacturer" in o:
            m = re.match(r"^([A-Za-z0-9]+)-\d+", cls)
            if m and m.group(1).lower() not in fam.lower():
                bad.append(f"{cls}: 家族 '{fam}' 与机型前缀不符")
    print(f"本体快检：{len(onto)} 类，异常 {len(bad)} 条")
    for b in bad[:20]:
        print("  ", b)
    for cls in list(onto)[:3]:
        print(f"\n[样本] {cls}: {json.dumps(onto[cls], ensure_ascii=False)[:400]}")




def stage_magent(args, classes, P, suite):
    pool, raw_log = {}, {}
    if os.path.exists(P["magent_pool"]):
        pool = json.load(open(P["magent_pool"], encoding="utf-8"))
        print(f"[resume] magent池已有 {len(pool)}/{len(classes)} 类")
    if os.path.exists(P["magent_raw"]):
        raw_log = json.load(open(P["magent_raw"], encoding="utf-8"))

    from e5_gen_pools import PROMPTS
    gen_tpl = PROMPTS[args.dataset]
    all_classes = ", ".join(classes)
    n_desc = DEFAULT_N[args.dataset]
    todo = [c for c in classes if c not in pool]
    print(f"[stage magent] 三角色生成：{len(todo)} 类（生成→批评→修订）")

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

    _run_arm(args, classes, P, pool, raw_log, one_magent, todo,
             "magent", P["magent_pool"], P["magent_raw"])




def stage_omni(args, classes, P, suite):
    pool, raw_log = {}, {}
    if os.path.exists(P["omni_pool"]):
        pool = json.load(open(P["omni_pool"], encoding="utf-8"))
        print(f"[resume] omni池已有 {len(pool)}/{len(classes)} 类")
    if os.path.exists(P["omni_raw"]):
        raw_log = json.load(open(P["omni_raw"], encoding="utf-8"))

    onto = json.load(open(P["onto"], encoding="utf-8"))
    if len(onto) < len(classes):
        print(f"[omni] 本体不完整（{len(onto)}/{len(classes)}），先跑 --stage onto")
        sys.exit(1)

    omni = build_omni_suite(args.dataset)
    all_classes = ", ".join(classes)
    n_desc = DEFAULT_N[args.dataset]
    todo = [c for c in classes if c not in pool]
    print(f"[stage omni] 本体引导三角色：{len(todo)} 类（本体生成→本体批评→本体修订）")

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
                           omni["critic_omni"].format(
                               all_classes=all_classes, cls=cls,
                               drafts=numbered, **omni["dk"], **f),
                           args.temperature, 1800, f"critic:{cls}")
        critique = parse_json_object(raw_c) or {}

        raw_r = call_retry(args.api_base, args.api_key, args.model,
                           omni["refiner_omni"].format(
                               cls=cls, drafts=numbered,
                               too_generic=critique.get("too_generic", []) or "none",
                               likely_wrong=critique.get("likely_wrong", []) or "none",
                               missing="; ".join(critique.get("missing", [])[:4]) or "none",
                               n=n_desc, **omni["dk"], **f),
                           args.temperature, gen_max_tokens(n_desc), f"refine:{cls}")
        descs, dropped = parse_desc_array(raw_r, n_desc)
        if len(descs) < max(5, n_desc // 3):
            descs = drafts
        return cls, descs, {"gen": raw_g, "critique": raw_c, "refined": raw_r,
                            "n_final": len(descs), "fallback": descs is drafts}

    _run_arm(args, classes, P, pool, raw_log, one_omni, todo,
             "omni", P["omni_pool"], P["omni_raw"])


def _run_arm(args, classes, P, pool, raw_log, one_fn, todo, arm, pool_path, raw_path):
    pass
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
            if i % 10 == 0:
                json.dump(pool, open(pool_path, "w", encoding="utf-8"),
                          indent=2, ensure_ascii=False)
                json.dump(raw_log, open(raw_path, "w", encoding="utf-8"),
                          indent=1, ensure_ascii=False)
                print(f"  {arm}池 {i}/{len(todo)}", flush=True)
    json.dump(pool, open(pool_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(raw_log, open(raw_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{arm} 池 -> {pool_path}（{len(pool)}/{len(classes)}，失败 {len(failed)}）")
    for f in failed[:5]:
        print("  FAIL:", f)




def stage_eval(ds):
    import torch
    from e5_probe import load_test, encode_pool, mean_protos, k1_joint_fast, _norm
    sys.path.insert(0, os.path.join(REPO, "vendor", "clip"))
    from vendor.clip import clip

    P = paths_for(ds)
    n_desc = DEFAULT_N[ds]
    feats, labels, classes = load_test(ds)
    model, _ = clip.load("ViT-B/32", device="cpu",
                         download_root=os.path.expanduser(r"~\.cache\clip"))
    model.eval()
    name2idx = {_norm(c): i for i, c in enumerate(classes)}

    res = {"dataset": ds, "n_desc_per_class": n_desc, "baseline": {}, "sanity": {}}
    refs = SANITY_REFS[ds]


    cupl_path = os.path.join(DESC_DIR, f"{ds}.json")
    cupl_descs = json.load(open(cupl_path, encoding="utf-8"))
    cupl_keys = list(cupl_descs.keys())
    cupl_feats = torch.load(os.path.join(REPO, "cache", f"{ds}_desc_vitb32.pt"),
                            map_location="cpu", weights_only=False)["feats"].float()
    cupl_spans, s = [], 0
    for k in cupl_keys:
        n = len(cupl_descs[k])
        cupl_spans.append((s, s + n))
        s += n
    assert cupl_feats.shape[0] == s, \
        f"CuPL 缓存条数 {cupl_feats.shape[0]} != 池条数 {s}（缓存过期？）"
    try:
        cupl_remap = torch.tensor([name2idx[_norm(k)] for k in cupl_keys])
    except KeyError as e:

        try:
            cupl_remap = torch.tensor(
                [name2idx[_norm(CUBL_ALIAS.get(k, k))] for k in cupl_keys])
            print(f"[eval] CuPL 键名经别名表对齐（{e} -> 别名命中）")
        except KeyError:
            raise
    protos = mean_protos(cupl_feats, cupl_spans, cupl_remap)
    cupl_direct = ((feats @ protos.t()).argmax(1) == labels).float().mean().item() * 100
    cupl_k1 = k1_joint_fast(feats, labels, cupl_feats, cupl_spans, cupl_remap) * 100
    res["baseline"]["cupl"] = {"direct": round(cupl_direct, 2), "k1": round(cupl_k1, 2)}
    print(f"[cupl] direct={cupl_direct:.2f}  k1={cupl_k1:.2f}")


    if os.path.exists(P["attr_pool"]):
        df, spans, remap = encode_pool(model, ds, "regen_attr_gpt54",
                                       P["attr_pool"], classes)
        if len(spans) == len(classes):
            protos = mean_protos(df, spans, remap)
            d = ((feats @ protos.t()).argmax(1) == labels).float().mean().item() * 100
            k1 = k1_joint_fast(feats, labels, df, spans, remap) * 100
            res["baseline"]["attr"] = {"direct": round(d, 2), "k1": round(k1, 2)}
            print(f"[attr] direct={d:.2f}  k1={k1:.2f}")
        else:
            print("[eval] attr 池类不齐，跳过 attr 基线")


    for name in ("cupl", "attr"):
        if name not in res["baseline"] or name not in refs:
            continue
        rd, rk = refs[name]
        for metric, ref_v in (("direct", rd), ("k1", rk)):
            if ref_v is None:
                continue
            drift = round(res["baseline"][name][metric] - ref_v, 2)
            res["sanity"][f"{name}_{metric}_drift"] = drift
            if abs(drift) > 0.3:
                print(f"  [sanity] {name}.{metric} drift={drift:+.2f} vs ref {ref_v}")


    for arm in ("onto", "magent", "omni"):
        pool_path = P[f"{arm}_pool"]
        if not os.path.exists(pool_path):
            print(f"[eval] 缺 {pool_path}，跳过 {arm}")
            continue

        df, spans, remap = encode_pool(model, ds, f"regen_{arm}", pool_path, classes)
        if len(spans) != len(classes):
            missing = sorted(set(classes) - set(json.load(
                open(pool_path, encoding="utf-8")).keys()))
            print(f"[eval] {arm} 池不完整（{len(spans)}/{len(classes)} 类），"
                  f"缺: {missing}")
            print(f"       先重跑 --stage {arm} 补齐后再评估")
            continue
        protos = mean_protos(df, spans, remap)
        direct = ((feats @ protos.t()).argmax(1) == labels).float().mean().item() * 100
        k1 = k1_joint_fast(feats, labels, df, spans, remap) * 100


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
        uni = torch.cat(uni)
        u_protos = mean_protos(uni, u_spans, cupl_remap)
        u_direct = ((feats @ u_protos.t()).argmax(1) == labels).float().mean().item() * 100
        res[arm] = {"direct": round(direct, 2), "k1": round(k1, 2),
                    "union_direct": round(u_direct, 2),
                    "union_gain_vs_cupl": round(u_direct - cupl_direct, 2)}
        print(f"[{arm}] direct={direct:.2f}  k1={k1:.2f}  "
              f"union_direct={u_direct:.2f} (cupl {cupl_direct:.2f})")


    verdict = {}
    for arm in ("onto", "magent", "omni"):
        if arm not in res:
            continue
        d_gain = res[arm]["direct"] - cupl_direct
        k_gain = res[arm]["k1"] - cupl_k1
        verdict[arm] = {
            "direct_gain": round(d_gain, 2), "k1_gain": round(k_gain, 2),
            "GO": bool(d_gain >= 0.8 or k_gain >= 1.5),
        }
    res["verdict"] = verdict
    res["go_criteria"] = (f"direct>=+0.8pt or k1>=+1.5pt vs CuPL "
                          f"({cupl_direct:.2f}/{cupl_k1:.2f})")


    if "attr" in res["baseline"]:
        attr_d, attr_k1 = res["baseline"]["attr"]["direct"], res["baseline"]["attr"]["k1"]
        fa = {}
        for arm, tag in (("onto", "structure"), ("magent", "interaction"), ("omni", "both")):
            if arm in res:
                fa[tag] = {"delta_direct": round(res[arm]["direct"] - attr_d, 2),
                           "delta_k1": round(res[arm]["k1"] - attr_k1, 2)}
        res["factor_vs_attr"] = fa

    json.dump(res, open(P["feas"], "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n==== Decision (vs CuPL) ====")
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    if "factor_vs_attr" in res:
        print("==== 因子分解（vs attr） ====")
        print(json.dumps(res["factor_vs_attr"], ensure_ascii=False, indent=1))
    print(f"-> {P['feas']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="aircraft")
    ap.add_argument("--stage", choices=["onto", "check", "magent", "omni", "eval"],
                    required=True)
    ap.add_argument("--api-base", default=os.environ.get("E5_API_BASE", "https://api.openai.com/v1"))
    ap.add_argument("--model", default=os.environ.get("E5_MODEL", "gpt-4o-mini"))
    ap.add_argument("--api-key", default=os.environ.get("E5_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ds = args.dataset
    P = paths_for(ds)
    suite = build_suite(ds)
    classes = load_classes(ds)
    n_desc = DEFAULT_N[ds]
    print(f"{ds}: {len(classes)} 类，每类 {n_desc} 条")
    if args.stage == "onto":
        stage_onto(args, classes, P, suite)
    elif args.stage == "check":
        stage_check(classes, P)
    elif args.stage == "magent":
        stage_magent(args, classes, P, suite)
    elif args.stage == "omni":
        stage_omni(args, classes, P, suite)
    elif args.stage == "eval":
        stage_eval(ds)


if __name__ == "__main__":
    main()
