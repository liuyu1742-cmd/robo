import json
from pathlib import Path

try:
    from tools.project_conditions import MANUAL_SOURCE, OPERATION_TO_VERB
except ModuleNotFoundError:
    from project_conditions import MANUAL_SOURCE, OPERATION_TO_VERB


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "project_action_dataset.json"
OUT = ROOT / "meta" / "project_condition_vocab.json"


def main():
    with DATASET.open("r", encoding="utf-8") as file:
        data = json.load(file)
    sources = list(dict.fromkeys(item["source"] for item in data))
    if MANUAL_SOURCE not in sources:
        sources.append(MANUAL_SOURCE)
    verbs = list(dict.fromkeys(item["verb"] for item in data))
    for verb in OPERATION_TO_VERB.values():
        if verb not in verbs:
            verbs.append(verb)
    vocab = {
        "source_file": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "counts": {"sources": len(sources), "verbs": len(verbs)},
        "sources": sources,
        "verbs": verbs,
    }
    with OUT.open("w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False, indent=2)
    print(json.dumps(vocab["counts"], ensure_ascii=False, indent=2))
    print(f"output: {OUT}")


if __name__ == "__main__":
    main()
