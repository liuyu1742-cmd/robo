import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "meta" / "compliance_catalog_15x120.json").open("r", encoding="utf-8") as file:
    _CATALOG = json.load(file)
with (ROOT / "datasets" / "standard_instruction_action_benchmark.json").open("r", encoding="utf-8") as file:
    _BENCHMARK = json.load(file)

TASK_VOCAB = {task["id"]: index for index, task in enumerate(_CATALOG["tasks"])}
_objects = [obj for task in _CATALOG["tasks"] for obj in task["objects"]]
OBJECT_VOCAB = {obj: index for index, obj in enumerate(_objects)}
_targets = list(dict.fromkeys(case["target"] for case in _BENCHMARK))
TARGET_VOCAB = {target: index for index, target in enumerate(_targets)}
TASK_TARGETS = {case["task"]: case["target"] for case in _BENCHMARK}

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
_verbs = []
for case in _BENCHMARK:
    for action in case["actions"]:
        verb = action.split("(", 1)[0]
        if verb not in _verbs:
            _verbs.append(verb)
ACTION_VOCAB = {"<pad>": PAD_ID, "<bos>": BOS_ID, "<eos>": EOS_ID}
ACTION_VOCAB.update({verb: index + 3 for index, verb in enumerate(_verbs)})
ID_TO_ACTION = {value: key for key, value in ACTION_VOCAB.items()}

ARG_PAD_ID, ARG_NONE_ID = 0, 1
_arguments = list(dict.fromkeys([*_objects, *_targets]))
ARG_VOCAB = {"<pad>": ARG_PAD_ID, "<none>": ARG_NONE_ID}
for argument in _arguments:
    if argument not in ARG_VOCAB:
        ARG_VOCAB[argument] = len(ARG_VOCAB)
ID_TO_ARG = {value: key for key, value in ARG_VOCAB.items()}


def _encode(value, vocab, kind):
    if value not in vocab:
        raise ValueError(f"Unknown {kind}: {value}. Available count: {len(vocab)}")
    return vocab[value]


def encode_task(task):
    return _encode(task, TASK_VOCAB, "task")


def encode_object(obj):
    return _encode(obj, OBJECT_VOCAB, "object")


def encode_target(target):
    return _encode(target, TARGET_VOCAB, "target")


def default_target(task):
    return _encode(task, TASK_TARGETS, "task target")


def parse_action(action):
    if "(" not in action or not action.endswith(")"):
        raise ValueError(f"Action must use verb(argument) syntax: {action}")
    return action[:-1].split("(", 1)


def encode_plan(actions):
    verb_ids, argument_ids = [BOS_ID], [ARG_NONE_ID]
    for action in actions:
        verb, argument = parse_action(action)
        verb_ids.append(_encode(verb, ACTION_VOCAB, "action"))
        argument_ids.append(_encode(argument, ARG_VOCAB, "argument"))
    verb_ids.append(EOS_ID)
    argument_ids.append(ARG_NONE_ID)
    return verb_ids, argument_ids


def encode_actions(actions):
    return encode_plan(actions)[0]


def decode_actions(action_ids):
    return [ID_TO_ACTION[action_id] for action_id in action_ids
            if action_id not in (PAD_ID, BOS_ID, EOS_ID)]


def decode_plan(plan):
    return [f"{ID_TO_ACTION[verb]}({ID_TO_ARG[argument]})"
            for verb, argument in plan]
