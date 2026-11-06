#!/usr/bin/env python3
"""
gguf_deepeval.py - Task accuracy for one GGUF model with deepeval + llama-cpp-python.

Usage:
    python gguf_deepeval.py path/to/model.gguf
    python gguf_deepeval.py model.gguf --tasks mmlu gsm8k --limit 20 --ngl 99

Output: one row per model appended to a CSV:
    model, mmlu, arc-challenge, ..., gsm8k   (accuracy 0-1)

Answer extraction uses deepeval's own schemas: deepeval calls
generate(prompt, schema=...) and reads `.answer`. We force llama.cpp to output
JSON matching that schema with a grammar, so no regex parsing is needed.

Requires: pip install deepeval datasets llama-cpp-python   (built with CUDA for GPU)
"""
import argparse
import csv
import json
from datetime import datetime
import re
import sys
from pathlib import Path

from llama_cpp import Llama, LlamaGrammar
from pydantic import ValidationError
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.dataset import Golden
from deepeval.benchmarks import MMLU, ARC, HellaSwag, Winogrande, TruthfulQA, GSM8K,MathQA
from deepeval.benchmarks.arc.mode import ARCMode

# deepeval still uses old un-namespaced dataset IDs, which new huggingface_hub
# rejects ("Repository id must be 'namespace/name'"). Remap them.
import datasets
_HF_RENAMES = {"gsm8k": "openai/gsm8k", "ai2_arc": "allenai/ai2_arc",
               "truthful_qa": "truthfulqa/truthful_qa"}
_orig_load_dataset = datasets.load_dataset


def _load_dataset(path, *args, **kwargs):
    return _orig_load_dataset(_HF_RENAMES.get(path, path), *args, **kwargs)
# AIME: the model's own final answer, e.g. "\boxed{204}"
BOXED_RE = re.compile(r"\\boxed\{\s*(\d+)\s*\}")
AIME_PROMPT = ("Solve the following math problem step by step. "
               "Put your final answer, an integer from 0 to 999, in \\boxed{{}}.\n\n"
               "Problem: {problem}\n\nSolution:")

datasets.load_dataset = _load_dataset

TASKS = ["mmlu", "arc-challenge", "arc-easy", "hellaswag", "winogrande", "truthfulqa", "gsm8k","mathqa"]
MAX_TOKENS = 32  # enough for {"answer": ...}
# GSM8K: the answer line the model writes itself, e.g. "**Answer**: 18"
ANSWER_RE = re.compile(r"\*\*Answer\*\*:?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")


def make_benchmark(task: str, limit):
    if task == "mmlu":           # limit = problems per subject (57 subjects)
        return MMLU()
    if task == "arc-challenge":
        return ARC(mode=ARCMode.CHALLENGE, n_problems=limit)
    if task == "arc-easy":
        return ARC(mode=ARCMode.EASY, n_problems=limit)
    if task == "hellaswag":      # limit = problems per category
        return HellaSwag(n_problems_per_task=limit)
    if task == "winogrande":
        return Winogrande(n_problems=limit) if limit else Winogrande()
    if task == "truthfulqa":
        return TruthfulQA(n_problems_per_task=limit)
    if task == "gsm8k":
        return GSM8K(n_problems=limit) if limit else GSM8K()
    if task == "mathqa":         # limit = problems per category (6 categories)
        return MathQA(n_problems_per_task=limit)

    raise ValueError(task)


class GGUFModel(DeepEvalBaseLLM):
    """Wraps a GGUF file (llama-cpp-python) for deepeval benchmarks."""

    def __init__(self, path: Path, n_gpu_layers: int, n_ctx: int):
        self.path, self.n_gpu_layers, self.n_ctx = path, n_gpu_layers, n_ctx
        self.grammars = {}   # one compiled grammar per deepeval schema
        self.debug_left = 0
        self.errors = 0
        self.task = ""
        self.cot_tokens = 512
        self.repeat_penalty = 1.0  # llama-cpp-python default (no penalty)
        self.truncated = 0
        self.no_answer = 0
        self.last_output, self.last_answer = "", None  # for the answers log
        super().__init__(model=path.stem)  # calls load_model()

    def load_model(self):
        return Llama(model_path=str(self.path), n_gpu_layers=self.n_gpu_layers,
                     n_ctx=self.n_ctx, verbose=False)

    def _grammar(self, schema):
        if schema not in self.grammars:
            js = json.dumps(schema.model_json_schema())
            self.grammars[schema] = LlamaGrammar.from_json_schema(js, verbose=False)
        return self.grammars[schema]

    def generate(self, prompt: str, schema=None):
        if self.task == "gsm8k" and schema is not None:
            return self._generate_gsm8k(prompt, schema)
        try:
            out = self.model.create_completion(
                prompt, max_tokens=MAX_TOKENS, temperature=0.7,
                grammar=self._grammar(schema) if schema else None)
            text = out["choices"][0]["text"]
        except ValueError as e:  # e.g. prompt longer than n_ctx
            self.errors += 1
            print(f"  [!] generation error: {e}")
            text = ""
        self._debug(prompt, text)
        self.last_output, self.last_answer = text, None

        if schema is None:
            return text.strip()
        try:
            res = schema.model_validate_json(text)         # deepeval reads .answer
        except ValidationError:                           # invalid -> scored wrong
            self.errors += 1
            res = schema.model_construct(answer=None)
        self.last_answer = {"answer": res.answer}
        return res

    def _generate_gsm8k(self, prompt: str, schema):
        """Free reasoning, then take ONLY the answer the model itself wrote.

        No second forced-answer step: if the model never writes "**Answer**: N"
        (e.g. it loops or hits --cot-tokens), the answer is None -> scored wrong.
        Nothing is ever taken from the prompt.
        """
        try:
            choice = self.model.create_completion(
                prompt, max_tokens=self.cot_tokens, temperature=0.0,
                repeat_penalty=self.repeat_penalty,
                stop=["**Problem**"])["choices"][0]
            text = choice["text"]
            if choice.get("finish_reason") == "length":
                self.truncated += 1
        except ValueError as e:
            self.errors += 1
            print(f"  [!] generation error: {e}")
            text = ""

        # the model's own answer line, in the few-shot format "**Answer**: 18"
        m = ANSWER_RE.search(text)
        answer = None
        if m:
            n = float(m.group(1).replace(",", ""))
            answer = int(n) if n.is_integer() else None  # GSM8K answers are integers
        if answer is None:
            self.no_answer += 1
        self._debug(prompt, text, final=json.dumps({"answer": answer}))
        self.last_output, self.last_answer = text, {"answer": answer}
        return schema.model_construct(answer=answer)

    def _debug(self, prompt: str, text: str, final: str = None):
        if self.debug_left > 0:  # full prompt + model output (+ answer sent to deepeval)
            self.debug_left -= 1
            bar = "=" * 70
            msg = f"\n{bar}\nPROMPT ({self.task})\n{bar}\n{prompt}\n{bar}\nOUTPUT\n{bar}\n{text}\n{bar}\n"
            if final is not None:
                msg += f"FINAL ANSWER (JSON)\n{bar}\n{final}\n{bar}\n"
            print(msg)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.path.stem


def live_predictions(bench, f, n_print: int, run_info: dict):
    """Wrap deepeval's per-question predict() so each scored question is
    appended to the JSONL file (and flushed) immediately, with the correct answer."""
    orig_predict = bench.predict
    count = [0]

    def predict(model, *a, **kw):
        res = orig_predict(model, *a, **kw)
        args = list(a) + list(kw.values())
        golden = next(x for x in args if isinstance(x, Golden))
        sub = next((x.value for x in args if hasattr(x, "value") and not isinstance(x, Golden)), "")
        count[0] += 1
        record = {
            **run_info, "n": count[0], "subtask": sub,
            "input": golden.input,
            "output": model.last_output,          # raw model text
            "final_answer": model.last_answer,    # JSON answer sent to deepeval
            "prediction": res["prediction"],
            "expected": golden.expected_output,
            "correct": bool(res["score"]),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        if count[0] <= n_print:
            mark = "OK " if res["score"] else "BAD"
            print(f"   [{mark}] predicted={str(res['prediction'])!r:8} correct={golden.expected_output!r}")
        return res  # deepeval unpacks this dict, so return it unchanged

    bench.predict = predict


def append_row(out: Path, row: dict):
    rows, fields = [], list(row)
    if out.exists():  # keep old rows, merge columns
        with out.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            fields = list(rows[0]) + [k for k in row if k not in rows[0]]
    rows.append(row)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Benchmark one GGUF with deepeval")
    p.add_argument("model", type=Path, help="path to .gguf file")
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    p.add_argument("--out", type=Path, default=Path("gguf_results.csv"))
    p.add_argument("--ngl", type=int, default=-1, help="GPU layers (-1 = all)")
    p.add_argument("--ctx", type=int, default=5096, help="context size (few-shot prompts are long)")
    p.add_argument("--limit", type=int, default=None, help="problems per task/subject (default: all)")
    p.add_argument("--debug", type=int, default=3, help="print full prompt + output for first N prompts per task (0 = off)")
    p.add_argument("--pred-dir", type=Path, default=Path("predictions-0.7"),
                   help="folder for per-question JSONL logs (appended, never overwritten)")
    p.add_argument("--cot-tokens", type=int, default=5000, help="max reasoning tokens for GSM8K")
    p.add_argument("--repeat-penalty", type=float, default=1.0,
                   help="GSM8K reasoning repeat penalty (1.0 = off; ~1.1 curbs loops)")
    args = p.parse_args()

    if not args.model.is_file():
        sys.exit(f"Model not found: {args.model}")

    model = GGUFModel(args.model, args.ngl, args.ctx)
    run_id = datetime.now().isoformat(timespec="seconds")  # tells appended runs apart
    row = {"model": args.model.stem}
    for task in args.tasks:
        print(f"\n[{args.model.stem}] {task}")
        model.debug_left = args.debug
        model.task = task
        model.cot_tokens = args.cot_tokens
        model.repeat_penalty = args.repeat_penalty
        bench = make_benchmark(task, args.limit)

        args.pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = args.pred_dir / f"{args.model.stem}__{task}.jsonl"
        print(f"  answers -> {pred_file} (appended live)")
        with pred_file.open("a", encoding="utf-8") as f:  # append, never overwrite
            live_predictions(bench, f, args.debug,
                             {"run": run_id, "model": args.model.stem, "task": task})
            bench.evaluate(model=model)
        row[task] = round(bench.overall_score, 4)
        print(f"  {task}: {row[task]}")

    if model.no_answer:
        print(f"\n[!] GSM8K: {model.no_answer} problems had no answer from the model -> None (scored wrong); "
              f"{model.truncated} of the outputs hit --cot-tokens.")
    if model.errors:
        print(f"\n[!] {model.errors} prompts failed (scored as wrong). Try a larger --ctx.")
    append_row(args.out, row)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()