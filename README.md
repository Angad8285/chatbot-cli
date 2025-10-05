# Local Command-Line Chatbot

Lightweight local chatbot supporting both legacy dialogue models (DialoGPT) and modern chat-template models (TinyLlama). Provides a sliding window memory, configurable decoding, and simple factual shortcuts (arithmetic, capital cities).

## Contents
1. Quick Start
2. Running & Basic Usage
3. Models: Chat vs Non‑Chat
4. Key CLI Flags
5. Recommended Presets
6. TinyLlama Usage
7. Shortcuts (Math & Capitals)
8. Troubleshooting Table
9. Saving & Exporting
10. Roadmap

---
## 1. Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py  # defaults to TinyLlama/TinyLlama-1.1B-Chat-v1.0 with tuned hyperparameters
```

Legacy (non-chat) example using DialoGPT-small:
```bash
python main.py --model microsoft/DialoGPT-small --temperature 0.55 --top-p 0.9 --top-k 50
```

## 2. Running & Basic Usage
Start the REPL:
```bash
python main.py --profile focused
```
Commands inside REPL:
`/help` `/reset` `/save [file]` `/config` `/exit`

## 3. Models: Chat vs Non‑Chat
| Aspect | Non‑Chat (DialoGPT) | Chat Template (TinyLlama) |
|--------|---------------------|---------------------------|
| Prompt format | Hand-built "User:/Bot:" lines | Tokenizer chat template | 
| System prompt | Injected text line | System role message | 
| Few-shot need | Helps a lot | Usually optional / minimal | 
| Factual QA | Weak | Better | 
| Long answers | Prone to drift | More stable |

## 4. Key CLI Flags (Selected)
`--model` choose model.
`--window` memory turns retained.
`--temperature` creativity (default 0.55).
`--top-p` nucleus cutoff (default 0.9).
`--top-k` hard candidate cap (default 50).
`--repetition-penalty` discourage loops (>1.0, default 1.18).
`--no-repeat-ngram` block exact n-grams (default 3).
`--max-new-tokens` length ceiling per reply (default 160).
`--min-new-tokens` floor to avoid one-word answers (default 10).
`--system-prompt` override persona.
`--torch-dtype` force precision (bfloat16/float16).

Defaults tuned for balanced factual Q&A while allowing moderate elaboration. Adjust downward for ultra-terse replies or upward (temperature/top-p) for ideation.

## 5. Recommended Presets
Focused Q&A:
```bash
python main.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
	--temperature 0.55 --top-p 0.9 --top-k 50 \
	--repetition-penalty 1.18 --no-repeat-ngram 3 \
	--max-new-tokens 160 --min-new-tokens 10
```
Creative Ideation:
```bash
python main.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
	--temperature 0.7 --top-p 0.95 --top-k 60 \
	--repetition-penalty 1.08 --max-new-tokens 180
```
Legacy Stable (DialoGPT medium):
```bash
python main.py --model microsoft/DialoGPT-medium \
	--temperature 0.5 --top-p 0.88 --repetition-penalty 1.22 \
	--no-repeat-ngram 3 --max-new-tokens 55
```

## 6. TinyLlama Usage
The tokenizer supplies a `chat_template`. We build a list of messages (system, user, assistant turns) then call `apply_chat_template(..., add_generation_prompt=True)`. Decoding is then standard `text-generation`.

## 7. Shortcuts (Math & Capitals)
Simple regex intercepts convert:
* Arithmetic like `what is 12 * 7` → computed locally.
* Capital queries (`capital of france`) → quick dictionary answer.

## 8. Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| One-word / blank replies | Too low min tokens | Increase `--min-new-tokens` (8–12) |
| Fragmented punctuation spam | High entropy tail | Lower `--temperature`, add `--top-k 50` |
| Repeated phrases | Weak penalties | Raise `--repetition-penalty` or `--no-repeat-ngram 4` |
| Hallucinated role labels | Legacy format drift | Use chat model or keep few-shot minimal |
| Wrong simple math | Model hallucination | Arithmetic shortcut already handles basic cases |
| Capital answer wrong | Outside dictionary | Extend capitals map or use larger model |

## 9. Saving & Exporting
Use `/save my.json` inside session to write transcript (system prompt & few-shot excluded).

## 10. Roadmap
- Optional: `--no-few-shot` flag
- External configurable shortcuts
- Retrieval augmentation stub
- Evaluation script for deterministic test set

---
*Status:* Active experimental prototype; expect iterative refinement.

