# Local Command-Line Chatbot (TinyLlama Focused)

Lightweight local chatbot optimized for a modern chat-template model (TinyLlama 1.1B Chat). Sliding window memory, configurable decoding, and simple factual shortcuts (arithmetic, capital cities). Legacy non-chat prompt support has been removed.

## Contents
1. Quick Start
2. Running & Basic Usage
3. Key CLI Flags
4. Recommended Presets
5. Chat Flow
6. Shortcuts (Math & Capitals)
7. Troubleshooting Table
8. Saving & Exporting
9. Roadmap

---
## 1. Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py  # loads TinyLlama with tuned defaults
```

To try another chat model (must have a chat template):
```bash
python main.py --model <other-chat-model>
```

## 2. Running & Basic Usage
Start the REPL:
```bash
python main.py --profile focused
```
Commands: `/help` `/reset` `/save [file]` `/config` `/exit`

## 3. Key CLI Flags (Selected)
`--model` model id (default TinyLlama).
`--window` memory turns retained.
`--temperature` creativity (default 0.55).
`--top-p` nucleus cutoff (default 0.9).
`--top-k` hard candidate cap (default 50).
`--repetition-penalty` discourage loops (default 1.18).
`--no-repeat-ngram` block exact n-grams (default 3).
`--max-new-tokens` reply max length (default 160).
`--min-new-tokens` reply min length (default 10).
`--system-prompt` override persona.
`--torch-dtype` precision (bfloat16/float16).

Defaults tuned for balanced factual Q&A while allowing moderate elaboration.

## 4. Recommended Presets
Focused Q&A:
```bash
python main.py --temperature 0.55 --top-p 0.9 --top-k 50 \
  --repetition-penalty 1.18 --no-repeat-ngram 3 \
  --max-new-tokens 160 --min-new-tokens 10
```
Creative Ideation:
```bash
python main.py --temperature 0.7 --top-p 0.95 --top-k 60 \
  --repetition-penalty 1.08 --max-new-tokens 180
```

## 5. Chat Flow
Messages (system + alternating user/assistant) are assembled and converted using the tokenizer's `chat_template` with `add_generation_prompt=True`.

## 6. Shortcuts (Math & Capitals)
- Arithmetic like `what is 12 * 7` computed locally.
- Capital queries (`capital of france`) answered via small dictionary.

## 7. Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| One-word / blank replies | Too low min tokens | Increase `--min-new-tokens` (8–12) |
| Fragmented punctuation spam | High entropy tail | Lower `--temperature`, add `--top-k 50` |
| Repeated phrases | Weak penalties | Raise `--repetition-penalty` or `--no-repeat-ngram 4` |
| Wrong simple math | Model hallucination | Arithmetic shortcut handles basics |
| Capital answer wrong | Outside dictionary | Extend capitals map or use larger model |

## 8. Saving & Exporting
`/save my.json` writes transcript (system prompt & few-shot excluded).

## 9. Roadmap
- Optional: `--no-few-shot` flag
- External configurable shortcuts
- Retrieval augmentation stub
- Evaluation script for deterministic test set

---
*Status:* Active experimental prototype (TinyLlama focused); expect iterative refinement.

