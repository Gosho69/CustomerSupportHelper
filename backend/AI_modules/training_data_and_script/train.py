import os, json, re
from statistics import mean
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from transformers.trainer_callback import EarlyStoppingCallback

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

TRAIN = "data/train_chat_en_gpt4.jsonl"
VAL   = "data/val_chat_en_gpt4.jsonl"

MODEL = "google/flan-t5-base"

REQUIRE_OVERALL = True
MAX_IN, MAX_OUT = 512, 192       
BATCH, GRAD_ACC = 2, 8 

SYSTEM = (
    "You are a call quality analyst. Analyze customer service call transcripts and provide:\n"
    "1. A clear, human-friendly SUMMARY (2-3 sentences) explaining: what the customer needed, "
    "how the agent helped, and what the outcome was.\n"
    "2. CUSTOMER TONE: The customer's emotional state (Positive, Negative, Neutral, Frustrated, Satisfied).\n"
    "3. AGENT TONE: The agent's approach (Positive, Professional, Apologetic, Helpful, Dismissive).\n"
    "4. RATINGS: Rate agent performance on a 1-5 scale where 1=poor, 3=average, 5=excellent.\n"
    "   - helpfulness: Did agent solve the problem?\n"
    "   - respect: Was agent courteous and professional?\n"
    "   - clarity: Was communication clear?\n"
    "   - adherence: Did agent follow procedures?\n"
    "   - overall: Overall service quality (required).\n\n"
    "Output as valid JSON with keys: summary, customer_tone, agent_tone, ratings."
)
TAIL = "Provide your analysis as valid JSON starting with '{' and ending with '}'."

def build_rows(path, tok):
    rows = []
    head = f"[SYSTEM]\n{SYSTEM}\n[/SYSTEM]\n[USER]\nTranscript:\n"
    tail = f"\n[/USER]\n{TAIL}"

    head_ids = tok(head, add_special_tokens=False).input_ids
    tail_ids = tok(tail, add_special_tokens=False).input_ids
    assert len(head_ids) + len(tail_ids) < MAX_IN - 32, "Instruction too long for MAX_IN. Shorten SYSTEM/TAIL."

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            msgs = o["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            target_raw = next(m["content"] for m in msgs if m["role"] == "assistant")
            tgt = json.loads(target_raw)

            if REQUIRE_OVERALL:
                ov = tgt.get("ratings", {}).get("overall")
                if not isinstance(ov, (int, float)) or not (1 <= int(ov) <= 5):
                    continue

            budget = MAX_IN - (len(head_ids) + len(tail_ids)) - 4
            if budget < 64: budget = 64
            u_ids = tok(user, add_special_tokens=False, truncation=True, max_length=budget+64).input_ids[:budget]
            user_trim = tok.decode(u_ids, skip_special_tokens=True)

            prompt = head + user_trim + tail
            rows.append({"input_text": prompt, "target_text": target_raw})
    return Dataset.from_list(rows)

tok = AutoTokenizer.from_pretrained(MODEL)
train_ds = build_rows(TRAIN, tok)
val_ds   = build_rows(VAL, tok)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
model.config.use_cache = False
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

def preprocess(b):
    x = tok(b["input_text"], max_length=MAX_IN, truncation=True)
    y = tok(text_target=b["target_text"], max_length=MAX_OUT, truncation=True)
    x["labels"] = y["input_ids"]
    return x

train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
val_tok   = val_ds.map(preprocess,   batched=True, remove_columns=val_ds.column_names)

def parse_json(txt):
    try: return json.loads(txt)
    except: 
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else None

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    
    vocab_size = tok.vocab_size
    preds = [[min(max(token_id, 0), vocab_size - 1) for token_id in seq] for seq in preds]
    
    pred_text = tok.batch_decode(preds, skip_special_tokens=True)
    labels[labels == -100] = tok.pad_token_id
    ref_text  = tok.batch_decode(labels, skip_special_tokens=True)

    n = len(pred_text)
    json_ok = 0; exact = 0; within1 = 0; paired = 0
    abs_err = []

    for p, r in zip(pred_text, ref_text):
        pj = parse_json(p); rj = parse_json(r)
        if pj and rj:
            json_ok += 1
            try:
                po = int(pj["ratings"]["overall"])
                ro = int(rj["ratings"]["overall"])
                paired += 1
                exact   += int(po == ro)
                within1 += int(abs(po - ro) <= 1)
                abs_err.append(abs(po - ro))
            except:
                pass

    mae = mean(abs_err) if abs_err else float("nan")
    return {
        "json_valid": json_ok / max(n,1),
        "overall_exact": exact / max(n,1),
        "overall_within1": within1 / max(n,1),
        "overall_mae": mae
    }

args = Seq2SeqTrainingArguments(
    output_dir="out-flan-sft",
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    learning_rate=3e-5,                 
    num_train_epochs=5,                 
    per_device_train_batch_size=1,     
    per_device_eval_batch_size=1,      
    gradient_accumulation_steps=16,     
    gradient_checkpointing=True,
    predict_with_generate=True,
    dataloader_pin_memory=False,
    max_grad_norm=0.5,
    fp16=False,                        
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    metric_for_best_model="overall_within1",
    greater_is_better=True,
    load_best_model_at_end=True,
    generation_max_length=MAX_OUT,
    generation_num_beams=6,         
)

trainer = Seq2SeqTrainer(
    model=model, tokenizer=tok, args=args,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tok, model=model),
    train_dataset=train_tok, eval_dataset=val_tok,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()
trainer.save_model("model/final")
tok.save_pretrained("model/final")