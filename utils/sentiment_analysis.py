# -*- coding: utf-8 -*-
"""
sentiment_stance_weibo.py
Step 3 : 情感 + 立场分析（微博）
-----------------------------------------------
输入 : weibo_cleaned.csv  (包含 clean_text 列)
输出 :
  weibo_sent_binary.csv   # neg_prob / pos_prob
  weibo_sent_emotion.csv  # 8-emotion 概率 (joy/anger/…/fear)
  weibo_stance.csv        # entail / neutral / contradict 概率
"""

import warnings, os, sys
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch, numpy as np
import torch.nn.functional as F
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ------------------------------------------------------------
# 0. 读入预处理微博
# ------------------------------------------------------------
# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'
OUTPUT_DIR = DATA_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CSV_FILE = DATA_DIR / "weibo_cleaned.csv"
assert CSV_FILE.exists(), f"找不到 {CSV_FILE}"

df = pd.read_csv(CSV_FILE)
df["clean_text"] = df["clean_text"].fillna("")
texts = df["clean_text"].tolist()
print(f"样本量：{len(df):,}")

DEVICE = 0 if torch.cuda.is_available() else -1   # GPU id or CPU

########################################################################
# 1. 二分类情感 Positive / Negative
########################################################################
print("\n🟢 Step 1 | Binary Sentiment (Pos/Neg)")

bin_model = "uer/roberta-base-finetuned-jd-binary-chinese"
sent_bin = pipeline(
    "sentiment-analysis",
    model=bin_model,
    tokenizer=bin_model,
    device=DEVICE,
    truncation=True,
    max_length=128,
)

neg_prob, pos_prob = [], []
for out in tqdm(sent_bin(texts, batch_size=32), total=len(df)):
    if out["label"].lower().startswith("pos"):
        pos_prob.append(out["score"])
        neg_prob.append(1 - out["score"])
    else:
        neg_prob.append(out["score"])
        pos_prob.append(1 - out["score"])

df_bin = df.copy()
df_bin["neg_prob"] = neg_prob
df_bin["pos_prob"] = pos_prob
df_bin.to_csv(OUTPUT_DIR / "weibo_sent_binary.csv", index=False, encoding='utf-8')
print("✅ Binary 情感结果已写入 weibo_sent_binary.csv")

########################################################################
# 2. 八类情绪 (Joy / Anger / Fear / …)
########################################################################
print("\n🟢 Step 2 | 8-Class Emotion")

emo_model = "Johnson8187/Chinese-Emotion"   # 预训练 8 类情绪
emo_pipe = pipeline(
    "text-classification",
    model=emo_model,
    tokenizer=emo_model,
    device=DEVICE,
    top_k=None,
    truncation=True,
    max_length=128,
)

# Emotion 标签顺序以模型 config.json 里的 id2label 为准
emo_labels = emo_pipe.model.config.id2label.values()  # 保证顺序
emo_columns = [f"emo_{lbl.lower()}" for lbl in emo_labels]

prob_mat = np.zeros((len(df), len(emo_labels)), dtype=np.float32)
for i, outs in enumerate(tqdm(emo_pipe(texts, batch_size=32),
                              total=len(df))):
    for out in outs:                        # outs 是 list[dict]
        idx = list(emo_labels).index(out["label"])
        prob_mat[i, idx] = out["score"]

df_emo = pd.concat([df, pd.DataFrame(prob_mat, columns=emo_columns)], axis=1)
df_emo.to_csv(OUTPUT_DIR / "weibo_sent_emotion.csv", index=False, encoding='utf-8')
print("✅ 8-情绪概率已写入 weibo_sent_emotion.csv")

########################################################################
# 3. 立场判定 (支持 / 反对 / 中立) —— NLI 零样本
########################################################################
print("\n🟢 Step 3 | Stance Zero-Shot (Entail / Neutral / Contradict)")

stance_model_name = "IDEA-CCNL/Erlangshen-Roberta-110M-NLI"
tok_stance = AutoTokenizer.from_pretrained(stance_model_name)
mdl_stance = AutoModelForSequenceClassification.from_pretrained(
    stance_model_name
).to(torch.device("cuda" if DEVICE >= 0 else "cpu")).eval()

# 定义议题 & 构造 hypothesis
TOPIC_CN = "恐婚或恐育"
HYP_ENTAIL = f"这条微博支持{TOPIC_CN}"
HYP_CONTRA = f"这条微博反对{TOPIC_CN}"

batch = 16
entail_p, neut_p, contra_p = [], [], []

with torch.no_grad():
    for i in tqdm(range(0, len(df), batch), desc="stance infer"):
        premise_batch = texts[i:i+batch]

        # 分别与两个假设做 NLI
        inputs_yes = tok_stance(premise_batch, [HYP_ENTAIL]*len(premise_batch),
                                padding=True, truncation=True,
                                max_length=128, return_tensors="pt")
        inputs_no  = tok_stance(premise_batch, [HYP_CONTRA]*len(premise_batch),
                                padding=True, truncation=True,
                                max_length=128, return_tensors="pt")

        inputs_yes = {k:v.to(mdl_stance.device) for k,v in inputs_yes.items()}
        inputs_no  = {k:v.to(mdl_stance.device) for k,v in inputs_no.items()}

        logit_yes = mdl_stance(**inputs_yes).logits          # [B,3]
        logit_no  = mdl_stance(**inputs_no).logits

        prob_yes = F.softmax(logit_yes, dim=-1).cpu().numpy()
        prob_no  = F.softmax(logit_no,  dim=-1).cpu().numpy()

        # 采用 Yes-Entail 概率作为"支持"，No-Contradict 作为"反对" (简化)
        entail_p.extend(prob_yes[:, 0])       # entail
        neut_p.extend((prob_yes[:, 1] + prob_no[:, 1]) / 2)  # neutral
        contra_p.extend(prob_no[:, 0])        # entail (反对命题)

df_stance = df.copy()
df_stance["stance_entail"] = entail_p
df_stance["stance_neutral"] = neut_p
df_stance["stance_contra"] = contra_p
df_stance.to_csv(OUTPUT_DIR / "weibo_stance.csv", index=False, encoding='utf-8')
print("✅ 立场概率已写入 weibo_stance.csv")

print("\n🎉 全部完成！结果文件：")
print("  • weibo_sent_binary.csv")
print("  • weibo_sent_emotion.csv")
print("  • weibo_stance.csv")
