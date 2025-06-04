# -*- coding: utf-8 -*-
"""
清洗 & 预处理：微博"年轻人恐婚恐育的真正原因是什么"话题
步骤：
1. 载入数据 ➜ 去重 / 过滤
2. 文本清洗（去链接、表情、@、标点等）
3. Datetime 解析
4. 分词 + 停用词过滤 + 同义词统一
5. 保存清洗结果
"""

import re
from pathlib import Path
import os

import jieba            # pip install jieba
import pandas as pd      # pip install pandas

# -------------------------------------------------
# Step 0 载入原始 CSV
# -------------------------------------------------
DATA_PATH   = r"E:\ccs\ccs\final\年轻人恐婚恐育的真正原因是什么.csv"   # ★根据实际路径调整
df_raw      = pd.read_csv(DATA_PATH)

print("列名预览：", df_raw.columns.tolist()[:10])  # ★确认字段

# 微博正文列名
TEXT_COL    = "content"
# 微博时间列名（若无可忽略 Step 3）
TIME_COL    = "created_at"

# -------------------------------------------------
# Step 1 基础去重、过滤短文本
# -------------------------------------------------
df = (
    df_raw
    .drop_duplicates(subset=[TEXT_COL])              # 内容重复
    .query(f"{TEXT_COL}.str.len() >= 10", engine="python")  # <10 字视为噪声
    .copy()
)

# -------------------------------------------------
# Step 2 文本清洗
# -------------------------------------------------
url_re      = re.compile(r"https?://\S+|www\.\S+")
emoji_re    = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
punct_re    = re.compile(r"[^\w\s]")                 # 非字母数字下划线

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = url_re.sub(" ", text)                     # 去链接
    text = emoji_re.sub(" ", text)                   # 去 emoji
    text = re.sub(r"//@.+", " ", text)               # 去"转发前缀"
    text = punct_re.sub(" ", text)                   # 去标点符号
    text = re.sub(r"\s+", " ", text).strip()         # 多空白压缩
    return text

df["clean_text"] = df[TEXT_COL].apply(clean_text)

# -------------------------------------------------
# Step 3 解析时间列（若有 TIME_COL）
# -------------------------------------------------
if TIME_COL in df.columns:
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.sort_values(TIME_COL)

# -------------------------------------------------
# Step 4 分词 + 停用词 + 同义词
# -------------------------------------------------
# 4.1 加载停用词表 ---------------------------------
STOP_PATH = "stopwords_zh.txt"      # 自备或公开停用词
stopwords = set()
if Path(STOP_PATH).exists():
    stopwords = set(Path(STOP_PATH).read_text(encoding="utf-8").splitlines())

# 4.2 同义词映射 ------------------------------------
SYN_MAP = {
    "生孩子": "生育",
    "要娃":   "生育",
    "不婚":   "恐婚",
    "不生":   "恐育",
    "结婚":   "婚姻",
    # ……继续补充
}

def unify(word: str) -> str:
    return SYN_MAP.get(word, word)

def tokenize(text: str):
    tokens = [
        unify(w)
        for w in jieba.lcut(text, cut_all=False)
        if w and w not in stopwords and len(w.strip()) > 1
    ]
    return tokens

df["tokens"]    = df["clean_text"].apply(tokenize)
df["token_str"] = df["tokens"].apply(lambda ts: " ".join(ts))  # 供 LDA/BERTopic 使用

# -------------------------------------------------
# Step 5 持久化
# -------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "weibo_cleaned.csv"
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"✅ 清洗完成，已保存为 {OUTPUT_FILE}")
