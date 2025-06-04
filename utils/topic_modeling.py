# -*- coding: utf-8 -*-
"""
step2_topic_detection.py
--------------------------------------------
主体议题自动发现
  • BERTopic (自动合并小主题)
  • LDA 基准 (k=18) + k 调优
  • 所有可视化写入 docs/outputs/
执行:
  python step2_topic_detection.py
"""

import warnings, ast, json, multiprocessing
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch

warnings.filterwarnings("ignore")
multiprocessing.freeze_support()

# ============================================================
# 0. 路径 & 读入数据
# ============================================================
SCRIPT_DIR   = Path(__file__).parent.absolute()
OUTPUTS_DIR  = SCRIPT_DIR / "docs" / "outputs";      OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR   = SCRIPT_DIR / "models";       MODELS_DIR.mkdir(exist_ok=True)
DATA_FILE    = SCRIPT_DIR / "weibo_cleaned.csv"
assert DATA_FILE.exists(), f"找不到 {DATA_FILE}"

df = pd.read_csv(DATA_FILE)

def parse_tokens(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            try: return ast.literal_eval(x)
            except Exception: pass
        return x.split()
    return []

df["tokens"] = df["tokens"].apply(parse_tokens)
df["clean_text"] = df["clean_text"].fillna("")

texts       = df["clean_text"].tolist()
token_lists = df["tokens"].tolist()
print(f"语料条数: {len(texts):,}")

# ============================================================
# 1. BERTopic
# ============================================================
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import jieba, re

print("\n🚀 1) 训练 BERTopic ...")

# 自定义分词 & 停用词
stopwords = set()
STOP_PATH = SCRIPT_DIR / "stopwords.txt"
if STOP_PATH.exists():
    stopwords = set(line.strip() for line in open(STOP_PATH, encoding="utf-8"))

def jieba_tokenizer(text):
    tokens = jieba.lcut(re.sub(r"\s+", "", str(text)))
    return [t for t in tokens if t and t not in stopwords and not t.isdigit()]

vectorizer = CountVectorizer(
    tokenizer=jieba_tokenizer,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95
)

sentence_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2", 
    device="cuda" if torch.cuda.is_available() else "cpu"
)

topic_model = BERTopic(
    language="chinese",
    embedding_model=sentence_model,
    vectorizer_model=vectorizer,
    min_topic_size=10,
    top_n_words=12,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(texts)

# ✅ 修复 reduce_topics 错误（加上参数名）
topic_model = topic_model.reduce_topics(docs=texts, nr_topics="auto")
topics = topic_model.topics_


# 保存
BER_DIR = MODELS_DIR / "bertopic_model"; BER_DIR.mkdir(exist_ok=True)
topic_model.save(str(BER_DIR / "model"))
topic_model.get_topic_info().to_csv(OUTPUTS_DIR / "bertopic_topic_info.csv", index=False, encoding="utf-8")

# 可视化
fig_bt = topic_model.visualize_topics(width=1200, height=700)
fig_bt.write_html(str(OUTPUTS_DIR / "bertopic_topics.html"))
print("✅ BERTopic 训练 & 可视化完成")

# ============================================================
# 2. LDA (k=18) + 可视化
# ============================================================
print("\n🚀 2) 训练 LDA(k=18) ...")
from gensim import corpora, models
from gensim.models import CoherenceModel

dictionary = corpora.Dictionary(token_lists)
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(tok) for tok in token_lists]

lda_18 = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=18,
    passes=10,
    random_state=42
)

LDA_DIR = MODELS_DIR / "lda_model"; LDA_DIR.mkdir(exist_ok=True)
lda_18.save(str(LDA_DIR / "lda_18.bin"))

with open(OUTPUTS_DIR / "lda_top_words_18.txt", "w", encoding="utf-8") as f:
    for i, topic in lda_18.show_topics(18, 15, formatted=False):
        f.write(f"Topic {i:02d}: {' '.join(w for w,_ in topic)}\n")

coh_18 = CoherenceModel(model=lda_18, texts=token_lists,
                        dictionary=dictionary, coherence='c_v',
                        processes=1).get_coherence()
print(f"🧐 LDA(k=18) coherence = {coh_18:.4f}")

# 可视化
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
vis18 = gensimvis.prepare(lda_18, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(vis18, str(OUTPUTS_DIR / "lda_vis_18.html"))
print("✅ LDA(k=18) 可视化完成")

# ============================================================
# 3. 自动调优 k
# ============================================================
print("\n🚀 3) 网格搜索 k (8–30, step 2) ...")
scores = []
best_k, best_c, best_model = None, -1, None
for k in tqdm(range(8, 31, 2)):
    m = models.LdaModel(corpus=corpus, id2word=dictionary,
                        num_topics=k, passes=10, random_state=42)
    c = CoherenceModel(model=m, texts=token_lists,
                       dictionary=dictionary,
                       coherence='c_v', processes=1).get_coherence()
    scores.append((k, c))
    if c > best_c:
        best_k, best_c, best_model = k, c, m

# 曲线
import matplotlib.pyplot as plt
ks, cvs = zip(*scores)
plt.figure(figsize=(8,4.5))
plt.plot(ks, cvs, marker='o')
plt.xlabel("Num Topics (k)")
plt.ylabel("Coherence (c_v)")
plt.title("LDA Coherence vs k")
plt.grid(alpha=.3)
plt.xticks(ks)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "lda_k_selection.png", dpi=150)
plt.close()

best_model.save(str(LDA_DIR / f"lda_best_k{best_k}.bin"))
print(f"🎯 最佳 k = {best_k} (c_v={best_c:.4f}) → 模型已保存")

print("\n✨ All done. 结果 & 可视化均写入 outputs/")



