# -*- coding: utf-8 -*-
"""
network_viz_final.py
------------------------------------------------------------
Step 4 ：关键词共现 + 用户互动（单 HTML，可手动暂停 / 稳定）
生成：
  outputs/co_network.html
  outputs/user_network.html   (若有 uid / retweeted_uid)
  outputs/network_stats.txt
依赖：
  pip install pandas networkx python-louvain pyvis==0.3.2 tqdm
"""

import ast, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from itertools import combinations
from collections import Counter

import pandas as pd, networkx as nx
import community as community_louvain          # Louvain
from pyvis.network import Network               # pyvis 0.3.2
from tqdm import tqdm

# ------------------------------------------------------------
# 通用工具
# ------------------------------------------------------------
ROOT = Path(__file__).parent
OUT  = ROOT / "outputs"; OUT.mkdir(exist_ok=True)

def parse_tokens(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        if x.startswith('[') and x.endswith(']'):           # "['婚姻','成本']"
            try: return ast.literal_eval(x)
            except Exception: pass
        return x.split()
    return []

def save_html_utf8(net: Network, path: Path):
    path.write_text(net.generate_html(), encoding="utf-8")

# ------------------------------------------------------------
# 读取数据
# ------------------------------------------------------------
CSV = ROOT / "weibo_cleaned.csv"
assert CSV.exists(), "❌ 找不到 weibo_cleaned.csv"

df = pd.read_csv(CSV)
df["tokens"] = df["tokens"].apply(parse_tokens)

# ------------------------------------------------------------
# Part A 关键词共现网络
# ------------------------------------------------------------
print("🔗 构建关键词共现网络 …")
MIN_W = 5
co_cnt = Counter()
for toks in tqdm(df["tokens"], desc="co-count"):
    for a, b in combinations(set(toks), 2):
        if a > b: a, b = b, a
        co_cnt[(a, b)] += 1

edges = [(a, b, w) for (a, b), w in co_cnt.items() if w >= MIN_W]
G = nx.Graph(); G.add_weighted_edges_from(edges)

stats = dict(
    kw_nodes   = G.number_of_nodes(),
    kw_edges   = G.number_of_edges(),
    kw_density = round(nx.density(G), 6),
)

part = community_louvain.best_partition(G, weight="weight")
nx.set_node_attributes(G, part, "comm")

net = Network("760px", "100%", bgcolor="#ffffff",
              directed=False, cdn_resources="in_line")
net.from_nx(G)
for n in net.nodes:
    deg = G.degree(n["id"])
    n.update(value=deg,
             group=G.nodes[n["id"]]["comm"],
             title=f"{n['id']}<br>degree: {deg}")

net.set_options("""
{
  "configure": { "enabled": true, "filter": ["physics"] },
  "physics":   { "enabled": true,
                 "barnesHut": { "gravitationalConstant": -15000 } },
  "nodes":     { "scaling": { "min": 6, "max": 30 } },
  "edges":     { "smooth": false },
  "interaction": {
    "dragNodes": true,
    "zoomView":  true,
    "dragView":  true,
    "navigationButtons": true
  }
}
""")

save_html_utf8(net, OUT / "co_network.html")
print("✅ 关键词共现图 →", OUT / "co_network.html")

# ------------------------------------------------------------
# Part B 用户互动网络（若有 uid / retweeted_uid）
# ------------------------------------------------------------
if {"uid", "retweeted_uid"}.issubset(df.columns):
    print("👥 构建用户互动网络 …")
    mask  = df["retweeted_uid"].notna()
    cnt   = Counter(zip(df.loc[mask,"uid"], df.loc[mask,"retweeted_uid"]))
    u_edges = [(u, v, w) for (u, v), w in cnt.items() if u != v]

    DG = nx.DiGraph(); DG.add_weighted_edges_from(u_edges)

    stats.update(
        user_nodes   = DG.number_of_nodes(),
        user_edges   = DG.number_of_edges(),
        user_density = round(nx.density(DG), 6),
    )

    part_u = community_louvain.best_partition(DG.to_undirected(), weight="weight")
    nx.set_node_attributes(DG, part_u, "comm")

    net_u = Network("760px", "100%", bgcolor="#ffffff",
                    directed=True, cdn_resources="in_line")
    net_u.from_nx(DG)
    for n in net_u.nodes:
        deg = DG.degree(n["id"])
        n.update(value=deg,
                 group=DG.nodes[n["id"]]["comm"],
                 title=f"user {n['id']}<br>degree: {deg}")

    net_u.set_options("""
    {
      "configure": { "enabled": true, "filter": ["physics"] },
      "physics":   { "enabled": true,
                     "repulsion": { "nodeDistance": 120 } },
      "nodes":     { "shape": "dot", "scaling": { "min": 4, "max": 24 } },
      "edges":     { "arrows": { "to": { "enabled": true } }, "smooth": false },
      "interaction": {
        "dragNodes": true,
        "zoomView":  true,
        "dragView":  true,
        "navigationButtons": true
      }
    }
    """)
    save_html_utf8(net_u, OUT / "user_network.html")
    print("✅ 用户互动图 →", OUT / "user_network.html")
else:
    print("⚠️ 未检测到 uid / retweeted_uid 列，跳过用户互动网络")

# ------------------------------------------------------------
# 保存指标
# ------------------------------------------------------------
with open(OUT / "network_stats.txt", "w", encoding="utf-8") as fw:
    for k, v in stats.items():
        fw.write(f"{k:22s}: {v}\n")
print("📑 指标写入 →", OUT / "network_stats.txt")

print("\n🎉 完成！HTML 打开后：\n   ▷ = 开启物理 ❚❚ = 暂停 ⟳ = Stabilize 并停止")
