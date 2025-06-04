# 情感/情绪/立场分析 + 静态PNG + 交互式Plotly仪表板
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------------- 目录 ----------------------
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------- 读数据 --------------------
df_binary  = pd.read_csv(OUTPUT_DIR / 'weibo_sent_binary.csv')
df_emotion = pd.read_csv(OUTPUT_DIR / 'weibo_sent_emotion.csv')
df_stance  = pd.read_csv(OUTPUT_DIR / 'weibo_stance.csv')

# --------------------------------- 1. 二分类情感 ----------------
print('二分类情感分析结果：')
print(f'样本总数：{len(df_binary)}')
print(f'平均正面概率：{df_binary.pos_prob.mean():.3f}')
print(f'平均负面概率：{df_binary.neg_prob.mean():.3f}')

# 静态直方图 (Matplotlib)
plt.figure(figsize=(10, 6))
plt.hist(df_binary.pos_prob, bins=20, alpha=0.5, label='正面概率')
plt.hist(df_binary.neg_prob, bins=20, alpha=0.5, label='负面概率')
plt.xlabel('概率'); plt.ylabel('频数'); plt.title('情感分布'); plt.legend()
plt.savefig(OUTPUT_DIR / 'sentiment_distribution.png'); plt.close()

# 交互直方图 (Plotly)
hist_fig = go.Figure()
hist_fig.add_trace(go.Histogram(
    x=df_binary.pos_prob, nbinsx=20, name='正面概率', opacity=0.6))
hist_fig.add_trace(go.Histogram(
    x=df_binary.neg_prob, nbinsx=20, name='负面概率', opacity=0.6))
hist_fig.update_layout(barmode='overlay',
                       title='情感分布直方图（可交互）',
                       xaxis_title='概率', yaxis_title='频数')
hist_fig.update_traces(marker_line_width=1)

# --------------------------------- 2. 情绪 --------------------
emo_cols = [c for c in df_emotion.columns if c.startswith('emo_')]
emo_labels = {
    'emo_label_0': '平淡',
    'emo_label_1': '关切',
    'emo_label_2': '开心',
    'emo_label_3': '愤怒',
    'emo_label_4': '悲伤',
    'emo_label_5': '疑问',
    'emo_label_6': '惊奇',
    'emo_label_7': '厌恶'
}
emo_means = df_emotion[emo_cols].mean().rename(emo_labels).sort_values(ascending=False)

print('\n各情绪平均概率：')
for label, value in emo_means.items():
    print(f'{label}: {value:.2%}')

# 静态条形图
plt.figure(figsize=(12, 6))
emo_means.plot(kind='bar')
plt.title('各情绪平均概率分布'); plt.xlabel('情绪类型'); plt.ylabel('平均概率')
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'emotion_distribution.png'); plt.close()

# 交互条形图
bar_fig = px.bar(emo_means, x=emo_means.index, y=emo_means.values,
                 labels={'x': '情绪类型', 'y': '平均概率'},
                 title='各情绪平均概率分布（可交互）')
bar_fig.update_layout(xaxis_tickangle=-35)

# --------------------------------- 3. 立场 --------------------
stance_means = {
    '支持': df_stance.stance_entail.mean(),
    '中立': df_stance.stance_neutral.mean(),
    '反对': df_stance.stance_contra.mean()
}
print('\n立场分布（平均概率）：')
for k, v in stance_means.items():
    print(f'{k}: {v:.3f}')

# 静态饼图
plt.figure(figsize=(8, 8))
plt.pie(stance_means.values(), labels=stance_means.keys(),
        autopct='%1.1f%%'); plt.title('立场分布')
plt.savefig(OUTPUT_DIR / 'stance_distribution.png'); plt.close()

# 交互饼图
pie_fig = px.pie(names=list(stance_means.keys()),
                 values=list(stance_means.values()),
                 title='立场分布（可交互）',
                 hole=0.3)

# --------------------------------- 4. 情绪 × 立场 相关系数 ----
emotion_stance_corr = pd.DataFrame(index=['支持', '中立', '反对'])
for emo in emo_cols:
    emotion_stance_corr[emo_labels[emo]] = [
        df_emotion[emo].corr(df_stance.stance_entail),
        df_emotion[emo].corr(df_stance.stance_neutral),
        df_emotion[emo].corr(df_stance.stance_contra)
    ]
print('\n情绪与立场相关系数：\n', emotion_stance_corr.round(3))

# 静态热力图
plt.figure(figsize=(10, 6))
sns.heatmap(emotion_stance_corr, annot=True, cmap='RdBu', center=0)
plt.title('情绪与立场相关系数'); plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'emotion_stance_correlation.png'); plt.close()

# 交互热力图
heat_fig = go.Figure(data=go.Heatmap(
    z=emotion_stance_corr.values,
    x=emotion_stance_corr.columns,
    y=emotion_stance_corr.index,
    colorscale='RdBu', zmid=0,
    hovertemplate="立场: %{y}<br>情绪: %{x}<br>相关系数: %{z:.3f}<extra></extra>"
))
heat_fig.update_layout(title='情绪 × 立场 相关系数热力图（可交互）',
                       xaxis_title='情绪类型', yaxis_title='立场')

# --------------------------------- 5. 生成 Plotly 仪表板 HTML ----
dashboard_path = OUTPUT_DIR / 'sentiment_dashboard.html'

# 使用 subplot 拼成一页，也可以单独文件。这里用简单 div 拼接：
from plotly.io import write_html

figs = {
    'hist': hist_fig,
    'bar':  bar_fig,
    'pie':  pie_fig,
    'heat': heat_fig
}

html_parts = ['<h1>微博情感·情绪·立场可交互可视化</h1>',
              '<p>点击图表可缩放、悬停查看数值。</p>']

for key, fig in figs.items():
    html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

dashboard_html = '\n<hr style="margin:40px 0;">\n'.join(html_parts)

# 写入文件
dashboard_path.write_text(dashboard_html, encoding='utf-8')
print(f'\n✅ 已生成交互式仪表板: {dashboard_path}')

# ------------------------------------------------------------
# 终端总结
print("\n静态PNG与交互HTML均保存至 outputs/ 目录。")
