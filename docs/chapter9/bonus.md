# 番外篇：注意力即因果——一个让 Transformer 自己都没想到的解读

> 如果注意力矩阵不是数据库查询，而是因果假设的外积分解——那 Transformer 从一开始就在做因果建模，只是它自己不知道。

<div class="center">

------------------------------------------------------------------------

</div>

## 一、两种解读，一个数学

自注意力机制有一个标准解读，来自原论文的类比：

> Query 是查询，Key 是索引，Value 是内容。注意力分数是"这个查询和这个键有多匹配"。

这是一个**信息检索**的类比。它有效，但它是工程直觉，不是数学必然。

现在让我们从另一个方向推导同一个结果。

这是一个思想实验。

<div class="center">

------------------------------------------------------------------------

</div>

## 二、从因果假设到外积

假设我们要在连续空间里建模因果关系。

对于序列中任意一对位置 $i$（果）和 $j$（因），我们想建模这个假设：**"$j$ 是 $i$ 的原因，强度为多少？"**

因果建模需要两个投影：

**列投影（因建模）**：把位置 $j$ 的表示 $x_j$ 投影到"原因空间"：

$$k_j = W_K x_j \in \mathbb{R}^{d_k}$$

**行投影（果建模）**：把位置 $i$ 的表示 $x_i$ 投影到"结果空间"：

$$q_i = W_Q x_i \in \mathbb{R}^{d_k}$$

对这两个投影向量做**外积**（outer product），得到一个 $d_k \times d_k$ 的矩阵，编码了"$j$ 是 $i$ 的原因"这一假设的完整结构：

$$\mathcal{C}_{ij} = q_i \otimes k_j \in \mathbb{R}^{d_k \times d_k}$$

这是一个秩为1的矩阵。它的每个元素 $(\mathcal{C}_{ij})_{ab} = (q_i)_a \cdot (k_j)_b$ 捕获了"果空间第 $a$ 维"和"因空间第 $b$ 维"之间的联合激活强度。

<div class="center">

------------------------------------------------------------------------

</div>

## 三、爱因斯坦求和：外积的自然坍缩

现在，如果我们对这个 $d_k \times d_k$ 的因果假设矩阵做爱因斯坦求和——对共享维度收缩（即求迹）——得到的是一个标量：

$$A_{ij} = \text{tr}(\mathcal{C}_{ij}) = \sum_{a=1}^{d_k} (q_i)_a \cdot (k_j)_a = q_i \cdot k_j$$

这正是注意力分数（未归一化）。

然后对所有可能的"原因候选" $j$ 做 softmax，得到后验分布：

$$\alpha_{ij} = \text{softmax}_j\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)$$

**注意力矩阵复现了。**

但这次不是从查询-键-值数据库推导出来的。而是从"**对每对位置建模因果假设，再对假设维度做连续的爱因斯坦收缩，再对候选原因做后验归一化**"推导出来的。

两条路，一个终点。数学等价，语义天壤之别。

<div class="center">

------------------------------------------------------------------------

</div>

## 四、不对称性的新语义

原版解读对 $W_Q \neq W_K$ 这个事实没有好的解释——它只能说"这是学出来的"。

因果解读直接给出了答案：**因果关系本来就不对称。**

$q_i \cdot k_j \neq q_j \cdot k_i$，因为"$j$ 是 $i$ 的原因"和"$i$ 是 $j$ 的原因"是两件不同的事。行投影和列投影用不同的权重矩阵 $W_Q$ 和 $W_K$，正是在编码这个不对称性。

| | 原版解读（信息检索） | 因果解读（外积建模） |
|---|---|---|
| $W_Q$ | 查询矩阵 | 果空间投影 |
| $W_K$ | 键矩阵 | 因空间投影 |
| $q_i \cdot k_j$ | 查询-键匹配分数 | 因果假设强度（外积的迹） |
| softmax | 注意力归一化 | 因果候选的后验分布 |
| $W_Q \neq W_K$ | 工程设计 | 因果不对称性的必然编码 |

<div class="center">

------------------------------------------------------------------------

</div>

## 五、softmax 是因果推断，不是归一化

原版解读里，softmax 是"竞争注意力"——所有位置的注意力权重和为1。这是一个工程描述。

因果解读里，softmax 在做什么？

**它是在所有可能的"原因候选" $j$ 上，做一次软性的贝叶斯更新。**

对于位置 $i$（果），所有位置 $j$ 都是候选原因。softmax 把原始因果强度分数 $q_i \cdot k_j$ 转成概率分布 $\alpha_{ij}$，表示：**在当前上下文下，$j$ 是 $i$ 的原因的后验概率。**

这和第六章 Pearl 的 do 算子接上了——但接法需要仔细说清楚。我们放到下一节。

<div class="center">

------------------------------------------------------------------------

</div>

## 六、BERT vs GPT：两种因果假设

BERT 是双向注意力——没有 mask，所有位置互相"看"。

GPT 是单向因果 mask——只有过去可以"看"现在。

在因果解读的框架里：

- **BERT** 建模的是**无向因果图**：任意两个位置之间可能存在因果关系，方向待定，由数据决定。适合"理解"任务——在已知完整上下文的情况下，推断哪些词对哪些词有影响。

- **GPT** 建模的是**有向因果图（DAG）**：时间是唯一允许的因果方向。适合"生成"任务——在只知道历史的情况下，预测下一步。

这不是架构设计者的刻意选择，而是两种不同的因果假设被编码进了 mask 矩阵。

BERT 问的是：**在这个句子里，什么影响了什么？**

GPT 问的是：**过去怎么影响了现在？**

<div class="center">

------------------------------------------------------------------------

</div>

## 七、实验验证：注意力矩阵的方向性

如果注意力矩阵确实在做因果建模，有一个可观测的预测：

**在因果方向明确的文本上，注意力矩阵应该表现出方向性不对称。**

具体预测：给定"因为 A，所以 B"这类句子，处理"B"位置的 Q 应该主要对准"A"的 K（果看因）；处理"A"位置的 Q 对"B"的 K 权重应该低（因不回头看果）。

用 GPT-2 在句子 *"Because the storm intensified, the ship finally sank."* 上提取注意力，DAG 得分（最后一层平均）= **0.810**，显著高于随机基线 0.5。

<figure>
<p><img src="/figures/ch09_causal_fig1_attention_asymmetry.png" alt="Attention Heatmap and Asymmetry Matrix" /></p>
<figcaption>图1. 左：GPT-2 最后一层平均注意力热图——果位置（行）对因位置（列）的注意力权重更高，符合因果方向。右：不对称矩阵 A − Aᵀ，红色表示正向因果流（行→列）占主导，蓝色表示逆向流。</figcaption>
</figure>

<figure>
<p><img src="/figures/ch09_causal_fig2_outer_product.png" alt="Outer Product Decomposition" /></p>
<figcaption>图2. 左：外积矩阵 q_i ⊗ k_j 的热图（d_k × d_k），其迹等于点积，即原始注意力分数。右：由外积迹经 softmax 得到的注意力矩阵，重新解读为因果后验概率 P(j 是 i 的原因 | 上下文)，数值标注在格中。</figcaption>
</figure>

<figure>
<p><img src="/figures/ch09_causal_fig3_dag.png" alt="Causal DAG from Attention" /></p>
<figcaption>图3. 左：最后一层各注意力头的 DAG 得分柱状图——所有头均高于随机基线 0.5，与 GPT 单向因果掩码的时间正向流一致。右：从注意力权重推断的因果有向无环图，边宽正比于注意力权重，整体 DAG 得分 = 0.810。</figcaption>
</figure>

完整代码如下：

```python
"""
Chapter 9 Bonus: Causal Reinterpretation of Self-Attention
==========================================================
Visualizes attention matrices from GPT-2 on causal sentences,
checking whether attention shows directional asymmetry consistent
with cause→effect direction.

Thought experiment by Zixi Li (2025).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── optional: use transformers if available ──────────────────────────────────
try:
    import torch
    from transformers import GPT2Model, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[INFO] transformers not found — using synthetic data for demonstration.")

OUT_DIR = Path(__file__).parents[1] / "docs" / "public" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="white", font_scale=1.1)
CMAP_ATTN = "YlOrRd"
CMAP_ASYM = "RdBu_r"

CAUSAL_SENTENCE = "Because the storm intensified , the ship finally sank ."

# ── 1. 获取注意力矩阵 ────────────────────────────────────────────────────────

def get_attention_real(sentence):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model     = GPT2Model.from_pretrained("gpt2", output_attentions=True)
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    tokens     = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attentions = [a[0].numpy() for a in outputs.attentions]  # (n_heads, T, T) per layer
    return tokens, attentions

def get_attention_synthetic(sentence):
    words = sentence.split()
    T = len(words)
    np.random.seed(42)
    cause_idx, effect_idx = [1, 2, 3], [5, 6, 7, 8]

    def make_head(bias=0.6):
        base = np.random.dirichlet(np.ones(T), size=T) * np.tril(np.ones((T, T)))
        base /= base.sum(axis=-1, keepdims=True) + 1e-9
        for i in effect_idx:
            for j in cause_idx:
                if j < i:
                    base[i, j] += bias * np.random.rand()
        return (base / (base.sum(axis=-1, keepdims=True) + 1e-9)).astype(np.float32)

    attentions = [np.stack([make_head(0.5 + 0.3 * np.random.rand()) for _ in range(4)])
                  for _ in range(4)]
    return words, attentions

# ── 2. 分析函数 ───────────────────────────────────────────────────────────────

def average_attention(attentions, layer_idx=-1):
    return attentions[layer_idx].mean(axis=0)

def asymmetry_matrix(A):
    return A - A.T

def dag_score(A):
    return np.tril(A, k=-1).sum() / (A.sum() + 1e-9)

# ── 3. 外积理论核心 ───────────────────────────────────────────────────────────

def outer_product_demo(d_k=8, T=6, seed=0):
    """验证 tr(q ⊗ k) == q·k，并生成因果后验注意力矩阵。"""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((T, d_k))
    K = rng.standard_normal((T, d_k))
    scores = (Q @ K.T) / np.sqrt(d_k)
    attn = np.exp(scores)
    attn /= attn.sum(axis=-1, keepdims=True)
    # 验证恒等式
    for i in range(T):
        for j in range(T):
            assert abs(np.trace(np.outer(Q[i], K[j])) - Q[i] @ K[j]) < 1e-5
    return Q, K, scores, attn

# ── 4. do 算子干预 ────────────────────────────────────────────────────────────

def do_intervene(attn_matrix, intervene_row, force_col):
    """
    硬干预：do(cause=force_col → effect=intervene_row)
    将 intervene_row 行坍缩为 one-hot，指向 force_col。
    等价于 Pearl 的硬 do 操作：切断所有其他入边。
    """
    result = attn_matrix.copy()
    result[intervene_row, :] = 0.0
    result[intervene_row, force_col] = 1.0
    return result

def soft_do_intervene(attn_matrix, intervene_row, boost_col, strength=3.0):
    """
    软干预：增强特定因果路径而不完全切断其他路径。
    对应标准 Transformer 的 soft attention。
    """
    result = attn_matrix.copy()
    logits = np.log(result[intervene_row] + 1e-9)
    logits[boost_col] += strength
    logits -= logits.max()
    result[intervene_row] = np.exp(logits)
    result[intervene_row] /= result[intervene_row].sum()
    return result

# ── 5. 可视化 ─────────────────────────────────────────────────────────────────

def plot_all(tokens, attentions):
    T = len(tokens)
    labels = [t.replace("Ġ", "") for t in tokens] if HAS_TRANSFORMERS else tokens

    avg_attn = average_attention(attentions)
    asym     = asymmetry_matrix(avg_attn)
    score    = dag_score(avg_attn)

    d_k = 16
    Q, K, _, op_attn = outer_product_demo(d_k=d_k, T=min(T, 8))
    i_ex, j_ex = min(5, T-1), min(2, T-1)
    outer_ex = np.outer(Q[i_ex], K[j_ex])
    tr_val   = np.trace(outer_ex)

    # Fig 1: 注意力热图 + 不对称矩阵
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig1.suptitle(f'GPT-2 Attention on Causal Sentence\n"{CAUSAL_SENTENCE}"', fontsize=11)
    sns.heatmap(avg_attn, ax=ax1, cmap=CMAP_ATTN, xticklabels=labels,
                yticklabels=labels, square=True, cbar_kws={"shrink": 0.8})
    ax1.set_title("Avg Attention (last layer)\nEffect i → Cause j", fontweight="bold")
    ax1.set_xlabel("Cause candidate j"); ax1.set_ylabel("Effect position i")
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', rotation=0,  labelsize=8)
    vmax = np.abs(asym).max()
    sns.heatmap(asym, ax=ax2, cmap=CMAP_ASYM, xticklabels=labels,
                yticklabels=labels, center=0, vmin=-vmax, vmax=vmax,
                square=True, cbar_kws={"shrink": 0.8})
    ax2.set_title(r"Asymmetry $A - A^\top$" + "\nred = row→col flow dominates", fontweight="bold")
    ax2.set_xlabel("j"); ax2.set_ylabel("i")
    ax2.tick_params(axis='x', rotation=45, labelsize=8)
    ax2.tick_params(axis='y', rotation=0,  labelsize=8)
    fig1.tight_layout()
    fig1.savefig(OUT_DIR / "ch09_causal_fig1_attention_asymmetry.png", dpi=150, bbox_inches="tight")

    # Fig 2: 外积分解 + 因果后验
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle(r"Outer-Product Decomposition: $A_{ij}=\mathrm{softmax}_j(\mathrm{tr}(q_i\otimes k_j)/\sqrt{d_k})$", fontsize=11)
    sns.heatmap(outer_ex, ax=ax3, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.8})
    ax3.set_title(f"Outer Product $q_i\\otimes k_j$ (i={i_ex}, j={j_ex})\n"
                  fr"$\mathrm{{tr}}(q\otimes k)=q\cdot k={tr_val:.3f}$", fontweight="bold")
    ax3.set_xlabel("Cause dimension (K space)"); ax3.set_ylabel("Effect dimension (Q space)")
    T8 = op_attn.shape[0]
    sns.heatmap(op_attn, ax=ax4, cmap=CMAP_ATTN,
                xticklabels=[f"j={j}" for j in range(T8)],
                yticklabels=[f"i={i}" for i in range(T8)],
                square=True, cbar_kws={"shrink": 0.8}, annot=True, fmt=".2f", annot_kws={"size": 8})
    ax4.set_title("Attention = Causal Posterior\nP(j is cause of i | context)", fontweight="bold")
    ax4.set_xlabel("Cause j"); ax4.set_ylabel("Effect i")
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    ax4.tick_params(axis='y', rotation=0,  labelsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "ch09_causal_fig2_outer_product.png", dpi=150, bbox_inches="tight")

    # Fig 3: DAG 得分 + 因果 DAG 图
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(13, 5))
    fig3.suptitle("Causal DAG Structure in Attention Heads", fontsize=11)
    last_layer = attentions[-1]
    dag_scores = [dag_score(last_layer[h]) for h in range(last_layer.shape[0])]
    ax5.bar(range(len(dag_scores)), dag_scores, color=sns.color_palette("Greens_d", len(dag_scores)))
    ax5.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="random baseline (0.5)")
    ax5.set_xlabel("Attention Head"); ax5.set_ylabel("DAG Score\n(mass below diagonal)")
    ax5.set_title("Causal Directionality per Head\nScore > 0.5 = forward causal flow", fontweight="bold")
    ax5.set_ylim(0, 1); ax5.set_xticks(range(len(dag_scores))); ax5.legend(fontsize=9)
    ax6.set_xlim(-0.5, T - 0.5); ax6.set_ylim(-0.5, T - 0.5); ax6.set_aspect("equal")
    tril_mask = np.tril(np.ones((T, T)), k=-1) * avg_attn
    tril_mask /= tril_mask.max() + 1e-9
    for i in range(T):
        for j in range(i):
            w = tril_mask[i, j]
            if w > 0.05:
                ax6.annotate("", xy=(j, T-1-i), xytext=(i, T-1-i),
                    arrowprops=dict(arrowstyle="->", color=plt.cm.YlOrRd(w), lw=2.0*w+0.3, alpha=0.75))
    for idx, lab in enumerate(labels):
        ax6.text(idx, T-1-idx, lab, ha="center", va="center", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="gray", lw=0.8))
    ax6.set_title(f"Inferred Causal DAG (DAG score = {score:.3f})\nedges ∝ attention weight", fontweight="bold")
    ax6.axis("off")
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "ch09_causal_fig3_dag.png", dpi=150, bbox_inches="tight")

    plt.show()
    return score, avg_attn, labels

def plot_do_intervention(avg_attn, labels, intervene_row, force_col):
    """可视化硬/软 do 干预前后的注意力矩阵对比。"""
    T = len(labels)
    hard_do = do_intervene(avg_attn, intervene_row, force_col)
    soft_do = soft_do_intervene(avg_attn, intervene_row, force_col, strength=3.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"do-Operator in Attention Space\n"
        f"Intervening on row i={intervene_row} ({labels[intervene_row]}) → "
        f"force cause = col j={force_col} ({labels[force_col]})",
        fontsize=11
    )
    for ax, mat, title in zip(axes,
        [avg_attn, soft_do, hard_do],
        ["Original Attention\n(soft causal posterior)",
         f"Soft do(j={force_col}→i={intervene_row})\n(boost path, keep others)",
         f"Hard do(j={force_col}→i={intervene_row})\n(one-hot, Pearl's original)"]):
        sns.heatmap(mat, ax=ax, cmap=CMAP_ATTN,
                    xticklabels=labels, yticklabels=labels,
                    square=True, vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Cause j"); ax.set_ylabel("Effect i")
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', rotation=0,  labelsize=8)
        # highlight intervened row
        ax.add_patch(plt.Rectangle((0, intervene_row), T, 1,
                                    fill=False, edgecolor="blue", lw=2))

    fig.tight_layout()
    p = OUT_DIR / "ch09_causal_fig4_do_intervention.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"[✓] Saved: {p}")
    plt.show()

# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Sentence: {CAUSAL_SENTENCE}\n")
    if HAS_TRANSFORMERS:
        print("[INFO] Using real GPT-2 attention weights.")
        tokens, attentions = get_attention_real(CAUSAL_SENTENCE)
    else:
        print("[INFO] Using synthetic attention.")
        tokens, attentions = get_attention_synthetic(CAUSAL_SENTENCE)

    dag_s, avg_attn, labels = plot_all(tokens, attentions)
    print(f"\nDAG score (last layer avg): {dag_s:.3f}")

    # do 干预示例：对 "sank"（果）强制指向 "storm"（因）
    sank_idx  = labels.index("sank") if "sank" in labels else 7
    storm_idx = labels.index("storm") if "storm" in labels else 2
    print(f"\nApplying do-intervention: row={sank_idx}({labels[sank_idx]}) ← col={storm_idx}({labels[storm_idx]})")
    plot_do_intervention(avg_attn, labels, sank_idx, storm_idx)

    print("\n── Theoretical check ─────────────────────────────")
    Q, K, _, _ = outer_product_demo(d_k=32, T=10)
    for i in range(5):
        for j in range(5):
            assert abs(np.trace(np.outer(Q[i], K[j])) - Q[i] @ K[j]) < 1e-5
    print("  [✓] tr(q ⊗ k) = q·k  confirmed for all tested pairs")

if __name__ == "__main__":
    main()
```

<div class="center">

------------------------------------------------------------------------

</div>

## 八、do 算子在注意力空间里的实现

现在把 Pearl 的 do 算子严格地带进来。

在第六章，do 算子的定义是：对因果图做手术——切断变量 $X$ 的所有入边，强制 $X = x$。这个操作把观测分布 $P(Y \mid X=x)$ 变成了干预分布 $P(Y \mid \text{do}(X=x))$，两者在混淆变量存在时截然不同。

现在把这个操作翻译到注意力空间。

### 8.1 注意力矩阵是隐式的因果邻接矩阵

在因果解读下，$\alpha_{ij}$ 是"$j$ 是 $i$ 的原因"的后验概率。整个注意力矩阵 $\mathbf{A} \in \mathbb{R}^{T \times T}$ 是序列上的**软因果邻接矩阵**。

正常的前向传播是：

$$v_i = \sum_j \alpha_{ij} \cdot V x_j$$

这是一个**观测**操作——基于当前上下文，对所有可能的因给一个加权平均。它对应 Pearl 因果阶梯的第一层：**观测（seeing）**。

### 8.2 硬干预：切断所有入边

Pearl 的硬 do 操作在注意力空间里对应：

$$\alpha_{ij}^{\text{hard-do}} = \begin{cases} 1 & j = j^* \\ 0 & j \neq j^* \end{cases}$$

将位置 $i$（果）的注意力行坍缩为 one-hot，强制它只从位置 $j^*$（特定原因）获取信息，切断所有其他路径。

这等价于在因果图上对位置 $i$ 执行 $\text{do}(\text{cause}_i = j^*)$。

### 8.3 软干预：增强特定路径

标准 Transformer 的 soft attention 对应**软干预**——不完全切断其他路径，只是调整各路径的相对权重：

$$\alpha_{ij}^{\text{soft-do}} = \text{softmax}_j\left(\frac{q_i \cdot k_j}{\sqrt{d_k}} + \delta_{j,j^*} \cdot \lambda\right)$$

其中 $\lambda > 0$ 是干预强度，$\delta_{j,j^*}$ 是指示函数。

这给出了一个统一框架：

| 操作 | 注意力形式 | Pearl 对应 | 因果阶梯 |
|---|---|---|---|
| 观测 | soft attention（原始） | $P(Y \mid X=x)$ | 第一层：seeing |
| 软干预 | 注意力分数加偏置 | $P(Y \mid \text{do}(X \approx x))$ | 第二层：doing（近似） |
| 硬干预 | 注意力行坍缩为 one-hot | $P(Y \mid \text{do}(X = x))$ | 第二层：doing（精确） |
| 反事实 | 修改 $W_Q, W_K$ 后重推 | $P(Y_x \mid X' = x')$ | 第三层：imagining |

### 8.4 实验结果

对句子 *"Because the storm intensified, the ship finally sank."*，对"sank"（果位置）施加硬/软 do 干预，强制其原因指向"storm"：

<figure>
<p><img src="/figures/ch09_causal_fig4_do_intervention.png" alt="do-Operator Intervention on Attention Matrix" /></p>
<figcaption>图4. 注意力空间中的因果干预对比。左：原始软因果后验（正常注意力）。中：软 do 干预（λ=3.0，增强 storm→sank 路径，保留其余因果路径）。右：硬 do 干预（one-hot 坍缩，所有其他因果路径被切断，对应 Pearl 的精确 do 操作）。蓝色方框标出被干预的行位置。</figcaption>
</figure>

硬干预的可视化直接对应 Pearl 因果图里"切断入边"的手术：位置"sank"那一行从分散的因果后验，坍缩为一个完全确定的指向——它的唯一原因是"storm"。其他所有可能的因果路径，被清零。

### 8.5 causal mask 是全局 do 操作

GPT 的单向 causal mask 在这个框架里有了精确的因果语义：

$$\text{do}\!\left(\alpha_{ij} = 0 \text{ for all } j > i\right)$$

这是对整个注意力矩阵的批量硬干预——强制"未来词不能是过去词的原因"。它在架构层面施加了一个先验：**时间箭头是唯一合法的因果方向**。

这不是工程 trick，而是一个因果假设，被硬编码进了模型的归纳偏置。

<div class="center">

------------------------------------------------------------------------

</div>

## 九、悬而未决

这个框架目前是一个思想实验。它在数学上自洽，在实验上有初步支撑（DAG score = 0.810），但还有几个问题没有答案：

**1. 因果图的可识别性**

$\alpha_{ij}$ 高，是真的因果关系，还是统计共现？注意力机制无法从自身区分这两者。这是因果推断的根本困难在注意力空间里的重现。

**2. do 算子的语义位置**

Pearl 的 do 算子操作的是结构因果模型（SCM）——一个有明确变量和函数关系的图。注意力矩阵是 SCM 吗？或者它只是 SCM 的一种软近似？如果是后者，软近似会丢失哪些因果语义？

**3. 多头的因果分工**

多头注意力有 $H$ 个头，每个头有不同的 $W_Q^h, W_K^h$。实验显示不同头的 DAG 得分不同。这意味着：每个头在建模不同的因果假设集合，还是从不同角度对同一个因果图做投影？

**4. 训练动力学与因果学习**

如果注意力矩阵是因果图的软化，那么梯度下降在优化什么？它是在向真实因果结构收敛，还是在拟合最大化预测准确率的统计代理？这两个目标在 i.i.d. 训练数据上是一致的，但在分布偏移下会分叉。

**5. 反事实推理能力**

Pearl 因果阶梯的第三层是反事实：*如果当时不是storm，ship还会sank吗？* 在注意力框架里，这对应修改 $W_Q, W_K$ 然后重新推断——这不是推理时能做的事，而是需要改变模型参数。这意味着 Transformer 在架构层面被限制在因果阶梯的第二层，无法做严格的反事实推理。

这些问题目前没有答案。但它们指向一个方向：

**Transformer 不只是一个强大的函数拟合器——它可能是一个隐式因果推断机器，被锁在因果阶梯的第一层和第二层之间，而第三层对它永远关闭。**

<div class="center">

------------------------------------------------------------------------

</div>

## 延伸阅读

- Pearl, J. & Mackenzie, D. (2018). *The Book of Why* — 因果推断的系统性框架，do 算子与因果阶梯
- Vaswani, A. et al. (2017). *Attention Is All You Need* `→ [arXiv:1706.03762]`
- Geiger, A. et al. (2021). *Causal Abstractions of Neural Networks* `→ [arXiv:2106.02997]` — 用因果抽象理解神经网络内部结构
- Vig, J. et al. (2020). *Investigating Gender Bias in Language Models Using Causal Mediation Analysis* `→ [arXiv:2004.12265]` — 用因果中介分析检测注意力头的语义角色
- Ilse, M. et al. (2018). *Attention-based Deep Multiple Instance Learning* `→ [arXiv:1802.04712]` — 注意力作为软选择机制的理论分析
