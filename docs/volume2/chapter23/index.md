# 第23章：因果推断的实现——从 do-calculus 到可运行的神经 SCM

> 理论告诉你 do 算子是什么。实现告诉你它是什么做成的。

---

第18章把 do 算子定义清楚了：$\mathsf{do}(X = v)$ 是把 $X$ 的结构方程替换为常数，删除所有入边，沿后代传播效应。这个定义是数学的，干净的，没有歧义。

但"删除入边"在计算机里是什么？"传播效应"是哪种数据结构上的哪种操作？"循环图是非法的"这个约束，能不能在代码运行之前就被检查出来，而不是等到运行时崩溃？

这一章的任务是把第18章的数学翻译成可以运行的代码——不是伪代码，而是真实的实现。工具是 [CocDo](https://github.com/lizixi-0x2F/CocDo)，一个把 Pearl 的 do-calculus 和 COC 类型论融合在一起的神经因果模型库。

翻译的过程会暴露一些在纯数学里看不见的问题：类型系统如何在结构层面排除循环？β-归约和矩阵传播是什么关系？梯度下降如何成为 do 算子的逆？

---

## 23.1 COC 类型论：让循环成为类型错误

第14章建立了形式系统的基础：命题、推断规则、证明。第15章发现了它的边界。但形式系统有一个特性在第18章没有被充分利用：**类型可以携带结构约束，使某些错误在语法层面就不可表达**。

CocDo 用的是 COC（Calculus of Constructions）的一个片段。核心思想只有一句话：

> 每条因果边 $X \to Y$ 被编码为依赖 Pi 类型 $\Pi(X : \text{Type}_i).\, \text{Type}_j$，要求 $i < j$。

$i$ 和 $j$ 是拓扑序中的层级——根节点是 $\text{Type}_0$，它的子节点是 $\text{Type}_1$，以此类推。要求 $i < j$ 意味着：**一条边只能从低层指向高层**。

循环图意味着什么？意味着存在一条路径 $X \to \cdots \to X$，使得 $X$ 的层级既要小于某个中间节点，又要大于它。这在类型系统里是矛盾的——不是运行时错误，而是类型检查失败：

```python
from cocdo.kernel.terms import Sort, Pi, Var
from cocdo.kernel.typing import type_of, Context

# 合法的边：X(层级0) → Y(层级1)
ctx: Context = {"X": Sort(0), "Y": Sort(1)}
edge = Pi("X", Sort(0), Sort(1))   # ✓ 0 < 1

# 非法的边：Y(层级1) → X(层级0)，构成循环
bad_edge = Pi("Y", Sort(1), Sort(0))  # type_of 会拒绝这个
```

这个设计的意义不只是工程上的防御。它说的是：**因果图的无环性不是一个运行时检查，而是一个类型不变量**。一个带循环的因果模型，在 CocDo 里根本无法被构造出来——就像在强类型语言里，你无法把字符串赋值给整数变量。

::: info 兔狲教授评
类型系统把约束从"运行时崩溃"提升到"编译时拒绝"。这不是技术细节，这是认识论的一步：你不是在检查模型是否合法，你是在让非法模型无法被表达。第14章说形式系统的价值是消除歧义；这里的价值是消除一整类错误的可能性。
:::

---

## 23.2 do() 作为 λ 演算的项替换

第18章说 $\mathsf{do}(X = v)$ 是"把 $X$ 的结构方程替换为常数方程 $X = v$"。在 λ 演算里，这个操作有一个精确的名字：**捕获避免替换**（capture-avoiding substitution），记作 $[v/X]M$——把项 $M$ 里所有自由出现的变量 $X$ 替换为 $v$。

CocDo 的实现：

```python
def subst(term, var: str, replacement):
    """把 term 里所有自由出现的 var 替换为 replacement。"""
    if isinstance(term, Var):
        return replacement if term.name == var else term
    elif isinstance(term, Lam):
        if term.var == var:          # 绑定变量遮蔽，停止替换
            return term
        return Lam(term.var, term.domain, subst(term.body, var, replacement))
    elif isinstance(term, App):
        return App(subst(term.func, var, replacement),
                   subst(term.arg,  var, replacement))
    return term  # Const, Sort 不含自由变量
```

"捕获避免"处理的是一个微妙的情况：如果 $M$ 里有一个 λ 绑定了和 $X$ 同名的变量，替换不应该进入那个绑定的作用域——否则会把本来指向外部的 $X$ 错误地替换掉。代码里的 `if term.var == var: return term` 就是这个检查。

替换之后，需要 **β-归约** 把结果化简到正常形式：

```python
def beta_reduce(term, steps=100):
    """按值调用归约到不动点。"""
    for _ in range(steps):
        reduced = _step(term)
        if reduced is term:   # 不动点：无法继续归约
            break
        term = reduced
    return term
```

`_step` 的核心规则是 β-规约：$(\lambda x. M)\, N \to [N/x]M$——把函数应用化简为替换。当 `Add`/`Mul` 的两个操作数都是带值的 `Const` 时，归约器直接计算：

```
App(App(Mul, Const(w=0.9)), Const(v=3.0))  →  Const(2.7)
```

这意味着结构方程 $E_j = \sum_i A_{ij} \cdot E_i + U_j$ 的传播，**发生在项语言内部**，不是一次独立的矩阵乘法。do 算子的语义和计算语义是同一件事。

::: info do() 与 β-归约的对应
| Pearl 的操作 | λ 演算操作 |
|-------------|-----------|
| 把 $X$ 的方程替换为 $X = v$ | `subst(mechanism, "X", Const(v))` |
| 删除 $X$ 的所有入边 | 替换后父节点项消失，不再出现在归约路径上 |
| 沿后代传播效应 | `beta_reduce` 按拓扑序归约到不动点 |
| 循环图非法 | Pi 类型要求层级严格递增，循环是 `TypeError` |
:::

---

## 23.3 NOTEARS：用梯度学习 DAG 结构

第18章的"悬而未决"提到了因果发现问题：能否从数据里自动推断因果图？传统方法（PC 算法、FCI）是组合搜索——在所有可能的 DAG 里找最符合数据的那个。节点数 $n$ 时，DAG 的数量是超指数的，搜索代价极高。

2018 年，Zheng 等人提出了 NOTEARS，把"是否是 DAG"这个离散约束转化为一个**连续可微的等式约束**：

$$h(A) = \mathrm{tr}(e^{A \circ A}) - n = 0 \iff A \text{ 是 DAG}$$

其中 $A \circ A$ 是逐元素平方，$e^M$ 是矩阵指数。这个等式成立当且仅当 $A$ 是有向无环图。

有了这个约束，因果发现变成了带等式约束的连续优化：

$$\min_A \mathcal{L}_{\text{recon}}(A) + \lambda h(A) + \frac{\rho}{2} h(A)^2$$

用增广拉格朗日法求解，每隔若干步收紧乘子 $\lambda$ 和惩罚系数 $\rho$。

CocDo 对稀疏图用轻量近似 $h(A) \approx \|A\|_F^2$，避免矩阵指数溢出：

```python
def acyclicity_loss(A: torch.Tensor) -> torch.Tensor:
    return (A * A).sum()   # ≈ tr(e^{A∘A}) - n，对稀疏 A 成立
```

**注意力即因果发现。** `CausalFFNN` 用低秩双线性打分：

$$\text{score}(i \to j) = \frac{(W_q h_i) \cdot (W_k h_j)^\top}{\sqrt{r}}$$

这和 Transformer 注意力在数学上完全相同。区别只有两点：用 sigmoid 而非 softmax（边独立竞争），以及对角线强制为零（变量不能是自身的原因）。

::: info 注意力是因果发现
Transformer 的注意力头在计算"token $i$ 对 token $j$ 的影响权重"——这正是因果权重矩阵 $A[i,j]$ 的定义。区别在于 Transformer 没有施加无环约束，也没有用 do-calculus 解释这些权重。CocDo 把这两件事补上了。
:::

---

## 23.4 梯度规划：把 argmin 当成 do 的逆

第18章的 do 算子是正向的：给定干预值 $v$，计算结果 $Y$。实践中更常见的问题是反向的：**给定目标 $Y = y^*$，找到最优干预值 $v^*$**。

$$v^* = \arg\min_v \sum_j \left(\|E_{\text{next}}[j]\| - y^*_j\right)^2$$

其中 $E_{\text{next}}$ 是 do 算子传播后的嵌入状态，用 L2 范数而非全向量比较，消除方向对齐问题。

`CausalPlanner` 的核心是一个可微的单步传播：

```python
# do-calculus 在嵌入空间的实现：
A_do  = A * (1 - col_mask)      # 把干预节点的入边列清零
E_do  = (1 - row_mask) * E + row_mask * interv_E   # 注入干预值
E_next = A_do.T @ E_do + U      # 结构方程传播
```

然后对干预值 $a$ 求梯度，用 Adam 优化：

```python
a = torch.tensor([0.0], requires_grad=True)
opt = torch.optim.Adam([a], lr=0.05)
for _ in range(200):
    E_next = planner._step(a, interv_nodes, E_init)
    energy = ((E_next[target_idx].norm(dim=-1) - scalar_targets) ** 2).sum()
    energy.backward(); opt.step()
```

整个计算图从目标一路反传到干预值，不需要采样，不需要强化学习。

::: info 兔狲教授评
强化学习把规划变成采样问题：试很多次，记住哪次好。梯度规划把规划变成微积分问题：沿梯度方向走。后者需要可微的世界模型——这正是神经 SCM 提供的。代价是世界必须可微，或者至少可以被可微模型近似。这个假设不总成立，但在嵌入空间里通常足够好。
:::

---

## 23.5 CausalSearch：推理王国作为因果知识图谱

前四节处理的是"变量"——标量或向量，有明确的数值含义。但知识也可以是因果结构的节点：一个段落依赖另一个段落，一个概念建立在另一个概念之上。

`demo_causal_search.py` 把推理王国的全部章节（22章，约1800段）用 BGE 嵌入，训练 `CausalFFNN` 学习段落间的因果权重矩阵 $A$，然后用 Pearl 三步法做检索：

**第一步：溯因（Abduction）**——找到与查询最近邻的段落 $j^*$。

**第二步：行动（Action）**——`do(j^* = \text{query\_emb})`：注入查询向量，清零 $j^*$ 的入边。

**第三步：预测（Prediction）**——$E_{\text{next}} = A_{\text{do}}^\top E_{\text{do}} + U$；按 $\Delta\|E_{\text{next}}\|$ 排序所有节点。

正 $\Delta$ = 下游激活（查询触发的知识链）；负 $\Delta$ = 上游前提（理解查询所需的概念）。

```
查询："Transformer 注意力与贝叶斯推断的关系"

[向量检索 RAG]
  1. ch9·Transformer 的成功触发了...  cos=0.763
  2. ch9·注意力作为因果性...          cos=0.682

[CausalSearch · Pearl 三步]
  溯因锚点 → ch9·Transformer 的成功...

  + 下游激活：
    ch17·贝叶斯更新与 ch14 的比较...  Δ=+2.69e-02
    ch20·PAC 与贝叶斯：ch17 的延续... Δ=+2.34e-02

  * CausalSearch 独有（RAG 遗漏）：
    ch17 贝叶斯推断 ×4，ch1 生成模型层，ch19 证明...
```

向量检索找的是"表面相似"；CausalSearch 找的是"因果相关"——沿学到的因果边传播，而不是在嵌入空间里量距离。

::: info 兔狲教授评
RAG 是第一层阶梯：关联。CausalSearch 是第二层：干预。你问的不是"哪些段落和这个查询相似"，而是"如果我把这个查询注入知识图谱，哪些节点会被激活"。这是两个完全不同的问题，只是碰巧都叫"检索"。
:::

---

## 23.6 回顾：一条从公理到代码的路

下卷走到这里，可以画一条完整的线：

| 章节 | 核心贡献 | 在 CocDo 里的对应 |
|------|---------|-----------------|
| 第14章 | 形式系统：命题、推断规则、证明 | COC 类型论：`Sort`、`Pi`、`Lam`、`App` |
| 第15章 | 一致性与完备性的边界 | 类型检查：循环图是 `TypeError`，不是运行时错误 |
| 第16章 | 线性逻辑：假设只能用一次 | `subst` 的捕获避免：替换后变量消失，不可重用 |
| 第17章 | 概率作为逻辑扩张 | 嵌入范数作为"信念强度"的连续表示 |
| 第18章 | do-calculus：干预的形式化 | `subst` + `beta_reduce` = do 算子 |
| 第19章 | 复杂度：推理的几何 | NOTEARS：把离散 DAG 搜索变成连续优化 |
| 第20章 | 启发式的形式合同 | 梯度规划：能量函数是"差不多对"的精确定义 |
| 第21章 | 学习作为逆推断 | `CausalFFNN`：从观测嵌入逆推因果权重矩阵 |
| 第22章 | 自指与涌现 | CausalSearch：系统用自身的章节作为知识图谱 |

这不是巧合。下卷的每一章都在问同一个问题的不同侧面：**推断是什么，它的边界在哪里，它能被实现吗？** CocDo 是这个问题的一个可运行的回答——不完整，但诚实。

---

## 思考题

**★ 热身**

1. 在 CocDo 里，为什么用 `sigmoid` 而不是 `softmax` 计算边权重 $A[i,j]$？这个选择对因果图的稀疏性有什么影响？
2. `subst` 函数里的"捕获避免"处理的是什么情况？举一个如果不做这个检查会出错的例子。

---

**★★ 推导**

考虑三节点图 $X \to Y \to Z$，结构方程为：

$$E_Y = w_{XY} \cdot E_X + U_Y, \quad E_Z = w_{YZ} \cdot E_Y + U_Z$$

1. 手动计算 $\mathsf{do}(X = v)$ 后 $E_Z$ 的值（用 $w_{XY}, w_{YZ}, v, U_Y, U_Z$ 表示）。
2. 用 CocDo 的 `step` 方法验证你的计算，设 `rollout_steps=2`。为什么需要 `rollout_steps=2` 而不是 1？
3. 如果只用 `rollout_steps=1`，$E_Z$ 的值会是什么？这对应 Pearl 阶梯的哪一层？

---

**★★★ 挑战**

NOTEARS 的无环约束 $h(A) = \mathrm{tr}(e^{A \circ A}) - n = 0$ 是一个充要条件。CocDo 用 $\|A\|_F^2$ 作为近似。

1. 证明：$\|A\|_F^2 = 0 \iff A = 0$（平凡 DAG）。这说明近似在什么情况下是精确的？
2. 构造一个非零的 DAG（至少有一条边），使得 $h(A) = 0$ 但 $\|A\|_F^2 > 0$。这说明近似在什么情况下会失效？
3. 在实际训练中，重建损失 $\mathcal{L}_{\text{recon}}$ 会推动 $A$ 解释数据，而 $\|A\|_F^2$ 会推动 $A$ 趋向零。这两个力的平衡点是什么？它和 L1 正则化有什么关系？

