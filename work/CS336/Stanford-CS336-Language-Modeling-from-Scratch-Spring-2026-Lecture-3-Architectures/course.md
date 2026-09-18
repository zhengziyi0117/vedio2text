# CS336 第三讲：Transformer 架构、超参数与稳定训练

这堂课不是介绍某一个“最新架构”，而是把现代自回归 Transformer 的设计空间摊开来看：哪些选择已经逐渐形成共识，哪些仍然可以调整，以及这些选择如何同时影响表达能力、GPU 利用率、推理成本和训练稳定性。核心内容包括 Pre-Norm、RMSNorm、GLU/SwiGLU、RoPE、模型宽深比、词表大小、正则化、Z-loss、QK-Norm、GQA 和滑动窗口注意力。

课程视频：[*Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 3: Architectures*](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&index=3)

## 先建立一个看架构的视角

讲者首先给出这堂课的定位：架构设计本身有点“不可名状”，因为你很难只依靠一套简洁的理论工具回答“这个模型应该怎么设计”。更现实的办法是观察大量已经训练成功的模型，找出它们共同保留的结构，再把仍然存在分歧的部分当作实验变量。真正理解架构的最好方式，依然是在较小规模上亲自训练模型，比较不同选择，而不是只记住某篇论文中的配置。

*[00:05](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=5)*

一个架构要同时满足几种互相牵制的要求：从数据中学习并泛化、能够在 GPU 上高效训练、不会在训练到一半时数值爆炸，还要在推理阶段拥有可接受的延迟和内存占用。因此，很多看起来“不够优雅”的结构，其实是多个工程约束共同留下的结果。读论文时应该把它看成一个多目标系统，而不是只问某个模块在抽象表达能力上是否更强。

*[04:58](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=298)*

从历史上看，Transformer 早期经历了很多探索；LLaMA 2 之后，大量模型开始采用相近的“LLaMA-like”骨架，只在少数模块上做变化。近几年的变化又出现了两个明显方向：一类改动是让大规模训练更稳定，另一类改动是为了更长的上下文和更低的推理成本。也就是说，现代架构并不是把原始 Transformer 推倒重来，而是在几个关键部位持续做系统性修补。

*[06:34](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=394)*

## 现代 Transformer 相比原始版本改了什么

原始 Transformer 的基本模块仍然非常稳固：注意力负责跨位置的信息交互，前馈网络负责逐位置的非线性变换，残差连接负责在层之间传递表示。现代语言模型最常见的变化集中在下面这些地方：

| 设计位置 | 原始 Transformer 的典型选择 | 现代模型中更常见的选择 | 主要动机 |
| --- | --- | --- | --- |
| 归一化位置 | 残差分支之后的 Post-Norm | 残差流之外的 Pre-Norm 或变体 | 改善深层梯度传播与训练稳定性 |
| 归一化形式 | LayerNorm | RMSNorm | 减少计算和数据搬运 |
| FFN 激活 | ReLU 或 GELU | GeGLU、SwiGLU 等门控单元 | 在相近参数规模下获得更好的效果 |
| 位置表示 | 正弦/余弦或绝对位置 embedding | RoPE 及其变体 | 让注意力更自然地表达相对位置 |
| 注意力头 | 每个头都有独立的 K、V | GQA/MQA | 减小 KV Cache 和解码内存带宽 |
| 长上下文 | 全量注意力 | 全注意力与局部注意力交替 | 控制上下文扩展带来的成本 |

这张表的重点不是“现代选择永远更好”，而是告诉我们应该把哪些地方作为实现和实验时的第一批变量。CS336 第一份作业所采用的 Pre-Norm、RoPE 和 SwiGLU，正好对应了现代语言模型中最常见的三类改动。

*[06:59](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=419)*

## Pre-Norm：保持残差流干净

### Post-Norm 与 Pre-Norm

设输入残差流为 $x$，一个 Transformer 子层为 $F$。原始 Transformer 的一种写法可以抽象为：

$$
x_{l+1} = \operatorname{Norm}\bigl(x_l + F(x_l)\bigr).
$$

归一化位于残差相加之后，这通常称为 Post-Norm，或者更准确地说，是把归一化放在残差流内部。现代语言模型更常见的写法是：

$$
x_{l+1} = x_l + F\bigl(\operatorname{Norm}(x_l)\bigr).
$$

这里的归一化发生在注意力或 FFN 计算之前，但没有把结果写回那条干净的残差通路，因此通常称为 Pre-Norm。也存在把归一化放在子层计算之后、但仍位于残差流外部的变体；讲者把它作为“计算之后的 Norm”与传统 Post-Norm 区分开来。

*[07:46](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=466)*

Pre-Norm 最直观的好处，是从底层到顶层存在一条近似恒等的残差路径。反向传播时，梯度可以直接沿着这条路径传递，而不必在每一层都穿过一次归一化。模型越深，这条路径越重要：它减少了梯度逐层衰减或突然放大的机会，也让信号传播更容易控制。

早期研究 Pre-Norm 的一个动机，是希望减少对 learning-rate warm-up 的依赖。原始 Post-Norm 在没有 warm-up 时更容易出现不稳定或收敛困难，而 Pre-Norm 往往具有更平滑的优化过程。即使现代训练仍然普遍使用 warm-up，Pre-Norm 对深层模型的稳定作用仍然保留下来。

*[09:53](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=593)*

这并不意味着“归一化越少越好”。如果训练出现异常梯度或注意力退化，工程上常见的做法反而是增加归一化位置：例如在 FFN 后、注意力的 Q/K 上额外做归一化。它们可能不是最简洁的理论结构，却经常能以较低代价换取更大的稳定性余量。

*[12:45](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=765)*

### RMSNorm：少做一件事，也许就能更快

LayerNorm 会在某个维度上减去均值、除以标准差，然后乘以可学习的缩放参数，通常还带有偏置。RMSNorm 则只根据均方根缩放：

$$
\operatorname{RMSNorm}(x)
= g \odot \frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}.
$$

它不减均值，也不需要同样形式的 bias。RMSNorm 在表达能力上看起来更弱，但大量语言模型实验表明，在典型语言建模任务上，它通常不会造成明显的质量损失。

RMSNorm 的价值不仅在 FLOPs。归一化这类操作的算术强度很低，需要频繁把激活从显存读入、计算统计量，再写回去。矩阵乘法可能占据绝大多数浮点运算，却不一定占据同等比例的实际运行时间；在某些小矩阵或受内存带宽限制的工作负载中，归一化的运行时间占比可以远高于它的 FLOPs 占比。因此，删掉均值计算和不必要的 bias，是一次系统层面的优化。

*[14:15](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=855)*

同样的思路也解释了为什么现代实现常常删除线性层的 bias。bias 参数带来的表达能力增益往往有限，却会增加额外的数据读取、kernel 调度和内存访问。架构选择中经常存在这种“算术上很小、系统上不免费”的操作，不能只用参数量或 FLOPs 判断它是否值得保留。

*[18:04](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=1084)*

## FFN 为什么从 ReLU 走向 SwiGLU

一个普通的 Transformer FFN 可以写成：

$$
\operatorname{FFN}(x)=W_2\,\phi(W_1x),
$$

其中 $\phi$ 可以是 ReLU 或 GELU。ReLU 足以训练出性能不错的语言模型，GELU 通过改变零点附近的梯度行为也能得到很好的结果；因此，门控结构不是“没有它就完全不能训练”。

门控线性单元的思路是，再增加一个投影产生门控向量，并进行逐元素相乘：

$$
\operatorname{GLU}(x)
=W_2\bigl(\phi(xW_1)\odot (xV)\bigr).
$$

当 $\phi$ 取 GELU 时得到 GeGLU；当 $\phi$ 取 Swish 时得到 SwiGLU。门控向量可以理解为对中间特征进行动态调制，让网络不只是对特征做固定的非线性变换，而是学习哪些通道应该被放大或抑制。Google 系模型中常见 GeGLU，而 LLaMA 系模型普遍采用 SwiGLU。

*[20:16](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=1216)*

门控结构的代价是多了一组参数：普通 FFN 主要有 $W_1,W_2$ 两个矩阵，GLU 则多出 $V$。如果保持中间维度不变，参数量会增加约 50%。因此，为了公平比较，常把 gated FFN 的中间维度缩小到原来的约 $2/3$，使总参数量与普通 FFN 接近。这个修正也是为什么现代模型里经常看到非门控 FFN 使用约 $4d$ 的中间维度，而 GLU 变体常见约 $2.5d$ 到 $2.7d$，有些 LLaMA 风格配置会选择更接近 $3.5d$ 的比例。

多篇受控实验的共同结论是：在参数量匹配的条件下，GLU 变体通常比对应的非门控激活更稳定地取得更低 loss。它不是理论上唯一正确的选择，GPT-3 使用 GELU 也能工作，某些模型甚至使用 squared ReLU；但从“默认实现”角度看，SwiGLU/GeGLU 已经成为非常强的基线。

*[24:12](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=1452)*

## 串行块与并行块：系统收益是否值得表达能力代价

标准 Transformer Block 通常先计算注意力，再把结果送入 FFN：

$$
x' = x + \operatorname{Attn}(\operatorname{Norm}(x)),
$$
$$
x'' = x' + \operatorname{FFN}(\operatorname{Norm}(x')).
$$

并行结构则让两个分支都读取同一个输入，再一起写回残差流：

$$
x' = x
  + \operatorname{Attn}(\operatorname{Norm}(x))
  + \operatorname{FFN}(\operatorname{Norm}(x)).
$$

并行结构可以共享部分归一化、合并矩阵乘法或减少同步等待，因此看起来很适合系统优化。GPT-J、PaLM 以及一些受 Google 影响的模型采用过类似设计。但它的代价是：在相同层数下，注意力和 FFN 不再形成两段连续的非线性变换，可以把它粗略理解为有效深度减少。随着串行结构的 kernel 融合和调度逐渐优化，并行结构带来的收益未必能抵消表达能力损失，所以近年来它没有成为主流。

这里的经验很值得记住：论文中报告的“某个结构更快”通常依赖具体硬件和实现；如果系统优化不断进步，一个曾经有明显优势的架构变体，可能会因为表达能力上的小损失而逐渐退出。

*[27:26](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=1646)*

## 位置表示：为什么 RoPE 成为主流

### 注意力本身不知道顺序

自注意力的核心是向量之间的内积。如果把输入 token 的顺序整体打乱，而没有额外的位置表示，注意力只能看到一组 token 及其相互相似度，却不知道谁在前、谁在后。因此位置依赖必须显式注入。

原始 Transformer 使用正弦和余弦位置编码；之后的模型也尝试过为每个绝对位置学习一个 embedding，或者直接给 attention score 加上与相对距离相关的 bias。它们都能工作，但“把位置加入 token 表示”和“让注意力分数只依赖相对位置”并不是一回事。

*[30:59](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=1859)*

如果希望一个位置编码真正体现相对位置，可以要求：

$$
\langle f(x,i), f(y,j)\rangle
=g(x,y,i-j),
$$

也就是两个带位置的向量做内积后，结果只与位置差有关，而不依赖它们同时向右平移了多少。简单地把绝对位置 embedding 加到词向量上，内积展开后会出现词向量与位置向量之间的交叉项，因此不天然满足这个性质。

### RoPE 的几何直觉

RoPE 的核心做法非常简洁：先得到不含位置信息的 query/key，然后根据 token 的位置把它们旋转一个角度。对二维向量来说，位置 $p$ 的旋转可以写成：

$$
R(p)=
\begin{bmatrix}
\cos(p\theta)&-\sin(p\theta)\\
\sin(p\theta)&\cos(p\theta)
\end{bmatrix}.
$$

如果两个向量位于位置 $i$ 和 $j$，那么：

$$
\langle R(i)q,R(j)k\rangle
=q^\top R(i)^\top R(j)k.
$$

旋转矩阵的正交性质使得 $R(i)^\top R(j)$ 只与 $j-i$ 有关，于是内积自然携带相对位置信息。直观地说，“我”和“知道”在句子中相距一个位置时，无论整个句子向右平移多少，它们之间的相对旋转都不变。

*[33:01](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=1981)*

真实的 hidden dimension 通常很高，因此 RoPE 会把向量切成许多二维坐标对，对每一对使用不同频率的旋转。高频坐标变化快，适合表示邻近 token 的局部关系；低频坐标变化慢，可以保留更长距离的信息。实现时通常只对 Q 和 K 应用 RoPE，而不对 V 旋转：注意力 score 由 QK 决定，V 负责被加权汇聚。

这里也解释了为什么 RoPE 看起来使用了 sin/cos，却不同于“把 sin/cos 直接加到 embedding 上”：RoPE 是乘法旋转，位置变化通过 Q/K 的内积进入 attention，避免了直接相加带来的额外交叉项。具体代码通常只需要根据 position ids 生成 cos/sin，再对 Q、K 的相邻坐标对执行二维旋转。

*[35:04](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=2104)*

## 超参数：有些重要，但通常比想象中更宽容

架构确定之后，真正实现模型还要选择 FFN 中间维度、注意力头数、层数、词表大小、dropout、weight decay 等超参数。初看这是一个很高维的搜索空间，但大量模型已经把其中一部分压缩成了比较窄的经验区间。下面的数值不是物理定律，而是一个合理的起点。

### FFN ratio

非门控 FFN 常以 $d_{\mathrm{ff}}\approx4d_{\mathrm{model}}$ 作为默认值。门控 FFN 因为增加了一组投影，如果要保持参数量相近，通常使用约 $2/3$ 的中间维度修正，于是常见比例落在 $2.5$ 到 $2.7$ 附近；一些配置为了强调 FFN 会使用约 $3.5$。T5 曾经尝试过约 $64d$ 的极端比例，背后的理由是让矩阵乘法更大、更容易利用硬件，但这种选择并没有成为后来模型的普遍标准。

控制实验通常显示，FFN ratio 存在一个比较平坦的“好区域”。只要不要偏离得太离谱，$2.5$、$3$、$4$ 之间未必会造成灾难性的质量差异；真正需要关注的往往是总参数量、训练 FLOPs 和 GPU 利用率。

*[43:41](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=2621)*

### 注意力头维度

多头注意力一般满足：

$$
d_{\mathrm{model}}=H_qd_{\mathrm{head}},
$$

其中 $H_q$ 是 query head 数量，$d_{\mathrm{head}}$ 是每个头的维度。也就是说，增加头数通常意味着每个头变窄，减少头数则意味着每个头变宽。绝大多数模型会让二者维持在一个相对稳定的比例附近，而不会任意扩大其中一个。

这也是一个相对宽容的超参数：只要矩阵形状适合硬件、每个头不是小到无法表达关系，模型通常能在一段范围内正常工作。实际训练时应先选一个常见配置，再用小规模消融确认是否值得改变，而不是从零搜索所有可能的头数。

*[49:57](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=2997)*

### 宽度、深度与 aspect ratio

模型放大时，常见做法不是任意增加层数和 hidden dimension，而是固定一个宽深比，再整体扩大模型。课程中给出的粗略经验是：

$$
\frac{d_{\mathrm{model}}}{N_{\mathrm{layers}}}\approx100.
$$

这个比例反映了表达能力和系统实现之间的折中。更深的模型可能有更多连续变换，但需要处理更多层间同步和 pipeline parallel；更宽的模型则更容易用 tensor parallel 切分矩阵。极端偏深的模型在并行化时更麻烦，极端偏宽的模型又可能浪费参数和显存带宽。因此，大量模型会落在一个相当宽的中间区域。

*[51:37](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=3097)*

### 词表大小和 tokenization 评价

早期面向单一英文语料的模型常用约三万规模的词表；多语言或面向生产部署的模型则常见十万到二十万甚至更大的词表。词表更大，通常可以让多语言文本或常见字符串使用更少 token，但 embedding 和输出投影也会变大，并会增加训练和推理时的内存压力。

比较不同 tokenizer 时，不能只比较 token-level perplexity，因为不同 tokenizer 对同一段文本产生的 token 数可能不同。只要 tokenizer 能无损表示同一批原始字节，并且用相同字节数做归一化，bits per byte 这类指标才更适合跨 tokenizer 比较。换句话说，评价单位必须固定，否则“每个 token 的 loss 更低”可能只是 token 切得更粗。

*[55:12](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=3312)*

### Dropout 与 weight decay

在计算受限的语言模型训练里，数据通常非常多，训练可能只遍历语料一次。单次 SGD 不容易像传统小数据集那样把全部训练样本记住，所以“防止过拟合”未必是 dropout 的主要价值。奇怪的是，很多现代训练仍然使用 weight decay，而它不一定只是在做经典意义上的正则化。

weight decay 会和学习率、学习率衰减以及优化器动力学相互作用。某些实验中，更强的 weight decay 配合学习率衰减，最终能到达更好的优化结果；这时它更像一个优化干预，而不是简单地把训练损失和验证损失拉开。因此，dropout 和 weight decay 不能只根据教科书里的正则化解释来判断，最好在目标训练设置上直接做消融。

*[59:20](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=3560)*

## 稳定训练：先盯住 softmax

随着模型和训练预算变大，稳定性的重要性甚至超过一点点最终 loss 改进。一次训练失败可能意味着数百万甚至更高的计算成本被浪费，因此现代架构会主动给可能爆炸的运算增加约束。softmax 是其中最需要小心的地方：它包含指数运算，还要除以归一化项；输入稍微变大，就可能产生极端值。

语言模型中至少有两个重要 softmax：输出 logits 上的 softmax，以及注意力分数上的 softmax。输出概率的对数可以写成：

$$
\log p_y=u_y-\log Z,
\qquad
Z=\sum_j \exp(u_j).
$$

即使 $u_y$ 本身没有异常，$Z$ 也可能因为指数和变得过大或过小。由于给所有 logits 加上同一个常数并不会改变 softmax 的最终概率，可以利用这个自由度约束归一化项。一个常见做法是加入 Z-loss：

$$
\mathcal{L}_{Z}=\lambda(\log Z)^2,
$$

它鼓励 $\log Z$ 靠近零，从而让输出 softmax 的数值范围更容易控制。这个技巧并不改变语言建模目标的主体，却能给训练增加稳定性余量。

*[65:01](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=3901)*

### QK-Norm

注意力的另一个危险点是 $QK^\top$。如果 Q 或 K 的范数逐渐变大，点积 logits 就会变大，进入 softmax 后可能出现过于尖锐或数值不稳定的分布。QK-Norm 的做法是在 Q、K 投影之后分别做 RMSNorm，再进行点积：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}
\left(
\frac{\operatorname{RMSNorm}(Q)\operatorname{RMSNorm}(K)^\top}
{\sqrt{d_{\mathrm{head}}}}
\right)V.
$$

这样做的直觉很直接：先把进入点积的两个向量限制在相近尺度，softmax 的输入就不会因为 Q/K 范数失控而突然膨胀。它通常不会显著改变模型表达能力，却能减少 attention degeneracy，并可能允许使用稍高的学习率。

*[69:53](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=4193)*

### Logit soft cap

如果希望更强地限制注意力 logits，可以使用 soft cap，例如：

$$
\operatorname{cap}(z)=c\tanh(z/c).
$$

无论输入多大，输出都会被限制在 $[-c,c]$ 附近。它比 QK-Norm 更强，因为它直接限制进入 softmax 的结果，能够提供更明确的数值上界；但相应地，模型也无法表达无限强的注意力偏好，可能损失一部分质量。因此它适合被看作“以表达能力换稳定性”的旋钮，而不是无条件开启的优化。

*[72:05](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=4325)*

## 推理阶段的注意力：KV Cache 与 GQA

训练或 prefill 阶段可以并行处理一整段序列，矩阵乘法规模大，GPU 通常有较好的算术强度。但自回归 decode 必须一个 token 一个 token 地生成。为了不重复计算历史 token，推理系统会缓存每一层过去的 Key 和 Value，这就是 KV Cache。

KV Cache 的内存量可以粗略写成：

$$
\text{KV memory}
\propto
2\times N_{\mathrm{layers}}\times T
\times H_{\mathrm{KV}}\times d_{\mathrm{head}}
\times \text{bytes},
$$

其中 $T$ 是已生成的上下文长度，$H_{\mathrm{KV}}$ 是 K/V head 数量。上下文越长、模型层数越多，缓存越大。解码时每生成一个 token，都要从显存读出历史 K/V，因此瓶颈经常不是 FLOPs，而是显存带宽和数据搬运。

*[74:10](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=4450)*

### MHA、MQA 与 GQA

标准 Multi-Head Attention（MHA）为每个 query head 都保存独立的 K 和 V。Multi-Query Attention（MQA）则让所有 query heads 共享一组 K/V，只保留多个不同的 Q。这样可以大幅缩小 KV Cache，减少解码阶段的内存读取，但也会损失一定表达能力，因为不同 query head 无法再观察完全不同的 key/value 子空间。

Grouped-Query Attention（GQA）位于二者之间：Q head 数量保持不变，但多个 Q head 分成若干组，每组共享一套 K/V。通过调整 $H_{\mathrm{KV}}$，可以连续控制表达能力和推理成本之间的折中：

| 形式 | Query heads | Key/Value heads | KV Cache | 表达能力/成本 |
| --- | ---: | ---: | ---: | --- |
| MHA | $H$ | $H$ | 最大 | 表达能力完整，缓存成本最高 |
| GQA | $H$ | $1 < H_{\mathrm{KV}} < H$ | 中等 | 常见的平衡点 |
| MQA | $H$ | $1$ | 最小 | 推理高效，但表达能力损失更明显 |

这不是一个只在推理阶段打开的开关，而是训练时就确定的注意力参数化方式。GQA 之所以成为现代模型的常见选择，是因为只减少一部分 K/V heads，就能获得大部分内存和带宽收益，同时保留接近 MHA 的质量。

*[80:58](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=4858)*

## 滑动窗口注意力与混合长上下文架构

全量因果注意力让每个位置都可以看到历史上的所有位置，但其注意力矩阵随序列长度呈二次增长，KV Cache 也会随上下文线性增长。一个更便宜的选择是滑动窗口注意力：每个位置只关注附近固定窗口内的 token。

只使用局部窗口会限制远距离信息传播，因此现代模型经常把全注意力和局部注意力交替放置。例如，每四层安排一层 full attention，中间三层使用 sliding-window attention。局部层先聚合邻近信息，后面的全局层再把这些局部结果交换到更远位置；这样可以在长上下文成本和全局建模能力之间取得折中。

*[85:12](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=5112)*

位置编码也可以跟着这个结构变化：有的设计让局部层保留位置编码，而全局层弱化甚至去掉 RoPE，使全局层更像对已经聚合好的内容做集合式交互。不同模型还会把便宜的局部注意力替换成状态空间模型等结构。例如课程中提到的 Qwen 3.5，会在 full attention 之间交替使用 gated DeltaNet；这种混合路线将在下一讲进一步展开。

长上下文仍然是当前架构探索最活跃的区域。它的共同主题不是“永远使用全局注意力”或“永远删除注意力”，而是把贵的全局交互和便宜的局部/状态空间计算组合起来，让模型在系统预算可接受的情况下保留长距离能力。

*[87:12](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=5232)*

## 把这堂课落到实现和调参上

如果要从零实现一个现代 decoder-only Transformer，可以把本讲压缩成一条实用基线：使用残差流之外的 RMSNorm；注意力和 FFN 采用 Pre-Norm；FFN 使用 SwiGLU 并按参数量匹配中间维度；对 Q/K 应用 RoPE；默认关闭不必要的 bias；推理侧使用 GQA 以控制 KV Cache；长上下文再考虑滑动窗口或混合注意力。

但这些不是不可修改的“神奇配方”。真正可靠的工作方式是先建立一个可运行的 baseline，再一次只改变一个变量，并同时记录 loss、吞吐、显存、梯度范数、attention logits 和最终评测。架构选择的实验必须把质量和系统成本放在同一张表里，否则很容易得到一个 loss 稍好、但训练时间或推理成本完全不可接受的结论。

*[88:40](https://www.youtube.com/watch?v=lVynu4bo1rY&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV&t=5320)*

## 一页速记

- **Pre-Norm**：把归一化放到子层计算之前，让残差流保留更直接的梯度通路。
- **RMSNorm**：省掉均值中心化和部分 bias，重点收益来自更少的数据搬运。
- **SwiGLU/GeGLU**：门控 FFN 通常比普通激活更强，比较时要用约 $2/3$ 的中间维度做参数匹配。
- **RoPE**：对 Q/K 的二维坐标对做按位置旋转，使内积自然携带相对位置信息。
- **FFN ratio**：非门控常从 $4d$ 起步，门控常从 $2.5d$—$3.5d$ 起步。
- **宽深比**：$d_{\mathrm{model}}/N_{\mathrm{layers}}\approx100$ 是一个粗略但实用的起点。
- **稳定性**：先检查 softmax、logits 和梯度范数；Z-loss、QK-Norm、soft cap 是不同强度的稳定化手段。
- **GQA**：保留多个 Q heads，但减少 K/V heads，用较小的质量代价换 KV Cache 和带宽收益。
- **长上下文**：full attention 与 sliding-window/state-space 层交替，是当前常见的混合方向。

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
