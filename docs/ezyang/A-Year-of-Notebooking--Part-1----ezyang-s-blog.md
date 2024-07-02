<!--yml

category: 未分类

date: 2024-07-01 18:17:45

-->

# A Year of Notebooking (Part 1) : ezyang’s blog

> 来源：[`blog.ezyang.com/2011/06/a-year-of-notebooking-part-1/`](http://blog.ezyang.com/2011/06/a-year-of-notebooking-part-1/)

今年，我已经积累了三本笔记本，记录了各种各样的笔记和思考。由于这些笔记本已经破烂不堪，我决定把它们的内容转移到这里。警告：它们可能有些不连贯！这是三本笔记本中的第一本。我建议你浏览一下各个章节的标题，看看有没有什么特别吸引你的内容。

### Tony Hoare：抽象分离代数

Tony Hoare 希望利用已经解决的“艰难工作”，将分离逻辑（例如 Hoare 三元组）的形式化放置到一个抽象代数中。其想法是通过将事物编码为一对，而不是三元组，我们可以利用代数中的众多结果。基本思想是我们取一个传统的三元组 `{p} q {r}`，并将其转换为一个有序半群关系 `p; q <= r`，其中 `;` 是一个单调运算。最终，我们得到了一个分离代数，这是一个带有额外星号操作符的单调格。公理的选择很重要：“这是抽象代数，所以你应该愿意接受这些公理，而无需考虑任何模型。”（在这里涂鸦写着：“梦中套梦作为数学多层次思维的隐喻。”）我们有一个同态（而不是同构）在实现和规范之间（右方向是简化，左方向是伽罗华连接）。实际上，正如观众中的一位评论者指出的那样，这被称为斯通对偶性——有点像两点决定一条线——带有逆变点和属性。我相信 Tony 在今年年初我去听这场讲座时已经对这个主题进行了一些思考，所以这些内容可能已经被后来的发现所取代。C'est la vie！

### Satnam Singh：多目标并行处理

我们能编写可以在多种类型硬件上执行的并行代码吗：例如，在传统 CPU 上进行矢量化操作，GPU 或 FPGA？他提出了一种可以嵌入到任何语言中的 EDSL（对于这种特定表示，是 C#），具有诸如`newFloatParallelArray`、`dx9Target.toArray1D(z)`和重载运算符的构造。在我的笔记中，我备注道：这种表示是否可以无标签地实现，或者我们总是需要在执行之前构建系统描述的成本？在异构处理器面前，将软件推向硬件尤为重要（例如 Metropolis）。萨特南姆是一个非常引人入胜的演讲者，[这里的引用](http://blog.ezyang.com/2010/10/quote-day/)很多都是归功于他——尽管我确实有一个引用是“希望这不会被引用”（别担心，我没有引用那句话）。跳舞是并行处理的一个比喻（尽管我不记得那个比喻是什么）。自修改硬件怎么样：我们将电路描述映射到内存，并让硬件重新编程 FPGA！

高层信息对优化至关重要：因此，我们可能希望有一个符号评估器，具有即时编译（除了在 FPGA 上我们无法做到）。内存访问融合很重要：我们想要摆脱意外的分号。`Array -> Stream / Shift = Delay`。研究想法：常见并发问题的几何编码。矩阵求逆是一个问题（所以不要反转矩阵，傻瓜），本地内存限制 GPU 与 FPGA，以及*能量*调度问题。

### Streambase

Streambase 是一家实现视觉事件流处理语言的公司。我曾经参与他们的编译器面试；虽然最终我拒绝了他们，但这是一个非常有趣的面试，我认为如果在那里工作会很有趣（尽管工作语言是 Java，我就不是很喜欢！）面试非常有趣：其中一个问题是，“向我解释单子。”天啊，我仍然不知道如何恰当地解释这个概念。（附注：人们确实非常喜欢副作用。围绕编写能在持久数据上运行的高性能程序的编程传说非常新颖，或许我可以说比围绕惰性求值的传说还要新颖。）

### 怀疑论

亚历山大·伯德的书《科学哲学》教会了我如何识别无效的怀疑，即使在不明显的情况下，比如休谟关于归纳问题。我的可靠主义论文相当不错；这是我在哲学辅导课中唯一一篇拿到一级的论文。像类型理论一样，理由的确证是分层的，有层层叠加的。

### 西蒙·佩顿·琼斯：不应该概括“Let”

Hindley-Milner 类型系统的程序员长期以来一直享受着实用类型推断的好处：我们通常期望推断出最一般的类型，并且在它们的使用位置进行表达式的语法替换总是可以类型检查的。当然，类型推断算法通常是 EXPTIME-complete 的，但类型理论家们不会因此而抱怨太多，因为对于更强大的逻辑，推断通常是不可判定的。（顺便说一句：字典构成了*运行时*证据。这是一个很好的思路。）有趣的是，在更高级别的类型特性存在的情况下，编写类型签名实际上可能会使类型检查器的工作变得更加困难，但它们会添加需要由约束求解器处理的局部相等性假设。广义的 let 意味着所有这些约束直到达到调用点才能解决。我们能否通过在解决相等约束时进行即时解决来解决这个问题？法国人有一篇关于这个问题的论文，但 Peyton Jones 建议如果你决定阅读这篇论文，最好随身带上一罐阿司匹林。在他的演讲后，一位研究生评论说 Hindley-Milner 在许多方面都是一个异常现象：Agda 的用户期望需要指定所有类型签名，除非在一些特殊情况下可以消除它们，而 Haskell 的用户则期望在特殊情况下不需要指定任何类型签名。

### 树的流处理

Document Object Model（DOM）的一个长期问题是它要求将整个文档加载到内存中。在像 PHP 文档手册这样的情况下，内存使用量可能超过一千兆字节。不幸的是，操作 DOM 的心理模型比操作 XML 标签事件流更自然。有没有办法自动将 DOM 的更改映射到流的更改上？我们希望构建两者之间的同构。我正在寻找 DOM 的函数表示，用于操作（对于 DOM 样式的事件编程模型，你仍然需要可变性）。"左边是小丑，右边是小丑"强调了局部和全局分析之间的差异。你可能会认为遍历令牌流的方法只是简单地遍历，使用拉链来跟踪你的位置。当然，这个拉链会占用内存（实际上，它形成了类似堆栈的东西，这正是你将令牌流转换为树的方式）。因此，我们可以高效地构建树表示而无需突变，但最终我们仍然得到了树表示。此时，我已经写下了“别再打自己了。”确实如此。我们能否利用领域特定知识，一个我承诺不再超出这一点的声明？将 DOM 操作投射到 XML 流操作中，并将其用作衡量某些事物成本的方法可能会很有利可图。当然，现在我应该做一次文献检索。

### 正则表达式编辑距离

给定一个正则表达式和一个不匹配的字符串，需要多少次编辑才能使字符串匹配？可能会有多个答案，算法应允许对不同的修改进行加权。

### Frank Tip：为 Web 应用生成测试和故障定位

Apollo 采用了测试 Web 应用程序的混合方法，结合了具体执行和符号执行。其理念是，大多数 Web 应用程序具有模糊的、早期的条件化，没有复杂的状态转换或循环。因此，我们在控制器上生成路径约束并解决它们，然后生成输入，使我们能够执行所有控制路径。数据即代码：我们想描述数据。我可能没有很仔细地听演讲，因为我写下了各种其他事情：“堆栈不是 STG 的正确调试机制”（嗯，是的，因为我们想知道我们*来自*哪里。不幸的是，知道我们*要去*哪里也不是很有用）和“我们可以使用执行跟踪自动生成 QuickCheck 缩减实现吗？”（一种自动化的测试用例最小化）以及最后的思考，“Haskell 不是一个适合运行时检查或故障定位的好语言。”

### Benjamin Pierce：类型与编程语言

如果有人制作出一个交互式可视化，展示在向类型系统添加新功能时类型系统如何生长和扩展，一种类型规则和操作语义的视觉差异，那将非常酷。

### 平滑趋势

作为一名博客作者，我的页面浏览量往往会非常波动，当我的文章被流行新闻网站推广时，访问量就会飙升（迄今为止，我的比特币文章已经有 22k 次浏览。不错！）但这并不能帮助我了解网站的长期趋势。有没有办法使这些波动趋势平滑，使得高峰仅仅成为更长期趋势上的“热门点”？

### 用户界面

我想要一个最小技术工作量最佳实践用户界面的圣经，实现起来简单且不会让用户太困惑的模式。对我来说，UI 设计有点太琐碎了。在智能界面的情况下，我们如何不让用户生气（例如 Google Instant）？我们有一个用户期望，即计算机不会猜测我想要什么。那太奇怪了。

### 第 32 页

我用大字写着：“证明局部定理。没有远程作用。”诺曼·拉姆齐在我和 Hoopl 一起工作时，几乎用了同样的话告诉我。我认为这是一个非常有力的想法。

### 分离逻辑与图形模型

我记录了一些数学定义，但它们并不完整。我不认为我写了什么特别有洞察力的东西。这反映了笔记的目的：你应该记录那些以后可能无法获取的东西，但你也应该确保跟进所有你说过会查找的完整信息。

### Jane Street

我有两页关于通过电话面试解决问题的涂鸦。我非常喜欢它们。一个是动态规划问题（一开始我对递归关系不太理解，但最终搞定了），第二个是在 OCaml 中实现函数式编程特性。实际上，我想写一篇关于后者的博客文章，但到目前为止，它一直留存在我的草稿箱中，等待重生的一天。在我的笔记中（第 74 页），我记录了现场面试的问题，不幸的是，我不能与你分享它们。

### Quote

“这就像雇用律师开车带你穿越城市。”我不记得具体的语境是什么了。

### Mohan Ganesalingam：数学语言

我真的很喜欢这个演讲。Mohan 研究将自然语言处理应用于一个比无限制的人类语料库更易处理的领域：数学语言的领域。为什么这个领域易于处理？数学在文本中定义了其词汇（数学术语必须明确定义），我们混合符号和自然语言，并且语法是受限的。Montague 语法与表义语义相对应。当然，像普通语言一样，数学语言也存在严重的歧义。我们有词汇歧义（“质数”可以描述数字、理想等），结构歧义（如果 p *生成* *某些多项式*的*分裂域* **在 F_0 上** —— F_0 是指生成还是多项式？），符号歧义（`d(x + y)`，这不仅仅是操作符重载，因为解析树可以改变：例如取`(A+B)=C`与`λ+(M=N)`作比较），以及符号和文本结合的歧义。事实证明，数学的语言类型系统，这是正确获取解析树所必需的，根本不是数学性的：整数、实数及其伴侣都归为一个大类别的数字，类型不是外延的（对象根据内容具有不同类型）。我们需要一个动态类型系统，而不是结构或名义类型系统，并且我们需要在解析过程中推断类型。

### 写笔记

从 12 月 1 日开始，我似乎需要写更具总结性的结尾段落，使用更短的句子。总结我的论点部分，详细描述实验内容，并不要忘记，历史数学的大部分是几何学。旨在用更少的句子表达更多的内容。阿门！

另一组笔记：所有问题都是陷阱：考官希望你思考被问的内容。思考事件周围的更广泛背景。你可能没有足够的时间与当代观点比较。**在你的文章中放置路标。** 小心不要发生非因果关系。冒号很好：它们增加了强调（但要小心使用）。**短句子！**

### Principia Mathematica

多么美妙的会议啊！有很多演讲，我本应该多记一些笔记，但这是我有的一些，一些引语和素描。

代数学家与分析学家。 “四个人骑自行车进来，然后再骑出去。” 数字作为时刻，而不是对象（尽管它不失一般性）。 “康托尔对此完全*无望*。”（关于零）。“数字是否从 0 或 1 开始？是和是。” 弗雷格和罗素最终给了零适当的地位。计数的误读：算术是否从计数开始？数字序列已经就位，相反，我们构造了同构。有一个错误的信念，我们从一开始数数。同构避免计数，给予零适当的地位，并且避开计数实际如何工作的问题（一个及物动词：预计数，我们必须决定计算什么）。与《逻辑漫游》中的普遍描述相反，哥德尔和罗素确实见过面。奎因逻辑和教会逻辑。“平方根二不是无理数”要求每个数字都是有理数或无理数。

我们为什么关心老年人？我们如何在哲学中取得进展？秩序是句法而不是语义：克里普克和塔尔斯基发展了一个真理的层次结构。自由变量推理有助于解决名词和典型模糊：对哲学问题的科学方法。“现在什么构成研究——也就是说，谷歌它。”名词模糊：断言“x 是偶数”，实际上是“对所有 x，x 是偶数。”引用：“从信中清楚地表明他没有看过《原理》的第五页。”单词“变量”是非常误导的，它不是变量名（进步！）“没有不确定的人。” 同指代代词。我们不能用这种方式表达推理规则。

类型：变量必须具有范围。几乎没有定理（所有的陈述都是模式）：我们想证明关于所有类型的事情，但不能因为矛盾而这样做。所以所有的变量通常是**类型**不明确的。第 2 卷中有关于无穷的论证，但是小世界给了你错误的数学（实证主义）。但是有一个聪明的想法：即使世界上的东西不够多，如果有 k 个东西，就有 2^k 类的东西等等。上升到层次结构。*这*就是典型模糊的解释。怀特海德认为理论是理论土地上的无意义字符串（一种宏）。斯特劳斯基在语言/元语言区分方面有所贡献！！“看”是确定类型的方式。逻各斯中心主义困境是你应该使用推理，但这种推理是在形式系统之外的。更高类型的操作符双关语，所有操作符都带有类型标签。类型的分层。

自由变量推理对典型模糊推理是相同的。量化推理的缩写（需要内部量化器的混乱规则），不定名（不能是变量名，不能导致不定事物），示意名（λ：正确的变量，现代的类型）。但是如果不让某人相信它（怀疑主义），看起来：如果正确的逻辑是类型理论和外部的，那么我们没有超出推理的立场。（这是一个单向方向。）**我认为有一种从内部谈论系统的方法。**我们有一种削弱的真理感：如果你已经相信它，那就没问题，但没有说服力。

下一堂课来自计算机科学家的世界。“可以说，编程语言越多地借鉴形式逻辑，它就越好。” 否则，这是“电工的临时创建”。计算机允许进行简单的形式化操作和正确性检查。但对于数学来说呢？并不是很多。可以通过算法检查证明（使用形式推理规则）。“因为这里有很多哲学家，我希望我能以适当模糊的方式回答问题。” 符号化允许我们机械地做“容易”的事情（怀特黑德的引用）。我们需要形式方法吗？在 1994 年，发现奔腾处理器在浮点除法中有错误。罗宾的猜想被错误地证明了。不同的证明系统：德布鲁因生成的证明由单独的检查器检查，LCF 将所有规则化简为由逻辑核心检查的原始推理。毕竟，为什么我们不证明我们的证明助手有效？HOL Light（Principia）只有 430 行代码。谢弗的笑话：拉姆赛化的类型。现在，形式逻辑正处于 20 世纪研究数学的边缘，证明只需要“1 万行代码”。形式证明的维护是一个大问题：我们需要中间的声明性抽象模型。看看 Flyspeck。

我在页边有些涂鸦：“逻辑中的引用？”（我认为这是线性逻辑），性能证明如何（保证在某个时间内运行，实时证明），或概率可检查的证明。也许复杂性理论在这里有所发言。

### 图灵机

他们有效访问的方法是……拉链。哦，天哪！

### GHC

我在这里的涂鸦大部分都看不清楚，但最初有些概念让我困扰：

+   栈布局，保持上下直线，信息表，以及栈指针的运动。现在我对这一切是如何工作有了相当清楚的理解，但开始时它确实相当神秘。

+   `CmmNode` 构造器有很多字段，与打印的 C-- 构成对应关系是非平凡的。

+   变量的大小。

+   标题，负载和代码。

+   指针标记，特别是关于存储在寄存器中、堆栈上的值，以及标签位在上下文中的含义（函数或数据）。我从未弄清楚压缩 GC 是如何工作的。

这结束了第一本笔记本。