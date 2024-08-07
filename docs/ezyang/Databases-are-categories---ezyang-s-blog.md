<!--yml

category: 未分类

日期：2024-07-01 18:18:17

-->

# 数据库即范畴：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/06/databases-are-categories/`](http://blog.ezyang.com/2010/06/databases-are-categories/)

*更新* 视频可以在这里找到：[Galois 技术讲座 Vimeo: 类别即数据库](http://vimeo.com/channels/galois#12428370)。

上周四，博士[大卫·斯皮瓦克](http://math.mit.edu/~dspivak/)在 Galois 的技术讲座上，演讲了《[类别即数据库](http://vimeo.com/channels/galois/12428370)》。他的幻灯片在[这里](http://math.mit.edu/~dspivak/informatics/talks/galois.pdf)，比他的论文《[单纯数据库](http://math.mit.edu/~dspivak/informatics/SD.pdf)》更加易于理解。这里简要介绍这个概念，适合那些对范畴论只有初步了解的人士。

在设计关系数据库时的一个重要练习是使用对象和关系的标记图进行对象建模。在视觉上，这涉及到绘制代表正在建模的对象的一堆框，并在对象之间画箭头显示它们可能具有的关系。然后，我们可以将这个对象模型作为关系数据库模式的基础。

软件工程课程中的一个例子模型如下：

当你脑中有一个对象模型的形象时，请考虑维基百科对范畴的定义：

> 在数学中，一个范畴是一个由**一组“对象”组成的代数结构，它们通过一组“箭头”相互连接**，具有两个基本属性：箭头的可结合性和每个对象存在一个身份箭头。

定义的其余部分可能看起来非常抽象，但希望粗体部分清晰地对应于我们之前绘制的框（对象）和箭头的图片。也许...

*数据库模式 = 范畴。*

不幸的是，一个有向图并不*完全*是一个范畴；使得范畴成为范畴的关键因素是箭头上的这两个属性，可结合性和身份性。如果我们真的想加强我们的论断，即模式是一个范畴，我们需要证明这些属性。

记住，我们的箭头是“关系”，即“X 占据 Y”或“X 是 Y 的关键”。我们的范畴必须有一个身份箭头，即某种关系“X 到 X”。那么，“X 就是 X”，一个几乎空洞的陈述，但绝对正确。身份箭头，*检查*。

我们还需要展示箭头的可结合性。两个箭头的组合很像他们在教你向量代数时所展示的：你拿一个箭头的头（从 X 到 Y），并将它与另一个箭头的尾（从 Y 到 Z）粘合在一起，你得到另一个箭头（从 X 到 Z）。如果“书有作者”和“作者有最喜欢的颜色”，我可以说“书的作者有最喜欢的颜色”。这个组合的陈述并不关心作者是谁... 只关心他最喜欢的颜色是什么。实际上，

*箭头组合 = 连接*

也就是说，范畴的一个基本特征，任何纯范畴论中的好结果都使用它，仿佛它是直观显而易见的特性，是那些在数据库教程的后半部分读者看起来并不显而易见的技术之一。

(*旁注.* 外键关系本质上是多对一的：外键字段只能指向另一个表中的一条记录，但许多行可以将该字段指向同一条记录。在关系建模时，我们经常使用多对多或一对多关系。然而，任何数据库管理员都知道，我们可以简单地将这些重新编写为多对一关系（在一对多情况下颠倒箭头，并引入新表以进行多对多关系）。)

当我们有一个模式时，我们也希望有数据来填充这个模式。事实证明，这也适合范畴论框架，尽管完整的解释不在本文范围内（建议查看幻灯片）。

*函子（C -> S）= 数据*

你为什么要关心这个？Spivak 提到了一些好的理由：

我会提到我自己的一个例子：SQL 虽然混乱，但是精确；它可以被输入计算机，并转化为可以进行实际工作的数据库。另一方面，关系模型是高层次的但有点模糊；开发者可能会抱怨，用箭头画图看起来并不是非常严格，形式主义并不真正帮助他们很多。

范畴论是精确的；它明确地赋予关系意义和结构，组合法则定义了哪些关系是可允许的，哪些是不允许的。范畴论不仅仅是关于箭头（如果只有箭头的话会相当无聊）；相反，它拥有许多领域的丰富成果，用一种通用语言表达，可以“翻译”为数据库术语。在许多情况下，重要的范畴论概念是数据库管理员传说中棘手的技术。当你谈论箭头时，你谈论的远不止箭头。这是非常有力的！
