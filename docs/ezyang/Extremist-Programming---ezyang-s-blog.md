<!--yml

category: 未分类

date: 2024-07-01 18:17:24

-->

# [极端编程](http://blog.ezyang.com/2012/11/extremist-programming/)：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/11/extremist-programming/`](http://blog.ezyang.com/2012/11/extremist-programming/)

*函数很棒。如果我们制作一种只有函数的编程语言呢？*

*对象是很棒的。如果我们制作一种编程语言，所有东西都是对象呢？*

*惰性评估很棒。如果我们制作一种编程语言，每种数据类型都是惰性的呢？*

**极端编程**（与极限编程无关）是将某些原则提升到高于一切的地位，并在各处应用它的行为。在尘埃落定后，人们经常会看到这种极端主义，并认为，“嗯，这有点有趣，但在 Y 中使用 X 显然是不合适的。你需要选择适合工作的正确工具！”

这里的关键是：有时你*应该*使用*错误的*工具——因为它可能是正确的工具，只是你还不知道而已。如果你不到处尝试使用函数，你可能不会意识到接受函数作为参数[1]或廉价 lambda[2]的函数的实用性。如果你不到处尝试使用对象，你可能不会意识到整数[3]和对象的类[4]也是对象。如果你不到处尝试使用惰性，你可能不会意识到纯度是更重要的语言特性[5]。

这导致两个建议：

1.  *在学习新原则时，尝试到处应用它。* 这样，你将更快地了解它何时适用，何时不适用，即使你对它的最初直觉是错误的。（另一方面，如果你没有意识到某种情况下该原则是适用的，正确的工具可能会导致你错失机会）。

1.  *在试图阐明某个原则的本质时，极端系统是最清晰的。* 如果你想知道使用惰性评估编程的感觉，你应该使用 Haskell，而不是一个具有可选惰性的语言。即使极端系统不太实用，它确实能更快地抓住问题的核心。

有很多情况下，极端主义是不合适的，但对于有趣的项目、小项目和研究，它确实可以教会你很多东西。在过去一年中，我最记得的一次互动是与 Adam Chlipala 合作时。我们正在 Coq 中进行一些证明，我一直采取逐步进行证明，然后再使用 Ltac 自动化的温和路线。亚当告诉我：“你应该从一开始就自动化证明，不要费心手动探索。” [6] 这是一条明智的建议，让我的生活好了很多：我想我只是不够极端而已！

*文件很棒。如果我们制作一个操作系统，所有东西都是文件呢？*

*Cons 单元格很棒。如果我们制作一种编程语言，所有东西都由 Cons 单元格构成呢？*

*数学很棒。如果我们制作一种编程语言，所有东西都来自数学呢？*

*数组真棒。如果我们创造一种编程语言，一切都是数组会怎样？*

* * *

[1] 高阶函数和组合子：这些往往不太流行，因为可能编写起来非常冗长，或者因为语言没有很好的词汇来描述高阶函数的接口。（类型在这里有些帮助。）

[2] 廉价的 lambda 对于方便使用许多功能是必要的，包括：单子、作用域分配（以及一般情况下的上下文）、回调、高阶函数。

[3] 考虑 Java 的早期版本，在整型和其他基本类型的自动装箱之前。

[4] Smalltalk 使用这个技术效果很好，JavaScript 也是如此。

[5] 这是我最喜欢的一个关于 Haskell 的叙述，出自 Simon Peyton Jones 的演讲 [Wearing the hair shirt](http://research.microsoft.com/en-us/um/people/simonpj/papers/haskell-retrospective/)（在这种情况下，指的是惰性）。

[6] 这是 Chlipala Coq 证明学派的核心，认识到有多么令人惊讶地能够欺骗经验丰富的计算机科学家手写等效于直线程序，而没有任何抽象。