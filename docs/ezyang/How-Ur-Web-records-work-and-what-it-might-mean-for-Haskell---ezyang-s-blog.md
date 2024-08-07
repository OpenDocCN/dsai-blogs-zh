<!--yml

category: 未分类

日期：2024-07-01 18:17:31

-->

# Ur/Web 记录的工作原理及其对 Haskell 可能意味着什么：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/04/how-urweb-records-work-and-what-it-might-mean-for-haskell/`](http://blog.ezyang.com/2012/04/how-urweb-records-work-and-what-it-might-mean-for-haskell/)

[Ur](http://www.impredicative.com/ur/)是一种编程语言，除其他外，它具有一个非常有趣的记录系统。记录系统在 Haskell 社区中是一个非常激烈辩论的话题，我注意到有人曾评论过“Ur/Web 有一个非常先进的记录系统。如果有人能够看一看 UR 实现论文，并尝试从 Haskell 的角度来梳理记录的解释，那将非常有帮助！”本文试图执行这种提炼，基于我与 Ur 记录系统互动和其主要存在原因之一：元编程的经验。（次要命名注意事项：Ur 是基础语言，而 Ur/Web 是用于 Web 编程的基础语言的专业化版本，它实际上也有一个编译器。为了技术上的精确性，我将在本文中始终将语言称为 Ur。）

### 记录和代数数据类型并不相同

在 Haskell 中，如果要定义记录，您必须去编写`data`声明：

```
data Foo = Foo { bar :: Int, baz :: Bool }

```

在 Ur 中，这两个概念是分开的：您可以定义一个代数数据类型（`Foo`构造函数），并且可以编写描述记录的类型（类型的`{ foo :: Int, bar :: Bool}`部分）。为了强调这一点，实际上有很多种方式可以在 Ur/Web 中拼写这个记录。我可以定义一个类型同义词：

```
type foo = { Bar : int, Baz : bool }

```

这对我来说没有提供保护，以免将其与结构上相似但语义上不同的`type qux = { Bar : int, Baz : bool }`混淆，或者我可以定义：

```
datatype foo = Foo of { Bar : int, Baz : bool }

```

这被展开为：

```
type foo' = { Bar : int, Baz : bool }
datatype foo = Foo of foo'

```

也就是说，这种数据类型只有一个构造函数，只接受一个参数，即记录！这个定义更接近原始 Haskell 定义的精神。（ML 用户可能熟悉这种风格；Ur 明显来自于这一传统。）

这种将代数数据类型与记录分离的设计意味着现在我们有了明显的记录构造设施（`let val x = { Bar = 2, Baz = true }`）和记录投影（`x.Bar`）；虽然如果我有一个数据类型，我必须先解包它才能从中投影。这些记录类型在排列上是唯一的（顺序无关紧要），这使它们比`HList`更有趣。它们也非常简洁：单元就是空记录类型`{}`，元组就是带有特殊字段名的记录：`1`，`2`等。

### 记录的类型和种类

现在，如果这就是 Ur 记录系统的全部内容，那就不会有什么意思。但实际上，字段 `#Bar` 在语言中是一个一流表达式，而花括号记录类型语法实际上是语法糖！解开这一点将要求我们定义相当多的新种类，以及大量的类型级别计算。

在纯 Haskell 中，我们只有一种种类：`*`，在 Ur 的术语中是 `Type`。值居住于类型，这些类型居住于这种种类。然而，Ur 的记录系统要求更多的外来种类：其中一种是 `Name` 种类，它表示记录字段名（`#Foo` 是其中之一）。然而，GHC 已经有了这个：它是 [最近添加的](http://hackage.haskell.org/trac/ghc/wiki/TypeNats/Basics) `Symbol` 种类。然而，GHC 没有的是 `{k}` 的种构造子，它是“类型级记录”的种类。如果值级别记录是包含数据的东西，那么类型级别记录就是*描述*值级别记录的东西。然而，它们并不是值级别记录的*类型*（因为如果它们是的话，它们的种类将是 `Type`）。让我们看一个具体的例子。

当我写：

```
type foo = { Bar : int, Baz : bool }

```

我真正要写的是：

```
type foo = $[ Bar = int, Baz = bool ]

```

`$` 是一个类型级别的操作符，被应用于表达式 `[ Bar = int, Baz = bool ]`，它是一个类型级别的记录，具体来说是 `{Type}` 的一种（记录的“值”是类型）。美元符号接受类型级别的记录，并将它们转换为 `Type`（以便它们实际上可以被值居住）。

这可能看起来是一个毫无意义的区分，直到你意识到，Ur 有类型级别的操作符，它们仅适用于类型级别的记录，而不是一般的类型。两个最重要的原始类型级别操作是连接和映射。它们的功能正如你所期望的：连接将两个记录放在一起，而映射将类型级别的函数应用于记录的每个成员：因此，我可以通过映射列表类型构造函数轻松地将 `[ Bar = int, Baz = bool ]` 转换为 `[ Bar = list int, Baz = list bool ]`。可扩展记录和元编程一举完成！

现在，请回想一下，字段名都存在于全局命名空间中。那么，如果我尝试执行 `[ Bar = bool ] ++ [ Bar = int ]` 会发生什么？Ur 类型检查器将拒绝这个声明，因为我没有提供这些记录“不相交”的（无法满足的）证明义务。一般来说，如果我有两个记录类型 `t1` 和 `t2`，我想要连接它们，我需要一个不相交证明 `[t1 ~ t2]`。处理不相交证明对于习惯于传统函数式编程语言的用户来说可能感觉相当不寻常，但对于依赖类型语言的用户来说并不那么奇怪。事实上，Ur/Web 编译器使处理不相交义务变得非常容易，如果可能的话会自动推断它们，并了解有关连接和映射的一些基本事实。

### 类型级别计算

Ur 记录系统至关重要地依赖于类型级计算来增强其表达能力：我们可以展开、收缩和映射记录，我们还可以利用“折叠器”，这些是利用类型级记录作为结构的函数，允许对记录进行通用折叠。有关这些更多信息，请参阅[类型级计算教程](http://www.impredicative.com/ur/tutorial/tlc.html)。但为了以用户友好的方式提供这些功能，Ur 关键依赖于编译器具有对这些运算符如何工作的某种了解，以避免用户解除大量微不足道的证明义务。

不幸的是，在这里，我必须承认对于其余的 Haskell 记录提案的工作方式以及这样一个记录系统如何与 Haskell 交互（Ur 确实有类型类，因此这种交互至少已经有了相当深入的研究。）我并不了解。虽然这个提案有一个在现有语言中有着明确定义的系统的好处，但它很复杂，并且显然是在追求完美。但我认为它对于理解除了类型级字符串之外可能需要添加的内容有所帮助，以实现[Gershom Bazerman 在这里的愿景](http://www.haskell.org/pipermail/glasgow-haskell-users/2011-December/021410.html)：

> 在我看来，只有一个基本缺失的语言特性，那就是适当类型化的类型级字符串（理想情况下，还能将这些字符串反映回值级）。鉴于此，模板 Haskell 和 HList 的种种技巧，我相信可以设计出相当多优雅的记录包。基于这种经验，我们可以决定哪些语法糖能够有助于完全省略 TH 层。
