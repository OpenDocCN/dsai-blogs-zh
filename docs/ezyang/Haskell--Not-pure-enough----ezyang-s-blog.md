<!--yml

category: 未分类

date: 2024-07-01 18:17:53

-->

# Haskell：不够纯粹？：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/05/haskell-not-pure-enough/`](http://blog.ezyang.com/2011/05/haskell-not-pure-enough/)

## Haskell：不够纯粹？

众所周知，`unsafePerformIO`是一个邪恶的工具，通过它，不纯的效果可以进入本来纯洁的 Haskell 代码。但是 Haskell 的其余部分真的那么纯粹吗？这里有一些问题需要问：

1.  `maxBound :: Int`的值是多少？

1.  `(\x y -> x / y == (3 / 7 :: Double))`，传入`3`和`7`作为参数时的值是多少？

1.  `os :: String`的值来自`System.Info`吗？

1.  `foldr (+) 0 [1..100] :: Int`的值是多少？

对于这些问题的每一个答案都是模糊不清的——或者你可以说它们是明确定义的，但你需要一些额外的信息来确定实际结果。

1.  Haskell 98 报告保证`Int`的值至少是`-2²⁹`到`2²⁹ - 1`。但是确切的值取决于你使用的 Haskell 实现（是否需要用于垃圾回收的位）以及你是在 32 位还是 64 位系统上。

1.  根据浮点寄存器的过度精度是否用于计算除法，或者是否遵循 IEEE 标准，此等式可能成立也可能不成立。

1.  程序运行的操作系统不同，此值将会改变。

1.  程序在运行时分配的栈空间不同，可能会返回结果，也可能会栈溢出。

在某些方面，这些构造以有趣的方式破坏了引用透明性：虽然它们的值在程序的单次执行期间保证一致，但它们可能在我们程序的不同编译和运行时执行之间有所变化。

这个合理吗？如果不合理，我们应该怎么说这些 Haskell 程序的语义？

在`#haskell`讨论了这个话题，我和一些参与者就此进行了热烈的讨论。我会尝试在这里总结一些观点。

+   *数学学派*认为所有这一切都非常不令人满意，他们的编程语言应该在所有编译和运行时执行中遵循一些精确的语义。人们应该使用任意大小的整数，如果需要模运算，要明确指定模数大小（`Int32`？ `Int64`？）。`os`简直是该放在`IO`罪恶箱中的一个悲剧。正如 tolkad 所说：“没有标准，你将迷失在未指定语义的海洋中。坚守规范的规则，否则你将被模糊性所吞噬。” 我们生活在的宇宙的局限性对数学家来说有些尴尬，但只要程序以一个漂亮的*栈溢出*崩溃，他们就愿意接受部分正确性的结果。一个有趣的子组是*分布式系统学派*，他们同样关心对计算环境所作的假设，但出于非常实际的原因。如果您的程序在异构机器上运行多个副本，则最好不要对传输中的指针大小做任何假设。

+   *编译时学派*认为数学方法在现实世界的编程中是不可行的：应该考虑编译编程。他们愿意在源代码程序中接受一些不确定性，但所有的歧义应该在程序编译后清除。如果他们感觉特别大胆，他们会根据编译时选项以多种含义编写程序。他们可以接受运行时确定的栈溢出，但对此也感到有些不舒服。这当然比`os`的情况要好，后者可能因运行时而异。数学家们用这样的例子取笑他们：“动态链接器或虚拟机怎么样，其中一些编译工作直到运行时才完成呢？”

+   *运行时学派*说：“对执行间的引用透明度无所谓”，只关心程序运行期间的内部一致性。他们不仅可以接受栈溢出，还可以接受命令行参数设置全局（纯粹！）变量，因为这些在执行期间不会改变（也许他们认为`getArgs`的签名应该是`[String]`而不是`IO [String]`），或者不安全地读取外部数据文件的内容在程序启动时。他们在文档中写道：“这个整数在应用程序的一次执行到另一次执行之间不需要保持一致。”其他人都有些发抖，但大多数人在某个时候都会沉迷于这种罪恶的快感。

所以，你属于哪个学派呢？

*附言.* 由于 Rob Harper 最近发布了另一篇[非常叛逆的博客文章](http://existentialtype.wordpress.com/2011/05/01/of-course-ml-has-monads/)，而且因为他的结尾言论与本文主题（纯度）有些关联，我觉得我忍不住要偷偷加上几句话。Rob Harper 说到：

> 那么为什么我们不默认这样做呢？因为这不是一个好主意。是的，起初听起来很美好，但后来你意识到这其实很可怕。一旦你进入 IO 单子，你就永远被困在那里，且被降为 Algol 风格的命令式编程。你不能轻易地在函数式和单子式风格之间转换，而不进行根本性的代码重构。而且你不可避免地需要使用 unsafePerformIO 来完成任何重要的工作。从实际角度来看，你失去了一个有用的概念——良性效应，这简直糟透了！

我认为 Harper 夸大了在 Haskell 中写函数式命令式程序的能力不足（从函数式到单子式的转换，在实践中确实很烦人，但相对来说是比较公式化的）。但这些实际上的关注确实影响了程序员的日常工作，正如我们在这里所看到的，纯度有各种各样的灰色阴影。在 Haskell 当前的情况上方和下方都有设计空间，但我认为认为纯度应该被完全放弃是错失了重点。