<!--yml

分类：未分类

日期：2024-07-01 18:17:47

-->

# 空间泄漏动物园：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/05/space-leak-zoo/`](http://blog.ezyang.com/2011/05/space-leak-zoo/)

## 空间泄漏动物园

感谢所有向我们提供了空间泄漏样本的人！我们的专家已经检查并分类了所有的泄漏，我们很高兴向公众开放空间泄漏动物园的大门！

这里存在几种不同类型的空间泄漏，但它们非常不同，访客最好不要混淆它们（如果在实际应用中遇到，处理它们的方法各不相同，使用错误的技术可能会加剧情况）。

+   *内存泄漏*是指程序无法将内存释放回操作系统。这是一种罕见的情况，因为现代垃圾收集基本上已经消除了这种情况。在本系列中我们不会看到任何例子，尽管 Haskell 程序在使用非括号化的低级 FFI 分配函数时可能会表现出这种类型的泄漏。

+   *强引用泄漏*是指程序保留了一个实际上永远不会再被使用的内存引用。纯度和惰性的结合使得这类泄漏在习惯用法的 Haskell 程序中并不常见。纯度通过不鼓励使用可变引用来避免这些泄漏，如果在适当时未清除，可变引用可能会泄漏内存。惰性通过使得在只需要部分数据结构时，无意中生成过多数据结构变得困难来避免这些泄漏：我们从一开始就使用更少的内存。当然，在 Haskell 中使用可变引用或严格性可能会重新引入这些错误（有时可以通过弱引用修复前者——这就是“强引用”泄漏的名称），而存活变量泄漏（后文有描述）是一种让对闭包不熟悉的人感到惊讶的强引用泄漏类型。

+   *thunk 泄漏*是指程序在内存中建立了大量未评估的 thunk，这些 thunk 本质上占据了大量空间。当堆剖析显示大量`THUNK`对象，或者在评估这些 thunk 链时导致栈溢出时，可以观察到这种情况。这些泄漏依赖于惰性评估，因此在命令式世界中相对罕见。通过引入适当的严格性，可以修复这些问题。

+   *活变量泄漏*是指某个闭包（无论是 thunk 还是 lambda）包含对程序员预期已经释放的内存的引用。它们的产生是因为 thunk 和函数中的内存引用往往是隐式的（通过活变量），而不是显式的（如数据记录的情况）。在函数结果显著较小的情况下，引入严格性可以修复这些泄漏。然而，这些不像 thunk 泄漏那么容易修复，因为您必须确保*所有*的引用都已丢弃。此外，评估内存大块的存在可能并不一定表示有活变量泄漏；相反，这可能意味着流处理失败。见下文。

+   *流泄漏*是指程序应该只需少量输入来生成少量输出，因此只使用少量内存，但事实并非如此。相反，大量输入被强制并保留在内存中。这些泄漏依赖于惰性和中间数据结构，但与 thunk 泄漏不同，引入严格性可能会加剧情况。您可以通过复制工作并仔细跟踪数据依赖关系来修复它们。

+   *堆栈溢出*是指程序积累了许多需要在当前执行之后执行的挂起操作。当您的程序耗尽堆栈空间时，可以观察到这种情况。严格来说，这不是空间泄漏，而是由于修复不当的 thunk 泄漏或流泄漏可能导致堆栈溢出，因此我们在此包括它。 （我们还强调这与 thunk 泄漏不同，尽管有时它们看起来相同。）这些泄漏依赖于递归。您可以通过将递归代码转换为迭代风格（可以进行尾调用优化）或使用更好的数据结构来修复它们。通常也会打开优化以帮助解决问题。

+   *选择器泄漏*是*thunk 泄漏*的一个子类，当一个 thunk 仅使用记录的一个子集时，但由于尚未评估，导致整个记录被保留。这些泄漏大多已被 GHC 的*选择器 thunks*的处理杀死，但它们有时也由于优化而偶尔显示。见下文。

+   *优化诱导泄漏*是这里任何泄漏的伪装版本，源代码声称没有泄漏，但编译器的优化引入了空间泄漏。这些非常难以识别；我们不会将它们放在宠物园中！（您可以通过向 GHC Trac 提交 bug 来修复它们。）

+   *线程泄漏*是指太多的 Haskell 线程还未终止。您可以通过堆分析的 TSO（线程状态对象）来识别这一点：TSO 代表线程状态对象。这些很有趣，因为线程可能不终止的原因有各种各样。

在接下来的文章中，我们将画一些图片，并且给出每种泄漏的例子。作为练习，我邀请感兴趣的读者对[上次我们看到的泄漏](http://blog.ezyang.com/2011/05/calling-all-space-leaks/)进行分类。

*更新.* 我已经将“thunk 泄漏”与我现在称之为“活变量泄漏”分开，并重新澄清了一些其他要点，特别是关于强引用。我将在后续文章中详细展开我认为它们之间关键概念差异的讨论。
