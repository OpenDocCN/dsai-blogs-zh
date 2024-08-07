<!--yml

category: 未分类

日期：2024-07-01 18:17:23

-->

# 什么是膜？：ezyang’s 博客

> 来源：[`blog.ezyang.com/2013/03/what-is-a-membran/`](http://blog.ezyang.com/2013/03/what-is-a-membran/)

如果你和某个特定群体一起呆得足够长（在我的情况下，是[ECMAScript TC39 委员会](http://wiki.ecmascript.org/doku.php)），你可能会听到“膜”这个术语被提起。最终，你会开始想知道，“嗯，膜到底是什么？”

就像许多聪明但简单的想法一样，膜最初作为[博士论文的脚注 [1]](http://www.erights.org/talks/thesis/)被引入。假设您正在构建分布式系统，在其中在两个独立节点之间传递对象的引用。如果我想将进程 `A` 中的 `foo` 的引用传递给进程 `B`，我几乎不能仅仅交出一个地址 - 内存空间不同！因此，我需要创建一个代表 `B` 中 `foo` 的包装对象 `wrappedFoo`，它知道如何访问 `A` 中的原始对象。到目前为止一切顺利。

现在问题来了：如果我将对 `wrappedFoo` 的引用传回到进程 `A` 中怎么办？如果我不够聪明，我可能会像最初那样做：在 `A` 中创建一个新的包装对象 `wrappedWrappedFoo`，它知道如何访问 `B` 中的 `wrappedFoo`。但这很愚蠢；实际上，当我再次返回到 `A` 时，我想要得到原始的 `foo` 对象。

这种包装和解包行为 *正是* 膜的本质。我们认为原始对象 `foo` 位于膜的“内部”（一个所谓的湿对象），当它离开膜时，它会被其自己的小膜包裹。然而，当对象返回到其原始膜时，包装会消失。就像生物学中一样！

还有最后一个操作，称为“门”：这发生在您在包装对象上调用方法时。由于包装对象实际上无法执行方法，它必须将请求转发给原始对象。然而，方法的 *参数* 在转发时需要被包装（或解包），正如您可能期望的那样。

在展示膜的基本原理时，我使用了类似 RPC 的系统，而更常见的用途是强制访问控制。膜非常重要；[Mozilla](https://developer.mozilla.org/en-US/docs/XPConnect_security_membranes) 在强制执行来自不同网站的对象之间访问限制时大量使用它们，但需要进行安全检查。（事实上，你知道 Mozilla 在他们的安全系统中使用基于能力的系统吗？挺有意思的！）需要注意的是，当我们解开膜时，我们跳过了安全检查——唯一可以接触未封装对象的对象是同一域中的对象。要获取更现代化的主题处理，请查看最近的一篇文章，[Trustworthy Proxies: Virtualizing Objects with Invariants](http://research.google.com/pubs/pub40736.html)，其中包含对膜的清晰解释。

[1] 嗯，实际上它是一个图；确切地说是第 71 页的图 9.3！
