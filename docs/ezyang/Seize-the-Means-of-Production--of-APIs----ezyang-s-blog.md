<!--yml

分类：未分类

日期：2024-07-01 18:17:04

-->

# 掌握生产资料（API 的）：ezyang 的博客

> 来源：[`blog.ezyang.com/2016/09/seize-the-means-of-production-of-apis/`](http://blog.ezyang.com/2016/09/seize-the-means-of-production-of-apis/)

有一个糟糕的 API 正在毁掉你的一天。你会怎么做？

在不进行道德判断的情况下，我想指出这两种方法之间确实存在很大的不同。在 Dropbox 的情况下，Dropbox 无法（直接）影响苹果为其操作系统提供的 API。因此，它别无选择，只能在现有 API 的*框架内*工作。（当苹果说跳时，你会问，“跳多高？”）但在亚当的情况下，POSIX 由开源项目 Linux 实现，通过一些[好主意](http://dune.scs.stanford.edu/)，亚当可以在 Linux 的*顶部*实现他的新接口（避免从头开始编写操作系统的必要性）。

API 跨越社会边界：有使用 API 生成软件的无产阶级，有控制 API 的资产阶级。当大公司成为“大人物”时，我们的选择只能是绕过他们糟糕的 API 或者付给他们足够多的钱来修复他们糟糕的 API。但当“大人物”是一个开源项目时，你的选择就会改变。当然，你可以绕过他们糟糕的 API。或者你可以**掌握生产资料**，从而使你能够修复这些糟糕的 API。

我所说的掌握生产资料是什么意思呢？确实，生产资料到底是什么？一个开源 API 并不孤立存在；它是由提供 API 的软件、为维护这些技术付出时间和专业知识的开发者、甚至是宣传平台共同使其有用。掌握生产资料就是要控制这些方面。如果你能说服体制认为你是开源软件的核心贡献者，那么你就能够修复这些糟糕的 API。如果你不愿意或者不能这样做，你仍然可以分支、提供商或者重写项目，但这并不是掌握生产资料，而是从头开始重新创建它。另一种可能性是在现有 API 的基础上构建你需要的抽象（就像亚当所做的那样），尽管你始终面临原始项目不关注你需求的风险。

一次又一次地，我看到与开源项目合作的人们拒绝掌握生产资料。相反，他们愿意写越来越复杂的变通方法来解决问题，这都是为了保持在界限内。你可能会说，“这只是做事的方法(TM)”，但在某个时候，你解决问题所做的工作量可能比直接修复它还要多。

所以停止吧。不要忍受糟糕的 API。不要局限自己。掌握生产资料吧。

**反对意见。**

1.  这篇文章所倡导的绝非别的，仅仅是无休止的无意义琐事；如果您认真对待这些建议，您将永远无法完成任何事情。

1.  虽然从总体上看，解决糟糕的 API 的成本可能超过了修复它的成本，但对于个体而言，成本通常较低。因此，即使您能够完美地预测解决方案与正确修复的成本，个体的激励也会阻止正确的修复。

1.  用户（包括开发人员）对他们使用的软件一无所知，并且缺乏设计更好的 API 的能力，即使他们知道痛点所在。

1.  很少能够单方面占领生产资料的控制权。在理想的世界中，要成为核心贡献者，仅仅展示对项目的持续、有用的贡献是足够的。我们都知道现实世界更加混乱。
