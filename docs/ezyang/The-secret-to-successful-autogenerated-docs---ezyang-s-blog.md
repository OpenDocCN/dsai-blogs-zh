<!--yml

类别：未分类

日期：2024-07-01 18:18:16

-->

# 成功的自动生成文档秘诀：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/06/secret-of-autogenerated-docs/`](http://blog.ezyang.com/2010/06/secret-of-autogenerated-docs/)

我在自动生成文档上有了相当成功的任期，既是作者又是读者。因此，当[雅各布·卡普兰·莫斯关于撰写“优秀文档”](http://jacobian.org/writing/great-documentation/what-to-write/)的文章在 Reddit 上重新浮出水面，并对自动生成的文档提出严厉批评时，我坐下来思考自动生成的文档为何让开发人员留下不快的印象。

我解释了莫斯的具体反对意见（除了断言它们“毫无价值”外）如下：

1.  它们通常不包含你正在寻找的信息（*“At best it’s a slightly improved version of simply browsing through the source”*），

1.  它们冗长（*“good for is filling printed pages when contracts dictate delivery of a certain number of pages of documentation”*），

1.  作者们跳过了“写作”部分（*“There’s no substitute for documentation written...”*），

1.  作者们跳过了“组织”部分（*“...organized...”*），

1.  作者们跳过了“编辑”部分（*“...and edited by hand.”*），

1.  它给了有文档的错觉（*“...it lets maintainers fool themselves into thinking they have documentation”*）。

因此，成功自动生成文档的*秘密*是：

记住你的受众。

毫无疑问，亲爱的读者，你正在挑眉看着我，心里想着，“当然你应该记住你的受众；这是他们在任何写作课程中总是教你的。你没有告诉我们任何有用的东西！”所以，让我详细说明一下。

*为什么开发人员会忘记“记住他们的受众”？* 自动生成的文档的一个定义特征是其来源于源代码：编程语言的代码行和文档块交织在一起。这有一定的好处：首先，将注释保持在描述的代码附近有助于防止随着代码变化而出现文档腐败，此外，源代码开发人员可以轻松访问与他们正在阅读的源文件相关的文档。但文档**经常**是面向不愿意阅读代码库的人的，因此同时编写代码和文档会使编写者陷入错误的思维模式。将此与坐下来看教程相比，文本流入一个空白文档，不受代码等琐事的影响。

这很遗憾，因为在最终用户开发人员文档的情况下（真正适合自动文档的唯一时机），最初编写代码的人最有可能拥有关于正在文档化的接口的相关知识。

*什么是“记住我的观众”的意思？* 简而言之，这意味着把自己放在最终用户的鞋子里，并问自己，“如果我想要找到关于这个 API 的信息，我会寻找什么？” 这可能很困难（不幸的是，这里没有秘密），但首先要考虑这个问题。

*在撰写文档块时如何记住观众？* 虽然如果我能一挥手就说，“我要以我的观众为中心撰写文档块”那就太好了，但我知道我会忘记，因为有一天我匆忙之下写了一个尖刻的文档块或者完全忘记了写文档块。如果在写代码后立即写文档，五分钟后发现那个函数做错了事情并需要剔除，这会很令人沮丧。

因此，我为自己设立了这两条规则：

1.  写代码后立即撰写文档不是必须的（还不如不写）。

    像许多使用高级语言的人一样，我喜欢用代码来原型化 API 设计。我会写一些东西，试着使用它，根据我的用例进行修改，再写一些，最终我既有了可工作的代码又有了可行的设计。如果我在原型设计时没有写文档，那也没关系，但当这一切结束时，我需要写文档（希望在代码还在我活跃的思维空间中）。在最后写文档的行为有助于最终确定 API，并可以提出最后的触及点。我还使用我的工具链告诉我何时留下了未记录的代码（使用 Sphinx 时，这是使用`coverage`插件）。

1.  在撰写文档时，不断查看最终用户将看到的输出。

    当你编辑任何包含格式的文本时，可能会有一个写作/预览的循环。这个循环应该延续到文档块中：你编辑你的文档块，运行你的文档构建脚本，然后在浏览器中查看结果。如果输出美观的话会很有帮助！这也意味着你的文档工具链应该智能地处理你做出的更改并重新编译所需的内容。检查最终用户会看到的内容有助于让你保持正确的心态，并且迫使你承认，“是的，这些文档实际上并不可接受。”

*我的自动文档生成器产生的输出冗长且杂乱无章！* 我一般发现 Python 或 Haskell 生成的自动化文档比 Java 或 C++生成的文档更易阅读。这些语言的主要区别在于 Python 和 Haskell 将它们的模块组织到文件中；因此，使用这些语言的程序员更容易记住模块文档块！

模块文档块是非常重要的一部分。如果你的代码写得好、命名得好，一个能力强的源码分析者通常可以在比读取你的文档块多几倍的时间内弄清楚某个特定函数的作用。模块是类和函数之上的第一个组织单位，*恰好*是文档开始变得最有用的地方。它是开发者渴望的“高级别文档”的第一形式。

因此，在 Python 和 Haskell 中，你可以把一个模块中涉及的所有功能写在一个文件中，并且可以在顶部放一个文档块，说明整个文件的作用。很简单！但是在 Java 和 C++中，每个文件都是一个类（通常是一个小类），所以你没有机会这样做。Java 和最近的 C++有命名空间，可以发挥类似的作用，但在 Java 中，对于实际上是目录的东西，你应该把文档块放在哪里呢？

还有大量冗长的污染来自自动文档工具尝试生成文档，用于那些本来不打算供最终用户使用的类和函数。Haddock（Haskell 自动文档工具）通过不为模块未导出的任何函数生成文档来强烈执行此规定。Sphinx（Python 自动文档工具）默认情况下会忽略以下划线开头的函数。那些需要大量类的 Java 文档的人应该仔细考虑哪些类实际上他们希望人们使用。

*最后的想法。* “自动生成的文档”这个词是一个误称：没有*自动*生成文档。相反，自动文档工具应该被视为一种有价值的文档构建工具，让你获得代码和注释的内聚性、格式化、相互链接等好处。
