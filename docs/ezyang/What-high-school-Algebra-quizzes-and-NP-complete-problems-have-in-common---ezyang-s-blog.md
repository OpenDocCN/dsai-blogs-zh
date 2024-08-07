<!--yml

category: 未分类

date: 2024-07-01 18:18:11

-->

# **高中代数测验和 NP 完全问题的共同点**：ezyang's 博客

> 来源：[`blog.ezyang.com/2010/08/what-high-school-algebra-quizzes-and-np-complete-problems-have-in-common/`](http://blog.ezyang.com/2010/08/what-high-school-algebra-quizzes-and-np-complete-problems-have-in-common/)

*我在 Galois 暑期实习中的经历*

*代数测验的世界*。作为一个高中生，我早在了解计算机科学之前就在使用计算机科学的概念。我记得参加数学测验——禁用计算器——面对一个困难的任务：大数的乘法。当涉及到铅笔和纸的算术时，我非常马虎——如果我不检查答案，我肯定会因为“愚蠢的错误”而失分。幸运的是，我知道以下的窍门：如果我将我的因数的数字相加（如果结果是十或更多，重新相加），这两个数的乘积应该与结果的数字之和相匹配。如果不匹配，我就知道我的答案是错的。直到后来我才发现这是校验和的一个非常基础的形式。

实际上，我重新发现的大部分技巧都是出于简单的学术需要：我的答案是否正确？事实上，虽然当时我不知道，但这个问题成为了我今年夏天在 Galois 实习的*基本基础*。

大约在我开始学习代数的时候，我开始注意到我的检查算术的技巧变得不够用了。如果老师让我计算多项式`(x + 2)(x - 3)(x - 5)`的展开式，我必须执行多步算术运算才能得到答案。检查每一步都很麻烦且容易出错——我深知我可能会对自己刚写的工作中的错误视而不见。我想要一种不同的方式来确保我的答案是正确的。

最终，我意识到我所要做的只是选择一个`x`的值，并将其代入原问题和答案`x³ - 6x² - x + 30`中。如果数值匹配，我会对我的答案相当有信心。我还意识到，如果我选择一个像`x = -2`这样的数，我甚至都不需要计算原始问题的值：答案显然是零！我“发明了”单元测试，并且借助这种技术，许多符号表达式都屈服于我的铅笔。（我作为一个刚入门的程序员独立学习了单元测试，但由于 PHP 程序员从不编写太多数学代码，我从未意识到这一点。）

* * *

*实际软件测试的世界。* 在这里，我们从代数测验的世界过渡到软件测试的世界。被测试的表达式比`x³ - 6x² - x + 30`更复杂，但大多数人仍然采用类似于高中时期的策略：他们手动挑选几个测试输入，以便能够合理地相信他们的新实现是正确的。如何知道程序的输出是正确的？对于许多简单的程序，被测试的功能足够简单，以至于测试人员能够心理上“知道”正确的结果，并手动记录下来——类似于挑选像`x = -2`这样特别容易让人类推断答案的输入。对于更复杂的程序，测试人员可能会使用参考实现来确定预期的行为应该是什么样子的。

测试如此只能显示 bug 的存在，而不能证明它们不存在。但是，正如许多软件公司发现的那样，这已经足够好了！如果程序员错过了一个重要的测试用例并且出现了 bug 报告，他会修复 bug 并添加一个回归测试来处理那个有 bug 的输入。因此，作为实用主义者，我们已经接受了这种状态：手动逐案测试（希望是自动化的）。*传统软件测试技术的现状基本上与高中生在代数测验中检查答案的方式是一样的。*比这更好的东西超越了理论计算机科学研究的障碍。

> *旁白。* 任何写过自动化测试的人都可以证明，自动化测试有两个主要任务：首先让你的代码能够自动测试（如果是算术比起内核驱动要容易得多），其次是想出一些有趣的情况来测试你的代码。对于后者来说，事实证明，虽然人类可以提出不错的边缘情况，但在提出随机测试用例方面他们真的非常糟糕。因此，一些极端实用的高科技测试技术包括让计算机生成随机输入。[模糊测试](http://en.wikipedia.org/wiki/Fuzz_testing)和[QuickCheck](http://en.wikipedia.org/wiki/QuickCheck)风格的测试都以此方法为特征，尽管模糊测试以无意义的输入为荣，而 QuickCheck 则努力生成有意义的输入。

* * *

*理论计算机科学的世界。* 批改你的代数测验的老师并没有像简单地选择几个随机数字，将它们代入你的答案中，看她是否得到正确答案那样简单。相反，她会将你的答案（程序本身）与答案卷上的标准答案（参考实现）进行比较，如果她能够判断答案相同，就会给你打分。如果你用费马最后定理来表达你的答案，她会因为你太过鲁莽而给你打分。

参考实现可能是错误的（答案键中的错误），但在这种情况下，它是我们判断程序是否“正确”的最佳标准。既然我们已经进入理论计算机科学的领域，我们可能会向[字面意思的精灵](http://tvtropes.org/pmwiki/pmwiki.php/Main/LiteralGenie)问这个问题：*通常能否确定两个程序是否等价？* 字面意思的精灵回答：“不！”这个问题是不可判定的：没有算法能够对所有输入回答这个问题。如果你能确定两个程序是否等价，你就能解决停机问题（无法解决问题的典型示例）：只需检查程序是否等价于一个无限循环的程序。

尽管工作中的理论家可能经常驯服无法计数的巨大无限，但对于工作中的程序员来说，处理的数量仍然非常有限——他们机器整数的大小、系统内存的数量、程序允许运行的时间。当你处理无限时，会出现各种奇怪的结果。例如，[赖斯定理](http://en.wikipedia.org/wiki/Rice's_theorem)声明，确定一个程序是否具有*任何*非平凡属性（即存在某些具有该属性的程序和某些没有该属性的程序）是不可判定的！如果我们加入一些合理的约束，比如“程序对所有输入都在多项式时间内终止”，那么这个问题的答案就是肯定的！但我们能否以比测试程序在每个输入上做相同事情更好的方式来做到这一点？

* * *

*更实际的计算机科学世界。* 我们已经放弃了足够的理论纯度，使得我们的问题对软件工程师再次变得有趣，但程序员要证明算法与其参考实现等效仍然非常困难。相比之下，用户很容易证明算法错误：他们只需给程序员一个输入，使得他的实现与参考实现不一致。

计算机科学家为这种情况起了一个名字：NP 问题，即可以在多项式时间内验证其解（在这种情况下，更像是反解：一个反例）。即使两个程序都在恒定时间内运行，如组合逻辑电路可能会（为了模拟这样一个电路，我们只需通过与电路中的门数量相同的门传播输入：没有依赖于输入），用来暴力检查等价性仍需指数时间。每次*增加*一个输入位，都会*加倍*需要检查的可能输入量。

实际上，电路非等效性的问题是 NP 完全的。我们一直在讨论程序等效性，但我们也可以讨论*问题*等效性，例如你可以将一个问题（图着色）转化为另一个问题（旅行推销员问题）。在 70 年代，计算机科学家花了大量时间证明需要“蛮力”的许多问题实际上都是同一个问题。斯蒂芬·库克引入了一个概念，即存在 NP 完全问题：NP 中的问题可以转化为其中的所有其他问题。最著名的 NP 完全问题的例子是 SAT，即给定一个带有布尔变量的逻辑公式，你询问是否存在变量的满足赋值，这些变量将导致该公式为真。

证明电路非等效性是 NP 完全的，我们需要展示它属于 NP（我们已经完成了），并且展示我们可以将某些其他 NP 完全问题转化为这个问题。使用 SAT 进行这个过程非常容易：编写一个程序，将 SAT 的布尔变量作为输入，并输出逻辑公式的结果，然后查看它是否等同于一个总是返回`false`的程序。

另一个方向稍微不那么微不足道，但从实际角度来看很重要：如果我们可以将我们的问题简化为 SAT 的一个实例，我可以向它投入一个高度优化的 SAT 求解器。可满足性问题同构于输出单个比特的逻辑电路。我们可以通过将电路合并成所谓的“miter”来将电路等效性问题转化为 SAT：我们将两个原始逻辑电路的输入组合成一个单一的集合，将其输入到两个电路中，然后测试两个电路之间对应的输出位是否相等（XOR），将整个结果进行 OR 运算。如果输出位在两个电路之间相同（所有的 XOR 返回 0），则生成电路输出 0，如果存在不匹配，则输出 1。

“很好”，你可能会想，“但我是程序员，不是硬件设计师。我的大多数程序不能仅用逻辑门来表达！” 这是正确的：要编码状态，你还需要锁存器，并且输入/输出需要通过特殊的输入和输出“端口”进行模拟。然而，有许多重要的问题纯粹是组合的：其中一个闪亮的例子是密码学，它保护你的钱，采用了大量复杂的数学并进行了无情的优化。

但仍然有一个持续的抱怨：即使我的程序只是逻辑电路，我也不想用 AND、OR 和 NOT 来编写它们。那看起来太痛苦了！

* * *

进入[Cryptol](http://www.galois.com/technology/communications_security/cryptol)，这是我在 Galois 公司工作的项目。Cryptol 自称如下：

> Cryptol 是用于编写密码算法规范的语言。它还是一个工具集，用于在 VHDL、C 和 Haskell 中生成高可靠性、高效的实现。Cryptol 工具包括对比参考规范与实现的等效性检查，无论实现是否从规范编译而来。

但是在我这个菜鸟实习生的谦虚观点中，真正使它显著的是，它可以将用 C、VHDL 或 Cryptol 等编程语言编写的程序转换为逻辑电路，或者我们所称的“形式模型”，然后你可以将其投放到一个 SAT 求解器中，后者会比暴力尝试所有可能的输入更明智地处理。有一次，我心想，“Cryptol 居然能工作真是个奇迹！”但它确实能在其密码算法问题域内非常成功地工作。传统软件测试的最新技术是手工编写的测试，只能显示实现中存在的缺陷；*Cryptol 的最新技术是完全自动化的测试，可以保证实现没有缺陷*。（当然，Cryptol 也可能有 bug，但这是高可靠性的生活方式。）

* * *

SAT 求解器可能是程序员手边最被低估的高科技工具之一。一个工业级别的 SAT 求解器可以在午餐时间内解决大多数 NP 完全问题，而 NP 类问题具有广泛的实际应用。然而，使用 SAT 求解器的常见障碍包括：

1.  没有简单的方法将你的问题转化为 SAT 问题，然后在高度优化的求解器之一上运行，这些求解器通常在学术界文档化不足且不友好。

1.  当你的 SAT 求解器通过或失败时（取决于什么是“错误”），生成友好的错误消息。

1.  说服你的团队，真的，你需要一个 SAT 求解器（而不是构建[你自己的，可能不那么高效的实现](http://algebraicthunk.net/~dburrows/blog/entry/package-management-sudoku/)）。

我的主要项目是通过构建名为[ABC，一个用于顺序合成和验证的系统](http://www.eecs.berkeley.edu/~alanmi/abc/)的绑定集来解决 Haskell 中的第一个问题，称为`abcBridge`。有人可能会观察到 Haskell 已经有了一些 SAT 求解库：ABC 之所以引人注目，是因为它采用了一种 SAT 的替代表述形式，即与非图（NAND 门能模拟所有布尔逻辑），以及一些处理 AIG 的新技术，比如 fraiging，这是一种高级策略，用于寻找电路中功能等效的子集。

项目本身非常有趣：由于我是从零开始构建这个库，所以在 API 决策上有很大的灵活性，但同时也深入了 Cryptol 代码库，需要将我的绑定与其集成。希望有幸能在实习结束时将代码作为开源发布。但当我的实习在两周后结束时，我会错过更多不仅仅是我的项目。我希望能跟进一篇关于我的实习的非技术性文章。请继续关注！

*事后诸事.* 嘿，这是我的第一百篇文章。甜蜜！
