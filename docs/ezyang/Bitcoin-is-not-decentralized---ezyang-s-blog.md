<!--yml

category: 未分类

date: 2024-07-01 18:17:46

-->

# 比特币并非去中心化：ezyang 的博客

> 来源：[`blog.ezyang.com/2011/06/bitcoin-is-not-decentralized/`](http://blog.ezyang.com/2011/06/bitcoin-is-not-decentralized/)

比特币由中本聪设计，主要客户端由一群来自[bitcoin.org](http://bitcoin.org/)的人开发。你在乎这些人是谁吗？理论上来说，你不应该在乎：他们所做的一切只是为开源协议开发一个开源客户端。任何人都可以开发自己的客户端（有些人已经这样做了），除了比特币网络中每个人的一致同意，没有人能改变协议。这是因为比特币网络被设计成去中心化的。

如果你相信比特币的长期可行性，你应该关心这些人是谁。虽然比特币本身是去中心化的，但从比特币到新货币的过渡不能是去中心化的。这一过渡的发生是由所有密码系统最终变得过时这一事实所保证的。谁将决定如何构建这种新货币？很可能是比特币的原始创造者，如果你在比特币中拥有重要的持有，你应该关心这些人是谁。

以下的文章将更仔细地阐述这一论点，如下：

1.  包括加密哈希在内的密码系统必须在最终被替换的前提下使用。有人可能会争辩说，“如果比特币的加密学被破解，金融行业的其余部分也会陷入麻烦” — 我们解释了为什么这对比特币是不相关的。我们还看到，如果比特币成为一个严肃的货币，合理地预期它将存在足够长的时间来发生这种过时。

1.  比特币社区流传着几种粗糙的过渡计划。我们描述了最常见的分散化和最常见的集中化变体，并解释了为什么分散化变体不能以非破坏性方式运行，同时吸引了经济学和具有类似属性的现有市场。

1.  我们更加仔细地审视这些分散化和集中化转变的影响，并评估与比特币作为新兴货币面临的其他风险相比，转变的风险。我们建议，虽然比特币的转变并不是一个核心关注点，但天真的去中心化的观念是一个需要打破的神话。

我已将这篇文章分成几个部分，以便那些对特定论点感兴趣的读者随意跳跃阅读。

### 密码系统的时限炸弹

“所有加密系统最终会变得过时。” 与货币相比，加密哈希是一个相对较新的发明，只能追溯到 20 世纪 70 年代。MD5 是 1991 年发明的，仅花了大约十五年的时间就[彻底被攻破了](http://valerieaurora.org/hash.html)。对于计算机程序员来说，加密技术的变化是不可避免的，并且系统设计时考虑到了这一点。例如，考虑一下用于安全互联网交易（包括金融交易）的 SSL 证书。这些证书需要定期更新，随着新证书的颁发，可以增加它们的保护级别，使用更新的密码或更长的密钥长度。大多数当前使用的加密技术遵循这种模式：密码和密钥可以相对容易地替换。

然而，比特币是特殊的。它实现去中心化的方式是将所有相关的技术细节嵌入到协议中。其中包括[哈希算法 SHA-256](https://en.bitcoin.it/wiki/Protocol_specification)。在比特币中“改变”哈希算法是完全不可能的；任何变动都会构成协议的变动，因此会导致一个全新的货币。不要相信任何告诉你相反的人。论点“如果比特币的加密被破解，其他金融行业也会陷入困境”是无关紧要的，因为其他金融机构可以控制它们使用的密码，并且可以轻松地更改它们：比特币却不能。由于 SHA-1 的弱点可能会影响 SHA-2 系列（其中 SHA-256 是一员），因此 SHA-3 的竞赛已经在进行中。

欺诈交易是否会持续足够长时间以变得实际？也许不会（毕竟，该货币可能在到达此阶段之前被许多其他潜在问题杀死）。然而，如果它确实变得建立起来，你可以期待它会是一个顽强的小家伙。货币会长期存在。

### 去中心化和中心化货币的过渡

比特币社区已经意识到转变将是必要的事实，尽管普遍的感觉是“我们到那时再想办法”，也有一些模糊的提议被浮出水面。冒着制造草人的风险，我现在想呈现我对两种最广泛提出的计划的看法。首先是去中心化计划：

> 由于加密系统不会一夜之间崩溃，一旦对 SHA-256 的关注达到足够高的程度，我们将创建一个使用更强加密哈希的新版本比特币。然后，我们将让市场决定这两种货币之间的汇率，并允许人们从一种货币转移到另一种货币。

这是分散的，因为任何人都可以提出一种新的货币：市场将决定最终哪一种会胜出。它也不可能以非破坏性的方式运作，因为任何想要将旧比特币兑换为新比特币的人都必须找到愿意购买的买家，而在某些时候，超级通胀将确保没有愿意购买的买家。所有现有的比特币将变得毫无价值。

此时，我们将稍作停留，进入中国的[月饼黑市场](http://marketplace.publicradio.org/display/web/2010/09/21/a-black-market-for-mooncakes-in-china/)，这是一个非常引人入胜的“货币”，与即将过时的比特币有很多相似之处。这个市场的前提是，虽然给予现金贿赂是非法的，但是赠送月饼券是合法的。因此，想要贿赂某人的人可以简单地“赠送”给他们一个月饼券，然后将其在黑市上出售，转换回现金。

参与月饼黑市的人必须小心，因为一旦中秋节到来，所有这些券必须兑换成月饼或变得毫无价值。随着日期的临近，你会看到对越来越贬值的券进行越来越疯狂的烫手山芋游戏。输家？他们最终会拥有大量的月饼。当然，有一个关键的区别，那就是比特币游戏的输家最终一无所有。

这是一个过渡吗？是的。它会造成混乱吗？绝对是。这当然不是你希望用于日常交易的货币做的事情。当然，对一些行业来说，这可能是可以接受的风险，我们将在最后一节中进一步分析这一点。

这是集中计划：

> 一旦对哈希算法的担忧达到足够高的程度，我们将创建一个新的比特币协议。这个协议不仅包括一个新的哈希算法，还基于某个日期的旧比特币经济价值：在那一点上，所有新的交易在新的比特币方案中都是无效的，并且使用该快照来确定每个人拥有的比特币数量。

还有一种变体，涉及到在他们设法切换之前对哈希算法进行主动攻击的情况，其中包括将特定的区块链标记为已知的良好区块链，并清除疑似的欺诈交易。

这个计划真的是集中的吗？是的：有人需要设计新的协议，说服所有客户接受它，并在新经济到来时统一切换到新的协议。比特币经济的分裂将会极大地破坏，对任何主要参与者都不利。比特币协议的任何其他更改（到那时可能会有很多提议）都可能对比特币经济产生重大影响。

### 影响和风险

在这里，我们评估了一个问题，“我真的在乎吗？” 短期内，不在乎。比特币有[许多，许多弱点](http://www.quora.com/Is-the-cryptocurrency-Bitcoin-a-good-idea?srid=pxt)将被检验。虽然我个人希望它会成功（它无疑是一个从未进行过的伟大实验），但我的评估是它的机会并不乐观。过度担心过渡并不是明智的时间利用方式。

然而，这并不意味着这不是一个重要的事实需要记住。比特币的未来取决于那些将设计其继任者的人。如果您在比特币上投入了大量资金，至少应该考虑谁拥有下一个王国的钥匙。更为紧迫的问题是[比特币客户端的单一文化](http://timothyblee.com/2011/04/19/bitcoins-collusion-problem)的影响（某人可能会推出一个更新，调整协议以达到不良目的）。使用比特币的人应尽快多样化其客户端。您应极度怀疑那些使他人能够将您的客户端从协议的一个版本翻转到另一个版本的更新。尽可能保持协议的不可变性，因为没有它，比特币根本不是去中心化的。

感谢 Nelson Elhage，Kevin Riggle，Shae Erisson 和 Russell O’Connor 阅读并评论本文草稿。

*更新.* 与主题无关的评论将会被严格审查。你已经被警告了。

*更新二.* 在 Hacker News 和 Reddit 的讨论中出现了一个可能的第三继任计划，即分散化的自启动货币。基本上，多种货币竞争入驻和采纳，但与仅通过汇率分开的两种完全独立的货币不同，这些货币在某种程度上被固定在旧比特币货币上（也许它们拒绝在某个日期之后的所有比特币交易，或者它们要求进行某种破坏性操作才能将旧比特币转换为新比特币——后者可能存在安全漏洞）。我没有分析过这种情况下的经济情况，我鼓励其他人接手。我的直觉是，它仍然会带来破坏性影响；也许会更多，因为这些货币的人为固定。
