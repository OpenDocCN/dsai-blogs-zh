<!--yml

category: 未分类

date: 2024-07-01 18:18:02

-->

# Reflexivity. Qed.：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/11/reflexivity-qed/`](http://blog.ezyang.com/2010/11/reflexivity-qed/)

*在其中讨论了 Mendeley、Software Foundations 和 Coq。*

有一天，我在`#haskell-blah`上抱怨过，整理我下载的所有论文（当然，还没有阅读）。当你从互联网上下载一篇论文时，它的名称可能会是一些非常不方便的东西，比如`2010.pdf`、`icfp10.pdf`或`paper.pdf`。因此，要找到一个你一个月前浏览过并模糊记得标题的论文，你需要某种组织系统。在使用 Mendeley 之前，我采用了`AuthorName-PaperTitle.pdf`的惯例，但是我总觉得从五个人的列表中挑选一个作者作为开头有些不好，而且我仍然找不到我要找的论文。

就在这个时候，有人（我没有日志，所以我不记得确切是谁）向我推荐了[Mendeley](http://www.mendeley.com/)。Mendeley 是免费的（就像啤酒一样的）软件，帮助你组织论文并将它们上传到云端；作为回报，他们获得了关于人们正在阅读的论文以及像我这样的元数据狂人策划他们数据库的各种有趣数据。

它不必做太多来改善我现有的临时命名方案。但它做得非常出色。在我将我的[paper 数据库](http://www.mendeley.com/groups/680221/edward-z-yang-s-paper-repository/overview/)转移到它之后，花一个下午整理 200 篇论文数据库变得相当容易，以确保所有的论文都附有合理的元数据。这些合理的元数据意味着你可以按作者（显然 Simon Peyton Jones 和 Chris Okasaki 是我最喜欢的两位作者之一）和会议对数据库进行切片（万一我真的写了一篇论文并需要找到发送的地方）。你还可以根据自己的主题对论文进行分类，如果像我这样拥有完全无关的研究文献群体，这将非常有用。简单而有效。

> 噢，我对 Mendeley 还是有一些抱怨的。它的 PDF 查看器有些欠缺：如果我向下翻页，它会完全跳到下一页而不是连续滚动；元数据提取可能会更好（基本上，它应该足够好以便能够在在线数据库中查找并填写数据库）；论文的工作流程应该更好（而不仅仅是一个*已读*或*未读*的切换，这完全没有用）；等等。但它足够好以提供价值，我愿意忽略这些小问题。

整理完所有文件后，我突然意识到最近并没有添加任何新文件到我的收藏中。文件要么通过朋友转发给我，要么我在寻找某个特定主题时会有相关的论文出现，但我实际上没有任何新的论文可供查看。为了解决这个问题，我决定挑选一些名字，去查看他们的最新出版物。

在此过程中，我注意到了[本杰明·皮尔斯的出版物](http://www.cis.upenn.edu/~bcpierce/papers/index.shtml#Recent)上的一个有趣的幻灯片。这份幻灯片是为一个名为[证明助手作为教学助理：从前线看](http://www.cis.upenn.edu/~bcpierce/papers/LambdaTA-ITP.pdf)的主题演讲准备的。我认为这是解决教学证明问题的一种非常迷人的方法，而且更好的是，课程笔记可以在线获取！

对我来说，精确地描述[软件基础](http://www.cis.upenn.edu/~bcpierce/sf/)是多么不可思议。我发现开始使用证明助手有点困难，因为不清楚应该用它们证明什么：选择太简单的东西感觉毫无意义，选择太难的东西又不知道如何着手解决问题。证明助手也相当复杂（这让我想起我曾在 Galois 听 Eric 和 Trevor 讨论证明策略的时候...那真是一个非常难懂的对话），所以如果你深入研究手册，你会发现自己掌握了许多工具，但并不知道如何全部运用起来。

软件基础之所以伟大，是因为它不是教你如何使用证明助手：它教你逻辑、函数式编程以及编程语言的基础，都是建立在 Coq 证明助手之上的。因此，你有一袋关于这些主题的有趣且基础的定理，想要证明它们，而这门课程则向你展示如何使用证明助手来证明它们。

这也是自学的一个相当理想的情况，因为与许多教科书的练习不同，你的 Coq 解释器会告诉你何时得到*正确的答案*。证明助手之所以有趣，正是因为它们有点像你可以在不知道解决方案的情况下创建并解决的谜题。因此，如果你有多余的时间，并且想学习如何使用证明助手但之前从未着手，我强烈推荐[去看看](http://www.cis.upenn.edu/~bcpierce/sf/)。