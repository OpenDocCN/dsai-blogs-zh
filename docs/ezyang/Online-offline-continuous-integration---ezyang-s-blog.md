<!--yml

category: 未分类

date: 2024-07-01 18:16:57

-->

# 在线/离线连续集成：ezyang 的博客

> 来源：[`blog.ezyang.com/2018/03/online-offline-continuous-integration/`](http://blog.ezyang.com/2018/03/online-offline-continuous-integration/)

如果您在连续集成脚本中使用过这些命令，请举手：

+   `apt install somepackage`

+   `pip install -r requirements.txt` 或 `pip install somepkg`

+   `conda install blah`

+   `cabal update` 或 `cabal install blah`

+   `git clone https://github.com/someguy/somerepo`

+   `wget http://some-website/thingy-latest.tgz`

您能说出问题在哪里吗？这些命令不可再现：取决于运行时机，它们可能会产生不同的结果。更隐蔽的是，*大多数*情况下它们给出的结果相同（或许对您的用例仍然有效的不同结果）。

**我知道，我们需要一个可再现的构建！** 工具作者对此问题的主要回应是夺取生产资料并用可再现的东西替换它。如果您生活在 npm/yarn 生态系统中，锁定文件确保每次构建时所有依赖项都以相同的方式重新下载（除非[不是这样的](http://blog.npmjs.org/post/141577284765/kik-left-pad-and-npm)）。如果您生活在 Stack 生态系统中，Stackage 发行版确保每次构建时都获取相同的 Hackage 包（除非[不是这样的...](https://www.snoyman.com/blog/2017/04/stackages-no-revisions-field)）。如果您生活在 Nix 生态系统中，这意味着您必须实际替换系统上的*所有*打包系统才能实现可再现性。

所以看起来：

1.  如果您完全依赖所使用的工具园区，事情可能是相当可再现的，但是在更新依赖项时，您仍然需要自行处理。

1.  一旦您走出园区，完全*由您*来确保可再现性。通常的“简便方法”往往不可复制。

**如果我们改变问题的方式呢？** 我们在讨论中假设*可再现性*是我们的终端价值。但事实并非如此：它是我们可以实现其他目标的机制。在连续集成的环境中，我们*真正*关心的是一个能够提供我们*信号*的系统，指示特定变更集是否正确或破坏了事物。一个不可再现的构建只会以一种方式干扰这一目标：如果某个随机依赖项已自行更新并破坏了您的构建。如果发生这种情况，您将*受阻*：在您解决依赖问题之前，您将无法得到清洁的信号。损坏窗户理论要求您放下一切并*修复构建*。

显然，我们*不在乎*我们的依赖关系在开发过程中是否在静默中升级；事实上，我们可能更喜欢这样，因为“自动”比“手动”少摩擦，至少在它工作时是这样的。我们*在乎的*是能够*阻止*如果已知会导致我们出现问题的升级，或者*回滚*如果后来发现它造成了一些问题。

**在线/离线持续集成。** 我们传统上认为持续集成构建是一个单一的流水线，从头到尾运行，为我们提供代码是否工作的信号。但我认为把 CI 流水线看作分成两个阶段更好：

1.  **在线环境配置。** 在这个阶段，你下载所有依赖于那些讨厌的第三方世界的外部软件，设置一个完整的构建环境。完成后，通过某种机制（例如文件系统快照或创建一个 Docker 镜像）*快照*这个环境。

1.  **离线实际构建和测试。** 在这个阶段，使用步骤（1）的快照环境，关闭你的互联网连接并运行实际的构建和测试。

关键在于你不必在每次构建时都运行步骤（1）（出于性能原因，你也不想这样做）。相反，由步骤（1）生成的不可变构建环境的快照系列使你能够回滚或锁定到所有依赖的特定版本，*而不必*使整个宇宙可复现。你可以每周设置一个定时任务来重建你的环境、运行测试，只有在一切顺利通过时才决定推进激活快照。在运行步骤（2）时，你并不一定要真的关闭互联网，但这可能有助于保持诚实。

**离线思考。** 在今天互联的世界中，很容易构建假设你始终连接到互联网的系统。然而，这样做会使你的工具受到现实世界的变化和嘈杂的影响。通过应用一个简单的原则：“我可以离线做什么；我必须在线做什么？”我们可以反向设计一个持续集成的设计，让你得到几乎和可复现性一样好的东西，而不必重新编写整个宇宙。毫无疑问，这是有价值的。
