<!--yml

类别：未分类

日期：2024-07-01 18:17:27

-->

# OfflineIMAP 很糟糕：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/08/offlineimap-sucks/`](http://blog.ezyang.com/2012/08/offlineimap-sucks/)

我将向你分享一个小秘密，只有使用和改进 OfflineIMAP 的人才能合理地了解：OfflineIMAP 很糟糕。当然，你仍然可以使用糟糕的软件（我经常这样做），但了解它的一些缺陷是很有用的，这样你可以决定是否愿意忍受它的糟糕表现。那么为什么 OfflineIMAP 糟糕呢？

> 这并不是一个真正有建设性的帖子。如果我真的想要有建设性，我会去修复所有这些问题。但所有这些都是需要大量精力才能解决的大问题... 不幸的是，我并不够在意这个软件。

### 项目健康状况不佳

最初的作者，[John Goerzen](http://www.complete.org/JohnGoerzen)，已经转向更“绿色”的、更像 Haskell 的牧场，而当前的维护团队在找到足够时间和专业知识来进行适当的软件维护方面有困难。这里是[最近一次维护者招募的通知](http://comments.gmane.org/gmane.mail.imap.offlineimap.general/5754)，因为两位共同维护者在正确跟踪所有提交的补丁方面都没有足够的空闲时间。似乎仍有足够多的人对让 OfflineIMAP 不陷入过时感兴趣，所以这个项目在可预见的未来应该会继续运作，但不应期望在代码库上进行任何重大新功能或密集工作。

### 几乎没有测试

OfflineIMAP 大部分历史上都没有测试。尽管现在有一个小小的测试套件，但它远远不够覆盖这样一个数据关键性程序所需的范围。开发者修复 OfflineIMAP 中的错误时，并不习惯添加新的回归测试。但或许更为恶劣的是，没有基础设施能够测试 OfflineIMAP 与尽可能多的 IMAP 服务器进行兼容性测试。这里才是真正会出现*严重*错误的地方，但这个项目却没有相关的基础设施。

### 对 UID 的过度依赖

OfflineIMAP 将 UID 作为确定两个消息是否对应的唯一依据。这在大多数情况下工作得很好，但也有例外。一旦出现问题，你将面临严重的后果。OfflineIMAP 不支持使用`Message-Id`头部或文件的校验和进行一致性检查，并且针对不支持`UIDPLUS`的服务器的`X-OfflineIMAP`补丁应该被淘汰。但它确实通过积累了大多数特例来使其在所有出现 UID 问题的怪异情况下正常工作。

### 空间复杂度差

OfflineIMAP 的内存使用量与收件箱中消息的数量成正比。对于大邮箱来说，这实际上意味着将数十万个元素加载到集合中并对其进行昂贵的操作（当我运行它时，OfflineIMAP 始终占用我的 CPU）。OfflineIMAP 应该能够在常量空间内运行，但在这个问题空间中没有考虑算法思想。它还有一个极其愚蠢的默认状态文件夹实现（认为每次上传文件时都重复写入 100MB 到磁盘），尽管通过设置 `status_backend = sqlite` 可以相对容易地解决这个问题。为什么它不是默认的？因为它仍然是实验性的。嗯...

### 未优化的关键路径

OfflineIMAP 从来没有真正被设计成为速度快的工具。即使在没有任何更改或只下载少量消息的普通情况下，同步所需的时间也很长。如果一个人的目标是尽快下载新消息，可以做很多调整，包括减少 IMAP 命令的数量（特别是冗余的选择和清除），减少对文件系统的访问次数，异步文件系统访问，不将下载的消息完整加载到内存中等等。一个推论是，OfflineIMAP 似乎并不真正了解它可以丢失哪些数据，以及在继续下一个操作之前必须执行 fsync 的数据："安全"操作只是随意地散布在代码中，没有明确定义的纪律。哦，还有 inotify 怎么样？

### 脑残的 IMAP 库

好吧，这个问题并不完全是 OfflineIMAP 的错，但是 `imaplib2` 实际上根本没有保护你免受 IMAP 协议（以及实际实现中如何实现）的尖锐边缘。你必须自己做所有的事情。这很愚蠢，当你忘记在写新的 IMAP 代码时检查 UIDVALIDITY 时，这将是一场灾难的食谱。此外，它几乎没有对 IMAP RFC 中关于命令响应的知识进行编码。在这里，更多的类型安全性确实会很有用：它将有助于强制人们考虑处理任何给定命令时的所有错误情况和所有可能发生的数据。

### 算法黑暗

OfflineIMAP 在其核心算法中有大量的调试输出和 UI 更新代码交织在一起，总体效果是很难判断所使用的算法的整体形状。如果算法比较微妙，并依赖于整个执行过程的一些全局属性来确保其正确性，这是不好的。有太多样板代码。

### 结论

总结来说，如果你想在一个表现良好、流行的、开源的 IMAP 服务器上使用 OfflineIMAP，其中一个维护者也偶尔使用它，并且你的收件箱中的消息数量相对较少，并且愿意接受 OfflineIMAP 作为一个不可修改的黑匣子，以一种完全神秘的方式同步，而且永远不想对 OfflineIMAP 进行修改，那么没有比这更好的选择了。对于其他所有人，嗯，祝你好运！也许对你来说会有所帮助！（对我来说大部分情况下都是如此。）
