<!--yml

category: 未分类

date: 2024-07-01 18:18:18

-->

# 我讨厌补丁：开源开发者的自白 : ezyang's 博客

> 来源：[`blog.ezyang.com/2010/05/i-hate-patches-confessions-of-an-open-source-developer/`](http://blog.ezyang.com/2010/05/i-hate-patches-confessions-of-an-open-source-developer/)

## 我讨厌补丁：

开源开发者的自白

众所周知，如果你*真的*希望将一个变更提交到开源项目中，你需要在 bug 报告中附上一个补丁。当然，你可能会抱怨*普通*用户没有任何编程经验，希望他们学习某些复杂的系统然后弄清楚如何做出他们寻求的变更是*不合理*的，但你只是不属于那些修复自己该死的软件并回馈他们使用的项目的黑客爱好者秘密社会。

我讨厌补丁。我感到*道义上的义务*去审查和修补它们，而这通常不仅仅是几个周期。

并非所有补丁都是相等的。我将它们分为以下层次：

+   *表面上不足.* 开发者对提交补丁一无所知；也许他们甚至还没有发现版本控制。他们往往会发送整个修改后的文件以及他们的更改。那些使用`diff -u`的人也不会去查看补丁的输出；他们提交的补丁会将空格与制表符互换，随意更改空白和过度的表面更改。许多开发者直接拒绝这些补丁。

+   *语义上不足.* 对于一些开发者来说，编写补丁的行为是拿一个源文件，尝试一个模糊可行的变更，看看变更是否达到了预期效果，如果没有，再试其他方法。在极端情况下，补丁是毫无意义的，根本不正确。更频繁地，提交的补丁未考虑到应用程序中的常见边界情况、适当的错误处理或与系统其他部分的交互。许多开发者会友好地回复这样的补丁，并要求用另一种方式进行修补。

+   *工程上不足.* 这个补丁写得很好，看起来不错，做了正确的事情。但是……他们没有添加测试来测试新的更改，也没有修复由功能差异引起的旧单元测试，并且没有在适当的位置为修复添加文档。许多开发者会专注于为补丁添加工程附加项。一些开发者没有这样的测试（咳咳，Linux 内核咳咳）。更少见的是，一些项目可以让提交者添加测试；通常这只发生在面向相当有识字编程的终端用户社区的项目中。

Git 邮件列表可以并且确实期望社区提交优秀的补丁；这是一个版本控制系统，这就是重点！一个主要由从未编写过单元测试或提交过上游审查统一差异的开发人员使用的 PHP 写的库，灵活性要小得多。对于 HTML Purifier 收到的大多数补丁，都未能解决表面问题。更糟糕的是，开发人员根本没有时间互动地改进补丁的最终版本：如果我回复补丁审查，他们永远无法使自己的补丁达到可以接受提交的水平，而不是一点点拔牙。但我觉得我的软件有问题，所以当我收到补丁时，我会去清理它，把它整合进来，重写其中一半，添加测试，然后发布这些更改。

所以，最终，即使维护者没有节省任何时间，软件也通过提交补丁得到了改进。所以是的，我讨厌补丁。也许我应该停止*脾气暴躁*，回到*改进*我的开源项目。