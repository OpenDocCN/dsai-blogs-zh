# 10 个 NLP 项目提升你的简历

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/10-nlp-projects>

[自然语言处理](https://web.archive.org/web/20221206061600/https://towardsdatascience.com/your-guide-to-natural-language-processing-nlp-48ea2511f6e1) (NLP)是一个非常令人兴奋的领域。在我们的日常生活中，NLP 项目和应用已经随处可见。从对话代理(亚马逊 Alexa)到情绪分析(Hubspot 的客户反馈分析功能)，语言识别和翻译(谷歌翻译)，拼写纠正(语法)，等等。

无论您是对 NLP 感兴趣的开发人员还是数据科学家，为什么不跳进游泳池的深水区，边做边学呢？

借助 PyTorch 和 TensorFlow 等知名框架，您只需启动一个 Python 笔记本，就可以在几分钟内处理最先进的深度学习模型。

在这篇文章中，我将通过推荐 10 个你现在就可以开始着手的伟大项目来帮助你练习 NLP 另外，这些项目中的每一个都将是你简历上的一大亮点！

## 自然语言处理领域的简史

这只是关于自然语言处理的一点背景知识，但是如果你不感兴趣的话，可以直接跳到项目上。

NLP 诞生于 20 世纪中叶。一个重要的历史里程碑是 1954 年的乔治敦实验，大约 60 个俄语句子被翻译成英语。

20 世纪 70 年代见证了许多聊天机器人概念的发展，这些概念基于处理输入信息的复杂的手工规则。在 20 世纪 80 年代后期，奇异值分解(SVD)被应用于向量空间模型，导致了潜在语义分析——一种用于确定语言中单词之间关系的无监督技术。

在过去的十年里(2010 年之后)，神经网络和深度学习一直在动摇 NLP 的世界。这些技术为最困难的自然语言处理任务(如机器翻译)提供了最先进的结果。2013 年，我们拿到了 [word2vec](https://web.archive.org/web/20221206061600/https://arxiv.org/pdf/1301.3781.pdf) 型号及其变种。这些基于神经网络的技术以这样一种方式对单词、句子和文档进行矢量化，即所生成的向量空间中向量之间的距离表示相应实体之间的意义差异。

在 2014 年，[序列到序列](https://web.archive.org/web/20221206061600/https://papers.nips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)模型被开发出来，并在机器翻译和自动摘要等困难任务中实现了显著改进。

后来人们发现长输入序列更难处理，这使我们想到了 [*注意力*](https://web.archive.org/web/20221206061600/https://arxiv.org/pdf/1706.03762.pdf) 技术。通过让模型关注与输出最相关的输入序列部分，这改进了序列间模型的性能。transformer 模型通过为编码器和解码器定义一个自我关注层，进一步改进了这一点。

这篇名为*“注意力是你所需要的一切”*的论文引入了注意力机制，也使得强大的深度学习语言模型得以创建，如:

*   **ULM-Fit–通用语言模型微调:**一种针对任何任务微调任何基于神经网络的语言模型的方法，在文本分类的上下文中演示。这种方法背后的一个关键概念是区别性微调，其中网络的不同层以不同的速率被训练。
*   **BERT–来自变压器的双向编码器表示:**通过保留编码器和丢弃解码器对变压器架构的修改，它依赖于字的屏蔽，然后需要作为训练度量准确预测。
*   **GPT 生成预训练的 Transformer:** 对 Transformer 的编码器-解码器架构的修改，旨在为 NLP 实现可微调的语言模型。它抛弃了编码器，保留了解码器和它们的自我关注子层。

近年来，自然语言处理领域发展最为迅速。在现代 NLP 范式中，迁移学习，我们可以将从一组任务中获得的知识调整/转移到另一组任务中。这是向 NLP 的完全民主化迈出的一大步，允许知识在新的环境中以以前所需资源的一小部分被重新使用。

## 为什么要构建 NLP 项目？

NLP 是人工智能、计算机科学和语言学的交叉领域。它处理与语言和信息相关的任务。理解和表达语言的意义是困难的。所以，如果你想在这个领域工作，你需要大量的实践。下面的项目将帮助你做到这一点。

构建真实世界的 NLP 项目是获得 NLP 技能并将理论知识转化为有价值的实践经验的最佳方式。

之后，当你申请 NLP 相关的工作时，你会比没有实践经验的人有很大的优势。任何人都可以在简历中加上“NLP 熟练程度”,但不是每个人都可以用一个你可以展示给招聘人员的实际项目来证明这一点。

好了，我们已经介绍够了。让我们来看看你现在就可以开始的 10 个 NLP 项目。我们有初级、中级和高级项目——选择你喜欢的项目，成为你一直想成为的 NLP 大师！

## 10 个 NLP 项目创意提升你的简历

我们将从初级项目开始，但是如果你已经在实践中完成了 NLP，你可以进入中级或高级项目。

### 初级 NLP 项目

1.  **营销情感分析**

这种类型的项目可以向你展示作为一名 NLP 专家是什么样的。在这个项目中，您希望了解客户如何评价竞争对手的产品，即他们喜欢什么和不喜欢什么。这是一个很好的商业案例。了解客户喜欢竞争产品的什么是改进你自己产品的好方法，所以这是许多公司正在积极尝试的事情。

为了完成这项任务，你将采用不同的 NLP 方法来更深入地了解客户的反馈和意见。

立即开始项目→ [转到项目存储库](https://web.archive.org/web/20221206061600/https://github.com/koosha-t/Sentiment-Analysis-NLP-for-Marketting)

在这个项目中，您希望创建一个模型，该模型可以预测将评论分类到不同的类别中。社交媒体上的评论往往是辱骂性和侮辱性的。组织通常希望确保对话不会变得太消极。这个项目是一个 Kaggle 挑战，参与者必须提出一个解决方案，使用 NLP 方法将有毒评论分类到几个类别中。

立即开始项目→ [转到项目存储库](https://web.archive.org/web/20221206061600/https://github.com/Prakhar-FF13/Toxic-Comments-Classification)

3.  **语言识别**

对于初学者来说，这是一个学习基本 NLP 概念和方法的好项目。我们可以很容易地看到 Chrome 或其他浏览器是如何检测网页编写语言的。有了机器学习，这个任务就简单多了。

您可以使用脸书的 fastText 模型构建自己的语言检测。

立即开始项目→ [转到项目存储库](https://web.archive.org/web/20221206061600/https://github.com/iamaziz/language-detection-fastText)

4.  **预测堆栈溢出的封闭问题**

如果你是任何类型的程序员，我不需要告诉你什么是堆栈溢出。它是任何程序员最好的朋友。

程序员一直在问很多关于栈溢出的问题，有些很棒，有些是重复的，浪费时间的，或者不完整的。因此，在这个项目中，您希望预测一个新问题是否会被关闭，以及关闭的原因。

数据集有几个特征，包括问题标题文本、问题正文文本、标签、帖子创建日期等等。

立即开始项目→ [转到数据集](https://web.archive.org/web/20221206061600/https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/data)

5.  **创建文本摘要器**

文本摘要是自然语言处理中最有趣的问题之一。作为人类，我们很难手动提取大型文本文档的摘要。

为了解决这个问题，我们使用自动文本摘要。这是一种识别文档中有意义的信息并在保留整体意义的同时对其进行总结的方法。

目的是在保留语义的同时，呈现原始文本的较短版本。

在这个项目中，您可以使用不同的传统和先进的方法来实现自动文本摘要，然后比较每种方法的结果，以得出哪种方法最适合您的语料库。

立即开始项目→ [转到项目存储库](https://web.archive.org/web/20221206061600/https://github.com/edubey/text-summarizer)

6.  **文档相似度(Quora 问题对相似度)**

Quora 是一个问答平台，在这里你可以找到各种各样的信息。网站上的每一条内容都是由用户生成的，人们可以互相学习经验和知识。

在这个项目中，Quora 要求 Kaggle 用户对问题对是否重复进行分类。

这项任务需要找到高质量的问题答案，从而改善 Quora 用户从作者到读者的体验。

立即开始项目→ [转到数据集](https://web.archive.org/web/20221206061600/https://www.kaggle.com/c/quora-question-pairs/overview)

7.  **转述检测任务**

复述检测是检查两个不同文本实体是否具有相同含义的任务。这个项目在机器翻译、自动剽窃检测、信息提取和摘要等领域有着广泛的应用。复述检测的方法分为两大类:基于相似性的方法和分类方法。

立即开始项目→ [转到项目存储库](https://web.archive.org/web/20221206061600/https://github.com/wasiahmad/paraphrase_identification)

### 高级 NLP 项目

8.  **生成研究论文标题**

这是一个非常创新的项目，你想产生科学论文的标题。在这个项目中，一个 GPT-2 在从 arXiv 中提取的 2000 多个文章标题上进行训练。你可以在其他事情上使用这个应用程序，比如产生歌词、对话等文本生成任务。从这个项目中，您还可以了解 web 抓取，因为您需要从研究论文中提取文本，以便将其提供给模型进行训练。

立即开始项目→ [转到项目存储库](https://web.archive.org/web/20221206061600/https://github.com/csinva/gpt2-paper-title-generator)

9.  **翻译和总结新闻**

你可以构建一个 web 应用程序，将新闻从阿拉伯语翻译成英语，并对它们进行总结，使用伟大的 Python 库，如[报纸](https://web.archive.org/web/20221206061600/https://github.com/codelucas/newspaper)、[变形金刚](https://web.archive.org/web/20221206061600/https://github.com/huggingface/transformers)和 [gradio](https://web.archive.org/web/20221206061600/https://gradio.app/) 。

其中:

立即开始项目→ [有用链接](https://web.archive.org/web/20221206061600/https://abidlabs.github.io/Summarize-News/)

10.  用于相似性检查的 RESTful API

这个项目是关于建立一个使用 NLP 技术的相似性检查 API。这个项目最酷的地方不仅在于实现 NLP 工具，还在于你将学会如何通过 docker 上传这个 API，并把它作为一个 web 应用程序来使用。通过这样做，您将学习如何构建一个完整的 NLP 应用程序。

立即开始项目→ [转到项目存储库](https://web.archive.org/web/20221206061600/https://github.com/thecraftman/Deploy-a-NLP-Similarity-API-using-Docker)

## 结论

就是这样！希望你能选择一个你感兴趣的项目。把手弄脏，开始学习 NLP 技能吧！建立真正的项目是在这方面做得更好的唯一最好的方法，也是改善你的简历的最好方法。

目前就这些。感谢阅读！