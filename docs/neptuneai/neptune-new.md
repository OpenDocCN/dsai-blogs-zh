# 海王星.新

> 原文：<https://web.archive.org/web/https://neptune.ai/blog/neptune-new>

首先，我们很抱歉！在过去的几个月里，我们在产品更新方面非常安静。您已经看到了 web 界面中的一个更新或 Python 客户端库中的一个 bug 修复，但仅此而已。

发生了什么事？

随着时间的推移，我们从您那里获得了大量的反馈。您询问了如何在 spot 实例和管道中使用 Neptune，如何以一种更加层次化的方式组织事物，以及许多其他问题，我们希望做到这一点。但是将这些改进一点一点地添加到当前产品中变得越来越困难(也越来越慢)。不要误解我，我们喜欢迭代，但有时你需要深呼吸，后退一步，重建基础。

简而言之，这就是当时的情况。然而，相信我——等待是值得的🙂

今天，我们很高兴地宣布，一个全新版本的 Neptune 已经为您准备好了，它具有许多新特性和改进的 Python API！

## 海王星有什么新的？

### 更好的组织，更大的灵活性，更多的定制

通常，当训练一个只有几个参数的简单模型时，你可以从头开始背诵它们。它们都可以显示在一个屏幕上。

然而，一旦添加了这个太多的参数，问题就出现了。当您除了按字母顺序对复杂的参数配置进行排序之外，还不能轻松地管理它们时，这就成了一种真正的痛苦。

你们对此都有不同的解决方案:

*   明确上传几个参数，其余的上传为 YAML 配置文件，
*   前缀的巧妙运用，
*   还有很多很多其他人。

所有这些都是海王星本身缺乏组织性和灵活性的权宜之计。这改变了！

有了新的 Neptune API，您可以**将所有的元数据分层组织到组织整齐的名称空间**(文件夹)中。对于所有元数据，我指的是所有元数据，不仅是参数，还包括日志、验证指标、二进制工件等。

你如何用代码做到这一点？新的 Neptune API 依赖于一个类似字典的接口。您可以用一种统一的方式跟踪所有元数据，而不是在那里传递参数，在那里传递属性，以这种方式传递度量，以那种方式传递工件。

```py
import neptune.new as neptune
run = neptune.init()

run['data/version/train'] = md5(train_set.data).hexdigest()
run['data/version/test'] = md5(test_set.data).hexdigest()
run["data/sample"].upload_files("data/sample_images")

PARAMS = {'lr': 0.005, 'momentum': 0.9}
run["model/params"] = PARAMS

run["model/visualization"].upload(File.as_image(model_vis))

for epoch in epochs:
    for batch in batches:
        [...]
        run["batch/accuracy"].log(accuracy)
        run["batch/loss"].log(accuracy)

run["test/total_accuracy"] = calculate_test_accuracy()

```

哦，这也意味着您不再需要担心预先设置参数—**您可以在方便的时间和地点更新它们**！

### 支持更多工作流:离线模式、恢复运行和 ML 管道

有时候你运行你的脚本，分析结果，然后继续前进。通常情况下，事情没那么简单:

*   也许你需要多训练几个纪元？
*   也许在第一次分析之后，您需要计算更多的验证指标？
*   也许你的 spot 实例死了，你需要恢复训练？

不要害怕。海王星已经为此做好了准备。更多🙂

使用新的 Python API，**您可以恢复对任何现有运行**的跟踪。您可以获取所有记录的元数据，更新它并记录新的。更重要的是，它现在是线程安全的，您可以将不同脚本的输出连接成一个。有了这些，你就可以**在并行和分布式计算中，使用带有 spot 实例的 Neptune，以及多步 ML 流水线。**

最后，如果你在无法永久访问互联网的地方训练你的模型，你会很高兴听到**我们增加了离线模式**。被跟踪的元数据将被保存在本地，**您可以在方便的时候批量上传**。如果您的互联网连接有点不稳定或发生意外中断——Neptune 现在会在多次重试后自动切换到离线模式，以确保您的数据始终安全。

### 更好的 web 界面:文件夹结构、更好的比较和拆分视图

您将注意到的第一件事是，新的 UI 利用了被跟踪元数据的组织的灵活性。以前，您必须在不同的部分之间跳转，才能完全理解生成模型的输入是什么。现在**你可以在一个对你最方便的层次结构中组织元数据**。

您将注意到的第二件事是，您可以看到所有运行及其元数据的表发生了变化。以前，比较不同运行的选项不仅有点难找到，而且变化意味着来回切换。最后，它只允许同时比较 10 次运行。现在，**你可以:**

*   **比较方式更多次运行(比如 100 次或更多)**，
*   添加和删除您想要实时显示的内容，
*   **在表格视图和比较之间切换或同时保持两者可见。**

酷吧？嘶，还有一件事。当您浏览特定跑步的详细信息时，您可能会注意到“添加新仪表板”按钮。现在还有点早，我们将添加更多类型的小部件，但你已经可以**构建自己的仪表板，以最适合你的方式可视化元数据**。请检查它，并让我们知道你的想法！

### 跑，跑，跑(再见实验)

从这篇文章开始，我就一直用**‘跑’这个词代替‘实验’**，这不是巧合。随着我们行业的发展，越来越多的 Neptune 用户关心 ML 模型的可操作性，我们需要与它一起发展。

当然，实验阶段是重要的(并且贴近我们的内心)，并且[实验跟踪](/web/20221206030431/https://neptune.ai/experiment-tracking)将仍然是 Neptune 的主要用例之一，但是越来越多的人将它用于其他事情，例如:

*   **模型注册表**
*   **监控模型再训练管道**
*   **监控生产中运行的模型**

我们希望更好地服务于您现有的使用案例。我们希望与您用来描述您的作品的命名更加一致。

称它为“实验”已经没有意义了，所以我们把它改成了“运行”。您将看到从实验到在 web 界面和新的 Python API 中运行的变化。

## 如何开始使用 neptune.new？

### 等等，我需要修改我的代码吗？

简而言之——不，**你现在不需要做任何事情，**你的跑步记录会被海王星跟踪，不会有任何问题。

新的 Python API 和改进的用户界面需要改变数据结构。在接下来的几周里，**我们将把现有的项目迁移到新的结构，**但是你已经可以尝试了，因为**所有的新项目都是使用新的结构创建的。**

当前的 Python API 将在迁移后继续工作，因此您不需要更改任何一行代码。在后台，我们悄悄地做我们的魔术，并确保事情为你工作。然而，新的 Python API 只与新的数据结构兼容，因此它只在项目迁移后才可用于项目。同样，改进的 web 界面也需要新的数据结构。您已经可以在一个新项目中试用它，并且一旦您的现有项目被迁移，它将可用于它们。

在未来的某个时候，我们计划在客户端库 1.0 版中，将新的 Python API 作为默认 API。然而，**我们将在很长一段时间内支持当前的 Python API**，这样你就可以在方便的时候进行切换。这是值得的转变，虽然，这是相当可怕的🙂我们准备了一本方便的[移民指南](https://web.archive.org/web/20221206030431/https://docs-beta.neptune.ai/migration-guide)来帮助你完成这个过程。

### 我想用 neptune.new，我现在该怎么做？

这非常简单:

#### 第一步:

**创建一个新项目**–您会注意到它有一个标记，表明它是用新结构创建的

#### 第二步:

**将 Neptune 客户端**库至少更新到 0.9 版本。只需在您的环境中运行:

```py
pip install --upgrade neptune-client
```

#### 第三步:

**查看** [**新文档**](https://web.archive.org/web/20221206030431/https://docs.neptune.ai/) 。如果你想尝试一下，重新开始——[快速入门部分](https://web.archive.org/web/20221206030431/https://docs.neptune.ai/getting-started/examples)是你最好的朋友。如果您想更新您现有的代码，我们准备了[迁移指南](https://web.archive.org/web/20221206030431/https://docs.neptune.ai/migration-guide)来帮助您。

#### 第四步:

**享受跟踪元数据的新方式！**

我们将在接下来的几周内将现有项目迁移到新的结构中，一旦您的项目被迁移，您也将能够使用新的 Python API。

### 新来海王星？

首先，你好，欢迎，感谢你读到这里！

如果你想知道到底是怎么回事，你可以:

…或者您可以:

**1。创建免费账户**

**2。安装 Neptune 客户端库**

```py
pip install neptune-client
```

**3。将日志添加到您的脚本中**

```py
import neptune.new as neptune

run = neptune.init(project="your_workspace/your_project")

run["JIRA"] = "NPT-952"
run["algorithm"] = "ConvNet"

params = {
    "batch_size": 64,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "optimizer": "Adam"
}
run["parameters"] = params

for epoch in range(100):
    run["train/accuracy"].log(epoch * 0.6)
    run["train/loss"].log(epoch * 0.4)

run["f1_score"] = 0.67

```

**4。在海王星看到它**

## 海王星的下一步是什么？

就这样吗？

类似 Dict 的 API，文件夹结构，离线模式，更好的比较，Neptune 团队去海滩喝 pia coladas(当然是通过 Zoom)？

不，我们很快会有更多的东西，这次会更快。

正如我之前提到的，越来越多的 Neptune 用户正在将他们的模型推向生产(恭喜你们！).我们不仅要支持它，而且要让事情变得更简单，就像我们对实验跟踪所做的那样。

在接下来的几个月里:

*   我们将通过对工件版本控制(数据和模型)的支持，使人们使用 Neptune 作为模型注册表的体验变得更好。
*   我们将添加对更多元数据类型的支持，这样您可以在 Neptune 中轻松地记录、显示和比较它；
*   我们将**添加与来自 MLOps 生态系统**的更多库的集成。

但总的来说，总的来说，我们将努力使 MLOps 工作流中的元数据的存储、显示、组织和查询更加容易。**我们将继续为 MLOps** **构建一个** [**元数据存储。**](/web/20221206030431/https://neptune.ai/product)