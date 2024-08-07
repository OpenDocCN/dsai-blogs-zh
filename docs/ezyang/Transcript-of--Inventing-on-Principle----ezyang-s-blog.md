<!--yml

category: 未分类

date: 2024-07-01 18:17:37

-->

# “Inventing on Principle” 的文本：ezyang 的博客

> 来源：[`blog.ezyang.com/2012/02/transcript-of-inventing-on-principleb/`](http://blog.ezyang.com/2012/02/transcript-of-inventing-on-principleb/)

[这里有一份 Github 的完整记录](https://github.com/ezyang/cusec2012-victor/blob/master/transcript.md)，Bret Victor 的["Inventing on Principle"](https://vimeo.com/36579366)。这是由我，An Yu 和 Tal Benisty 转录的。

下面是一份转录副本，我将努力与 Github 的副本保持一致。原始内容根据[CC-BY](http://creativecommons.org/licenses/by/3.0/)许可。

* * *

[[0:07]] 所以，与之前的会议不同，我没有任何奖品要颁发。我只是要告诉你如何过你的生活。

[[0:14]] 这个演讲实际上是关于一种生活方式，大多数人不会谈论的。当你接近你的职业生涯时，你会听到很多关于追随你的激情，或者做你喜欢的事情的建议。我将谈论一些有点不同的事情。我将谈论追随一个原则 —— 找到你工作的一个指导原则，你认为是重要和必要的，并用它来指导你的所作所为。

[[0:46]] 这次演讲分为三个部分。我首先会讲述指导我大部分工作的原则，并试图让你体验到它的成果。我还将讨论一些其他以这种方式生活的人；他们的原则是什么，他们相信什么。但这些只是例子，帮助你思考你自己的信念以及你想如何过你的生活。

[[1:10]] 所以让我开始：对我来说，思想非常重要。我认为将思想带入这个世界是人们做的最重要的事情之一。而且我认为伟大的思想，无论是伟大的艺术、故事、发明、科学理论，这些东西都拥有它们自己的生命，赋予我们作为人生命的意义。所以，我经常考虑人们如何创造思想以及思想如何成长。特别是，什么样的工具能够为思想的成长创造一个健康的环境。多年来，我花了很多时间制作创意工具，使用创意工具，并且深思熟虑。我得出了一个结论：创作者需要与他们正在创造的东西有即时的连接。这是我的原则。创作者需要与他们创造的东西有即时的连接。我的意思是，当你在创造某物时，如果你做了一个改变或者做了一个决定，你需要立即看到那个效果。不能有延迟，也不能有任何隐藏的东西。创作者必须能够看到他们在做什么。接下来，我将展示一系列案例，我注意到这个原则被违反了，并且我会告诉你我是如何解决这个问题的，然后我会谈论我进行这项工作的更大背景。

[[2:32]] 那么，首先，让我们考虑一下编码。编码是怎么工作的：你在文本编辑器中输入一堆代码，想象每一行代码会做什么。然后你编译和运行，然后出现了一些东西。所以在这种情况下，这只是 JavaScript，绘制到 Canvas，绘制了这个小场景，有一棵树。但是如果场景有什么问题，或者如果我去做修改，或者如果我有更多的想法，我必须回到代码，编辑代码，编译和运行，看看它的样子。如果有什么问题，我就回到代码中去。我的大部分时间都花在工作中，盲目地在文本编辑器中工作，没有与我实际想要制作的东西有直接的连接。

[[3:20]] 所以我觉得这违背了我所坚持的原则，即创作者需要与他们正在创作的内容有直接的联系，因此我试图设计出一个编码环境，更符合我这个原则。所以我这里有这样一幅图在一边，代码在另一边，这部分画天空，这部分画山，这部分画树，当我对代码进行任何更改时，图像立即改变。因此代码和图像始终保持同步；没有编译和运行。我只需改变代码中的东西，就能看到图像中的变化。既然我们在代码和图像之间有了这种即时连接，我们可以开始考虑除了键入之外的其他改变方式。例如，这里的这个数字是树枝的长度。如果我想控制这个数字，我只需用鼠标指向它，按住控制键，就可以向上或向下调整。因此，我可以看到大树枝和小树枝的样子，我可以在艺术上找到感觉。这在代码的任何部分都很有效，我只需指向它，向上或向下调整。这些数字中的一些，我知道它们是做什么用的，但看到它们做到这一点还是有些惊讶的。还有一些完全让我惊讶。[笑声] 其他一些则完全出乎意料。[更多笑声]

[[4:48]] 所以在这里，我有一个循环，我数到十六，我在每根树枝上放了十六朵小粉色花朵。我可以减少花朵数量或增加花朵数量。但是，看看我在这里做什么：我只是在大约二十左右左右上下移动数字：它有这种非常有趣的闪烁效果；它看起来好像风在树上吹过。我第一次看到这个效果时，我立即开始考虑如何将这种效果用于动画。如果每次改变都要编译和运行，我怎么会发现这个呢？艺术的很多，创作的很多都是发现，如果你看不到你在做什么，你就什么都发现不了。

\[\[5:33\]\] 所以我已经展示了调整代码，现在让我们添加一些代码。我想在天空上放一个太阳，所以我会到 drawSky 函数的末尾，我想要填充一个圆，所以我开始输入 context.fillCircle，一开始我就得到了一个自动完成列表，显示了不同的填充方法。所以这些是我可以在那里输入的内容：fillCircle、fillRect、fillText。当我在这个自动完成列表中上下移动时，我立即看到每个方法在做什么。所以，我不必从方法名想象它会做什么。我也不必查看文档，我只是看到它，立即明白。

\[\[6:12\]\] 所以我想要一个圆，我要调整 x 坐标和 y 坐标，稍微改变半径。看起来差不多了。可能应该是黄色的，所以我要设置填充样式，context.fillStyle，和之前一样自动完成，选择 fillStyle，默认给了我白色，我可以像改变任何数字一样改变颜色代码，按住控制键，我得到一个颜色调色板。所以我可以选择一个漂亮的黄色来画我的太阳。虽然，白色也挺有趣的，我觉得。我有点没想到会是这样。但是，用白色，现在看起来像是月亮了，对吧？嘿，看，现在是夜晚了！[笑声] 所以这种即时连接使得想法可以表达并发展出以前不可能的方式。

[[7:17]] 但是这里仍然存在一个问题，我认为，就是我有这个图片，还有这里的代码，我必须在脑海中维护两者之间的映射关系。所以我有所有这些行的代码，但是光看这一行，我不立刻知道它做了什么。所以我可以这样做。我按住选项键，我的光标变成了一个放大镜，现在当我滚动每一行代码时，它在图片中高亮显示了这一行所绘制的内容。所以，如果我想知道这个函数中发生了什么，我只需滚动函数并查看高亮部分。所以这里有两个调用 drawMountain 的地方；我不知道哪个是哪个；好吧，这是那座山，那是那座山。这也必须反过来工作；如果我看到图片的一部分，我必须知道是哪段代码负责绘制它。所以我做同样的事情；我按住选项键，现在当我移动图片的每个像素时，你会在右侧看到它跳到了绘制该像素的代码行。所以这绘制了天空，那绘制了树，那绘制了花朵。因此，维护这种映射非常重要，但也非常有用，用于导航。所以你知道，我想把太阳弄大一点；我跳到那里，然后把它弄大一点。或者我想把树抬高一点；我跳到那里，把树抬高一点；我想把山抬高一点，所以我跳到那里，把山抬高一点；我可以在想到它们的时候就做出这些改变，这对创造过程非常重要。能够在想到一个想法时立即尝试它是如此重要。如果在这种反馈循环中有任何延迟，在思考某事和看到它、并在此基础上构建之间，那么这些想法的世界将永远不存在。这些是我们无法思考的想法。

[[9:36]] 对我来说，思想非常重要。而关于思想的一点是，思想起步都很小。思想一开始微小、脆弱。为了发展和成熟，思想需要一个创作者可以培养它们的环境。某种程度上，需要照顾它们、喂养它们，并塑造它们的成长。对我来说，这就是即时连接原则的意义所在。因为思想对我如此珍贵，所以当我看到这一原则被违反时，当我看到思想因为创作者无法看清自己的所作所为而夭折或者停滞不前时，我觉得这是不对的。这不是违反某些界面指导方针或者违背某些最佳实践的意义上的不对，而是更深层次的错误。稍后我会回来讨论这个问题，但我想先展示另一个遵循这一原则的例子。

\[\[10:26\]\] 所以在这段代码中，没有状态，没有持久状态，没有时间，没有互动性。我在思考如何以符合我拥有的原则的方式处理编码中的这些方面：创作者需要即时连接。所以这里有一个小平台游戏。这是我的小家伙，他可以四处跑动，可以跳跃，可以死亡[笑声]。他的代码在这里。所以这段代码让他四处跑动，这段让他跳跃，这段让他与物体碰撞……而在这里，我有一些为这只小乌龟写的代码。现在乌龟现在没做什么，因为我还没完成他的代码，所以，我现在就去做这件事。每当他的 x 位置每次间隔时间增加他的方向乘以时间间隔的六十分之一的秒再乘以一些速度，哪怕是一点？可以快，可以慢，如果是负的，他就会向后走。[笑声]而这些都是我可以用来做其他敌人的想法，但我认为乌龟应该是慢的，所以让我们为我们的乌龟设定这个速度。而在这里，我有一些代码说，当我的小家伙与乌龟碰撞时，他会获得一些 Y 速度，所以他会弹到空中，而乌龟则被踩扁了。所以看起来是这样的。而乌龟会在一段时间后起身。

\[\[12:01\]\] 问题是，我不希望玩家能够在这里爬上去。我希望玩家能弹跳乌龟，并穿过这里下面的小通道。然后他将不得不绕过去解决谜题之类的事情，然后再回来拿到星星。所以，现在乌龟的弹性太大了。当然，我可以简单地在代码中调整它，现在我可以尝试，但现在它的弹性不够。所以虽然我可以在运行时调整代码而无需停止和重新编译并找到我的位置，这是很好的，但我不能立即看到我需要看到的东西，也就是他是否能够跳过去。

\[\[12:43\]\] 所以这是我要做的。我要弹跳乌龟，并暂停游戏。所以我暂停游戏，现在这里有一个滑块，让我可以倒回时间。现在，我可以倒回到在我跳跃之前，并修改代码，让它不那么有弹性，现在，当我向前移动时，它会模拟使用相同的输入控制，相同的键盘命令录制如前，但使用新代码。[掌声]

[[13:20]] 这还不够好。 [笑声] 我需要能立即看到变化。我需要立即看到我的反弹是否正确。不要再用这些东西了。如果你有一个时间过程，并且想要立即看到变化，你必须把时间映射到空间上。所以这是我要做的事情。我要弹起我的海龟，暂停游戏，现在按下这里的按钮，显示我的家伙的轨迹。所以现在我可以看到他去过的地方。当我倒带时，他面前的这条轨迹就是他将要去的地方。这是他的未来。当我改变代码时，我改变了他的未来。 [喘息声] 所以我可以找到我需要的确切值，这样当我播放时，他就可以顺利进入那里。 [鼓掌声]

[[14:26]] 因此，创作者需要能够看到他们正在做的事情。如果你正在设计一个嵌入时间的东西，你需要能够控制时间。你需要能够跨越时间看清楚，否则你就是在盲目设计。

[[14:40]] 当我玩这个的时候，我注意到玩重力很有趣。所以我可以稍微把重力搞负一点，他就开始浮起来。 [笑声] 我可以玩弄一下，试着让他停在那里。你可能可以围绕这个机制做一个完整的游戏，重力操控。事实上，我敢打赌，我可以调整这段代码的任何部分，想出一个游戏的点子。即使我只是注释掉代码中的第一个语句，现在我的家伙就不能向左移动了 - 他只能向右移动。这听起来有点傻，但是 Terry Cavanagh 实际上围绕这个概念制作了一个美丽的游戏，叫做《别回头》。Terry Cavanagh，他还做了另一个非常出色的游戏，你可能见过，叫做《VVVVVV》，用六个 v 拼写。而且，这款游戏的工作方式是你不能跳跃。相反，你只能翻转身体，向上而不是向下掉落。所以它有点像这样。你可以走在天花板上，或者在地面上走。所以你有这些看起来有点像这样的关卡，然后你在这样的地形上行走...你必须学会如何穿越这样的地形。所以如果你像那样有一个，你就不能跳过它。你必须翻转过来，然后翻转过来；他利用这个概念获得了大量的游戏体验。

\[\[16:07\]] 所以再次，能够在你想到它们的时候尝试想法。[暂停] 这个例子，以及上一个关于树的例子，这两个都是非常视觉化的程序；我们可以通过看图片如何改变来看到我们的变化。所以我在思考，我们如何能够写更符合这个原则的更抽象的编码。我们如何写一个通用算法，以便我们能够看到我们在做什么。所以举个例子，让我们看看二分查找。关于二分查找的超快速刷新：你有一个有序值数组，还有一个关键字，这是你试图在数组中定位的值。你要跟踪两个变量，它们是你认为该值可能存在的下界和上界；现在它可以是任何地方。然后你查看这个范围的中间 - 如果找到的值太小，那么关键字必须在后面。查看范围的中间，如果找到的值太大，关键字必须在前面。你继续将你的范围细分，直到你锁定你正在寻找的值。在代码中，二分查找看起来像这样。从我的角度来看，你什么也看不到。你什么也看不到。我看到了 'array' 这个词，但我实际上看不到一个数组。因此，为了编写这样的代码，你必须在脑海中想象一个数组，并且基本上你必须玩电脑。你必须在脑海中模拟每一行代码在计算机上的操作会做什么。在很大程度上，我们认为是熟练的软件工程师的人，其实就是那些非常擅长玩电脑的人。但是如果我们在计算机上写我们的代码...为什么我们要在脑海中模拟计算机会做什么？为什么计算机不只是做它...并且展示给我们呢？

\[\[18:06\]] 所以。让我们写二分查找。函数"binary search"接受一个关键字和一个数组。然后在这边，它说："好的，它接受一个关键字和一个数组，比如什么？给我一个例子；我需要一些东西来处理。" 所以，例如，我的数组可能是 'a', 'b', 'c', 'd', 'e', 'f'。比如说我们正在寻找 'd'。现在让我们开始编码。下界初始为零。在这边它说 'low equals zero'，没什么了不起的。上界初始为数组的末尾，所以 high equals 数组长度减一。在这边，它说 'high equals five'。所以我有了我的代码中的抽象公式。在这边，它给了我对应于这些示例参数的具体值。所以我不必在脑海中维护这个图像；它只是展示给我看。

[[19:09]] 现在我需要数组中间的索引，所以我将取这两者的平均值。Mid 等于 low 加 high 除以二，但...显然这不对。2.5 不是一个有效的数组索引。我想我需要四舍五入一下。所以我会加上 floor 函数并将其向下取整到 2。我刚刚输入时就捕捉到了这个 bug，而不是写整个函数和二十个单元测试。所以现在我从数组中取得了值...然后我需要分割我的范围，所以这里有一个 if 语句，我将它粘贴在这里。在这种情况下，我找到的值小于关键字，因此它采取了 if 语句的第一个分支。这调整了下限。当然，如果关键字更小，那么它将采取 if 语句的第二个分支并调整上限。或者，如果关键字是 'c'，那么我们可能第一次就找到它，并返回索引。

[[20:14]] 这是这个算法的第一次迭代。现在我们需要做的是循环。我们已经细分了数组，我们需要继续细分直到我们找到我们要找的东西。所以，我们需要循环；我将简单循环。当 1 时，执行所有这些操作。现在我们有三列对应于这个循环的三次迭代。所以这第一列就是你之前看到的。Low 和 high 跨越整个数组，我们找到了一个 'c'，它太低了，所以我们调整了下限，并循环到这里。第二次迭代，边界更紧；我们找到了一个 'e'。调整上限。第三次迭代，循环到这里；low 和 high 是一样的。我们已经缩小到一个单一的候选者 - 确实是我们正在寻找的关键字，并返回这个索引。所以这里没有隐藏的东西；你可以在每个点上清楚地看到算法正在做什么。我可以一直尝试不同的关键字，所以我可以看到算法对这些不同的输入参数的行为。

\[\[21:20\]\] 通过分析这些数据，我可以对这个算法的运作方式有所直观理解。所以我在这里尝试不同的键，比如我试试找一个'g'。结果看起来有些不同。实际上并没有返回。原因是，我在查找一个数组中不存在的键。唯一能跳出这个循环的方法，就是找到这个键。所以它在这里陷入了无限循环。因此我们可以看看发生了什么问题，算法出了什么差错。前几次迭代看起来没问题，但这次迭代看起来奇怪，因为低位大于高位。我们的范围完全崩溃了。所以如果我们到了这一步，那么我们知道这个键找不到。我看到了这个错误的条件，然后我说：“哦，这不对；低位必须小于或等于高位。”好的，我只需把这个作为我的 while 语句的条件。低位小于等于高位，然后就能跳出循环，我会返回一些信号来表示找不到。所以这里有三次循环迭代，找不到，我们返回一个未找到的值。这就像是在不盲目的情况下编写算法可能会是什么样子。[掌声]

\[\[22:45\]\] 所以我有这样一个原则，即创作者需要能够看到他们在做什么。他们需要与他们正在创造的内容有直接的联系。我试图通过三个编码示例来展示这个原则，但这只是因为这是一个软件工程的会议，我以为我应该谈论编程。但对我来说，这个原则与特定的编程无关。它与任何类型的创作都有关。所以我想展示给你们几个更多的演示，只是为了展示我在这里的想法的广度。

\[\[23:17\]\] 所以，首先，让我们看看工程的另一个分支。这里我有一张我画的电子电路图。我还没画完，所以让我完成一下。然后我们加 2。现在我们有一个工作中的电路。我是说我假设这是一个工作中的电路。我实际上没有看到任何东西在这里工作。所以这与编写代码完全相同，我们在一个静态的表示中工作。但是我们实际上关心的是数据。变量的值，所以我们在这里看不到那些。现在在一个电路中，变量是这些不同导线上的电压。所以每根导线都有一个随时间变化的电压，我们必须能够看到这一点。如果我在实验台上构建这个电路，物理上构建它，我至少可以拿一个示波器，看看这些不同导线上发生了什么，这里，或者这里。所以至少，我应该能够做到这一点。所以我这里有这根导线上电压随时间变化的图。你可以看到它是高的，低的，高的和低的，所以这显然是振荡的。如果我物理构建这个，我也能看到电路在做什么。在这种情况下，我有这两个 LED 灯在这里上面。这些是 LED 灯，小灯，据推测它们有原因在那里。我可以点击播放，看它实时模拟出来的情况。所以现在你可以看到电路在做什么。

\[\[24:50\]\] 为了设计这样一个电路，你必须理解每根导线上的电压。你必须理解整个电路中所有电压的变化。就像编码一样，要么环境向你展示了这一点，要么你在脑海中模拟它。而我有更重要的事情要用我的头脑来做，而不是模拟电子在做什么。所以我要做的是，我会把它们分开一点。所以同样的电路，稍微分开一点，我要添加每个节点的电压。所以现在你可以看到整个电路中的每个电压。而且我甚至可以点击播放，看它们实时模拟出来。

\[\[25:30\]\] 虽然，我更喜欢的是，只需将鼠标移动到上面，我可以查看对我来说有趣的区域，并查看数值。我可以比较任意两个节点。因此，如果你看看这边的节点，而我在此节点上方悬停，你会看到我悬停的那个节点的阴影叠加在上面。实际上，我悬停的节点的阴影叠加在所有节点上。所以，我只需将鼠标悬停在其中一个节点上，就能比较任意两个节点。

\[\[26:00\]\] 而且，我可以立即看到我的更改结果。所以，这里有一个 70k 电阻。我想改变它的值，我只需点击并拖动它，现在我立即看到波形立即变化。而且你会注意到，当我点击并拖动时，它会留下我开始拖动前波形的阴影，这样我就可以比较。我可以立即看到我的更改结果。

\[\[26:26\]\] 信息设计的两大黄金法则：展示数据，展示比较。这就是我在这里做的一切。但即使这样还不够好。我们在这里看到的是电压，但在电子学中实际上有两种数据类型。有电压和电流。我们看不到的是电流，流过每个组件。为了设计电路，你需要理解电压和电流的两者。你需要理解它们之间的相互作用。这就是模拟设计的内容。

\[\[26:51\]\] 所以我要把它们稍微分开一点。现在我要用随时间变化的电流图来替换每个组件。所以每个蓝色的方框代表一个组件。你可以看到每个组件是哪一个，因为它在角落里有一个小徽章，一个小图标，但现在你可以看到电路中的一切。你可以看到电流如何变化，你可以看到电压和电流如何变化。没有什么是隐藏的，没有什么需要在你的脑海中模拟。

\[\[27:22\]\] 所以这里我们有一种不同的电路表示方式。总的来说，你可以用这些块绘制任何电路，而不是用小波浪形符号制成，它是由数据制成的。我认为重要的是要问：为什么我们一开始就有这些波浪形符号？它们为什么存在？它们存在是因为用铅笔在纸上很容易画出来。但这不是纸。所以当你有了一种新的媒介，你必须重新思考这些事情。你必须考虑如何利用这种新媒介让我们对我们正在制作的东西有更直接的联系。这种新媒介如何让我们以一种方式工作，我们可以看到我们正在做什么。

\[\[28:00\]\] 对编程而言情况基本相同。我们当前对计算机程序的理解——一系列文本定义，你交给编译器——这是直接源自上世纪 50 年代 Fortran 和 ALGOL 的。那些语言是为穿孔卡设计的。所以你会在一叠卡片上打出程序，交给计算机操作员（就是底部图片中的那位），然后过一段时间再回来。所以当时根本没有交互性。这种假设已经深深融入我们当前对编程的概念中。

\[\[28:34\]\] C 语言是为电传打字机设计的。上面的是 Ken Thompson 和 Dennis Ritchie。Ritchie 创造了 C 语言。这张图片中没有显示视频显示器。Ritchie 基本上是在一台能够回显的高级打字机上打字。每当你使用控制台或终端窗口时，你正在模拟电传打字机。即使今天，人们仍然认为 REPL 或交互式顶层是交互式编程的最佳体验。因为在电传打字机上这是你能做的最好的事情。

[[29:06]] 我还有一个演示想要展示，因为我想强调这个原则，即即时连接，不仅仅是工程，而是任何类型的创作。所以我要跳到一个完全不同的领域，让我们想想动画。

[[29:22]] 所以我这里有一幅画，上面画了一棵树和一片叶子，我想用一个小视频来表现叶子慢慢飘落到树上的过程。在传统动画软件比如 Flash 中，通常的做法是使用关键帧。你基本上要指定叶子在不同时间点的位置，然后点击播放，看看效果如何。所以，我要说：在第 20 帧，我要创建一个关键帧，叶子应该在那里。然后在第 40 帧，再创建一个关键帧，叶子应该在那里，但我完全是在瞎猜。我看不到动作，感受不到时间，只是随意地把事物放在时间和空间中。

[[30:12]] 所以我在不同的时间点有了这片叶子，然后我要添加一个补间动画，告诉 Flash 如何连接这些点。然后我会点击播放，看看效果。看起来很荒谬，就像台球在来回弹跳。

[[30:32]] 而问题是，我其实知道我想要什么，对吧？就是一片叶子从树上飘落下来。我甚至可以用手来表演：叶子从树上飘落。但是 Flash 不知道如何听取我的手势。也许有一种新的媒介，能够理解我的手势。

[[30:57]] 所以我要向大家展示的是我制作的一个小应用程序，用于进行动画制作。我们目前没有准备好从 iPad 上进行实时演示，所以我只是给你们播放一个我制作视频的视频。这个场景的表现方式是树叶会从树上飘落下来，镜头会移动过去，兔子会做一些动作。有两点需要注意：首先，这一切会非常快速；其次，我几乎会始终使用双手。我有不同的图层，背景、中景和前景。我用左手拇指选择要移动的图层。我要把我的叶子移动到它的位置上。我要把我的兔子移到舞台外并开始运行时间。现在我要演示叶子从树上飘落下来的动作。回放，看看效果如何。动作看起来很好，但是叶子需要有点摇晃。所以我要拿出一个旋转控制器，回放，找到叶子即将脱落的位置，记录下旋转。我在那里添加了一个小翻转，因为那一刻感觉对了。停止，因为我想要移动视角。所以我要一次性拖动很多层，把所有图层都拉成一个列表，我降低了背景层的灵敏度，这样它们移动得更慢，产生一种视差效果。我只想水平移动，所以我拿出一个水平拖动器，看看效果如何。我不太喜欢这种视差效果，所以我稍微调整了灵敏度，再试一次，我喜欢这样的效果，所以我准备好继续了，我回放到开头，这样我可以再次进入作品的节奏中。叶子着陆后，我等待了一拍，然后开始移动视角。我不知道我等了多少帧，也不知道过了多长时间，我就是在感觉对的时候行动了。

[[32:50]] 所以我移动视角到这个冬季场景，并慢慢停了下来。然后我回放，因为我想给我的兔子做点什么。我扔掉这些工具，因为我用完了。然后等到我觉得我的兔子应该移动了，它就跳走了。我有几种不同的姿势给我的兔子。所以我拿出它们。然后找到兔子即将离开地面的点。就是这里。我改变它的姿势，并在它跳跃时切换姿势。然后我回放，因为我想看看效果如何，我会把它全屏给你看。这就是作品。

[[33:50]] 所以我用手做了这个，只用了 2 分钟，就像演奏一个乐器一样。我和我试图创作的东西之间有非常直接的联系。[掌声]

\[\[34:08\]\] 这个工具的灵感之一是几年前我尝试制作的一部动画。虽然不是那个动画，但它也是从一片叶子从树上飘落开始的。我花了一整天在 Flash 里尝试关键帧那片叶子。做不到。所以就这样结束了。我仍然保留着我的分镜头。有时我会播放我为这个作品写的音乐。但这件作品本身锁在我的脑海中。所以我总是想到数以百万计的作品锁在数以百万计的头脑中。不仅仅是动画，不仅仅是艺术，而是所有种类的想法。包括非常重要的想法，改变世界的发明，拯救生命的科学发现。所有这些想法都必须得到培育。如果没有一个能够让它们在其中生长的环境，或者它们的创造者可以通过即时的连接来培育它们，那么许多这些想法将不会出现。或者它们将发育不良。

\[\[35:14\]\] 所以我有这样一个原则，创作者需要即时的连接，我刚刚展示的所有演示都只是我四处观察，注意到这个原则被违反的地方，并试图修复它们。这真的就是我做的。我只是遵循这个指导原则，它引导我去做我必须做的事情。

\[\[35:40\]\] 但我并没有多说这个故事最重要的部分，那就是为什么。为什么我有这个原则。为什么我这样做。

\[\[35:51\]\] 当我看到这个原则被违反时，我不把它看作是一个机会。当我看到创作者被他们的工具限制，他们的想法受到损害时，我不会说：哦，好的，一个制造产品的机会。一个开始业务的机会。或者一个进行研究或为某个领域做贡献的机会。我并不因找到问题而感到兴奋。我参与这个并不是为了制造东西的乐趣。对我来说，想法是非常珍贵的。当我看到想法消失时，我感到痛心。我看到了一场悲剧。对我来说，这感觉像是一种道德错误，像是一种不公正。如果我觉得有什么事我能做的，我感觉这是我的责任去做。不是机会，而是责任。

\[\[36:44\]\] 现在这只是我的看法。我并不要求你像我一样相信这个。我在这里要说的是，我使用的这些词语：不公正，责任，道德错误，这些不是我们在技术领域通常听到的词语。我们确实听到这些词与社会问题相关联。所以像审查制度、性别歧视、环境破坏这类事情。我们都认识到这些是道德错误。大多数人不会看到侵犯公民权利就想：“哦，好的，一个机会。” 我希望不是。

[[37:23]] 相反，我们非常幸运地有历史上的人们认识到这些社会不公，并认为解决这些问题是他们的责任。因此，有这样一种活动主义生活方式，这些人致力于为他们所信仰的事业而战。这次演讲的目的是告诉你，这种活动主义生活方式不仅仅适用于社会活动主义。作为技术专家，你可以认识到世界上的不公正。你可以对一个更好的世界有所设想。你可以致力于为一个原则而战。社会活动家通常通过组织来斗争，但你可以通过发明来斗争。

[[38:07]] 现在我想告诉你一些其他以这种方式生活过的人，首先是 Larry Tesler。Larry 在他的一生中做了许多了不起的事情，但我要告诉你的是他在上世纪 70 年代中期在施乐帕克研究中心（Xerox PARC）所做的工作。当时，个人计算机并不存在。个人计算的概念非常年轻，Larry 和他在 PARC 的同事们认为它们具有变革潜力，个人计算可以改变人们的思维和生活方式。我认为在座的每一个人都会同意，他们对此的预见是正确的。

[[38:43]] 但是当时，软件界面设计是基于模式的。所以，比如在文本编辑器中，你不能像在打字机上那样直接打字然后文字出现在屏幕上。你会处于命令模式，如果你想插入文本，你得按下`I`进入插入模式，然后按`Escape`退出到命令模式，或者也许你会按`A`进入追加模式。或者如果你想移动文本，你会按`M`进入移动模式，然后你得选择并且处于选择和移动事物的模式。Larry 观察人们使用电脑——实际上，他们开创了软件用户研究的概念，这也是他的另一个成就——但是他发现，即使经过培训和使用几周后，许多人对使用电脑仍然感到不舒服。

[[39:30]] 他相信这些模式是问题所在。模式的复杂性是许多人无法跨越的一种障碍。因此，这在某种程度上威胁了个人计算机的梦想。所以拉里把消除软件中的模式作为他的个人使命，并确立了一个原则：不应让任何人被困在模式中。他的口号是“不要让我进入模式”，他还把它印在了 T 恤上。这个原则影响了他所做的每一件事情。他在所有的工作中都思考着这个问题。最终，他开发了一个名为 Gypsy 的文本编辑器，基本上就是今天我们所知道的文本编辑方式。有一个插入点。当你输入时，单词会出现在屏幕上。要选择文本，他发明了无模式选择，即点击和拖动。所以你只需点击并拖动你想选择的文本，就像使用荧光笔一样 —— 这是拖动的最早应用之一。要移动文本，他发明了所谓的剪切、复制、粘贴命令。你选择并剪切。稍后你随时可以粘贴。你永远不会被困在模式中，也不必在模式之间切换。当你在键盘上按下 W 键时，屏幕上就会显示 W。始终如此。

[[40:48]] 他观察人们使用他的软件，发现从未见过计算机的人（当时大多数人）可以在半小时内使用起来。这显然是一种能够让人们与计算机连接的变革性改变。他关于无模式的理念传播到了同时在 PARC 发明的桌面界面的其余部分。今天，这些理念在计算体验中已经根深蒂固，以至于我们现在几乎视之为理所当然。

[[41:20]] 现在我说拉里将消除模式作为他的个人使命。这确实是他的话，如果你认为他在夸大的话，这是拉里过去 30 年的车牌。当然，如今拉里有一个网站，位于 nomodes.com，他还在 Twitter 上：@nomodes。所以就像我说的，拉里在他的职业生涯中做了很多令人惊叹的工作，但他的自我认同显然与这个事业密切相关。

[[41:46]] 所以我想问一下：拉里到底做了什么？我们如何最好地描述拉里做了什么？一个典型的传记可能会说拉里·特斯勒发明了剪切、复制、粘贴。这是事实，但我认为这其实很误导，因为这种发明与说托马斯·爱迪生发明了留声机截然不同。爱迪生基本上是偶然发现了音频录制技术，然后把它作为一种新奇事物来开发。他列出了他技术的可能应用清单，但并没有任何文化意图。而拉里所做的完全是对特定文化背景的反应。

\[\[42:41\]\] 另一个你可能听到的事情是 Larry Tesler 解决了无模式文本操作的问题。解决了这个问题。显然，这是真的，他花了很长时间研究这个问题，最终解决了它。但我认为这真的很误导，因为他解决的这个问题只存在于他自己的头脑中。没有其他人认为这是一个问题。对其他人来说，模式只是计算机运行的方式。这跟我们认为双臂有什么问题一样。这只是生活的一个事实。

\[\[43:18\]\] 所以 Larry 首先做的事情是他认识到了文化中未被承认的错误。事实上，许多重大的社会变革也是从这样开始的。所以，150 年前，伊丽莎白·卡迪·斯坦顿必须站出来说：女性应该投票。其他人都说，“那太疯狂了，你在说什么啊？”今天，我们认识到性别歧视是错误的。但在那时，它是社会的一部分，是看不见的。她不得不认识到这一点，并且不得不与之斗争。对我来说，这比托马斯·爱迪生发明一堆随意技术然后申请专利的模式更接近 Larry 所做的事情。

\[\[44:01\]\] 现在明确一下，我并没有对这两个人的相对重要性或影响力做出任何评判，我只是谈论他们的动机和方法。他们两人都认识到了文化上的错误，他们设想了一个没有这个错误的世界，并且致力于为一个原则而战。她通过组织来斗争，他通过发明来斗争。

\[\[44:23\]\] 计算机领域的许多开创性人物也有类似的动机。当然，包括道格·恩格尔巴特。道格·恩格尔巴特基本上发明了交互式计算。把信息放在屏幕上的概念。通过不同方式查看信息。指向事物并操作它们。他在几乎没有人听说过实时与计算机交互的时代就提出了所有这些概念。今天，他最知名的是鼠标的发明者，但他真正发明的是这种全新的处理知识方式。他从一开始就明确的目标是使人类能够解决世界的紧急问题。他有一个愿景，他称之为知识工作者利用复杂而强大的信息工具来利用他们的集体智慧。他之所以涉足计算机，完全是因为他有一种直觉，认为这些新的东西称为计算机的东西可以帮助他实现这一愿景。他所做的几乎一切都是为了追求这个愿景而单刀直入地推动。

[[45:26]] 这里是艾伦·凯。艾伦·凯在施乐帕克研究所负责实验室管理，我们从那里得到了桌面界面，如窗口、图标、命令菜单等。他还发明了面向对象的编程以及许多其他东西。他的目标，我引用他的话，是要“扩展人类的影响力，并为一个急需新思维的摇摇欲坠的文明带来新的方式”。是不是很伟大？他的方法是通过儿童。他相信，如果儿童能够流利地运用计算机的思维方式，也就是说，如果编程成为像阅读和写作一样的基本素养，那么他们长大后就会具备新形式的批判性思维，以及理解世界的新方式。我们将拥有一个更加开明的社会，类似于识字带给社会的变化。他所做的一切，他发明的一切，都是出于追求这一愿景、这一目标，并且通过与皮亚杰、蒙特梭利、杰罗姆·布鲁纳等人采纳的原则来实现，这些人研究了儿童的思维方式。

[[46:37]] 而与软件活动主义最为广泛联系在一起的人物可能是理查德·斯托曼。斯托曼启动了 GNU 项目，这在今天构成了任何 Linux 系统的一个重要组成部分。他还创建了自由软件基金会，编写了 GCC、GPL 等等。他的原则是软件必须自由，即自由的意义上，并且他对此表达了非常明确的含义。他一直非常清楚地认为软件自由是一种道德上的对与错，并且在自己的生活中采取了特别毫不妥协的态度。

[[47:10]] 所有这些极具影响力的人物都将他们的一生奉献给了为特定理想而战，他们对对错有着非常清晰的认识。通常情况下，他们会与不承认他们所认为的错误的权威或主流进行斗争。今天，世界仍然远未实现任何他们理想中的状态，因此他们仍然看到一个危机四伏的世界，他们继续奋斗。他们一直在奋斗。

[[47:41]] 现在我不是在说你必须过这种生活方式。我也不是说你应该过这种生活方式。我要说的是你可以过这种生活方式。这是一种可供选择的生活方式，而且不会经常听到。你的职业顾问不会建议你开始一个个人的十字军东征。在社交领域可能会，但在技术领域不会。相反，世界会试图让你通过一项技能来定义自己。

[[48:08]] 这就是为什么你在大学里有一个主修专业。这就是为什么你有一个职称。你是一名软件工程师。你可能会专门成为数据库工程师或前端工程师，并且会被要求设计前端。这可能是有价值的，如果你想要在追求卓越并练习一项技能上花费你的一生，你可以选择这条路。这是一位工匠的路径。这是最常见的路径。你真正听说的另一条路径就是问题解决者的路径。所以我将创业精神和学术研究看作是这个硬币的两面。有这个领域。有在这个领域中的一系列问题，或市场上的需求。你进入其中，选择一个问题，你解决它，你在那里做出你的贡献。也许后来，你选择另一个问题，你解决它，你在那里做出你的贡献。同样，这可能是有价值的和有意义的，如果这是你想做的，那么你可以选择这条路。

[[49:04]] 但我没看到 Larry Tesler 走过这两条路之一。我不会说他为用户体验设计领域做出了贡献，因为那时还没有这样的领域。他没有选择解决某个开放性问题，而是提出了一些只存在于他自己头脑中的问题，而且没人认可。当然，他也没有以他的手艺定义自己，而是以他的事业。以他为维护的原则。我敢肯定，如果你去查维基百科，会说他是计算机科学家或者用户体验领域的某种东西，但对我来说，这就像说 Elizabeth Cady Stanton 是一个社区组织者一样。不，Elizabeth Cady Stanton 确立了妇女选举权的原则。那才是她的身份。那是她选择的身份，而 Larry Tesler 确立了无模态原则。他有这个愿景，他实现了这个愿景。

\[\[50:01\]\] 所以，你可以选择这种生活。或者也许它最终会选择你。这可能不会立即发生。找到一个原则可能需要时间，因为找到一个原则本质上是一种自我发现，你试图弄清楚你的生活应该是关于什么。你想作为一个人站在什么位置。对我来说花了像十年的时间。在我真正理解我的原则之前，我的二十岁过得很艰难。当我年轻的时候，我觉得我必须以这种方式生活，但我只能偶尔看到对我重要的东西，但没有大局观。这对我来说非常困扰。我所要做的就是做很多事情。做许多事情，做许多不同类型的事情。学习许多事情，体验许多，许多事情。并利用所有这些经验来分析自己。将所有这些经验作为一种分析自己的方式。把所有这些经验拿来问自己：这与我产生共鸣吗？这是否排斥我？我是否不在乎？积累这些经验，因为某种原因我对它们有很强烈的感觉，并试图理解其中的意义。试图弄清楚其中的秘密成分，这些让我如此强烈反应的经验中到底是什么。

\[\[51:16\]\] 现在我认为每个人都是不同的。而我谈论过的所有人都有他们自己的起源故事，你可以去了解。我只想说，局限于练习一项技能可能会使你难以获得那种似乎对这种工作如此有价值的广泛经验。

\[\[51:35\]\] 最后，如果你选择遵循一个原则，这个原则不能仅仅是你相信的一些老生常谈。你会听到很多人说他们想要让软件更易于使用。或者他们想要让用户感到愉悦。或者他们想要简化事物。这是一个当前非常流行的想法。每个人都想要简化事物。这些都是很好的想法，也许会给你一个方向，但它们太模糊了，不足以直接采取行动。拉里·特斯勒喜欢简单。但他的原则是这个特定的见解：没有人应该被困在一种模式中。这是一个强有力的原则，因为它给了他一种新的看待世界的方式。它以一种相当客观的方式将世界划分为对和错。所以，他可以看着某人选择文本，然后问：这个人处于一种模式中吗？是或否？如果是，他必须对此做些什么。同样地，我相信创作者需要强大的工具。这是一个很好的想法，但它并没有真正帮我什么忙。我的原则是创作者需要这种即时的联系。所以我可以看着你改变一行代码，然后问：你立即看到了那个改变的效果吗？是或否？如果不是，我得对此做些什么。

[[52:52]] 而且，我给你展示的所有演示都是我做到了这一点，都是我遵循这个原则，并让它带领我做到了我需要做的事情。因此，如果你有一个指导原则和具体的洞见，它将引导你。你会始终知道你所做的是否正确。

[[53:19]] 生活有许多种方式。也许你在生活中最重要的认识就是，你的生活的每一个方面都是一个选择。但是也有默认的选择。你可以选择在生活中懒散地前行，接受已经为你铺好的道路。你可以选择接受世界的现状。但你不必这样。如果你觉得世界上有什么是不对的，而你又有一个更好世界的愿景，你可以找到你的指导原则。你可以为一个事业而战。所以在这次演讲之后，我希望你花一点时间思考对你而言重要的事情。你相信什么。你可能会为何而奋斗。

[[54:06]] 谢谢。[掌声]
