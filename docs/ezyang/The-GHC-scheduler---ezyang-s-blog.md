<!--yml

category: 未分类

date: 2024-07-01 18:17:23

-->

# GHC 调度器：ezyang 的博客

> 来源：[`blog.ezyang.com/2013/01/the-ghc-scheduler/`](http://blog.ezyang.com/2013/01/the-ghc-scheduler/)

我想谈谈 GHC 线程调度的一些细节，这些细节是在为 GHC 实现步幅调度的过程中发现的。大多数选择仅仅是*实现*细节，并不属于任何规范的一部分。虽然这些选择不应该依赖，但了解它们是值得的，因为许多这些细节是通过许多性能 bug、基准测试和其他斗争而积累起来的。在本文中，我将尝试提供一些历史性见解，解释为什么会做出许多选择。这些见解应该适用于任何希望实现*绿色线程*的系统，这些线程比传统的操作系统线程使用的内存少。由于篇幅限制，我不会讨论 STM 或者 sparks（尽管它们也非常有趣）。

### 线程的解剖

首先，我想先讨论一些关于运行时系统的简要背景，并指出一些可能不直观的设计选择。在 GHC 中，线程由 TSO（线程状态对象）表示，即 `includes/rts/storage/TSO.h` 中的 `StgTSO` 结构体。 [1] 在 Haskell 中，TSO 可以作为 `ThreadId` 对象进行传递。结构体名称前的 `Stg` 表示 TSO 像 Haskell 中的其他闭包一样*会被垃圾回收*。TSO 和与之分配的栈（STACK）构成线程的主要内存开销。默认的栈大小由 GC 标志 `-ki` 控制，默认为 1k。 [2] 线程由 Capabilities 运行，可以将其视为 GHC 管理的虚拟核心。Capabilities 又映射到真正的操作系统线程或任务，尽管我们不会详细讨论它们。

作为被垃圾回收的对象对 TSO 有两个主要影响。首先，TSO *不是* GC 根，因此如果没有任何东西持有它们（例如[死锁的情况下](http://blog.ezyang.com/2011/07/blockedindefinitelyonmvar/)），它们将会被 GC 回收，并且它们的空间在执行完成后不会*自动*回收。 [3] 通常情况下，TSO 将由 Capability 的运行队列（一个 GC 根）保留，或者在某些并发变量的等待线程列表中，例如 MVar。其次，TSO 必须被视为*可变*对象，因此它们受到生成式垃圾回收器中任何可变对象所需的传统 GC 写屏障的约束。 [4] `dirty` 位跟踪 TSO 是否已被修改；当线程运行时或者修改 TSO 的任何指针字段时，它总是被设置。`setTSOLink` 和 `setTSOPrev` 设置的两个字段对调度器特别重要。

### 运行队列

运行队列是调度程序的核心，因为任何可运行的线程在调度程序实际弹出它并运行之前都会进入运行队列。每个能力有一个 `rts/Capability.h`（在旧时代，存在全局运行队列，但对于多线程进程性能表现不佳），它被实现为一个双向链表 `run_queue_hd` 和 `run_queue_tl`。[6] 头部和尾部指针意味着队列实际上是一个双端队列：这很重要，因为调度程序通常必须处理以某种方式中断的线程，并应该让这些线程重新回到队列上。链接本身位于 TSO 上，并通过 `setTSOLink` 和 `setTSOPrev` 进行修改，因此修改队列会使涉及的 TSO 变脏。[7] 否则，运行队列完全归调度程序所有。如果存在空闲的能力，并且我们的运行队列中有多于一个线程，线程将被推送到其他队列中使用 `schedulePushWork`。

线程被放在 *前* (`pushOnRunQueue`) 的情况包括：

+   发生堆栈溢出；

+   发生堆溢出；[8]

+   任务尝试运行一个线程，但它是 [绑定](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Control-Concurrent.html#v:forkOS)，而当前任务不是正确的任务；

+   线程与黑洞关联（正在评估的 thunk），并且另一个可能在另一个能力上的线程已经阻塞在其评估上（见 [ticket #3838](http://hackage.haskell.org/trac/ghc/ticket/3838)）；

+   在线程化运行时，如果一个线程被中断，因为另一个能力需要执行停止世界 GC（参见提交 `6d18141d8`）；

+   在非线程化运行时，当一个等待 IO 的线程取消阻塞时。

线程在 *后* (`appendToRunQueue`) 放置的情况包括抢占或是新线程的情况；特别是，如果

+   线程通过上下文切换标志被抢占（例如来自另一个线程的消息，定时器触发，线程协作性地放弃等；另请参阅 [8] 了解这与堆溢出的交互方式）；

+   它是一个新线程（因此大量线程创建不会饿死旧线程，请参见 `conc004` 和提交 `05881ecab`）；

+   线程变为非阻塞状态；

+   线程被迁移到另一个能力（尽管在这种情况下，队列本来就是空的）；

+   线程完成，但出于某些原因我们需要保留它（这与内部调用相关，尽管我不完全确定具体情况；如果您知道，请告诉我！）

### 结论

GHC 调度器相当复杂！大部分当前行为是针对特定问题而创建的：正确的选择*并不*显而易见！我希望本文能成为任何未来对 GHC 调度器感兴趣的黑客的宝贵参考，以及对需要为其运行时系统实现调度器的其他人的有价值参考。大部分历史数据来源于评论（尽管我找到了一些过时的评论），大量使用`git blame`和与 Bug 跟踪器交叉引用——这些都是弄清楚“嗯，为什么这段代码这样做？”的有用方法。在本文中，我希望我在一定程度上回答了这个问题。

* * *

[1] `StgTSO`的初始化在`rts/Threads.c`的`createThread`中处理；然后，此函数由`rts/RtsAPI.c`中的`createGenThread`、`createIOThread`和`createStrictIOThread`调用。这些函数设置了初始堆栈状态，控制线程实际运行时执行的内容。这些函数是由`fork#`和其他 primops（primops 的入口点位于`rts/PrimOps.cmm`中）调用的函数。

[2] 实际上，你可用的堆栈比这个稍小，因为这个大小还包括`StgTSO`结构体的大小。（然而，这只适用于将大量线程分配到一个块中，一旦发生 GC，TSO 和堆栈将不再相邻。）

[3] 这里是一个演示如何通过使用稳定指针来保留`ThreadId`（这会强制它们指向的对象永不被 GC 回收）可能导致内存泄漏的示例程序：

```
import Control.Concurrent
import Control.Monad
import Foreign.StablePtr

n = 400000
main = do
    ms <- replicateM n (newEmptyMVar >>= \m -> (forkIO (putMVar m ()) >>= newStablePtr) >> return m)
    mapM_ takeMVar ms

```

程序的[堆剖析](http://heap.ezyang.com/view/6e310e2e2e2c11ff3a7cc8ff0f5c205e51a8a188)显示，即使在 MVars 排空后，TSO/STACK 对象仍未被释放。

[4] 用于分代 GC 的写屏障不是指多线程执行的[内存屏障](http://en.wikipedia.org/wiki/Memory_barrier)，而是当旧代中的可变引用发生变化并且可能指向年轻代中的对象时，通知垃圾收集器的操作。在小型收集期间，不会遍历旧代，因此如果旧代可能指向年轻代的对象，我们可能会错过年轻对象仍然存活的事实，即使它没有从其他年轻对象中引用。在 GHC 中，通过将对象添加到能力的[可变列表](http://hackage.haskell.org/trac/ghc/wiki/StgObjectTypes)（`mut_list`）来实现写屏障，如果对象不在最年轻的代中。（一些对象，如`MutArr#`，始终在可变列表上；在这种情况下，可能不需要写屏障。但请参阅[5]以了解更多详细信息。）对象通常会跟踪它们的脏状态，以便它们不会多次将自己添加到可变列表中。（意外添加对象多次是无害的，但意味着 GC 必须额外遍历可变列表。）此外，如果我们可以保证新引用不指向年轻代（例如，它是像`END_TSO_QUEUE`这样的静态闭包），则不需要标记该对象为脏。毫无疑问，搞清楚这些东西是相当棘手的！

[5] 这里有一点不太光彩的故事。通过`rts/sm/Scav.c`中的`scavenge_mutable_list`将对象永久添加到可变列表中，如果它在那里看到它，则将这样的对象无条件地重新添加到可变列表中。对象如何首次添加到可变列表中呢？它并非在创建时放置在列表上；相反，在最年轻代的第一次小型 GC 时，清扫 GC 注意到该对象，并通过`gct->failed_to_evac = rtsTrue`将其放置在可变列表上。我们如何最终释放对象？可变列表被视为一组根指针，但仅进行*清扫*，而不是疏散。如果可变列表上的项目最终未被疏散，则将其清除。（这确实意味着，其元素直到下一次 GC 才会被释放。）总是扫描这些数组真的很低效吗？是的，这曾经是一个问题（票号 #650），现在通过卡片标记来缓解。同样的故事也适用于[TSOs（票号 #1589）](http://hackage.haskell.org/trac/ghc/ticket/1589)，但修复的方法是正确应用写屏障，并且不将对象永久地保留在可变列表中；这在存在大量线程时显著提高了性能（即使不清扫它们的指针，遍历庞大的可变列表仍然很痛苦）。创建大量小型可变数组很可能是令人头疼的。

[6] 曾经它是单向链表，但修复 [ticket #3838](http://hackage.haskell.org/trac/ghc/ticket/3838) 需要从运行队列中移除 TSOs 的能力。

[7] 由于这些字段总是被 GC 遍历，保证它们不包含 NULL 指针或垃圾非常重要。相反，我们将它们设置为静态闭包 `END_TSO_QUEUE`。因为这个闭包保证不在年轻代中，这就是为什么在设置完这个字段之后不需要污染 TSO 的原因。

[8] 有时，堆溢出和上下文切换同时发生。如果线程请求了一个大块，我们仍然总是把它放在前面（因为我们不希望另一个线程窃取我们的大块）；然而，否则，上下文切换优先，并且线程被移动到队列的末尾——上下文切换尽可能晚地检查。 (见提交 `05881ecab`)
