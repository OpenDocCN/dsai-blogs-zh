<!--yml

类别：未分类

日期：2024-07-01 18:18:13

-->

# 有效管理外部指针：ezyang 的博客

> 来源：[`blog.ezyang.com/2010/07/managing-foreign-pointers-effectively/`](http://blog.ezyang.com/2010/07/managing-foreign-pointers-effectively/)

## 有效管理外部指针

[Foreign.ForeignPtr](http://haskell.org/ghc/docs/6.12.2/html/libraries/base-4.2.0.1/Foreign-ForeignPtr.html)是您可以向 C 库挥动的魔棒，使它们突然变得可垃圾回收。虽然事实并非如此简单，但它确实*相当简单*。以下是在使用 Haskell FFI 有效管理外部指针时的一些来自前线的快速提示：

+   尽早使用它们。一旦从外部导入函数传递给您一个您预期要释放的指针，您应该在做任何其他操作之前将其包装在 ForeignPtr 中：这一责任完全在低级绑定中。找到您必须作为`FunPtr`导入的函数。如果您正在使用 c2hs，请将您的指针声明为`foreign`。

+   作为上述观点的例外，如果 C 库提供了多种释放指针的方式，则可能需要小心处理；一个示例是一个接受指针并销毁它的函数（可能不释放内存，而是重新使用它），并返回一个新的指针。如果将其包装在 ForeignPtr 中，当它被垃圾回收时，您将面临双重释放问题。如果这是主要操作模式，请考虑使用`ForeignPtr (Ptr a)`和自定义释放函数，该函数在外部 ForeignPtr 中释放内部指针后释放外部指针。如果在释放指针时没有逻辑连续性，可以使用`StablePtr`来保持您的`ForeignPtr`永远不会被垃圾回收，但这实际上是一种内存泄漏。一旦是外部指针，就永远是外部指针，因此如果不能承诺直到垃圾回收分手，请不要使用它们。

+   您可以将外部指针作为不透明引用传递给用户代码，这可能导致新类型的普及。定义`withOpaqueType`非常有用，这样您就不必每次自己的代码窥视黑匣子时都进行模式匹配，然后使用`withForeignPtr`。

+   要小心使用库的`free`等效方法。尽管在由 libc 统一的系统上，您可能可以通过对获取的`int*`数组使用`free`来摆脱问题（因为大多数库在内部使用`malloc`），但这段代码不具备可移植性，如果尝试在 Windows 上编译，则几乎肯定会崩溃。当然，复杂的结构可能需要更复杂的释放策略。（实际上，这是我在测试自己的库在 Windows 上时遇到的唯一错误，直到我记起雷蒙德的博客文章才感到非常沮丧。）

+   如果你有指向由另一个指针进行内存管理的数据的指针，而这另一个指针位于 ForeignPtr 内部，则必须极其小心以防止在你还有这些指针的情况下释放 ForeignPtr。有几种方法可以解决这个问题：

    +   在具有秩-2 类型的 Monad 中捕获子指针（参见`ST` Monad 的示例），并要求该 Monad 在`withForeignPtr`内运行，以确保主指针在子指针存在时保持活动状态，并保证子指针不会泄漏出上下文。

    +   使用`Foreign.ForeignPtr.Concurrent`可以进行有趣的事情，它允许你将 Haskell 代码用作最终器：参考计数和依赖跟踪（只要你的最终器能在主最终器运行后继续运行）。我觉得这种方式非常不理想，你能得到的保证并不总是很好。

+   如果不需要将指针释放到外界，最好别这么做！[Simon Marlow](http://article.gmane.org/gmane.comp.lang.haskell.glasgow.user/7107) 承认最终器可能会带来各种麻烦，如果你可以只给用户一个括号函数，你应该考虑这种方式。这样做可以使你的内存使用和对象生命周期更加可预测。