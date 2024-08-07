<!--yml

category: 未分类

date: 2024-07-01 18:18:20

-->

# Haskell 中的设计模式：ezyang’s 博客

> 来源：[`blog.ezyang.com/2010/05/design-patterns-in-haskel/`](http://blog.ezyang.com/2010/05/design-patterns-in-haskel/)

*注意：保存注意事项*。列出了如何在函数式编程语言中等效实现四人帮设计模式的清单。这是面向对象程序员处理函数式编程概念的短语手册。

在其对经典作品*设计模式*的介绍中，四人帮说：“编程语言的选择很重要，因为它影响一个人的视角。我们的模式假设具有 Smalltalk/C++级别的语言特性，而这种选择决定了什么可以轻松实现，什么不可以。如果我们假设过程式语言，我们可能会包括名为‘继承’、‘封装’和‘多态性’的设计模式。”

在函数式编程语言中，什么容易实现，什么难以实现？我决定重新审视所有 23 个原始的四人帮设计模式，从这个角度出发。我希望这些结果对希望学习函数式编程的面向对象程序员有所帮助。

[策略](http://zh.wikipedia.org/wiki/策略模式)。*一级函数和 lambda*。任何可能放置为类成员的额外数据通常使用闭包（将数据存储在 lambda 函数的环境中）或柯里化（为函数的参数创建隐式闭包）来实现。策略也非常强大，因为它们是多态的；函数类型的类型同义词可以发挥类似的作用。Java 已经认识到匿名函数是一个好主意，并且添加了匿名类的功能，这在这方面经常被使用。

[工厂方法](http://zh.wikipedia.org/wiki/工厂方法模式)和[模板方法](http://zh.wikipedia.org/wiki/模板方法模式)。*高阶函数*。不要创建子类，只需传递想要变化行为的函数。

[抽象工厂](http://zh.wikipedia.org/wiki/抽象工厂模式)，[建造者](http://zh.wikipedia.org/wiki/建造者模式)和[桥接](http://zh.wikipedia.org/wiki/桥接模式)。*类型类*和*智能构造函数*。类型类能够定义创建其实例的函数；一个函数只需承诺返回某种类型的值 `TypeClass a => a`，并且仅使用类型类公开的（构造函数等）函数，就能利用这一特性。如果你不仅仅是构造值，而是通过通用类型类接口操纵它们，你就拥有了一个桥接。智能构造函数是在基本数据构造函数之上构建的函数，可以做“更多”事情，无论是不变性检查、封装还是更简单的 API，这对应于工厂提供的更高级方法。

[适配器模式](http://en.wikipedia.org/wiki/Adapter_pattern)，[装饰者模式](http://en.wikipedia.org/wiki/Decorator_pattern)和[责任链模式](http://en.wikipedia.org/wiki/Chain-of-responsibility_pattern)。*组合*和*提升*。函数组合可用于形成函数之间的数据管道；外部函数可以夹在期望的类型转换函数之间，或者一个函数可以与另一个函数组合以使其执行更多操作。如果签名保持不变，则一个或多个函数是*端态的*。如果函数具有副作用，则可能是 Kleisli 箭头组合（更通俗地称为单子函数组合）。多个函数可以使用 Reader 单子处理相同的输入。

[访问者模式](http://en.wikipedia.org/wiki/Visitor_pattern)。*等式函数*。经常是*可折叠*的。许多函数式语言喜欢以数学等式风格将相同操作在不同数据构造器上分组，这意味着类似的行为被聚集在一起。传统的按“类”分组行为是通过*类型类*来实现的。访问者通常将它们操作的数据结构折叠成更小的值，这在折叠函数族中可以看到。

[解释器模式](http://en.wikipedia.org/wiki/Interpreter_pattern)。*函数*。经常通过*嵌入式领域特定语言*绕过。代数数据类型使得轻量级抽象语法树易于构造。正如访问者经常与解释器一起使用，你可能会用模式匹配编写你的解释函数。更好的做法是，不要再想出另一种数据类型；只需使用函数和中缀操作符来表达你的意图。与此密切相关的是...

[命令模式](http://en.wikipedia.org/wiki/Command_pattern)。*单子*。另见*代数数据类型*，经常是*广义代数数据类型（GADT）*。纯语言不会运行你的`IO`，直到`main`触及它，因此你可以自由地传递类型为`IO a`的值，而不必担心实际引起副作用，尽管这些函数难以序列化（命令背后的常见动机）。通过高阶函数再次实现对要执行的操作的参数化。GADT 有点臃肿，但可以在像[Prompt monad (PDF)](http://themonadreader.files.wordpress.com/2010/01/issue15.pdf)这样的地方看到，其中 GADT 用于表示另一个函数将其解释为`IO`单子的操作；类型给出了静态强制执行的保证，这种数据类型中允许做什么操作。

[组合模式](http://en.wikipedia.org/wiki/Composite_pattern)。递归*代数数据类型*。特别突出，因为没有内置继承。

[迭代器模式](http://en.wikipedia.org/wiki/Iterator_pattern)。*惰性列表。* 迭代器公开了对数据结构逐个元素访问的接口，而不暴露其外部结构；列表是这种访问的 API，惰性意味着在需要时我们不会计算整个流。涉及 IO 时，你可能会使用真正的迭代器。

[原型模式](http://en.wikipedia.org/wiki/Prototype_pattern)。*不可变性。* 修改默认复制。

[享元模式](http://en.wikipedia.org/wiki/Flyweight_pattern)。*记忆化* 和 *常量适用表达形式（CAF）。* 而不是计算表达式的结果，创建一个包含所有可能输入值的结果的数据结构（或者，只是最大的记忆）。因为它是惰性的，结果在需要时才计算；因为它是一个合法的数据结构，所以在后续计算中返回相同的结果。CAF 描述的表达式可以提升到程序的顶层，并且其结果可以被所有引用它的其他代码共享。

[状态模式](http://en.wikipedia.org/wiki/State_pattern) 和 [备忘录模式](http://en.wikipedia.org/wiki/Memento_pattern)。不必要；状态具有显式表示，因此可以随意修改，它可以包括函数，可以更改以改变行为。状态作为函数（而不是对象或枚举），如果你愿意。备忘录提供的封装是通过隐藏适当的构造函数或析构函数实现的。你可以在适当的单子（例如 Undo 单子）中轻松自动管理过去和未来状态。

[单例模式](http://en.wikipedia.org/wiki/Singleton_pattern)。不必要；除了在单子中的全局状态之外，还可以通过单子的类型来强制只存在一个记录实例；函数存在于全局命名空间并且总是可访问的。

[外观模式](http://en.wikipedia.org/wiki/Facade_pattern)。*函数。* 一般来说不太普遍，因为函数式编程侧重于输入输出，使得直接使用函数的版本非常简短。高泛化度可能需要更用户友好的接口，通常通过更多的函数实现。

[观察者模式](http://en.wikipedia.org/wiki/Observer_pattern)。诸如通道、异步异常和可变变量等许多并发机制之一。参见*函数式反应式编程*。

[代理模式](http://en.wikipedia.org/wiki/Proxy_pattern)。*包装数据类型，* *惰性* 和 *垃圾回收器。* 参见参考单子类型（IORef，STRef），它们提供更传统的指针语义。惰性意味着结构总是按需创建的，垃圾回收意味着智能引用是不必要的。你还可以包装一个数据类型，并且只发布强制执行额外限制的访问器。

[中介者模式](http://en.wikipedia.org/wiki/Mediator_pattern)。 *单子堆栈*。虽然讨论对象之间的交互并不实用，因为更偏好无状态代码，但单子堆栈经常用于为在复杂环境中执行操作的代码提供统一接口。

欢迎评论和建议；我将会保持这篇文章的更新。
