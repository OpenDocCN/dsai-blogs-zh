<!--yml
category: 未分类
date: 2024-07-01 18:18:20
-->

# Design Patterns in Haskell : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/05/design-patterns-in-haskel/](http://blog.ezyang.com/2010/05/design-patterns-in-haskel/)

*Attention Conservation Notice.* A listing of how Gang of Four design patterns might be equivalently implemented in Haskell. A phrasebook for object-oriented programmers dealing with functional programming concepts.

In their introduction to seminal work *Design Patterns*, the Gang of Four say, "The choice of programming language is important because it influences one's point of view. Our patterns assume Smalltalk/C++-level language features, and that choice determines what can and cannot be implemented easily. If we assumed procedural languages, we might have included design patterns called 'Inheritance,' 'Encapsulation,' and 'Polymorphism.'"

What is easy and what is hard to implement in a functional programming language? I decided to revisit all 23 original Gang of Four design patterns under that lense. My hope is that these results will be useful to Object Oriented Programmers seeking to learn the ways of Functional Programming.

[Strategy](http://en.wikipedia.org/wiki/Strategy_pattern). *First class functions and lambdas.* Any extra data that might be placed as class members is traditionally implemented using closures (which stash the data in a lambda function's environment) or currying (which create implicit closures for function's arguments). Strategies are also powerful because they are polymorphic; type synonyms for function types can play a similar role. Java has recognized anonymous functions as a good idea, and have added facilities for anonymous classes, which are frequently used in this capacity.

[Factory Method](http://en.wikipedia.org/wiki/Factory_method_pattern) and [Template Method](http://en.wikipedia.org/wiki/Template_method_pattern). *Higher-order functions.* Instead of making a subclass, just pass the the function you'd like to vary the behavior of with the function.

[Abstract Factory](http://en.wikipedia.org/wiki/Abstract_factory_pattern), [Builder](http://en.wikipedia.org/wiki/Builder_pattern) and [Bridge](http://en.wikipedia.org/wiki/Bridge_pattern). *Type classes* and *smart constructors.* Type classes are capable of defining functions which creating instances of themselves; all a function needs to do to take advantage of this is to commit itself to returning some value of type `TypeClass a => a` and using only (constructor et alia) functions that the type class exposes. If you're not just constructing values but manipulating them with the general type class interface, you have a Bridge. Smart constructors are functions built on top of the basic data constructor that can do "more", whether this is invariant checking, encapsulation or an easier API, this can correspond to more advanced methods that a factory provides.

[Adapter](http://en.wikipedia.org/wiki/Adapter_pattern), [Decorator](http://en.wikipedia.org/wiki/Decorator_pattern) and [Chain of Responsibility](http://en.wikipedia.org/wiki/Chain-of-responsibility_pattern). *Composition* and *lifting.* Function composition can be used to form a pipeline of data between functions; a foreign function can be sandwiched between two functions that convert to and from the type the function expects, or a function can be composed with another to make it do more things. If the signature stays the same, one or more of the functions was *endomorphic.* If the functions have side effect, it may be Kleisli arrow composition (more plainly spoken as monadic function composition.) Multiple functions can handle the same input using the Reader monad.

[Visitor](http://en.wikipedia.org/wiki/Visitor_pattern). *Equational functions.* Frequently *foldable.* Many functional languages favor grouping the same operation on different data constructors together, in a mathematical equational style. This means similar behaviors are grouped together. Traditional grouping of behavior by "class" is implemented with *type classes.* Visitors typically collapse the data structures they operate on into smaller values, this is seen in the fold family of functions.

[Interpreter](http://en.wikipedia.org/wiki/Interpreter_pattern). *Functions*. Frequently circumvented with an *embedded domain specific language.* Algebraic data types make light-weight abstract syntax trees easy to formulate. Just as Visitor is often used with Interpeter, you'll probably write your interpreting functions with pattern matching. Even better, don't come up with another data type; just use functions and infix operators to say what you mean. Closely related to...

[Command](http://en.wikipedia.org/wiki/Command_pattern). *Monads.* See also *algebraic data types*, frequently *generalized (GADT)*. A pure language will not run your `IO` until `main` touches it, so you can freely pass values of type `IO a` without fear of actually causing the side-effect, though these functions are difficult to serialize (a common motivation behind Command). Parametrization of the action to perform is once again achieved through higher-order functions. GADTs are a little more bulky, but can be seen in places like the [Prompt monad (PDF)](http://themonadreader.files.wordpress.com/2010/01/issue15.pdf), where a GADT is used to represent actions that another function interprets into the `IO` monad; the type gives a statically enforced guarantee of what operations in this data type are allowed to do.

[Composite](http://en.wikipedia.org/wiki/Composite_pattern). Recursive *algebraic data types.* Especially prominent since there's no built-in inheritance.

[Iterator](http://en.wikipedia.org/wiki/Iterator_pattern). *Lazy lists.* Iterators expose an element-by-element access of a data structure without exposing it's external structure; the list is the API for this sort of access and laziness means we don't compute the entirety of the stream until it is necessary. When IO is involved, you might use a real iterator.

[Prototype](http://en.wikipedia.org/wiki/Prototype_pattern). *Immutability.* Modification copies by default.

[Flyweight](http://en.wikipedia.org/wiki/Flyweight_pattern). *Memoising* and *constant applicative forms (CAF).* Instead of calculating the result of an expression, create a data structure that contains all of the results for all possible input values (or perhaps, just the maximum memo). Because it is lazy, the result is not computed until it is needed; because it is a legitimate data structure, the same result is returned on successive computations. CAFs describe expressions that can be lifted into the top-level of a program and whose result can be shared by all other code that references it.

[State](http://en.wikipedia.org/wiki/State_pattern) and [Memento](http://en.wikipedia.org/wiki/Memento_pattern). Unnecessary; state has an explicit representation and thus can always be arbitrarily modified, and it can include functions, which can be changed to change behavior. State as a function (rather than an object or an enumeration), if you will. The encapsulation provided by Memento is achieved by hiding the appropriate constructors or destructors. You can easily automatically manage past and future states in an appropriate monad such as the Undo monad.

[Singleton](http://en.wikipedia.org/wiki/Singleton_pattern). Unnecessary; there is no global state except in a monad, and the monad's type can enforce that only one instance of a record is present; functions exist in a global namespace and are always accessible.

[Facade](http://en.wikipedia.org/wiki/Facade_pattern). *Functions.* Generally less prevalent, since function programming focuses on input-output, which makes the straight-forward version use of a function very short. High generality can require more user friendly interfaces, typically implemented with, well, more functions.

[Observer](http://en.wikipedia.org/wiki/Observer_pattern). One of many concurrency mechanisms, such as channels, asynchronous exceptions and mutable variables. See also *functional reactive programming.*

[Proxy](http://en.wikipedia.org/wiki/Proxy_pattern). *Wrapped data types,* *laziness* and *garbage collector.* See also ref monadic types (IORef, STRef), which give more traditional pointer semantics. Laziness means structures are always created on demand, garbage collection means smart references are not necessary. You can also wrap a data type and only publish accessors that enforce extra restrictions.

[Mediator](http://en.wikipedia.org/wiki/Mediator_pattern). *Monad stacks*. While it's not useful to talk about interactions between objects, due to a preference for stateless code, monad stacks are frequently used to provide a unified interface for code that performs operations in a complex environment.

Comments and suggestions appreciated; I'll be keeping this post up-to-date.