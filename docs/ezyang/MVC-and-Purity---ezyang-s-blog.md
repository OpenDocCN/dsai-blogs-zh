<!--yml
category: 未分类
date: 2024-07-01 18:18:15
-->

# MVC and Purity : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/07/mvc-and-purity/](http://blog.ezyang.com/2010/07/mvc-and-purity/)

*Attention conservation notice.* Purely functional programming demonstrates the same practices recommended by object-oriented MVC practice.

[Model-View-Controller](http://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller) is a widely used object-oriented design pattern for organizing functionality in an application with a user interface. I first ran across it in my early days programming web applications. The Model/View separation made deep intuitive sense to me as a PHP programmer: without it, you’d end up with spaghetti templates with HTML print statements interleaved with MySQL queries. But Controller [was always a little wishy-washy](http://www.c2.com/cgi/wiki?WhatsaControllerAnyway). What exactly did it do? It was some sort of “glue” code, the kind of stuff that bound together the Model and View and gave them orders. But this was always a sort of half-hearted answer for me ([where should input validation go?](http://discuss.joelonsoftware.com/default.asp?design.4.354410.6)), and soon I left the world of web applications, my questions unanswered.

Having been exposed to purely functional programming, I now believe that the controller and model/view separation is precisely the separation between side-effectful code (IO) and pure code.

*The controller depends on the model and the view, but the model and view should not (directly) depend on the controller.* Pure code and impure code don't mix freely. In particular, you're not allowed to reference impure code from pure code (unless you use `unsafePerformIO`). However, impure code can call pure code (although there may be some [technical details](http://www.haskell.org/ghc/docs/6.12.2/html/libraries/base-4.2.0.1/Control-Monad.html) involved), and the resulting code is impure. So, if the Controller is impure code and the Model/View is pure code, separating the two is simply making sure that if we have any code that is impure, we've extracted as much of the pure computation out of it as possible. Stated differently, if I have a function that reads and writes data, and there are lines in it that don't have anything to do with IO, I should move them into their own function. Maybe those lines are the templating system, in which case it’s View; maybe those lines are running some complicated equation, in which case it’s Model. Pure/impure doesn't capture the model/view distinction.

*The controller receives input and initiates a response.* So, the controller is input-output, i.e. IO.

*The controller handles events that affect the model or the view.* Pure code sort of lives in a vacuum: it can do computation, but it can't do anything useful, since it can’t have any side effects and thus has no way for us to tell it what to compute, or to view the result of the computation. Impure code is the way we get anything done by handing off this information to our pure code.

There are several possible objections to this division. Here are a few of them:

*Most object-oriented models are stateful, and state is not pure!* There is a common misconception that state isn't pure, possibly arising from the fact that both IO and State are monads. However, I can turn a state monad into a single, pure value by simply running the state machine: code that is stateful is monadic, but it is also pure, since it doesn't have any external side effects. Shared state is a bit trickier, and usually not pure.

*Controller code doesn’t have to be impure and here’s an example.* Here I’ll indulge in a bit of prescriptivism: I bet you have a model, but one that is only tangentially related to your core business logic. If you have code that parses binary strings into message objects (but doesn’t actually handle transmitting or receiving those binary strings on the network), you have a mini-model of network messages. You should probably keep it separate from your real model, but for testability you should also keep it separate from your network code. Separation of concerns may be malleable, but the little IO in your type signature is always honest.

Some parting words about the purity hair-shirt: it is fairly widely acknowledged that busting out the MVC pattern makes your application more complex initially, and in a purely functional language, you’re forced to respect the distinction from the very beginning. Thus, writing small programs can be frustrating in a purely functional language because you don’t want to use the bulky but scalable engineering practice yet, and the language is making you do so from the very beginning. Haskell gives you a lot of rope to make it pleasant again, but it takes a while to get used to. On the plus side, when your program grows, this separation will continue to be enforced, and a messy refactoring may be averted.