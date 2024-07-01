<!--yml
category: 未分类
date: 2024-07-01 18:17:17
-->

# If you’re using lift, you’re doing it wrong (probably) : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/09/if-youre-using-lift-youre-doing-it-wrong-probably/](http://blog.ezyang.com/2013/09/if-youre-using-lift-youre-doing-it-wrong-probably/)

## If you’re using lift, you’re doing it wrong (probably)

David Darais asked me to make this public service announcement: *If you're using lift, you're doing it wrong.* This request was prompted by several talks at ICFP about alternatives to monad transformers in Haskell, which all began their talk with the motivation, "Everyone hates lifting their operations up the monad stack; therefore, we need another way of organizing effects." This [StackOverflow question](http://stackoverflow.com/questions/9054731/avoiding-lift-with-monad-transformers) describes the standard technique that `mtl` uses to remove the use of lift in most monadic code.

Now, as most things go, the situation is a bit more nuanced than just "never use lift", and a technically incorrect quip at the beginning of a talk does not negate the motivation behind other effect systems. Here are some of the nuances:

*   As everyone is well aware, when a monad transformer shows up multiple times in the monad stack, the automatic type class resolution mechanism doesn't work, and you need to explicitly say which monad transformer you want to interact with.
*   This mechanism only works if the monadic operations you are interacting with are suitably generalized to begin with, e.g. `MonadReader a m => m a` rather than `Monad m => ReaderT m a` or `Reader a`. This is especially evident for the `IO` monad, where most people have not generalized their definitions to `MonadIO`. Fortunately, it is generally the case that only one `liftIO` is necessary.

And of course, there are still many reasons why you would want to ditch monad transformers:

*   Type-class instances are inherently unordered, and thus a generalized `MonadCont m, MonadState m => m a` monadic value says nothing about what order the two relevant monads are composed. But the order of this composition has an important semantic effect on how the monad proceeds (does the state transfer or reset over continuation jumps). Thus, monad transformers can have subtle interactions with one another, when sometimes you want *non-interfering* effects that are truly commutative with one another. And indeed, when you are using the type class approach, you usually use only monads that commute with one another.
*   The interference between different monad transformers makes it difficult to lift certain functions. For example, the type of `mask :: ((forall a. IO a -> IO a) -> IO b) -> IO b`. If we think operationally what has to happen when IO is composed with State, the lifter has to some how arrange for the state to transfer all the way into the code that runs with exceptions restored. That's very tricky to do in a general way. It gets even worse when these callbacks are [invoked multiple times.](http://blog.ezyang.com/2012/01/monadbasecontrol-is-unsound/)
*   At the end of the day, while the use of type classes makes the monad stack somewhat abstract and allows the elision of lifts, most of this code is written with some specific monad stack in mind. Thus, it is very rare for nontrivial programs to make use of multiple effects in a modular way, or for effects to be instantiated (i.e. a concrete monad selected) without concretizing the rest of the monad stack.

Monad transformers have problems, let's argue against them for the right reasons!