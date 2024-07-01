<!--yml
category: 未分类
date: 2024-07-01 18:17:36
-->

# Modelling IO: MonadIO and beyond : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/01/modelling-io/](http://blog.ezyang.com/2012/01/modelling-io/)

The MonadIO problem is, at the surface, a simple one: we would like to take some function signature that contains `IO`, and replace all instances of `IO` with some other IO-backed monad `m`. The MonadIO typeclass itself allows us to transform a value of form `IO a` to `m a` (and, by composition, any function with an `IO a` as the result). This interface is uncontroversial and quite flexible; it’s been in the bootstrap libraries ever since it was [created in 2001](https://github.com/ghc/packages-base/commit/7f1f4e7a695c402ddd3a1dc2cc7114e649a78ebc) (originally in base, though it migrated to transformers later). However, it was soon discovered that when there were many functions with forms like `IO a -> IO a`, which we wanted to convert into `m a -> m a`; MonadIO had no provision for handling arguments in the *negative* position of functions. This was particularly troublesome in the case of exception handling, where these higher-order functions were *primitive*. Thus, the community began searching for a new type class which captured more of IO.

While the semantics of lift were well understood (by the transformer laws), it wasn’t clear what a more powerful mechanism looked like. So, early attacks at the problem took the approach of picking a few distinguished functions which we wanted, placing them in a typeclass, and manually implementing lifted versions of them. This lead to the development of the already existing `MonadError` class into a more specialized `MonadCatchIO` class. However, Anders Kaseorg realized that there was a common pattern to the implementation of the lifted versions of these functions, which he factored out into the `MonadMorphIO` class. This approach was refined into the `MonadPeelIO` and `MonadTransControlIO` typeclasses. However, only `MonadError` was in the core, and it had failed to achieve widespread acceptance due to some fundamental problems.

I believe it is important and desirable for the community of library writers to converge on one of these type classes, for the primary reason that it is important for them to implement exception handling properly, a task which is impossible to do if you want to export an interface that requires only `MonadIO`. I fully expected monad-control to be the “winner”, being the end at a long lineage of type classes. However, I think it would be more accurate to describe `MonadError` and `MonadCatchIO` as one school of thought, and `MonadMorphIO`, `MOnadPeelIO` and `MonadTransControlIO` as another.

In this blog post, I’d like to examine and contrast these two schools of thought. A type class is an interface: it defines operations that some object supports, as well as laws that this object abides by. The utility in a type class is both in its generality (the ability to support multiple implementations with one interface) as well as its precision (the restriction on permissible implementations by *laws*, making it easier to reason about code that uses an interface). This is the essential tension: and these two schools have very different conclusions about how it should be resolved.

### Modelling exceptions

This general technique can be described as picking a few functions to generalize in a type class. Since a type class with less functions is preferable to one with more (for generality reasons), `MonadError` and `MonadCatchIO` have a very particular emphasis on exceptions:

```
class (Monad m) => MonadError e m | m -> e where
  throwError :: e -> m a
  catchError :: m a -> (e -> m a) -> m a

class MonadIO m => MonadCatchIO m where
  catch   :: Exception e => m a -> (e -> m a) -> m a
  block   :: m a -> m a
  unblock :: m a -> m a

```

Unfortunately, these functions are marred by some problems:

*   MonadError encapsulates an abstract notion of errors which does not necessarily include asynchronous exceptions. That is to say, `catchError undefined h` will not necessarily run the exception handler `h`.
*   MonadError is inadequate for robust handling of asynchronous exceptions, because it does not contain an interface for `mask`; this makes it difficult to write bracketing functions robustly.
*   MonadCatchIO explicitly only handles asynchronous exceptions, which means any pure error handling is not handled by it. This is the “finalizers are sometimes skipped” problem.
*   MonadCatchIO, via the `MonadIO` constraint, requires the API to support lifting arbitrary IO actions to the monad (whereas a monad designer may create a restricted IO backed monad, limiting what IO actions the user has access to.)
*   MonadCatchIO exports the outdated `block` and `unblock` function, while modern code should use `mask` instead.
*   MonadCatchIO exports an instance for the `ContT` transformer. However, continuations and exceptions are [known to have nontrivial interactions](http://hpaste.org/56921) which require extra care to handle properly.

In some sense, `MonadError` is a non-sequitur, because it isn’t tied to IO at all; perfectly valid instances of it exist for non-IO backed monads as well. `MonadCatchIO` is closer; the latter three points are not fatal ones could be easily accounted for:

```
class MonadException m where
  throwM  :: Exception e => e -> m a
  catch   :: Exception e => m a -> (e -> m a) -> m a
  mask    :: ((forall a. m a -> m a) -> m b) -> m b

```

(With a removal of the `ContT` instance.) However, the “finalizers are sometimes skipped” problem is a bit more problematic. In effect, it is the fact that there may exist zeros which a given instance of `MonadCatchIO` may not know about. It has been argued that [these zeros are none of MonadCatchIO’s business](http://www.haskell.org/pipermail/haskell-cafe/2010-October/085079.html); one inference you might draw from this is that if you have short-circuiting which you would like to respect finalizers installed using `MonadException`, it should be implemented using asynchronous exceptions. In other words, `ErrorT` is a bad idea.

However, there is another perspective you can take: `MonadException` is not tied just to asynchronous exceptions, but any zero-like value which obeys the same laws that exceptions obey. The semantics of these exceptions are described in the paper [Asynchronous Exceptions in Haskell](http://community.haskell.org/~simonmar/papers/async.pdf). They specify exactly the interaction of masking, throw and catch, as well as how interrupts can be introduced by other threads. In this view, whether or not this behavior is prescribed by the RTS or by passing pure values around is an implementation detail: as long as an instance is written properly, zeros will be properly handled. This also means that it is no longer acceptable to provide a `MonadException` instance for `ErrorT e`, unless we also have an underlying `MonadException` for the inner monad: we can’t forget about exceptions on the lower layers!

There is one last problem with this approach: once the primitives have been selected, huge swaths of the standard library have to be redefined by “copy pasting” their definitions but having them refer to the generalized versions. This is a significant practical hurdle for implementing a library based on this principle: it’s simply not enough to tack a `liftIO` to the beginning of a function!

I think an emphasis on the semantics of the defined type class will be critical for the future of this lineage of typeclasses; this is an emphasis that hasn’t really existed in the past. From this perspective, we define with our typeclass not only a way to access otherwise inaccessible functions in IO, but also how these functions should behave. We are, in effect, modeling a subset of IO. I think Conal Elliott [would be proud](http://conal.net/blog/posts/notions-of-purity-in-haskell).

> There is a [lively debate](http://comments.gmane.org/gmane.comp.lang.haskell.cafe/93834) going on about extensions to the original semantics of asynchronous exceptions, allowing for the notion of “recoverable” and “unrecoverable” errors. (It’s nearer to the end of the thread.)

### Threading pure effects

This technique can be described as generalizing the a common implementation technique which was used to implement many of the original functions in `MonadCatchIO`. These are a rather odd set of signatures:

```
class Monad m => MonadMorphIO m where
  morphIO :: (forall b. (m a -> IO b) -> IO b) -> m a

class MonadIO m => MonadPeelIO m where
  peelIO :: m (m a -> IO (m a))

class MonadBase b m => MonadBaseControl b m | m -> b where
  data StM m :: * -> *
  liftBaseWith :: (RunInBase m b -> b a) -> m a
  restoreM :: StM m a → m a
type RunInBase m b = forall a. m a -> b (StM m a)

```

The key intuition behind these typeclasses is that they utilize *polymorphism* in the IO function that is being lifted in order to *thread the pure effects* of the monad stack on top of IO. You can see this as the universal quantification in `morphIO`, the return type of `peelIO` (which is `IO (m a)`, not `IO a`), and the `StM` associated type in `MonadBaseControl`. For example, `Int -> StateT s IO a`, is equivalent to the type `Int -> s -> IO (s, a)`. We can partially apply this function with the current state to get `Int -> IO (s, a)`; it should be clear then that as long as the IO function we’re lifting lets us smuggle out arbitrary values, we can smuggle out our updated state and reincorporate it when the lifted function finishes. The set of functions which are amenable to this technique are precisely the ones for which this threaded is possible.

As I described in [this post](http://blog.ezyang.com/2012/01/monadbasecontrol-is-unsound/), this means that you won’t be able to get any transformer stack effects if they aren’t returned by the function. So perhaps a better word for MonadBaseControl is not that it is unsound (although it does admit strange behavior) but that it is incomplete: it cannot lift all IO functions to a form where the base monad effects and the transformer effects always occur in lockstep.

This has some interesting implications. For example, this forgetfulness is in fact precisely the reason why a lifted bracketing function will always run no matter if there are other zeros: `finally` by definition is only aware of asynchronous exceptions. This makes monad-control lifted functions very explicitly only handling asynchronous exceptions: a lifted `catch` function will not catch an ErrorT zero. However, if you manually implement `finally` using lifted versions of the more primitive functions, finalizers may be dropped.

It also suggests an alternate implementation strategy for monad-control: rather than thread the state through the return type of a function, it could instead be embedded in a hidden IORef, and read out at the end of the computation. In effect, we would like to *embed* the semantics of the pure monad transformer stack inside IO. Some care must be taken in the `forkIO` case, however: the IORefs need to also be duplicated appropriately, in order to maintain thread locality, or MVars used instead, in order to allow coherent non-local communication.

It is well known that MonadBaseControl does not admit a reasonable instance for ContT. Mikhail Vorozhtsov has argued that this is too restrictive. The difficulty is that while unbounded continuations do not play nice with exceptions, limited use of continuation passing style can be combined with exceptions in a sensible way. Unfortunately, monad-control makes no provision for this case: the function it asks a user to implement is too powerful. It seems the typeclasses explicitly modeling a subset of IO are, in some sense, more general! It also highlights the fact that these type classes are first and foremost driven by an abstraction of a common implementation pattern, rather than any sort of semantics.

### Conclusion

I hope this essay has made clear why I think of MonadBaseControl as an implementation strategy, and not as a reasonable *interface* to program against. MonadException is a more reasonable interface, which has a semantics, but faces significant implementation hurdles.