<!--yml
category: 未分类
date: 2024-07-01 18:17:38
-->

# Why iteratees are hard to understand : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/01/why-iteratees-are-hard-to-understand/](http://blog.ezyang.com/2012/01/why-iteratees-are-hard-to-understand/)

There are two primary reasons why the low-level implementations of iteratees, enumerators and enumeratees tend to be hard to understand: *purely functional implementation* and *inversion of control*. The strangeness of these features is further exacerbated by the fact that users are encouraged to think of iteratees as sinks, enumerators as sources, and enumeratees as transformers. This intuition works well for clients of iteratee libraries but confuses people interested in digging into the internals.

In this article, I’d like to explain the strangeness imposed by the *purely functional implementation* by comparing it to an implementation you might see in a traditional, *imperative*, object-oriented language. We’ll see that concepts which are obvious and easy in an imperative setting are less-obvious but only slightly harder in a purely functional setting.

### Types

*The following discussion uses nomenclature from the enumerator library, since at the time of the writing it seems to be the most popular implementation of iteratees currently in use.*

The fundamental entity behind an iteratee is the `Step`. The usual intuition is that is represents the “state” of an iteratee, which is either done or waiting for more input. But we’ve cautioned against excessive reliance on metaphors, so let’s look at the types instead:

```
data Step a b = Continue (Stream a -> m (Step a b)) | Yield b
type Iteratee     a b =                  m (Step a b)
type Enumerator   a b = Step a b ->      m (Step a b)
type Enumeratee o a b = Step a b -> Step o (Step a b)

```

I have made some extremely important simplifications from the enumerator library, most of important of which is explicitly writing out the `Step` data type where we would have seen an `Iteratee` instead and making `Enumeratee` a pure function. The goal of the next three sections is to explain what each of these type signatures means; we’ll do this by analogy to the imperative equivalents of iteratees. The imperative programs should feel intuitive to most programmers, and the hope is that the pure encoding should only be a hop away from there.

### Step/Iteratee

We would like to design an object that is either waiting for input or finished with some result. The following might be a proposed interface:

```
interface Iteratee<A,B> {
  void put(Stream<A>);
  Maybe<B> result();
}

```

This implementation critically relies on the identity of an object of type `Iteratee`, which maintains this identity across arbitrary calls to `put`. For our purposes, we need to translate `put :: IORef s -> Stream a -> IO ()` (first argument is the Iteratee) into a purely functional interface. Fortunately, it’s not too difficult to see how to do this if we understand how the `State` monad works: we replace the old type with `put :: s -> Stream a -> s`, which takes the original state of the iteratee (`s = Step a b`) and some input, and transforms it into a new state. The final definition `put :: Step a b -> Stream a -> m (Step a b)` also accomodates the fact that an iteratee may have some other side-effects when it receives data, but we are under no compulsion to use this monad instance; if we set it to the identity monad our iteratee has no side effects (`StateT` may be the more apt term here). In fact, this is precisely the accessor for the field in the `Continue` constructor.

### Enumerator

We would like to design an object that takes an iteratee and feeds it input. It’s pretty simple, just a function which mutates its input:

```
void Enumerator(Iteratee<A,B>);

```

What does the type of an enumerator have to say on the matter?

```
type Enumerator a b = Step a b -> m (Step a b)

```

If we interpret this as a state transition function, it’s clear that an enumerator is a function that *transforms* an iteratee from one state to another, much like the `put`. Unlike the `put`, however, the enumerator takes no input from a stream and may possibly cause multiple state transitions: it’s a big step, with all of the intermediate states hidden from view.

The nature of this transformation is not specified, but a common interpretation is that the enumerator repeatedly feeds an input to the continuation in step. An execution might unfold to something like this:

```
-- s :: Step a b
-- x0, x1 :: Stream a
case s of
    Yield r -> return (Yield r)
    Continue k -> do
        s' <- k x0
        case s' of
            Yield r -> return (Yield r)
            Continue k -> do
                s'' <- k x1
                return s''

```

Notice that our type signature is not:

```
type Enumerator a b = Step a b -> m ()

```

as the imperative API might suggest. Such a function would manage to run the iteratee (and trigger any of its attendant side effects), but we’d lose the return result of the iteratee. This slight modification wouldn’t do either:

```
type Enumerator a b = Step a b -> m (Maybe b)

```

The problem here is that if the enumerator didn’t actually manage to finish running the iteratee, we’ve lost the end state of the iteratee (it was never returned!) This means you can’t concatenate enumerators together.

> It should now be clear why I have unfolded all of the `Iteratee` definitions: in the `enumerator` library, the simple correspondence between enumerators and side-effectful state transformers is obscured by an unfortunate type signature:
> 
> ```
> type Enumerator a b = Step a b -> Iteratee a b
> 
> ```
> 
> Oleg’s original treatment is much clearer on this matter, as he defines the steps themselves to *be* the iteratees.

### Enumeratee

At last, we are now prepared to tackle the most complicated structure, the enumeratee. Our imperative hat tells us a class like this might work:

```
interface Enumeratee<O,A,B> implements Iteratee<O,B> {
    Enumeratee(Iteratee<A,B>);
    bool done();
    // inherited from Iteratee<O,B>
    void put(Stream<O>);
    Maybe<B> result();
}

```

Like our original `Iteratee` class, it sports a `put` and `result` operation, but upon construction it wraps another `Iteratee`: in this sense it is an *adapter* from elements of type `O` to elements of type `A`. A call to the outer `put` with an object of type `O` may result in zero, one or many calls to put with an object of type `A` on the inside `Iteratee`; the call to `result` is simply passed through. An `Enumeratee` may also decide that it is “done”, that is, it will never call `put` on the inner iteratee again; the `done` method may be useful for testing for this case.

Before we move on to the types, it’s worth reflecting what stateful objects are involved in this imperative formulation: they are the outer `Enumeratee` and the inner `Iteratee`. We need to maintain *two*, not *one* states. The imperative formulation naturally manages these for us (after all, we still have access to the inner iteratee even as the enumeratee is running), but we’ll have to manually arrange for this is in the purely functional implementation.

Here is the type for `Enumeratee`:

```
type Enumeratee o a b = Step a b -> Step o (Step a b)

```

It’s easy to see why the first argument is `Step a b`; this is the inner iteratee that we are wrapping around. It’s less easy to see why `Step o (Step a b)` is the correct return type. Since our imperative interface results in an object which implements the `Iteratee<O,B>` interface, we might be tempted to write this signature instead:

```
type Enumeratee o a b = Step a b -> Step o b

```

But remember; we need to keep track of two states! We have the outer state, but what of the inner one? In a situation similar reminiscent of our alternate universe `Enumerator` earlier, the state of the inner iteratee is lost forever. Perhaps this is not a big deal if this enumeratee is intended to be used for the rest of the input (i.e. `done` always returns false), but it is quite important if we need to stop using the `Enumeratee` and then continue operating on the stream `Step a b`.

By the design of iteratees, we can only get a result out of an iteratee once it finishes. This forces us to return the state in the second parameter, giving us the final type:

```
type Enumeratee o a b = Step a b -> Step o (Step a b)

```

“But wait!” you might say, “If the iteratee only returns a result at the very end, doesn’t this mean that the inner iteratee only gets updated at the end?” By the power of *inversion of control*, however, this is not the case: as the enumeratee receives values and updates its own state, it also executes and updates the internal iteratee. The intermediate inner states exist; they are simply not accessible to us. (This is in contrast to the imperative version, for which we can peek at the inner iteratee!)

> Another good question is “Why does the `enumerator` library have an extra monad snuck in `Enumeratee`?”, i.e.
> 
> ```
> Step a b -> m (Step o (Step a b))
> 
> ```
> 
> My understanding is that the monad is unnecessary, but may be useful if your `Enumeratee` needs to be able to perform a side-effect prior to receiving any input, e.g. for initialization.

### Conclusion

Unfortunately, I can’t claim very much novelty here: all of these topics are covered in [Oleg’s notes](http://okmij.org/ftp/Haskell/Iteratee/IterateeIO-talk-notes.pdf). I hope, however, that this presentation with reference to the imperative analogues of iteratees makes the choice of types clearer.

There are some important implications of using this pure encoding, similar to the differences between using IORefs and using the state monad:

*   Iteratees can be forked and run on different threads while preserving isolation of local state, and
*   Old copies of the iteratee state can be kept around, and resumed later as a form of backtracking (swapping a bad input for a newer one).

These assurances would not be possible in the case of simple mutable references. There is one important caveat, however, which is that while the pure component of an iteratee is easily reversed, we cannot take back any destructive side-effects performed in the monad. In the case of forking, this means any side-effects must be atomic; in the case of backtracking, we must be able to rollback side-effects. As far as I can tell, the art of writing iteratees that take advantage of this style is not well studied but, in my opinion, well worth investigating. I’ll close by noting that one of the theses behind the new conduits is that purity is not important for supporting most stream processing. In my opinion, the jury is still out.