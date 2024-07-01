<!--yml
category: 未分类
date: 2024-07-01 18:17:53
-->

# IO evaluates the Haskell Heap : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/04/io-evaluates-the-haskell-heap/](http://blog.ezyang.com/2011/04/io-evaluates-the-haskell-heap/)

In today’s post, we focus on *you*, the unwitting person rooting around the Haskell heap to open a present. After all, presents in the Haskell heap do not spontaneously unwrap themselves.

Someone has to open the first present.

If the Haskell heap doesn’t interact with the outside world, no presents need to be opened: thus IO functions are the ones that will open presents. What presents they will open is not necessarily obvious for many functions, so we’ll focus on one function that makes it particularly obvious: `evaluate`. Which tells you to...

...open a present.

If you get a primitive value, you’re done. But, of course, you might get a gift card (constructor):

Will you open the rest of the presents? Despite that deep, dissatisfaction inside you, the answer is no. `evaluate` only asks you to open one present. If it’s already opened, there’s nothing for you to do.

> Advanced tip: If you want to evaluate more things, make a present containing a ghost who will open those things for you! A frequently used example of this when lazy IO is involved was `evaluate (length xs)`, but don’t worry too much if you don’t understand that yet: I haven’t actually said how we make presents yet!

Even though we’re only opening one present, many things can happen, as described in the last post. It could execute some IO...

This is our direct window into evaluation as it evolves: when we run programs normally, we can’t see the presents being opened up; but if we ask the ghost to also shout out when it is disturbed, we get back this information. And in fact, this is precisely what `Debug.Trace` does!

There are other ways to see what evaluation is going on. A present could blow up: this is the exploding booby-trapped present, also known as “bottom”.

Perhaps the explosion was caused by an `undefined` or `error "Foobar"`.

Boom.

* * *

We’ll end on a practical note. As we’ve mentioned, you can only be sure that a present has been opened if you’ve explicitly asked for it to be opened from IO. Otherwise, ghosts might play tricks on you. After all, you can’t actually *see* the Haskell heap, so there’s no way to directly tell if a present has been opened or not.

If you’re unsure when a thunk is being evaluated, add a trace statement to it. If ghosts are being lazy behind your back, the trace statement will never show up.

More frequently, however, the trace statement will show up; it’ll just be later than you expect (the ghosts may be lazy, but they’ll eventually get the job done.) So it’s useful to prematurely terminate your program or add extra print statements demarcating various stages of your program.

Last time: [Evaluation on the Haskell Heap](http://blog.ezyang.com/2011/04/evaluation-on-the-haskell-heap/)

Next time: [Implementing the Haskell Heap in Python, v1](http://blog.ezyang.com/2011/04/implementing-the-haskell-heap-in-python-v1/)

*Technical notes.* Contrary to what I’ve said earlier, there’s no theoretical reason why we couldn’t spontaneously evaluate thunks on the heap: this evaluation approach is called *speculative evaluation.* Somewhat confusingly, IO actions themselves can be thunks as well: this corresponds to passing around values of `IO a` without actually “running” them. But since I’m not here to talk about monads, I’ll simply ignore the existence of presents that contain `IO` actions—they work the same way, but you have to keep the levels of indirection straight. And finally, of course infinite loops also count as bottom, but the image of opening one present for the rest of eternity is not as flashy as an exploding present.

This work is licensed under a [Creative Commons Attribution-ShareAlike 3.0 Unported License](http://creativecommons.org/licenses/by-sa/3.0/).