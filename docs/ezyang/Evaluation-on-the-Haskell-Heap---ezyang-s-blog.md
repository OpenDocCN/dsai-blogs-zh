<!--yml
category: 未分类
date: 2024-07-01 18:17:54
-->

# Evaluation on the Haskell Heap : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/04/evaluation-on-the-haskell-heap/](http://blog.ezyang.com/2011/04/evaluation-on-the-haskell-heap/)

*The Ghost of the Christmas Present*

In today’s post, we’ll do a brief survey of the various things that can happen when you open haunted presents in the Haskell heap. Asides from constants and things that have already been evaluated, mostly everything on the Haskell heap is haunted. The real question is what the ghost haunting the present does.

In the simplest case, almost nothing!

Unlike gift-cards, you have to open the next present (Haskell doesn’t let you evaluate a thunk, and then decide not to follow the indirection...)

More commonly, the ghost was *lazy* and, when woken up, has to open other presents to figure out what was in your present in the first place!

Simple primitive operations need to open all of the presents involved.

But the ghost may also open another present for no particular reason...

or execute some IO...

Note that any presents he opens may trigger more ghosts:

Resulting in a veritable ghost jamboree, all to open one present!

The fact that opening a present (thunk) can cause such a cascading effect is precisely what makes the timing of lazy evaluation surprising to people who are used to all of the objects in their heap being unwrapped (evaluated) already. So the key to getting rid of this surprise is understanding when a ghost will decide it needs to unwrap a present (strictness analysis) and whether or not your presents are unwrapped already (amortized analysis).

Last time: [The Haskell Heap](http://blog.ezyang.com/2011/04/the-haskell-heap/)

Next time: [IO evaluates the Haskell Heap](http://blog.ezyang.com/2011/04/io-evaluates-the-haskell-heap/)

This work is licensed under a [Creative Commons Attribution-ShareAlike 3.0 Unported License](http://creativecommons.org/licenses/by-sa/3.0/).