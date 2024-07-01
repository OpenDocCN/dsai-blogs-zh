<!--yml
category: 未分类
date: 2024-07-01 18:17:53
-->

# The Haskell Heap : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/04/the-haskell-heap/](http://blog.ezyang.com/2011/04/the-haskell-heap/)

## The Haskell Heap

The Haskell heap is a rather strange place. It’s not like the heap of a traditional, strictly evaluated language...

...which contains a lot of junk! (Plain old data.)

In the Haskell heap, every item is wrapped up nicely in a box: the Haskell heap is a heap of *presents* (thunks).

When you actually want what’s inside the present, you *open it up* (evaluate it).

Presents tend to have names, and sometimes when you open a present, you get a *gift card* (data constructor). Gift cards have two traits: they have a name (the `Just` gift card or the `Right` gift card), and they tell you where the rest of your presents are. There might be more than one (the tuple gift card), if you’re a lucky duck!

But just as gift cards can lie around unused (that’s how the gift card companies make money!), you don’t have to redeem those presents.

Presents on the Haskell heap are rather mischievous. Some presents explode when you open them, others are haunted by ghosts that open other presents when disturbed.

Understanding what happens when you open a present is key to understanding the time and space behavior of Haskell programs.

In this series, Edward makes a foray into the webcomic world in order to illustrate the key operational concepts of evaluation in a lazily evaluated language. I hope you enjoy it!

Next time: [Evaluation on the Haskell Heap](http://blog.ezyang.com/2011/04/evaluation-on-the-haskell-heap/)

> *Technical notes.* Technically speaking, this series should be “The GHC Heap.” However, I’ll try to avoid as many GHC-isms as possible, and simply offer a metaphor for operationally reasoning about any kind of lazy language. Originally, the series was titled “Bomberman Teaches Lazy Evaluation,” but while I’ve preserved the bomb metaphor for thunks that error or don’t terminate, I like the present metaphor better: it in particular captures several critical aspects of laziness: it captures the evaluated/non-evaluated distinction and the fact that once a present is opened, it’s opened for everyone. The use of the term “boxed” is a little suggestive: indeed, boxed or *lifted* values in GHC are precisely the ones that can be nonterminating, whereas unboxed values are more akin to what you’d see in C’s heap. However, languages like Java also use the term boxed to refer to primitive values that look like objects. For clarity’s sake, we won’t be using the term boxed from now on (indeed, we won’t mention unboxed types).

This work is licensed under a [Creative Commons Attribution-ShareAlike 3.0 Unported License](http://creativecommons.org/licenses/by-sa/3.0/).