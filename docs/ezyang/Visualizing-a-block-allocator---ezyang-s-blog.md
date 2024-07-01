<!--yml
category: 未分类
date: 2024-07-01 18:17:16
-->

# Visualizing a block allocator : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/10/visualizing-a-block-allocator/](http://blog.ezyang.com/2013/10/visualizing-a-block-allocator/)

## Visualizing a block allocator

GHC’s [block allocator](http://ghc.haskell.org/trac/ghc/wiki/Commentary/Rts/Storage/BlockAlloc) is a pretty nifty piece of low-level infrastructure. It offers a much more flexible way of managing a heap, rather than trying to jam it all in one contiguous block of memory, and is probably something that should be of general interest to anyone who is implementing low-level code like a runtime. The core idea behind it is quite old (BIBOP: Big Bag of Pages), and is useful for any situation where you have a number of objects that are tagged with the same descriptor, and you don’t want to pay the cost of the tag on each object.

Managing objects larger than pages is a bit tricky, however, and so I wrote a document visualizing the situation to help explain it to myself. I figured it might be of general interest, so you can get it here: [http://web.mit.edu/~ezyang/Public/blocks.pdf](http://web.mit.edu/~ezyang/Public/blocks.pdf)

Some day I’ll convert it into wikiable form, but I don’t feel like Gimp'ing the images today...