<!--yml
category: 未分类
date: 2024-07-01 18:18:21
-->

# Association maps in mit-scheme : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/04/association-maps-in-mit-scheme/](http://blog.ezyang.com/2010/04/association-maps-in-mit-scheme/)

## Association maps in mit-scheme

I recently some did some benchmarking of persistent data structures in mit-scheme for my UROP. There were a few questions we were interested in:

1.  For what association sizes does a fancier data structure beat out your plain old association list?
2.  What is the price of persistence? That is, how many times slower are persistent data structures as compared to your plain old hash table?
3.  What is the best persistent data structure?

These are by no means authoritative results; I still need to carefully comb through the harness and code for correctness. But they already have some interesting implications, so I thought I'd share. The implementations tested are:

All implementations use `eq?` for key comparison.

Unsurprisingly, assoc beats out everyone else, since all it has to do is a simple cons. However, there are some strange spikes at regular intervals, which I am not sure of the origin; it might be the garbage collector kicking in.

Of course, you pay back the cheap updates in assoc with a linear lookup time; the story also holds true for weight-balanced trees, which have fast inserts but the slowest lookups.

The hamt really flies when the key isn't present, even beating out hash-tables until 15 elements or so.

Source code for running the benchmarks, our home-grown implementations, and graphing can be found at the [scheme-hamt repository](http://github.com/ezyang/scheme-hamt).