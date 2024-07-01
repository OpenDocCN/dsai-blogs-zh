<!--yml
category: 未分类
date: 2024-07-01 18:17:15
-->

# PEPM’14: The HERMIT in the Stream : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/01/pepm14-the-hermit-in-the-stream/](http://blog.ezyang.com/2014/01/pepm14-the-hermit-in-the-stream/)

POPL is almost upon us! I’ll be [live-Tumblr-ing](http://ezyang.tumblr.com/) it when the conference comes upon us proper, but in the meantime, I thought I’d write a little bit about one paper in the colocated PEPM'14 program: [The HERMIT in the Stream](http://www.ittc.ku.edu/~afarmer/concatmap-pepm14.pdf), by Andrew Farmer, Christian Höner zu Sierdissen and Andy Gill. This paper presents an implementation of an optimization scheme for fusing away use of the concatMap combinator in the [stream fusion framework](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.104.7401), which was developed using the [HERMIT optimization framework](http://www.ittc.ku.edu/csdl/fpg/software/hermit.html). The HERMIT project has been chugging along for some time now, and a stream of papers of various applications of the framework have been trickling out (as anyone who was at the Haskell implementors workshop can attest.)

“But wait,” you may ask, “don’t we already have [stream fusion](http://hackage.haskell.org/package/stream-fusion)?” You’d be right: but while stream fusion is available as a library, it has not replaced the default fusion system that ships with GHC: foldr/build fusion. What makes fusion scheme good? One important metric is the number of list combinators it supports. Stream fusion nearly dominates foldr/build fusion, except for the case of concatMap, a problem which has resisted resolution for seven years and has prevented GHC from switching to using stream fusion as its default.

As it turns out, we’ve known how to optimize concatMap for a long time; [Duncan Coutts gave a basic outline in his thesis.](http://community.haskell.org/~duncan/thesis.pdf) The primary contribution of this paper was a [prototype implementation of this optimization](https://github.com/xich/hermit-streamfusion), including an elucidation of the important technical details (increasing the applicability of the original rule, necessary modifications to the simplifier, and rules for desugaring list comprehensions). The paper also offers some microbenchmarks and real world benchmarks arguing for the importance of optimizing concatMap.

I was glad to see this paper, since it is an important milestone on the way to replacing foldr/build fusion with stream fusion in the GHC standard libraries. It also seems the development of this optimization was greatly assisted by the use HERMIT, which seems like a good validation for HERMIT (though the paper does not go into very much detail about how HERMIT assisted in the process of developing this optimization).

There is something slightly unsatisfying with the optimization as stated in the paper, which can be best articulated by considering the paper from the perspective of a prospective implementor of stream fusion. She has two choices:

*   She can try to use the HERMIT system directly. However, HERMIT induces a 5-20x compilation slowdown, which is quite discouraging for real use. This slowdown is probably not fundamental, and will be erased in due time, but that is certainly not the case today. The limited implementation of stream fusion in the prototype (they don’t implement all of the combinators, just enough so they could run their numbers) also recommends against direct use of the system.
*   She can directly incorporate the rules as stated into a compiler. This would require special-case code to apply the non-semantics preserving simplifications only to streams, and essentially would require a reimplementation of the system, with the guidance offered by this paper. But this special-case code is of limited applicability beyond its utility for concatMap, which is a negative mark.

So, it seems, at least from the perspective of an average GHC user, we will have to wait a bit longer before stream fusion is in our hands. Still, I agree that the microbenchmarks and [ADPFusion](http://hackage.haskell.org/package/ADPfusion) case study show the viability of the approach, and the general principle of the novel simplification rules seems reasonable, if a little ad hoc.

One note if you’re reading the nofib performance section: the experiment was done comparing their system to foldr/build, so the delta is mostly indicative of the benefit of stream fusion (in the text, they point out which benchmarks benefitted the most from concatMap fusion). Regardless, it’s a pretty cool paper: check it out!