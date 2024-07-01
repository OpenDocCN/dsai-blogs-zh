<!--yml
category: 未分类
date: 2024-07-01 18:17:57
-->

# On cargo culting and hacking : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/02/on-cargo-culting-and-hacking/](http://blog.ezyang.com/2011/02/on-cargo-culting-and-hacking/)

## On cargo culting and hacking

*two inflammatory vignettes*

The term *to cargo cult* is one with derogatory connotations: it indicates the act of imitating the superficial exterior without actually understanding the underlying causal structure of the situation. The implication is that one should try to understand what one is doing, before doing it. There is, however, an ounce of truth in the practice of cargo culting: when you are in a situation in which you legitimately do not know what’s going on (e.g. the context of an experiment), it is safest to preserve as many superficial traits as possible, in case a “superficial” trait in fact has a deep, non-obvious connection to the system being studied. But in this regard, beneficial “cargo culting” is nothing like the islanders throwing up airstrips in hopes of attracting planes—understanding what conditions are applicable for this treatment is often the mark of experience: the novice ignores conditions that should be preserved and does not know how to probe deeper.

* * *

*Hacking* is the art of accidental generalization. It is developing a program under a single set of conditions (a hard-coded test input, a particular directory structure, a single URL) and (perhaps) hoping it will work in the more general case. Anything that gets in the way of specificity—proofs, types, security, verbosity, edge-cases, thinking—is the enemy for pure creation. It is the art of the present, and much profit and pleasure can be derived from it. It is the art of laser precision, each problem dealt with as it comes. It is an art that becomes more acceptable engineering practice with experience: one develops little internal censors that continually pipe up with mental flags where you need to give a little extra to make the generalization work. Novices are recommended to bring their check-lists along.