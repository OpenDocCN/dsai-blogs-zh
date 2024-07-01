<!--yml
category: 未分类
date: 2024-07-01 18:17:18
-->

# Cost semantics for STG in modern GHC : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/09/cost-semantics-for-stg-in-modern-ghc/](http://blog.ezyang.com/2013/09/cost-semantics-for-stg-in-modern-ghc/)

## Cost semantics for STG in modern GHC

One of the problems with academic publishing is that it’s hard to keep old papers up-to-date. This is the certainly case for this [1995 Sansom paper on profiling non-strict, higher-order functional languages](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.43.6277). While the basic ideas of the paper still hold, the actual implementation of cost centers in GHC has changed quite a bit, perhaps the most dramatic change being the introduction of cost center stacks. So while the old paper is good for giving you the basic idea of how profiling in GHC works, if you really want to know the details, the paper offers little guidance.

So what do you do when your cost semantics are out-of-date? Why, update them of course! I present an [updated cost-semantics for STG in modern GHC (PDF)](https://github.com/ezyang/stg-spec/raw/master/stg-spec.pdf) ([GitHub](https://github.com/ezyang/stg-spec)). Eventually these will go in the GHC repository proper, alongside [core-spec](http://typesandkinds.wordpress.com/2012/12/03/a-formalization-of-ghcs-core-language/) which is a similar document for Core. However, I haven't done any proofs with these semantics yet, so they are probably a little buggy.

Despite the lack of proofs, the formalization has been helpful already: I’ve already spotted one bug in the current implementation (remarked upon in the document). I’ve also identified a potential refactoring based on the way the rules are currently setup. Please let me know about any other bugs you find!