<!--yml
category: 未分类
date: 2024-07-01 18:17:40
-->

# Obviously Correct : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/10/obviously-correct/](http://blog.ezyang.com/2011/10/obviously-correct/)

What do automatic memory management, static types and purity have in common? They are methods which take advantage of the fact that we can make programs *obviously correct* (for some partial definition of correctness) upon visual inspection. Code using automatic memory management is *obviously correct* for a class of memory bugs. Code using static types is *obviously correct* for a class of type bugs. Code using purity (no mutable references or side effects) is *obviously correct* for a class of concurrency bugs. When I take advantage of any of these techniques, I don’t have to *prove* my code has no bugs: it just is, automatically!

Unfortunately, there's a catch. What all of these “obviously correct” methodologies ask you do is to sacrifice varying degrees of expressiveness at their altar. No more pointer tricks. No more playing fast and loose with data representation. No more mutation. If this expressiveness was something most people really didn’t want anyway (e.g. memory management), it is happily traded away. But if it’s something they *want*, well, as language designers, we’re making it harder for people to do things that they want to do, and it shouldn’t surprise us when they grab their torches and pitchforks and storm the ivory tower, assertions about correctness and maintainability be damned.

It seems to me that we must fight fire with fire: if we’re going to take away features, we better be giving them compelling new features. With static types you also get pattern matching, QuickCheck style property testing, and performance benefits. With purity, you get software transactional memory and speculative evaluation. Discovering and implementing more of these “killer apps” is the key to adoption. (Some research that I’m currently doing with Adam Chlipala is leveraging purity to offer automatic caching for web applications. It’s not much, but I think it’s in the right direction.)

I still have a fanatical devotion to correctness. But these days, I suspect that for most people, it’s something bitter, like medicine, to be taken with some better tasting features. That’s fine. Our challenge, as programming language researchers, is to exploit correctness to bring tangible benefits now, rather than nebulous maintainability benefits later.

*Thanks Nelson Elhage and Keegan McAllister for their comments.*

* * *

*Postscript: Performance of static types versus dynamic types.* An earlier draft of this post pointed at [Quora’s decision to move to Scala from Python](http://www.quora.com/Is-the-Quora-team-considering-adopting-Scala-Why) as a clear indicator of this fact. Unfortunately, as several pre-readers pointed out, there are too many confounding factors to make this claim definitive: CPython was never explicitly engineered for performance, whereas the JVM had decades of work poured into it. So I’ll have to leave you with a more theoretical argument for the performance of static types: the optimization techniques of runtime just-in-time compilers for dynamic compilers involves identifying sections of code which are actually statically typed, and compiling them into the form a static compiler will. So, if you know this information ahead of time, you will always do better than if you know this information later: it's only a question of degree. (Of course, this doesn't address the possibility that JIT can identify information that would have been difficult to determine statically.)

*Postscript: Shared transactional memory.* Joe Duffy had a great [retrospective on transactional memory](http://www.bluebytesoftware.com/blog/2010/01/03/ABriefRetrospectiveOnTransactionalMemory.aspx) and the experience he had attempting to implement it for Microsoft’s stack. And despite a great enthusiasm for this idea, it’s interesting to note this quote:

> Throughout all of this, we searched and searched for the killer TM app. It’s unfair to pin this on TM, because the industry as a whole still searches for a killer concurrency app. But as we uncovered more successes in the latter, I became less and less convinced that the killer concurrency apps we will see broadly deployed in the next 5 years needed TM. Most enjoyed natural isolation, like embarrassingly parallel image processing apps. If you had sharing, you were doing something wrong.

Richard Tibbetts points out that concurrency is often addressed at an architectural level *lower* than what most working programmers want to deal with, and so while STM is a killer application for those platforms, most developers don't want to think about concurrency at all.