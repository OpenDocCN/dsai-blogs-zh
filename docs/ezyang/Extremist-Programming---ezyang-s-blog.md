<!--yml
category: 未分类
date: 2024-07-01 18:17:24
-->

# Extremist Programming : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/11/extremist-programming/](http://blog.ezyang.com/2012/11/extremist-programming/)

*Functions are awesome. What if we made a PL that only had functions?*

*Objects are awesome. What if we made a PL where everything was an object?*

*Lazy evaluation is awesome. What if we made a PL where every data type was lazy?*

**Extremist programming** (no relation to extreme programming) is the act of taking some principle, elevating it above everything else and applying it everywhere. After the dust settles, people often look at this extremism and think, “Well, that was kind of interesting, but using X in Y was clearly inappropriate. You need to use the right tool for the job!”

Here’s the catch: sometimes you *should* use the *wrong* tool for the job—because it might be the right tool, and you just don’t know it yet. If you aren’t trying to use functions everywhere, you might not realize the utility of functions that take functions as arguments [1] or cheap lambdas [2]. If you aren’t trying to use objects everywhere, you might not realize that both integers [3] and the class of an object [4] are also objects. If you aren’t trying to use laziness everywhere, you might not realize that purity is an even more important language feature [5].

This leads to two recommendations:

1.  *When learning a new principle, try to apply it everywhere.* That way, you’ll learn more quickly where it does and doesn’t work well, even if your initial intuitions about it are wrong. (The right tool for the job, on the other hand, will lead you to missed opportunities, if you don’t realize that the principle is applicable in some situation).
2.  *When trying to articulate the essence of some principle, an extremist system is clearest.* If you want to know what it is like to program with lazy evaluation, you want to use Haskell, not a language with optional laziness. Even if the extremist system is less practical, it really gets to the core of the issue much more quickly.

There are a lot of situations where extremism is inappropriate, but for fun projects, small projects and research, it can really teach you a lot. One of the most memorable interactions I had in the last year was while working with Adam Chlipala. We were working on some proofs in Coq, and I had been taking the moderate route of doing proofs step-by-step first, and then with Ltac automation once I knew the shape of the proof. Adam told me: “You should automate the proofs from the very beginning, don’t bother with the manual exploration.” [6] It was sage advice that made my life a lot better: I guess I just wasn’t extremist enough!

*Files are awesome. What if we made an OS where everything was a file?*

*Cons cells are awesome. What if we made a PL where everything was made of cons cells?*

*Mathematics is awesome. What if we made a PL where everything came from math?*

*Arrays are awesome. What if we made a PL where everything was an array?*

* * *

[1] Higher-order functions and combinators: these tend to not see very much airplay because they might be very verbose to write, or because the language doesn't have a very good vocabulary for saying what the interface of a higher-order function is. (Types help a bit here.)

[2] Cheap lambdas are necessary for the convenient use of many features, including: monads, scoped allocation (and contexts in general), callbacks, higher-order functions.

[3] Consider early versions of Java prior to the autoboxing of integer and other primitive types.

[4] Smalltalk used this to good effect, as does JavaScript.

[5] This is one of my favorite narratives about Haskell, it comes from Simon Peyton Jones’ presentation [Wearing the hair shirt](http://research.microsoft.com/en-us/um/people/simonpj/papers/haskell-retrospective/) (in this case, laziness).

[6] This is the essence of the Chlipala school of Coq proving, in recognition of how astonishingly easy it is to trick experienced computer scientists into writing the equivalents of straight-line programs by hand, without any abstractions.