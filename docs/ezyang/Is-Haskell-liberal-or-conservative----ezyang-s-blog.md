<!--yml
category: 未分类
date: 2024-07-01 18:17:29
-->

# Is Haskell liberal or conservative? : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/08/is-haskell-liberal-or-conservative/](http://blog.ezyang.com/2012/08/is-haskell-liberal-or-conservative/)

## Is Haskell liberal or conservative?

Steve Yegge has posted a [fun article](https://plus.google.com/u/0/110981030061712822816/posts/KaSKeg4vQtz) attempting to apply the liberal and conservative labels to software engineering. It is, of course, a gross oversimplification (which Yegge admits). For example, he concludes that Haskell must be “extreme conservative”, mostly pointing at its extreme emphasis on safety. This completely misses one of the best things about Haskell, which is that *we do crazy shit that no one in their right mind would do without Haskell’s safety features.*

So I thought I’d channel some Yegge and take a walk through the criteria proposed for assessing how conservative a user of a language is, and try to answer them to the best of my ability with my ”Haskell hat” on:

1.  *Software should aim to be bug free before it launches.* Yes. Though, “Beware of bugs in the above code; I have only proved it correct, not tried it.”
2.  *Programmers should be protected from errors.* Yes. **But**, Yegge then adds the sentence: “Many language features are inherently error-prone and dangerous, and should be disallowed for all the code we write.” This is not the approach that Haskell takes: if you want continuations with mutable state, Haskell will give them to you. (Try doing that in Python.) It doesn’t *disallow* language features, just make them more wordy (`unsafePerformIO`) or harder to use. Haskell has a healthy belief in escape hatches.
3.  *Programmers have difficulty learning new syntax.* **No.** Haskell is completely on the wrong side of the fence here, with arbitrary infix operators; and even more extremist languages (e.g. Coq) go even further with arbitrary grammar productions. Of course, the reason for this is not syntax for its own sake, but syntax for the sake of closely modeling existing syntax that mathematicians and other practitioners already use. So we allow operator overloading, but only when it is backed up by algebraic laws. We allow metaprogramming, though I suspect it’s currently used sparingly only because it’s so unwieldy (but *culturally*, I think the Haskell community is very open to the idea of metaprogramming).
4.  *Production code must be safety-checked by a compiler.* Yes. **But,** anyone who has used a dependently typed language has a much higher standard of what “safety-checked” means, and we regularly play fast and loose with invariants that we decided would be too annoying to statically encode. Note that Yegge claims the opposite of compiler safety-checking is *succinctness*, which is a completely false myth perpetuated by non-Hindley Milner type systems with their lack of type inference.
5.  *Data stores must adhere to a well-defined, published schema.* Well-defined? Yes. Published? No. The emphasis that Haskell has on static checking mean that people writing data types are a lot more willing to update them as the needs of the application change, and don’t really mind global refactoring of the database because it’s so damn easy to get right.
6.  *Public interfaces should be rigorously modeled.* Yes. (though *cough* “ideally object oriented” *cough*)
7.  *Production systems should never have dangerous or risky back-doors.* **Accidental.** The lack of tooling here means that it’s pretty difficult to snoop into a running compiled executable and fiddle around with internal data: this is a big sore point for the current Haskell ecosystem. But in the abstract, we’re pretty flexible: XMonad, for example, can be restarted to run arbitrary new code *while preserving the entirety of your working state*.
8.  *If there is ANY doubt as to the safety of a component, it cannot be allowed in production.* This is something of a personal question, and really depends on your project, and not so much on the language itself. Haskell is great for safety critical projects, but I also use it for one-off scripts.
9.  *Fast is better than slow.* **No.** Haskell code has the opportunity to be really fast, and it tends to be quite zippy from the get go. But we’ve emphasized features (laziness and abstraction) which are known to cause performance problems, and most Haskellers take the approach of only optimizing when our (very awesome) profiler yells at us. Some Haskellers reflexively add `! {-# UNPACK #-}` to their data types, but I don’t—at least, not until I decide my code is too slow.

Haskell has a lot of features which show up in Yegge’s “Liberal Stuff”. Here are some of them:

*   Eval: We love coding up interpreters, which are like type-safe evals.
*   Metaprogramming: Template Haskell.
*   Dynamic scoping: Reader monad.
*   all-errors-are-warnings: We can [delay type errors to runtime!](http://hackage.haskell.org/trac/ghc/ticket/5624).
*   Reflection and dynamic invocation: `class Data`.
*   RTTI: I hear it’s called a “dictionary”.
*   The C preprocessor: Indispensable, begrudgingly.
*   Lisp macros: Why use macros when you can do it properly in Template Haskell!
*   Domain-specific languages: Haskell eats EDSLs for lunch.
*   Optional parameters: It’s called combinator libraries.
*   Extensible syntax: Fuck yeah infix!
*   Auto-casting: Numeric literals, anyone?
*   Automatic stringification: `class Show` and deriving.
*   Sixty-pass compilers: GHC does *a lot* of passes.
*   Whole-namespace imports: Yep (and it's both convenient and kind of annoying).

The feeling I get from this conversation is that most people think “Haskell” and “static typing” and while thinking about how horrible it is to write traditional dynamically typed code in Haskell, forget that Haskell is actually a surprisingly liberal language prizing understandability, succinctness and risk-taking. Is Haskell liberal or conservative? I think of it as an interesting point in the design space which treats some conservative viewpoints as foundational, and then sees how far it can run from there. *It’s folded so far right, it came around left again.*