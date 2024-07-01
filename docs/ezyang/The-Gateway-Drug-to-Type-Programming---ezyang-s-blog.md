<!--yml
category: 未分类
date: 2024-07-01 18:18:12
-->

# The Gateway Drug to Type Programming : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/08/the-gateway-drug-to-type-programming/](http://blog.ezyang.com/2010/08/the-gateway-drug-to-type-programming/)

[David Powell](http://blog.ezyang.com/2010/07/suggestion-box/#comment-789) asks,

> There seems to be decent detailed information about each of these [type extensions], which can be overwhelming when you’re not sure where to start. I’d like to know how these extensions relate to each other; do they solve the same problems, or are they mutually exclusive?

Having only used a subset of GHC’s type extensions (many of them added only because the compiler told me to), I’m unfortunately terribly unqualified to answer this question. In the cases where I’ve gone out of my way to add a language extension, most of the time it’s been because I was following some specific recipe that called for that type. (Examples of the former include FlexibleInstances, MultiParamTypeClasses, and FlexibleContexts; examples of the latter include GADTs and EmptyDataDecl).

There is, however, one language extension that I have found myself increasingly relying on and experimenting with—you could call it my gateway drug to type level programming. This extension is `Rank2Types`. ([Tim Carstens](http://intoverflow.wordpress.com/2010/06/30/haskell-features-id-like-to-see-in-other-languages/) appears to be equally gaga at this feature.)

The reason why this feature speaks so powerfully to me is that it lets me encode an invariant that I see all the time in imperative code: *when a resource is released, you should not use it anymore.* Whether for memory, files or network connections, the resource handle is ubiquitous. But normally, you can only write:

```
FILE *fh = fopen("foobar.txt", "r");
fread(buf, sizeof(char), 100, fh);
fclose(fh);
// ...
fread(buf, sizeof(char), 2, fh); // oops

```

so you rely on the file handle being available in a small enough scope so that it’s obvious if you’re using it incorrectly, or if the handle is to be available in a global context, you add runtime checks that it’s hasn’t been closed already and hope that no one’s messed it up a thousand lines of code away.

So the moment I realized that I could actually enforce this statically, I was thrilled. *What other invariants can I move from runtime to compile time?* Luckily, the system I was working on offered more opportunities for type-level invariant enforcement, stepping from “released resources cannot be reused” to “components bound to one resource should not be mixed with another resource” and “exception to previous rule: components can be used for another resource, but only if the target resource came from the source resource, and you need to call a translation function.” These are fairly complicated invariants, and I was quite pleased when I found that I was able to encode these in the type system. In fact, this was a turning point: *I’d moved beyond cookbook types to type programming.*

So, how do you discover your gateway drug to type programming? I feel that right now, there are two ways:

*   Consider all type system features and extensions to be intrinsically useful, study each of them to learn their capabilities and obvious use-cases, and hope that at some point you know the primitives well enough to start fitting them together. (As for many other things, I feel that knowing the fundamentals is the only way to get to truly understand a system, but I personally find this approach very daunting.)
*   Get acquainted with the canonical use-cases for any given type system feature and extension, accumulating a cookbook-like repository of type system possibilities. Stumble upon a real problem that is precisely the use-case, implement it, and then start tinkering at the edges to extend what you can do. (This is how I got hooked, but it has also left me at a loss as to a methodology—a common framework of thought as opposed to isolated instances of cleverness.)

In fact, this seems quite similar to the learning process for any programming language. There are several types of learning materials that I would love to know about:

*   A comprehensive cookbook of type level encodings of invariants that are normally checked at runtime. It would show the low-tech, runtime-verified program, and then show the abstractions and transformations necessary to move the invariant to types. It would collect all of the proposed use-cases that all of the various literature has explored for various type extensions under a uniform skin, a kind of Patterns book. A catalog of Oleg’s work would be a good place to start.
*   When I reuse a type variable in an expression such as `Foo a -> Foo a`, I’ve state that whatever type the left side is, the right side must be the same too. You might usually associate `a` with a usual type like `Int` or `Char`, and `Foo` as some sort of container. But we can put stranger types in this slot. If `Foo` uses `a` as a phantom type, I can use empty types to distinguish among a fixed set of types without any obligation to supply a corresponding value to `Foo`. If I use `Rank2Types` to make `a` bound to another universally quantified type `forall b. b`, I’ve a unique label which can be passed along but can’t be forged. What is actually going on here? What does the “types as propositions” (Curry-Howard) viewpoint say about this?
*   What kinds of type programming result in manageable error messages, and what types of type programming result in [infamous error messages](http://intoverflow.wordpress.com/2010/05/21/announcing-potential-x86-64-assembler-as-a-haskell-edsl/#more-385)? When I first embarked on my API design advantage, a fellow engineer at Galois warned me, “If you have to sacrifice some static analysis for a simpler type system, do it. Things like type level numbers are not worth it.” I may have wandered too far off into the bushes already!

I’m sure that some of this literature exists already, and would love to see it. Bring on the types!