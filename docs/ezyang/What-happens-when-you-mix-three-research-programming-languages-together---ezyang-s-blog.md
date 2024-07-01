<!--yml
category: 未分类
date: 2024-07-01 18:17:30
-->

# What happens when you mix three research programming languages together : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/05/what-happens-when-you-mix-three-research-programming-languages-together/](http://blog.ezyang.com/2012/05/what-happens-when-you-mix-three-research-programming-languages-together/)

“...so that’s what we’re going to build!”

“Cool! What language are you going to write it in?”

“Well, we were thinking we were going to need three programming languages...”

“...three?”

“...and they’ll be research programming languages too...”

“Are you out of your mind?”

* * *

This was the conversation in streaming through my head when I decided that I would be writing my latest software project in Coq, Haskell and Ur/Web. I had reasonably good reasons for the choice: I wanted Coq because I didn’t actually want to implement a theorem prover from scratch, I wanted Ur/Web because I didn’t actually want to hand write JavaScript to get an AJAX interface, and I wanted Haskell because I didn’t want to write a bucket of C to get Ur/Web and Coq to talk to each other. But taken altogether the whole thing seemed a bit ludicrous, like an unholy fusion of a trinity of research programming languages.

In the end, it worked out quite well. Now, what this means depends on your expectations: it was not the case that “everything worked out of the box and had very nice instructions attached.” However, if it was the case that:

*   No single issue ended up requiring an unbounded amount of time and yak shaving,
*   Any patches written made it into upstream, improving the situation of the software for future developers, and
*   The time spent on engineering grease is less than the time it would have taken to build the system with inferior languages,
*   Everyone involved in the project is willing to learn all of the languages involved (easy if it’s only one person),

then yes, it worked “quite well”. In this post, I’d like to describe in a little more detail what happened when I put these three languages together and speculate wildly about general maxims that might apply when someone is doing something similar.

### Coq

While Coq is a research language, it is also in very wide use among academics, and most of its instability lies in advanced features that I did not use in my project. So the primary issues I encountered with Coq were not bugs, but in integrating it with the system (namely, making it talk to Haskell).

**Maxim 1.** *Interchange formats will be undocumented and just good enough to get the job done.*

Coq is already designed to allow for communication between processes (this is how the Proof General/Emacs and Coq talk to each other), but the format between coqtop and Proof General was undocumented, ad hoc, and didn't transmit enough information for my application. In the face of such a situation, there are two ways to proceed: grit your teeth and implement the bad protocol or patch the compiler to make a better one. I chose the latter, and learned something very interesting:

**Maxim 2.** *In ML-like languages, it’s very easy to make simple but far reaching changes to a codebase, due to the assistance of the typechecker.*

Making the changes to the frontend was very simple; there was nothing deep about the change, and a combination of the typechecker and grep allowed me to pull off the patch with zero debugging. With a few XML tags at a few key spots, I got output reasonable enough to build the rest of the system with.

*Aside.* Later, I learned that coqide in recent versions of Coq (8.4 and later) has another interchange format. Moving forward, it is probably the correct mechanism to interact with Coq interactively, though this is made somewhat more difficult by the fact that the interchange format is undocumented; however, I've [filed a bug](https://coq.inria.fr/bugs/show_bug.cgi?id=2777). With any luck, it will hopefully do better than my patch. My patch was originally intended to be a partial implementation of PGIP, a generic interchange format for interacting with theorem provers, but the Coq developers and I later discovered that the PGIP project is inactive, and the other user, Isabelle, has discontinued using their PGIP backend. (Sometimes standards don’t help!)

### Ur/Web

Ur/Web is comparatively less used, and accordingly we ran into a variety of bugs and other infelicities spanning all parts of the system, from the frontend to the compiler. Were they blockers? No!

**Maxim 3.** *A deterministically reproducible bug in some core functionality will get fixed very quickly by an active original author of the code.*

This maxim doesn’t apply to fundamental limitations in design (where the fix will take a lot of elbow grease, though the author will usually have a good idea when that’s the case), but other bugs of this type, I found I could get freakishly quick turnaround times for fixes. While I may attribute part of this to the fact that my advisor was the one who wrote the compiler, I don’t think that’s all there is to it. There is a certain pride that comes with an interesting, tricky bit of code you wrote, that makes it an irresistible little puzzle when someone shows you a bug. And we *love* little puzzles.

There’s also a corollary:

**Maxim 4.** *The less interesting a problem is to the academic, the more likely it is you’ll be able to fix it yourself.*

Academics are somewhat allergic to problems that they’re not interested in and which aren’t vital for their research. This means they don’t like working on these bits, but it also means that they’ve probably kept it simple, which means you’re more likely to be able to figure it out. (A good typechecker also really helps! See maxim 2.) There was a simple bug with serving 404s from FastCGI's compiled by Ur/Web, which had a very simple fix; I also made some modifications to Ur/Web made it runnable without having to `make install` first. Maintainers of active research software tend to be quite receptive to these "engineering" patches, which serve no direct research purpose. I consider these contributes to be a vital component of being a good citizen of the open source community.

### Haskell

OK, Haskell is not really “just” a research language anymore; it is also a very flexible general purpose language which has seen quite a bit of real world use and can be treated as an “ordinary” language in that respect. This made it a good choice for gluing the two other languages together; it can do just about anything, and has very good FFI support for calling into and out of Haskell. This brings us to our next maxim:

**Maxim 5.** *An FFI is a crucial feature for any DSL, and should be a top priority among tasks involved in preparing a language for general usage.*

Having Haskell and Ur/Web talk to each other through their FFIs was key for making this all work. Ur/Web is a domain specific language for writing web applications, and among other things it does not include robust systems libraries (e.g. executing external processes and interfacing with them). Most languages will have this problem, since library support takes a bit of work to add, but Ur/Web has a second problem: all side-effectful transactions need to also be able to be rolled back, and this is rather hard to achieve for general input-output. However, with an FFI, we can implement any code which needs this library support in a more suitable language (Haskell), wrap it up in an API which gives the appropriate transactional guarantees, and let Ur/Web use it. Without it, we would not have been able to use Ur/Web: it’s an extremely flexible escape hatch.

Specifying an FFI also is a good way of demonstrating how your language is *different* from C: it forces you to think about what invariants you expect foreign functions to have (referential transparency? thread-safety?): these invariants are exactly the ones that get automatically fulfilled by code written in your language. That’s pretty cool!

However, because functions which manipulate C pointers are non-transactional, Ur/Web is limited to FFI functions which handle basic C types, e.g. integers and strings. Thus the question of parsing becomes one of utmost importance for Ur/Web, as strings are the preferred interchange format for complex structures. While different languages will have different situations, in general:

**Maxim 6.** *Make sure you know how to do parsing in all of the languages involved.*

### Conclusion

I’ve presented six maxims of research polyglottism:

1.  Interchange formats will be undocumented and just good enough to get the job done.
2.  In ML-like languages, it’s very easy to make simple but far reaching changes to a codebase, due to the assistance of the typechecker.
3.  A deterministically reproducible bug in some core functionality will get fixed very quickly by an active original author of the code.
4.  The less interesting a problem is to the academic, the more likely it is you’ll be able to fix it yourself.
5.  An FFI is a crucial feature for any DSL, and should be a top priority among tasks involved in preparing a language for general usage.
6.  Make sure you know how to do parsing in all of the languages involved.

If you keep all of these maxims in mind, I believe that the tradeoff between some extra bugfixing and yak shaving for the benefits of the research programming language is a compelling one, and one that should be considered seriously. Yes, you have to be willing to muck around with the innards of all the tools you use, but for any sufficiently important tool, this is inevitably true. And what is a more important tool than your compiler?

*Postscript.* The application in question is [Logitext](http://logitext.ezyang.scripts.mit.edu/main).