<!--yml
category: 未分类
date: 2024-07-01 18:17:53
-->

# Haskell: Not pure enough? : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/05/haskell-not-pure-enough/](http://blog.ezyang.com/2011/05/haskell-not-pure-enough/)

## Haskell: Not pure enough?

It is well known that `unsafePerformIO` is an evil tool by which impure effects can make their way into otherwise pristine Haskell code. But is the rest of Haskell really that pure? Here are a few questions to ask:

1.  What is the value of `maxBound :: Int`?
2.  What is the value of `\x y -> x / y == (3 / 7 :: Double)` with `3` and `7` passed in as arguments?
3.  What is the value of `os :: String` from `System.Info`?
4.  What is the value of `foldr (+) 0 [1..100] :: Int`?

The answers to each of these questions are ill-defined—or you might say they’re well defined, but you need a little extra information to figure out what the actual result is.

1.  The Haskell 98 Report guarantees that the value of `Int` is at least `-2^29` to `2^29 - 1`. But the precise value depends on what implementation of Haskell you’re using (does it need a bit for garbage collection purposes) and whether or not you’re on a 32-bit or 64-bit system.
2.  Depending on whether or not the excess precision of your floating point registers is used to calculate the division, or if the IEEE standard is adhered to, this equality may or may not hold.
3.  Depending on what operating system the program is run on this value will change.
4.  Depending on the stack space allotted to this program at runtime, it may return a result or it may stack overflow.

In some respects, these constructs break referential transparency in an interesting way: while their values are guaranteed to be consistent during a single execution of the program, they may vary between different compilations and runtime executions of our program.

Is this kosher? And if it’s not, what are we supposed to say about the semantics of these Haskell programs?

The topic came up on `#haskell`, and I and a number of participants had a lively discussion about the topic. I’ll try to distill a few of the viewpoints here.

*   The *mathematical school* says that all of this is very unsatisfactory, and that their programming languages should adhere to some precise semantics over all compilations and runtime executions. People ought to use arbitrary-size integers, and if they need modular arithmetic specify explicitly how big the modulus is (`Int32`? `Int64`?) `os` is an abomination that should have been put in the `IO` sin bin. As tolkad puts it, “Without a standard you are lost, adrift in a sea of unspecified semantics. Hold fast to the rules of the specification lest you be consumed by ambiguity.” Limitations of the universe we live in are something of an embarrassment to the mathematician, but as long as the program crashes with a nice *stack overflow* they’re willing to live with a partial correctness result. An interesting subgroup is the *distributed systems school* which also care about the assumptions that are being made about the computing environment, but for a very practical reason. If multiple copies of your program are running on heterogeneous machines, you better not make any assumptions about pointer size on the wire.
*   The *compile time school* says that the mathematical approach is untenable for real world programming: one should program with compilation in mind. They’re willing to put up with a little bit of uncertainty in their source code programs, but all of the ambiguity should be cleared up once the program is compiled. If they’re feeling particularly cavalier, they’ll write their program with several meanings in mind, depending on the compile time options. They’re willing to put up with stack overflows, which are runtime determined, but are also a little uncomfortable with it. It is certainly better than the situation with `os`, which could vary from runtime to runtime. The mathematicians make fun of them with examples like, “What about a dynamic linker or virtual machine, where some of the compilation is left off until runtime?”
*   The *run time school* says “Sod referential transparency across executions” and only care about internal consistency across a program run. Not only are they OK with stack overflows, they’re also OK with command line arguments setting global (pure!) variables, since those don’t change within the executions (they perhaps think `getArgs` should have had the signature `[String]`, not `IO [String]`), or variables that unsafely read in the contents of an external data file at program startup. They [write things in docs](http://hackage.haskell.org/packages/archive/hashable/1.1.1.0/doc/html/Data-Hashable.html) like “This integer need not remain consistent from one execution of an application to another execution of the same application.” Everyone else sort of shudders, but it’s a sort of guilty pleasure that most people have indulged in at some point or another.

So, which school are you a member of?

*Postscript.* Since Rob Harper has recently posted another [wonderfully iconoclastic blog post](http://existentialtype.wordpress.com/2011/05/01/of-course-ml-has-monads/), and because his ending remarks are tangentially related to the topic of this post (purity), I thought I couldn’t help but sneak in a few remarks. Rob Harper states:

> So why don’t we do this by default? Because it’s not such a great idea. Yes, I know it sounds wonderful at first, but then you realize that it’s pretty horrible. Once you’re in the IO monad, you’re stuck there forever, and are reduced to Algol-style imperative programming. You cannot easily convert between functional and monadic style without a radical restructuring of code. And you inevitably need unsafePerformIO to get anything serious done. In practical terms, you are deprived of the useful concept of a benign effect, and that just stinks!

I think Harper overstates the inability to write functional-style imperative programs in Haskell (conversions from functional to monadic style, while definitely annoying in practice, are relatively formulaic.) But these practical concerns do influence the day-to-day work of programmers, and as we’ve seen here, purity comes in all sorts of shades of gray. There is design space both upwards and downwards of Haskell’s current situation, but I think to say that purity should be thrown out entirely is missing the point.