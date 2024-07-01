<!--yml
category: 未分类
date: 2024-07-01 18:18:10
-->

# My type signature overfloweth : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/09/my-type-signature-overfloweth/](http://blog.ezyang.com/2010/09/my-type-signature-overfloweth/)

I’ve recently started researching the use of *session types* for practical coding, a thought that has been in the back of my mind ever since I was part of a team that built a networked collaborative text editor and spent a lot of time closely vetting the server and the client to ensure that they had implemented the correct protocols. The essence of such protocols is often relatively simple, but can quickly become complicated in the presence of error flow (for example, resynchronizing after a disconnection). Error conditions also happen to be difficult to automatically test! Thus, static types seem like an attractive way of tackling this task.

There are three implementations of session types in Haskell: [sessions](http://hackage.haskell.org/package/sessions), [full-sessions](http://hackage.haskell.org/package/full-sessions) and [simple-sessions](http://hackage.haskell.org/package/simple-sessions). If you were feeling particularly naive, you might try going to the [Haddock page](http://hackage.haskell.org/packages/archive/sessions/2008.7.18/doc/html/Control-Concurrent-Session.html) to get a feel for what the API looks like. Before you continue reading, please inspect that page.

* * *

Done gouging your eyes out? Let’s proceed.

In an interview in *Coders at Work*, Simon Peyton Jones mentioned that one of the notable benefits of types is that it gives a concise, crisp description of what a function might do. That API is anything from concise and crisp, and it’s certainly not something that I could figure out just by looking at the corresponding function definition. Accordingly, one of the key selling points of current encodings of session types is that they do not break type inference: we give up on our user understanding what the gaggle of typeclasses means, and only expect to transfer one bit of information, “Do the protocols match?”

This is not a problem that is fundamental to session types: any functionality that makes extensive use typeclasses can easily fall prey to these long type signatures. I have two (rather half-baked) thoughts on how this complexity might be rendered more nicely to the user, although not eliminated:

*   A favorite pastime of type system hackers is a type-level encoding of naturals, using Peano numbers `Z` and `S a`, attached to something like `Vector (S (S Z))`. Vector is a type constructor of kind `* -> *`. However, since there is only one primitive kind in Haskell, we could actually pass any type to Vector, say `Vector Int`, which would be a nonsensical. One way to prevent this from occurring is to declare our Peano numbers instances of a typeclass `Nat`, and then declare `Nat a => Vector a`. But, since `a` is used precisely once in any such a statement, wouldn’t it be great if instead we could write `Vector :: Nat -> *`? If you need to specify type equality, you could imagine some sort of type pattern matching `concat :: Vector a -> Vector b -> Vector c with c ~ a :+: b`. [Collapsing types and kinds](http://byorgey.wordpress.com/2010/08/05/typed-type-level-programming-in-haskell-part-iv-collapsing-types-and-kinds/) is an interesting step in this direction.
*   When mathematicians present proofs, they might explicitly specify “for all F such that F is a field...”, but more frequently, they’ll say something like, “in the following proof, assume the following variable naming conventions.” With this, they get to avoid having to repeatedly explicitly redeclare what all of their variable names mean. An analogous system for type variables would go a long way towards reducing long type signatures.

But actually, that has nothing to do with what I’m currently looking at.

* * *

Here’s what I am looking at: session types suffer from another type signature explosion phenomenon: any function in the protocol contains, in its type, a complete specification of the entire protocol continuing from that point in time. As [Neubauer and Thiemann admit (PDF)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.70.7370&rep=rep1&type=pdf), the “session type corresponding to full SMTP is quite unreadable.” The two lines of inquiry I am pursuing are as follows:

*   Can building exception support into session types (currently an open problem) allow for much simpler session types by allowing most cases to elide the session types corresponding to error cases?
*   Can we use `type` to permit a single global specification of the protocol, which individual functions then simply refer to? Do we need something a little more powerful?

At this point, I’ve just been doing thinking and paper reading, but I hope to start hacking on code soon. I’d love to hear your thoughts though.