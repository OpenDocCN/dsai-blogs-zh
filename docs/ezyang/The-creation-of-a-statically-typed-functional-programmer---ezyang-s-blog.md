<!--yml
category: 未分类
date: 2024-07-01 18:17:57
-->

# The creation of a statically-typed functional programmer : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/03/the-creation-of-a-statically-typed-functional-programmer/](http://blog.ezyang.com/2011/03/the-creation-of-a-statically-typed-functional-programmer/)

## The creation of a statically-typed functional programmer

The bug bit me in early 2009, during MIT’s independent activities period; really, it was two bugs. The first was [6.184](http://web.mit.edu/alexmv/6.001/), the re-animated introductory computer science class taught in Scheme—for obvious reasons. But I don’t think that was sufficient: I seemed to recall thinking Scheme was interesting but not a language I actually wanted to code in. The second was a comment made by Anders Kaseorg after I finished delivering a talk [Introduction to Web Application Security](http://mit.edu/~ezyang/Public/iap/intro-to-was.html) (one of the few things that, as a freshman at MIT, I thought I knew well enough to give a lecture on). One of the emphases of the talk was all about *types*: that is, the fact that “string” doesn’t adequately represent the semantic content of most bits of text that float around in our applications these days. Haskell came up as a way of making your compiler make sure you didn’t mix up HTML with plain text.

Something must have clicked. That February, I wrote:

> Wow. Haskell is pretty.

To which someone replied:

> Don't look too hard into the sun, your eyes will get burned.

And thus a statically-typed functional programmer was born.

*Postscript.* My first application in Haskell was a Laplace solver, with which I also learned about monads (because a map lookup returned a `Maybe` value, and Anders decided it would be a good idea to talk about do-notation and bind to elucidate how to handle it. I probably didn’t understand the explanation the first time around, but I did manage to get the program working.)