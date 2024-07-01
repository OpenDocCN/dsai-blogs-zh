<!--yml
category: 未分类
date: 2024-07-01 18:17:36
-->

# Anatomy of “You could have invented…” : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/02/anatomy-of-you-could-have-invented/](http://blog.ezyang.com/2012/02/anatomy-of-you-could-have-invented/)

## Anatomy of “You could have invented…”

The *You could have invented...* article follows a particular scheme:

1.  Introduce an easy to understand problem,
2.  Attempt to solve the problem, but get stuck doing it the "obvious" way,
3.  Introduce an easy to understand insight,
4.  Methodically work out the rest of the details, arriving at the final result.

Why does framing the problem this way help?

*   While the details involved in step 4 result in a structure which is not necessarily obvious (thus giving the impression that the concept is hard to understand), the insight is very easy to understand and the rest is just "monkey-work". The *method of deriving the solution* is more compressible than the *solution itself*, so it is easier to learn.
*   Picking a very specific, easy-to-understand problem helps ground us in a concrete example, whereas the resulting structure might be too general to get a good intuition off of.

It's very important that the problem is easy to understand, and the process of "working out the details" is simple. Otherwise, the presentation feels contrived. This method is also inappropriate when the audience is in fact smart enough to just look at the end-result and understand on an intuitive level what is going on. Usually, this is because they have *already seen the examples.* But for the rest of us, it is a remarkably effective method of pedagogy.

I'll take this opportunity to dissect two particular examples. The first is Dan Piponi’s canonical article, [You Could Have Invented Monads](http://blog.sigfpe.com/2006/08/you-could-have-invented-monads-and.html). Here are the four steps:

1.  Suppose we want to debug pure functions, with type signature `f :: Float -> Float`, so that they can also return a string message what happened, `f' :: Float -> (Float, String)`.
2.  If we want to compose these functions, the obvious solution of threading the state manually is really annoying.
3.  We can *abstract* this pattern out as a higher-order function.
4.  We can do the same recipe on a number of other examples, and then show that it generalizes. The generalization is a Monad.

The second is article of mine, [You could have invented Zippers](http://blog.ezyang.com/2010/04/you-could-have-invented-zippers/):

1.  I want to do two things: have access to both the parent and children of nodes in a tree, and do a *persistent* update of the tree (e.g. without mutation.)
2.  If we do the obvious thing, we have to update all the nodes in the tree.
3.  We only need to flip one pointer, so that it points to the parent.
4.  We can create a new data structure which holds this information (which, quite frankly, is a little ugly), and then show this procedure generalizes. The generalization is a Zipper.

So the next time you try to explain something that seems complicated on the outside, but simple on the inside, give this method a spin! Next time: *You could have invented fractional cascading.*

*Postscript.* This is a common way well-written academic papers are structured, though very few of them are titled as such. One noticeable difference, however, is that often the “detail work” is not obvious, or requires some novel technical methods. Sometimes, a researcher comes across a really important technical method, and it diffuses throughout the community, to the point where it is obvious to anyone working in the field. In some respects, this is one thing that characterizes a truly successful paper.