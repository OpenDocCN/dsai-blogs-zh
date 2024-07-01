<!--yml
category: 未分类
date: 2024-07-01 18:18:08
-->

# Data is Code : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/09/data-is-code/](http://blog.ezyang.com/2010/09/data-is-code/)

Yesterday I had the pleasure of attending a colloquium given by [Chung-Chieh Shan](http://www.cs.rutgers.edu/~ccshan/) on [Embedding Probabilistic Languages](http://www.cs.rutgers.edu/news/colloquia/?action=view&colloquium_id=4263&organization_id=1). A full account for the talk can be found in [this paper](http://okmij.org/ftp/kakuritu/dsl-paper.pdf), so I want to focus in on one specific big idea: the idea that *data is code.*

* * *

Lispers are well acquainted with the mantra, “code is data,” the notion that behind every source code listing there is a data structure of cons-cells and tags representing the code that can constructed, modified and evaluated. With this framework, a very small set of data is code: `'(cons 1 (cons 2 ()))` is code but `'((.5 ((.5 #t) (.5 #f))) (.5 ((.5 #t))))` isn’t.

Under what circumstances could the latter be code? Consider the following question (a hopefully unambiguous phrasing of the [Boy-Girl paradox](http://en.wikipedia.org/wiki/Boy_or_Girl_paradox)):

> You close your eyes. I hand you a red ball or a blue ball. Then, I will hand you a red ball or a blue ball. You then peek and discover that at least one of the balls is red. What are the odds that the first one was red?

Those of you familiar with probability might go write up the probability table and conclude the answer is `2/3`, but for those who are less convinced, you might go write up some code to simulate the situation:

```
a <- dist [(.5, red), (.5, blue)]
b <- dist [(.5, red), (.5, blue)]
if a != red && b != red
  then fail
  else a == red

```

Where `dist` is some function that randomly picks a variable from a distribution, and `fail` reports a contradiction and ignores the generated universe. This code is data, but it is data in a much deeper way than just an abstract syntax tree. In particular, it encodes the *tree of inference* `'((.5 ((.5 #t) (.5 #f))) (.5 ((.5 #t))))`:

```
     O               O
    / \             / \
   /   \           /   \
  R     B        .5    .5
 / \   / \       / \   / \
RR RB BR BB    .25.25.25
                #t #t #f

```

* * *

> *Aside.* Interested Haskellers may now find it instructive to go off and write the naive and continuation passing implementations of the probability monad suggested by the above code, a monad which, when run, returns a list of the probabilities of all possible outcomes. It is an interesting technical detail, which will possibly be the subject of a future blog post, but it's treated quite well in sections 2.2, 2.3 and 2.4 of the [above linked paper](http://okmij.org/ftp/kakuritu/dsl-paper.pdf) and fairly standard practice in the continuation-using community.

Now, I haven’t really shown you how data is code; rather, I’ve shown how code can map onto an “abstract syntax tree” representation or an “inference tree” representation. However, unlike an AST, we shouldn’t naively build out the entire inference tree: inference trees whose nodes have many children can branch out exponentially, and we’d run out of memory before we could do what is called *exact inference*: attempt to build out the entire inference tree and look at every result.

However, if we follow the mantra that “data is code” and we represent our tree as a *lazy* data structure, where each child of a node is actually a continuation that says “build out this subtree for me,” we recover an efficient representation. These continuation can, themselves, contain more continuations, which are to be placed at the leaves of the subtree, to be applied with the value of the leaf. Thus our *data* structure is, for the most part, represented by *code.* (This is in fact how all lazy data structures work, but it’s particularly poignant in this case.)

Even more powerfully, first-class support for delimited continuations means that you can take a regular function `() -> e` and reify it into a (partial) tree structure, with more continuations as children ready to themselves be reified. We can, of course, evaluate this tree structure to turn it back into a function. (Monads in Haskell cheat a little bit in that, since lambdas are everywhere, you get this representation for free from the abstraction’s interface.)

* * *

What I find really fascinating is that a whole class of algorithms for efficient probabilistic inference become *obvious* when recast on top of an inference tree. For example:

*   Variable and bucket elimination corresponds to memoizing continuations,
*   Rejection sampling corresponds to randomly traversing paths down our tree, discarding samples that result in contradictions (`fail`), and
*   Importance sampling corresponds to randomly traversing a path but switching to another branch if one branch fails.

Being a shallow embedding, we unfortunately can’t do things like compare if two continuations are equal or do complex code analysis. But some preliminary experimental results show that this approach is competitive with existing, custom built inference engines.

* * *

There’s a bigger story to be told here, one about DSL compilers, where we give users the tools to easily implement their own languages, thereby increasing their expressiveness and productivity, but *also* allow them to implement their own optimizations, thereby not trading away speed that usually is associated with just writing an interpreter for your language. We’d like to leverage the existing compiler framework but add enhancements for our own problem domain as appropriate. We’d like to give behavioral specifications for our problem domains and teach the compiler how to figure out the details. It’s not feasible to write a compiler that fits everyone, but everyone can have the compiler spirit in them—and I think that will have an exciting and liberating effect on software engineering.