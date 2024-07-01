<!--yml
category: 未分类
date: 2024-07-01 18:17:56
-->

# On expressivity : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/03/on-expressivity/](http://blog.ezyang.com/2011/03/on-expressivity/)

*Wherein I make fun of functional programming advocates.*

In this essay, I’d like to discuss the ideologies of “imperative programming” and “functional programming” in terms of the language features they lean on: in particular, the mechanisms by which they allow developers to express themselves in less code. I propose that the set of features that make up imperative programming constitute a dominant programming monoculture that is partially incompatible with functional programming’s favored features, requiring functional programming advocates to do funny things to gain the attention of the programmers.

To first give a flavor of *expressiveness*, here are some frequently seen language features that increase expressiveness:

*   Macros
*   Concurrency
*   Mutation
*   Indirection
*   Laziness
*   Dynamic typing
*   Polymorphism
*   Higher-order functions
*   Exceptions
*   Eval
*   Input/Output
*   Continuations
*   Anonymous functions

A few of those entries might make you laugh, because you might not understand how you could program without them. You may recognize a few that your favorite language supports well, a few that your language supports less well, and a few your language has no support for. The culture around the language will also have its folklore about what kinds of features are acceptable for use and which ones are not (think Pythonic or *JavaScript: The Good Parts*). The language you choose determines which features you get to know well.

Being expressive has a cost, which most developers figure out with a little experience in the field. There is a sort of natural selection going on here: language features that are well supported by languages, that other programmers know how to use, and that allow the job to get done are favored—in particular the community effect reinforces the winner. As a result, we have developer monoculture that is mostly comfortable with mutation, input/output, indirection, exceptions, polymorphism, etc. But even the bread-and-butter of current programming practice doesn’t come without cost: think about the famed division between “those who get pointers and those who don’t”, or the runtime costs of using exceptions in C++, or the representation complications of polymorphism (e.g. autoboxing in Java and Go’s lack thereof).

When someone does *functional programming advocacy*, what they’re really doing is asking you to look more closely at some of the other mechanisms we have for increasing expressiveness. You might feel like these are the only voices you hear, because there’s not much point advocating something that everyone uses already. And you might feel like the enthusiasm is unjustified, because the feature seems bizarrely complicated (continuations, anyone?) or you’ve tried using it in your favorite language, and there’s nothing more painful than seeing someone try to do functional programming in Python. Fact of the matter is, it’s not easy to add these extra language features to the existing monoculture. The new features interact in very complex and subtle ways.

This is why a functional programming advocate will often ask you to give up some of your old tools of expression. They will ask you to give up shared-state mutation, because otherwise handling concurrency is really damn hard. They will ask you to give up dynamic typing, because otherwise higher-order functions become much harder to reason about. The rhetoric will edge on the side of “stop doing that!” because it’s the common practice—they don’t actually mean “stop it entirely” but to the poor functional programming advocate it seems like a little bit of hyperbole is necessary to get the point across—and you can do some pretty amazing things with these extra features.

I encourage programmers to learn about as many ways to express themselves as possible, even if their language or workplace won’t necessarily allow them to use the method. The reasons are manifold:

1.  “Any complex enough program eventually contains a poorly written implementation of Lisp.” Like it or not, eventually you will be faced with a hard problem that is handily dispatched by one of these well studied language features, and if you’re going to have to implement it by hand, you might as well know how it’s going to look like from the start. As the Gang of Four once said, language features that are not actually supported by the language often show up as design patterns; knowing the pattern makes your code clearer and cleaner.
2.  Conversely, if you are reading someone else’s code and they resort to using one of these patterns, knowing how the feature should work will greatly aid comprehension.
3.  Libraries and frameworks are considered essential to the working developer’s toolbox, yet they seem to grow and be obsoleted at a dizzying rate. Language features are eternal: the anonymous functions of 1936 (when Alonzo Church invented the lambda calculus) are still the anonymous functions of today.
4.  Language features are fun to learn about! Unlike “yet another API to memorize”, a language feature will tickle your brain and make you think very hard about what is going on.

**tl;dr** Certain programming language features increase developer expressiveness, and the “imperative programming methodology” captures the dominant monoculture containing those language features in wide use. But there are other ways of expressing oneself, and programmers are encouraged to explore these methods even when practical use necessitates them to stop using some of their favorite expressive tools.