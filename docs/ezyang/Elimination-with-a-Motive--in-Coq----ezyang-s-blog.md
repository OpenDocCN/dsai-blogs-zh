<!--yml
category: 未分类
date: 2024-07-01 18:17:14
-->

# Elimination with a Motive (in Coq) : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/05/elimination-with-a-motive-in-coq/](http://blog.ezyang.com/2014/05/elimination-with-a-motive-in-coq/)

## Elimination with a Motive (in Coq)

Elimination rules play an important role in computations over datatypes in proof assistants like Coq. In his paper "Elimination with a Motive", Conor McBride argued that "we should exploit a hypothesis not in terms of its immediate consequences, but in terms of the leverage it exerts on an arbitrary goal: we should give elimination a motive." In other words, proofs in a refinement setting (backwards reasoning) should use their goals to guide elimination.

I recently had the opportunity to reread this historical paper, and in the process, I thought it would be nice to port the examples to Coq. Here is the result:

> [http://web.mit.edu/~ezyang/Public/motive/motive.html](http://web.mit.edu/~ezyang/Public/motive/motive.html)

It's basically a short tutorial motivating John Major equality (also known as heterogenous equality.) The linked text is essentially an annotated version of the first part of the paper—I reused most of the text, adding comments here and there as necessary. The source is also available at:

> [http://web.mit.edu/~ezyang/Public/motive/motive.v](http://web.mit.edu/~ezyang/Public/motive/motive.v)