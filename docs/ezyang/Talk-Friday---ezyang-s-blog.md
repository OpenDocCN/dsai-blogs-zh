<!--yml
category: 未分类
date: 2024-07-01 18:18:01
-->

# Talk Friday : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/12/talk-friday/](http://blog.ezyang.com/2010/12/talk-friday/)

I’ve had the pleasure of attending a number of really interesting talks over the past few months, so many that I couldn’t find time to write thorough articles for each of them as I did over the summer. So you’ll have to forgive me for putting two of them in compressed form here. There is something of a common theme of recasting a problem on a different input domain in order to achieve results, as I hope will become evident by these summaries.

* * *

*A Language for Mathematics* by [Mohan Ganesalingam](http://people.pwf.cam.ac.uk/mg262/). *Big idea:* Apply linguistics and natural language processing techniques to mathematical language—the type found in textbooks and proofs.

Ganesalingam has big goals: his long term project is to “enable computers to do mathematics in the same way that humans do.” “But wait,” you may say, “aren’t we already approaching this with proof assistants?” Unfortunately, the answer to this is no: proof assistants are quite good at capturing rigorous formal reasoning, but are terrible at actually capturing the soft ideas that mathematicians gesture at when writing proofs and textbooks. The first step in this program is understanding this mathematical language—thus, the title of his talk.

Why do we have any reason to believe that this program will be any more successful than current research in linguistics and NLP? After all, most papers and textbooks use English interspersed with mathematical notation, and grand ideas about semantic analysis have given way to more effective but theoretically less appealing statistical methods. Ganesalingam makes some key observations here: in essence, mathematical language has the right dose of formality to make traditionally hard problems tractable. Only a small lexicon is necessary, and then mathematical terms can be defined in terms of other mathematical terms, and in many cases, there is a clear semantics for a mathematical statement: we can in principle translate it into a statement in higher order logic.

Further reading: [Slides for a similar presentation that was given at Stanford](http://people.pwf.cam.ac.uk/mg262/CSLI%20talk.pdf), [an informal non-technical introduction](http://people.pwf.cam.ac.uk/mg262/GanesalingamMsum.pdf), [author's homepage](http://people.pwf.cam.ac.uk/mg262/).

* * *

*Evaluating Formulas on Graphs* by [Anuj Dawar](http://www.cl.cam.ac.uk/~ad260/). There are really two big ideas here. *Big idea 1.* Generalize graph problems into the question “does this first-order logical formula hold on this graph?”, treating your algorithm as a function on two inputs: the graph and the logical formula. *Big idea 2.* Use graph structure theory to characterize what input spaces of graphs we can efficiently solve these FO formulas for.

First big idea: the study of graph problems is frequently focused on an individual graph problem at a time: after all, being able to assume a concrete problem instance makes it easier to reason about things. What Dawar’s talk introduces is a way to talk about large classes of graph problems by bundling them up into logics (of various shapes and sizes.) Existential second-order logic gives you all NP problems (Fagin); first-order logic is more restrictive but admits better analysis. Separating out the formula from your problem also lets you apply parametrized complexity theory: the formula is an input to your algorithm, and you set it constant or vary it. Unfortunately, the problem (even for fixed graphs) is still PSPACE-complete, so we need another way to get a grip on the problem.

Second big idea: restrict the input graphs in order to make the algorithms tractable. This involves a bit of graph theory knowledge which I’m not going to attempt to summarize, but there are some really nice results in this area:

*   Seese (1996): For the class of graphs with degree bounded by k, every FO definable property is decidable in linear time.
*   Frick and Grohe (2001): For the class of graphs of local tree-width bounded by a function f, every FO definable property is decidable in quadratic time.
*   Flum and Grohe (2001): For the class of graphs excluding K_k as a minor, every FO definable property is decidable in O(n^5).

One oddball fact is that Flum and Grohe’s O(n^5) bound on complexity has a constant factor which may not be computable.

By the end, we get to the edge of research: he introduces a new class of graphs, *nowhere dense* graphs, motivates why we have good reason to think this characterizes tractability, and says that they hope to establish FO is fixed parameter tractable.

A quick aside: one of the things I really enjoy about well-written theoretical research talks is that they often introduce me to subfields of computer science that I would not have otherwise encountered. This presentation was a whirlwind introduction to graph theory and parametrized complexity theory, both topics I probably would not have otherwise considered interesting, but afterwards I had tasted enough of to want to investigate further. I think it is quite commendable for a researcher doing highly abstract work to also be giving seminars working up the background knowledge necessary to understand their results.

Further reading: [Full course on these topics](http://phdopen.mimuw.edu.pl/index.php?page=z10w1)