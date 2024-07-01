<!--yml
category: 未分类
date: 2024-07-01 18:17:58
-->

# Picturing Hoopl transfer/rewrite functions : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/02/picturing-hoopl-transferrewrite-functions/](http://blog.ezyang.com/2011/02/picturing-hoopl-transferrewrite-functions/)

## Picturing Hoopl transfer/rewrite functions

[Hoopl](http://hackage.haskell.org/package/hoopl) is a “higher order optimization library.” Why is it called “higher order?” Because all a user of Hoopl needs to do is write the various bits and pieces of an optimization, and Hoopl will glue it all together, the same way someone using a fold only needs to write the action of the function on one element, and the fold will glue it all together.

Unfortunately, if you’re not familiar with the structure of the problem that your higher order functions fit into, code written in this style can be a little incomprehensible. Fortunately, Hoopl’s two primary higher-order ingredients: transfer functions (which collect data about the program) and rewrite functions (which use the data to rewrite the program) are fairly easy to visualize.