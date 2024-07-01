<!--yml
category: 未分类
date: 2024-07-01 18:17:21
-->

# The AST Typing Problem : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/05/the-ast-typing-problem/](http://blog.ezyang.com/2013/05/the-ast-typing-problem/)

This [Lambda the Ultimate post (dated 2010)](http://lambda-the-ultimate.org/node/4170) describes a rather universal problem faced by compiler writers: how does one go about adding “extra information” (e.g. types) to an AST? (The post itself divides the problem into three components: adding the information to the data types, using the information to inform the construction of the node, and using the information to inform the destruction of a node—but I’m really only interested in the question of how you define your data type, not do things to it.) In this post, I want to sum up ways of solving the problem which were described in this post, and also take a look at what some real world compilers do. The running example lambda calculus looks like the following:

```
data Exp = Num Int
         | Bool Bool
         | Var Var
         | If Exp Exp Exp
         | Lambda Var Exp
         | App Exp Exp
data Type = TyInt | TyBool | TyArrow Type Type

```

### Separate IR where nodes are decorated with types

The low-tech solution: if you need a new version of the IR with more information, just define a new IR type where each node can also carry the information. A trick to make these definitions more concise is to make a mutually recursive data structure. [[1]](http://lambda-the-ultimate.org/node/4170#comment-63834)

```
type TExp = (TExp', Type)
data TExp' = TNum Int
           | TBool Bool
           | TVar Var
           | TIf TExp TExp TExp
           | TLambda Var TExp
           | TApp TExp TExp

```

Despite (or perhaps because of) it’s simplicity, this approach is extremely popular among many compilers, especially in the ML community. A few examples include OCaml (parsetree/typedtree), MLton (AST/CoreML) and Ikarus Scheme. Part of the reason for this is that the transition from frontend language to typed language also comes with some other changes, and when a new AST is defined those changes can be combined in too.

### Nullable field

The unprincipled solution: use one AST, but have an optional field in which you can slot in the information. [[2]](http://lambda-the-ultimate.org/node/4170#comment-63832)

```
type TExp = (TExp', Maybe Type)
data TExp' = TNum Int
           | TBool Bool
           | TVar Var
           | TIf TExp TExp TExp
           | TLambda Var TExp
           | TApp TExp TExp

```

Presented without further comment.

### Explicit typing

While closely related to the separate IR solution, an explicitly typed IR takes the approach of not decorating each node with a type, but arranging that the type of any given node can be quickly computed using only local information. [[3]](http://lambda-the-ultimate.org/node/4170#comment-63884)

```
data TExp = TNum Int
          | TBool Bool
          | TVar Var
          | TIf TExp TExp TExp
          | TLambda Var Type TExp
          | TApp TExp TExp

```

Here, the difference between `TExp` and `Exp` is very slight; the `TLambda` is annotated with an explicit type for the binder. As far as type-checking is concerned, this makes a world of difference: we no longer need to look outside a lambda to figure out what the binder could be.

Forcing your IR to be explicitly typed is often a good idea for metatheoretic reasons, as complicated type systems often don’t have decidable inference algorithms. Both GHC’s core IR, Ur/Web's core and Coq are explicitly typed in this way.

### Two-level types

By deferring when you tie the knot of a recursive data-structure, you can arrange for the base functor to do double-duty for the untyped and typed representations. [[4]](http://lambda-the-ultimate.org/node/4170#comment-63836)

```
data ExpF a = Num Int
            | Bool Bool
            | Var Var
            | If a a a
            | Lambda Var a
            | App a a
newtype Exp = Exp (ExpF Exp)
newtype TExp = TExp (ExpF TExp, Type)

```

The Coq kernel uses this to define its expression type, although it doesn’t use it to define an untyped variant.

### (Lazy) Attribute grammars

I don’t claim to understand this approach too well, but essentially it is a programming model distinct from usual algebraic data types which associates attributes over nodes of a tree. In some sense, it can be thought as a memoized function from AST nodes to the attributes. Many compilers do utlize maps, but only for top-level declarations. [[5]](http://lambda-the-ultimate.org/node/4170#comment-63903)