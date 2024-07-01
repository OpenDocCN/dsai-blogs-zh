<!--yml
category: 未分类
date: 2024-07-01 18:17:15
-->

# Equality, roughly speaking : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/01/equality-roughly-speaking/](http://blog.ezyang.com/2014/01/equality-roughly-speaking/)

## Equality, roughly speaking

In Software Foundations, equality is [defined in this way](http://www.cis.upenn.edu/~bcpierce/sf/Logic.html#lab220):

> Even Coq's equality relation is not built in. It has (roughly) the following inductive definition.
> 
> ```
> Inductive eq0 {X:Type} : X -> X -> Prop :=
>   refl_equal0 : forall x, eq0 x x.
> 
> ```

*Why the roughly?* Well, as it turns out, Coq defines equality a little differently (reformatted to match the Software Foundations presentation):

```
Inductive eq1 {X:Type} (x:X) : X -> Prop :=
  refl_equal1 : eq1 x x.

```

What’s the difference? The trick is to look at the induction principles that Coq generates for each of these:

```
eq0_ind
   : forall (X : Type) (P : X -> X -> Prop),
     (forall x : X, P x x) -> forall y y0 : X, eq0 y y0 -> P y y0

eq1_ind
   : forall (X : Type) (x : X) (P : X -> Prop),
     P x -> forall y : X, eq1 x y -> P y

```

During our Homotopy Type Theory reading group, Jeremy pointed out that the difference between these two principles is exactly the difference between path induction (eq0) and based path induction (eq1). (This is covered in the [Homotopy Type Theory book](http://homotopytypetheory.org/book/) in section 1.12) So, Coq uses the slightly weirder definition because it happens to be a bit more convenient. (I’m sure this is folklore, but I sure didn’t notice this until now! For more reading, check out this [excellent blog post by Dan Licata](http://homotopytypetheory.org/2011/04/10/just-kidding-understanding-identity-elimination-in-homotopy-type-theory/).)