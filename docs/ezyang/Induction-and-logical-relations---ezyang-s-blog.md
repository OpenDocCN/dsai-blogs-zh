<!--yml
category: 未分类
date: 2024-07-01 18:17:18
-->

# Induction and logical relations : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/09/induction-and-logical-relations/](http://blog.ezyang.com/2013/09/induction-and-logical-relations/)

Logical relations are a proof technique which allow you to prove things such as normalization (*all programs terminate*) and program equivalence (*these two programs are observationally equivalent under all program contexts*). If you haven't ever encountered these before, I highly recommend [Amal Ahmed's OPLSS lectures](http://www.cs.uoregon.edu/research/summerschool/summer13/curriculum.html) on the subject; you can find videos and notes from yours truly. (You should also be able to access her lectures from previous years.) This post is an excuse to talk about [a formalization of two logical relations proofs in Agda](https://github.com/ezyang/lr-agda/blob/master/STLC-CBV.agda) I worked on during OPLSS and the weeks afterwards. I'm not going to walk through the code, but I do want expand on two points about logical relations:

1.  They work when simple induction would not, and
2.  The logical relation is not an inductive definition.

The full development is in the [lr-agda repository on GitHub](https://github.com/ezyang/lr-agda/). Many thanks to Dan Licata for providing the initial development as a homework assignment for his OPLSS Agda course and for bushwhacking the substitution lemmas which are often the greatest impediment to doing proofs about the lambda calculus.

* * *

If you didn't know any better, you might try to prove normalization inductively, as follows:

> To show that all programs normalize to a value, let us proceed inductively on the typing derivations. For example, in the application case, we need to show `e1 e2` normalizes to some value `v`, given that `e1` normalizes to `v1` and `e2` normalizes to `v2`. Well, the type of `v1` is `t1 -> t2`, which means `v1 = λx. e'`. Uh oh: this should step to `e'[v2/x]`, but I don’t know anything about this expression (`e'` could be anything). Stuck!

What is the extra *oomph* that logical relations gives you, that allows you to prove what was previously unprovable by usual induction? Let's think about our second proof sketch: the problem was that we didn't know anything `e'`. If we knew something extra about it, say, "Well, for some appropriate v, e'[v/x] will normalize," then we'd be able to make the proof go through. So if this definition of `WN` was our old proof goal:

```
WN : (τ : Tp) → [] ⊢ τ → Set
WN τ e = e ⇓

```

then what we'd like to do is extend this definition to include that "extra stuff":

```
WN : (τ : Tp) → [] ⊢ τ → Set
WN τ e = e ⇓ × WN' τ e

```

> At this point, it would be good to make some remarks about how to read the Agda code here. WN is a *type family*, that is, `WN τ e` is the unary logical relation of type `τ` for expression `e`. The type of a type is `Tp`, which is a simple inductive definition; the type of a term is a more complicated `[] ⊢ τ` (utilizing Agda's mixfix operators; without them, you might write it `Expr [] τ`), which not only tells us that `e` is an expression, but *well-typed*, having the type `τ` under the empty context `[]`. (This is an instance of the general pattern, where an inductive definition coincides with a well-formedness derivation, in this case the typing derivation.) `e ⇓` is another mixfix operator, which is defined to be a traditional normalization (there exists some value v such that e reduces to v, e.g. `Σ (λ v → value v × e ↦* v)`).

But what is this extra stuff? In the case of simple types, e.g. booleans, we don't actually need anything extra, since we're never going to try to apply it like a function:

```
WN' : (τ : Tp) → [] ⊢ τ → Set
WN' bool e = Unit

```

For a function type, let us say that a function is WN (i.e. in the logical relation) if, when given a WN argument, it produces a WN result. (Because we refer to WN, this is in fact a mutually recursive definition.) This statement is in fact the key proof idea!

```
WN' (τ1 ⇒ τ2) e = (e1 : [] ⊢ τ1) → WN τ1 e1 → WN τ2 (app e e1)

```

There are a number of further details, but essentially, when you redo the proof, proving WN instead of plain old normalization, you no longer get stuck on the application case. Great! The flip-side, however, is that the proof in the lambda case is no longer trivial; you have to do a bit of work to show that the extra stuff (WN') holds. This was described to me as the "balloon" principle.

The two sides of the balloon are the "use of the inductive hypothesis" and the "proof obligation". When you have a weak inductive hypothesis, it doesn't give you very much, but you don't have to work as hard to prove it either. When you strengthen the inductive hypothesis, you can prove more things with it; however, your proof obligation is correspondingly increased. In the context of a normalization proof, the "use of the inductive hypothesis" shows up in the application case, and the "proof obligation" shows up in the lambda case. When you attempt the straightforward inductive proof, the lambda case is trivial, but the inductive hypothesis is so weak that the application case is impossible. In the logical relations proof, the application case falls out easily from the induction hypothesis, but you have to do more work in the lambda case.

* * *

We now briefly take a step back, to remark about the way we have defined the WN' type family, on our way to the discussion of why WN' is not an *inductive definition*. In Agda, there are often two ways of defining a type family: it can be done as a recursive function, or it can be done as an inductive definitions. A simple example of this is in the definition of a length indexed list. The standard inductive definition goes like this:

```
data Vec : Set → Nat → Set where
  vnil : {A : Set} → Vec A 0
  vcons : {A : Set} → A → Vec A n → Vec A (S n)

```

But I could also build the list out of regular old products, using a recursive function on the index:

```
Vec : Set → Nat → Set
Vec A 0 = Unit
Vec A (S n) = A × Vec A n

```

The two different encodings have their own strengths and weaknesses: using a recursive function often means that certain equalities are definitional (whereas you'd have to prove a lemma with the inductive definition), but an inductive definition lets you do case-analysis on the different possibilities.

Sometimes, it is simply not possible to do the inductive definition, and this is the case for a logical relation. This strengthening of the inductive hypothesis is to blame:

```
data WN' : τ → [] ⊢ τ → Set where
  WN/bool : {e : [] ⊢ bool} → WN' bool e
  WN/⇒ : {e1 : [] ⊢ τ1} → (WN τ1 e1 → WN τ2 (app e e1)) → WN' (τ1 ⇒ τ2) e

```

Agda doesn't complain about inductive-recursive definitions (though one should beware: they are not too well metatheoretically grounded at the moment), but it will complain about this definition. The problem is [a familiar one](http://blog.ezyang.com/2012/09/y-combinator-and-strict-positivity/): WN does not occur in strictly positive positions; in particular, it shows up as an argument to a function contained by the WN/⇒ constructor. So we can't use this!

As it turns out, the inability to define the logical relation inductively is not a big deal for normalization. However, it causes a bit more headaches for more complicated logical relations proofs, e.g. for equivalence of programs. When considering program equivalence, you need a binary relation relating values to values, saying when two values are equal. This can be stated very naturally inductively:

```
data V'⟦_⟧ : (τ : Tp) → [] ⊢ τ → [] ⊢ τ → Set where
   V/bool-#t : V'⟦ bool ⟧ #t #t
   V/bool-#f : V'⟦ bool ⟧ #f #f
   V/⇒ : {τ₁ τ₂ : Tp} {e₁ e₂ : [] ,, τ₁ ⊢ τ₂}
     → ((v₁ v₂ : [] ⊢ τ₁) → V'⟦ τ₁ ⟧ v₁ v₂ → E⟦ τ₂ ⟧ (subst1 v₁ e₁) (subst1 v₂ e₂))
     → V'⟦ τ₁ ⇒ τ₂ ⟧ (lam e₁) (lam e₂)

```

We define the relation by type. If a value is a boolean, then we say that `#t` (true) is related to itself, and `#f` is related to itself. If the value is a function, then we say that a lambda term is related to another lambda term if, when applied to two related values, the results are also related. This "function" is directly analogous to the extra stuff we added for the normalization proof. (If you like, you can mentally replace "related" with "equal", but this is misleading since it doesn't capture what is going on in the function case). But this fails the strict positivity check, so we have to define it recursively instead:

```
V⟦_⟧ : (τ : Tp) → [] ⊢ τ → [] ⊢ τ → Set
V⟦ bool ⟧    #t #t = Unit
V⟦ bool ⟧    #f #f = Unit
V⟦ bool ⟧ _ _  = Void
V⟦ τ₁ ⇒ τ₂ ⟧ (lam e) (lam e') = (v v' : [] ⊢ τ₁) → V⟦ τ₁ ⟧ v v' → E⟦ τ₂ ⟧ (subst1 v e) (subst1 v' e')
V⟦ τ₁ ⇒ τ₂ ⟧ _ _ = Void

```

Notice that the definition here is much less nice than the inductive definition: we need two fall through cases which assert a contradiction when two things could not possibly be equal, e.g. `#t` could not possibly be equal to `#f`. Furthermore, supposing that we are given V as a hypothesis while performing a proof, we can no longer just case-split on it to find out what kind of information we have; we have to laboriously first case split over the type and expression, at which point the function reduces. To give you a sense of how horrible this is, consider this function which converts from a inductive definition to the recursive definitions:

```
pV : {τ : Tp} → {e e' : [] ⊢ τ} → V⟦ τ ⟧ e e' → V'⟦ τ ⟧ e e'
pV {bool} {#t} {#t} V = V/bool-#t
pV {bool} {#t} {#f} ()
pV {bool} {#t} {if _ then _ else _} ()
pV {bool} {#t} {var _} ()
pV {bool} {#t} {app _ _} ()
pV {bool} {#f} {#t} ()
pV {bool} {#f} {#f} V = V/bool-#f
pV {bool} {#f} {if _ then _ else _} ()
pV {bool} {#f} {var _} ()
pV {bool} {#f} {app _ _} ()
pV {bool} {if _ then _ else _} ()
pV {bool} {var _} ()
pV {bool} {app _ _} ()
pV {_ ⇒ _} {if _ then _ else _} ()
pV {_ ⇒ _} {var _} ()
pV {_ ⇒ _} {lam _} {if _ then _ else _} ()
pV {_ ⇒ _} {lam _} {var _} ()
pV {_ ⇒ _} {lam _} {lam _} f = V/⇒ (\ v v' V → pE (f v v' V))
pV {_ ⇒ _} {lam _} {app _ _} ()
pV {_ ⇒ _} {app _ _} ()

```

Good grief! Perhaps the situation could be improved by improving how Agda handles wildcards in pattern matching, but at the moment, all of this is necessary.

"But wait, Edward!" you might say, "Didn't you just say you couldn't define it inductively?" Indeed, this function does not operate on the inductive definition I presented previously, but a slightly modified one, which banishes the non-strictly-positive occurrence by replacing it with V, the recursive definition:

```
V/⇒ : {τ₁ τ₂ : Tp} {e e' : [] ,, τ₁ ⊢ τ₂}
  → (V : (v v' : [] ⊢ τ₁) → V⟦ τ₁ ⟧ v v' {- the critical position! -} → E'⟦ τ₂ ⟧ (subst1 v e) (subst1 v' e'))
  → V'⟦ τ₁ ⇒ τ₂ ⟧ (lam e) (lam e')

```

This conversion function helps a lot, because agda-mode interacts a lot more nicely with inductive definitions (`C-c C-c` works!) than with recursive definitions in cases like this.

* * *

Why do logical relations in Agda? (or any proof assistant, for that matter?) Proofs using logical relations often follow the pattern of defining an appropriate logical relation for your problem, and then a lot of book-keeping to actually push the relation through a proof. Computers are great for doing book-keeping, and I think it is hugely informative to work through a logical relation proof in a proof assistant. An interesting challenge would be to extend this framework to a non-terminating language (adding step-indexes to the relation: the very pinnacle of book-keeping) or extending the lambda calculus with polymorphism (which requires some other interesting techniques for logical relations).