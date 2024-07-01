<!--yml
category: 未分类
date: 2024-07-01 18:18:16
-->

# Well-founded recursion in Agda : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/06/well-founded-recursion-in-agda/](http://blog.ezyang.com/2010/06/well-founded-recursion-in-agda/)

Last Tuesday, Eric Mertens gave the Galois tech talk [Introducing Well-Founded Recursion](http://www.galois.com/blog/2010/06/11/tech-talk-introducing-well-founded-recursion/). I have to admit, most of this went over my head the first time I heard it. Here are some notes that I ended up writing for myself as I stepped through the code again. I suggest reading the [slides](http://code.galois.com/talk/2010/10-06-mertens.pdf) first to get a feel for the presentation. These notes are oriented towards a Haskell programmer who feels comfortable with the type system, but not Curry-Howard comfortable with the type system.

```
> module Quicksort where
>
> open import Data.Nat public using (ℕ; suc; zero)
> open import Data.List public using (List; _∷_; []; _++_; [_]; length; partition)
> open import Data.Bool public using (Bool; true; false)
> open import Data.Product public using (_×_; _,_; proj₁; proj₂)

```

Agda is a proof assistant based on intuitionistic type theory; that is, the Curry-Howard isomorphism. The *Curry-Howard isomorphism* states that things that look like types and data can also be treated as propositions and proofs, and one of the keys to understanding well-founded recursion in Agda is to freely interchange between the two, because we will use the type system as a way of making propositions about our code, which Agda will use when checking it. We’ll try to present both perspectives of the types and propositions.

Types : Data :: Propositions : Proofs

Agda needs to be convinced that your proof works: in particular, Agda wants to know if you've covered all the cases (exhaustive pattern matching, totality) and if you aren't going to procrastinate on the answer (termination). Agda is very clever when it comes to case checking: if it knows that a case couldn't be fulfilled in practice, because its type represents a falsehood, it will not ask you to fill it out. However, the termination checker frequently needs help, which is where well-founded recursion comes in.

*Warmups.*

Our first data type today is top: the type inhabited by precisely one value, unit. This is () in Haskell. Data inhabits a type the way a proof exists for a proposition; you can think of a type as a “house” in which there reside any number of inhabitants, the data types. Frequently infinitely many. You’ll see Set pop a lot: rigorously, it's the type of “small” types, with Set₁ being larger, Set₂ larger still, and so forth...

```
> data ⊤ : Set where unit : ⊤

```

Bottom is the type inhabited by nothing at all. If no proof exists for the proposition, it is false! In the same way, the top proposition is vacuously true, since we said so! At the value level, this is undefined or error “foobar” in Haskell; at the type level, it's be called Void, though no one actually uses that in real code. In Agda, these are one and the same.

```
> data ⊥ : Set where

```

We pulled in natural numbers from Data.Nat, but here's what a minimal definition would look like:

```
data ℕ : Set where
  zero : ℕ
  suc : ℕ → ℕ

```

It is worth dwelling that Agda numeric constants such as 0 or 2 are syntax sugar for zero and suc (suc zero). They also may show up in types, since Agda is dependently typed. (In Haskell, you’d have to push the definition of natural numbers into the type system; here we can write a normal data definition and then lift them up automatically. [Power to the working class!](http://strictlypositive.org/winging-jpgs/))

This function does something very strange:

```
> Rel : Set → Set₁
> Rel A = A → A → Set

```

In fact, it is equivalent to this expanded version:

```
Rel A = (_ : A) → (_ : A) → (_ : Set)

```

So the resulting type is not A → A → Set, but rather, it is something *whose* type is A, something else *whose* type is A, and as a result something whose type is Set. In Haskell terms, this is not a type function of kind `* → *`; this is more like an illegal `* -> (a -> a -> *)`.

Here is an example of a simple relation: less-than for natural numbers.

```
> data _<_ (m : ℕ) : ℕ → Set where
>   <-base : m < suc m
>   <-step : {n : ℕ} → m < n → m < suc n

```

Not so simple Agda syntax:

*   The (m : ℕ) indicates that _<_ is parametrized by m, making m, a value of type ℕ, available throughout our data constructors. Parametrization means it is also required to be the first argument of _<_; at this point, you should check all of the type signatures of the constructors and see that they really are of form m<_
*   The {n : ℕ} indicates an “implicit” parameter, which means when we go to invoke <-step, we don't need to pass it; Agda will automatically figure it out from a later argument, in this case, m < n.
*   Remember that “for all x : A, y : B”, is the same as providing a total function f(x : A) : B. So there's a convenient shorthand ∀ x → which is equivalent to (x : _) → (the underscore means any type is ok.)

With the syntax out of the way, the mathematical intent of this expression should be clear: for any number, we automatically get a proof m<m+1; and with m<n → m<n+1, we can inductively get the rest of the proofs. If you squint, you can also see what is meant in terms of data: <-base is a nullary constructor, whereas <-step is a recursive constructor.

Let's prove that 3 < 5\. We start off with <-base : 3 < 4 (how did we know we should start there and not with 4 < 5? Notice that m, our parameter, is 3: this is a hint that all of our types will be parametrized by 3.) Apply step once: 3 < suc 4 = 3 < 5, QED.

```
> example₁ : 3 < 5
> example₁ = <-step <-base

```

Recall that true propositions are inhabited by data types, whereas false propositions are not. How can we invert them? In logic, we could say, “Suppose that the proposition was the case; derive a contradiction.” In types, we use the empty function: the function that has no domain, and thus while existing happily, can’t take any inputs. A function has no domain only if it’s input type is not inhabited, so the only way we can avoid having to give a contradiction is to... not let them ask the question in the first place!

```
> _≮_ : Rel ℕ
> a ≮ b = a < b → ⊥

```

() denotes falsity, in this case () : 5 < 0, which clearly can never be true, since <-base doesn’t pattern match against it (suc m != 0). A point worth mentioning is that Agda requires your programs to be complete, but doesn't ask you to pattern match against absurdities.

```
> example₂ : 5 ≮ 2
> example₂ (<-step (<-step ()))

```

*Well-foundedness.*

We introduce a little Agda notation; modules let us parametrize over some variable over an extended block then just the constructors of a ‘data’ declaration. Members of a module can be accessed ala WF.Well-founded A (rest of the arguments). This is quite convenient and idiomatic, though not strictly necessary; we could have just parametrized all of the members accordingly. We also happen to be parametrizing over a type.

```
> module WF {A : Set} (_<_ : Rel A) where

```

Logically, what it means for an element to be accessible is that for all y such that y < x, y is accessible. From a data and logic view, it states if you want me to give you Acc x, the data/proof you want, you'll have to give me a proof that for all y, if you give me a proof that y < x, I can determine Acc y. Now that we're trying to prove properties about our types and functions, treating our data types as strictly data is making less and less sense.

```
>   data Acc (x : A) : Set where
>     acc : (∀ y → y < x → Acc y) → Acc x

```

The entire type A is well-founded if all elements in it are accessible. Alternatively, the entire type A is well-founded if, given an element in it, I can produce an accessibility proof for that element. Note that its type is Set; this a type, the proposition I want to prove!

```
>   Well-founded : Set
>   Well-founded = ∀ x → Acc x

```

Proof for well-foundedness on naturals related by less-than.

```
> <-ℕ-wf : WF.Well-founded _<_
> <-ℕ-wf x = WF.acc (aux x)
>   where
>     aux : ∀ x y → y < x → WF.Acc _<_ y
>     --  : (x : _) → (∀ y → y < x → WF.Acc _<_ y)
>     aux .(suc y) y <-base = <-ℕ-wf y

```

Base case, (e.g. x=5, y=4). This, conveniently enough, triggers the well-founded structural recursion on ℕ by checking if y is well-founded now.

```
>     aux .(suc x) y (<-step {x} y<x) = aux x y y<x

```

The structural recursion here is on _<_; we are peeling back the layers of <-step until y<x = <-base, as might be the case for 3<4 (but not 3<6). We're essentially appealing to a weaker proof that is still sufficient to prove what we're interested in. Notice that we are also recursing on x; actually, whatever we know about x, we knew from y<x (less information content!), so we indicate that with a dot. Eventually, x will be small enough that y is not much smaller than x (<-base).

Where do we deal with zero? Consider aux zero : ∀ y -> y < zero → WF.Acc _<_ y. This is the empty function, since y < zero = ⊥ (no ℕ is less than zero!) In fact, this is how we get away with not writing cases for yx (the upper triangle): it's equivalent to y≮x which are all bottom, and give us empty functions for free.

In fact, there is a double-structural recursion going on here, one x, and one on y<x. The structural recursion on x is on just aux, but once we conclude <-base, we do a different structural recursion on y with <-ℕ-wf. This fills out the bottom right triangle on the xy-plane split by y=x-1; the upper left triangle is not interesting, since it's just a barren wasteland of bottom.

Standard mathematical trick: if you can reduce your problem into another that you've already solved, you solved your problem!

```
> module Inverse-image-Well-founded { A B }
>   -- Should actually used ≺, but I decided it looked to similar to < for comfort.
>   (_<_ : Rel B)(f : A → B) where
>   _⊰_ : Rel A
>   x ⊰ y = f x < f y
>
>   ii-acc : ∀ {x} → WF.Acc _<_ (f x) → WF.Acc _⊰_ x
>   ii-acc (WF.acc g) = WF.acc (λ y fy<fx → ii-acc (g (f y) fy<fx))

```

The types have to look right, so we unwrap our old proof g and wrap it into a new lambda, pushing the reduction via f into our proof (i.e. WF.acc data constructor).

```
>   ii-wf : WF.Well-founded _<_ → WF.Well-founded _⊰_
>   ii-wf wf x = ii-acc (wf (f x))
>   --    wf = λ x → ii-acc (wf (f x))
>   -- I.e. of course the construction ii-acc will work for any x.

```

Here, we finally use our machinery to prove that lists, compared with their lengths, are well-founded.

```
> module <-on-length-Well-founded { A } where
>   open Inverse-image-Well-founded { List A } _<_ length public
>   wf : WF.Well-founded _⊰_
>   wf = ii-wf <-ℕ-wf

```

A little bit of scaffolding code that does not actually “change” the proof, but changes the propositions. We’ll need this for the PartitionLemma.

```
> s<s : ∀ {a b} → a < b → suc a < suc b
> s<s <-base = <-base
> s<s (<-step y) = <-step (s<s y)

```

Show that partitioning a list doesn't increase its size.

```
> module PartitionLemma { A } where
>   _≼_ : Rel (List A)
>   x ≼ y = length x < suc (length y) -- succ to let us reuse <

```

For all predicates and lists, each partition's length is less than or equal to the original length of the list. proj₁ and proj₂ are Haskell fst and snd.

```
>   partition-size : (p : A → Bool) (xs : List A)
>     → proj₁ (partition p xs) ≼ xs
>     × proj₂ (partition p xs) ≼ xs

```

Though we've expressed our proposition in terms of ≼, we still use the original < constructor. <-base actually means equality, in this context!

```
>   partition-size p [] = <-base , <-base
>   partition-size p (x ∷ xs)
>     with p x | partition p xs | partition-size p xs
>   ... | true | as , bs | as-size , bs-size = s<s as-size , <-step bs-size
>   ... | false | as , bs | as-size , bs-size = <-step as-size , s<s bs-size

```

And finally, Quicksort.

```
> module Quick {A} (p : A → A → Bool) where

```

Open the presents (proofs).

```
>   open <-on-length-Well-founded
>   open PartitionLemma
>   quicksort' : (xs : List A) → WF.Acc _⊰_ xs → List A
>   quicksort' [] _ = []
>   quicksort' (x ∷ xs) (WF.acc g) ::

```

From the partition lemma, we get small ≼ xs and big ≼ xs. By making length well-founded, we are now able to “glue” together the layer of indirection: x ∷ xs originally was strictly smallers and structurally recursive, and the partition lemma lets us say to the termination checker that small, big and xs are essentially the same.

```
>     with partition (p x) xs | partition-size (p x) xs
>   ... | small , big | small-size , big-size = small' ++ [ x ] ++ big'
>     where
>       small' = quicksort' small (g small small-size)
>       big' = quicksort' big (g big big-size)
>   quicksort : List A → List A
>   quicksort xs = quicksort' xs (wf xs)

```