<!--yml
category: 未分类
date: 2024-07-01 18:18:21
-->

# Inessential guide to fclabels : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/04/inessential-guide-to-fclabels/](http://blog.ezyang.com/2010/04/inessential-guide-to-fclabels/)

Last time I did an [Inessential guide to data-accessor](http://blog.ezyang.com/2010/04/inessential-guide-to-data-accessor/) and everyone told me, "You should use fclabels instead!" So here's the partner guide, the inessential guide to fclabels. Like data-accessor the goal is to make record access and editing not suck. However, it gives you some more useful abstractions. It uses Template Haskell on top of your records, so it is not compatible with data-accessor.

*Identification.* There are three tell-tale signs:

1.  Type signatures that contain `:->` in them ("Oh, that kind of looks like a function arrow... but it's not? Curious!"),
2.  Records that contain fields with a leading underscore (as opposed to data-accessor's convention of an trailing underscore), and
3.  An `import Prelude hiding (id, (.), mod)`, with an import from `Control.Category` to replace them.

*Interpreting types.* A *label* is signified by `r :-> a` which contains a getter `r -> a` and a setter `a -> r -> r`. Internally, a wrapped label is simply a *point*, a structure consisting of `r -> a` and `b -> r -> r`, with `a` required to be equal to `b`. (As we will see later, a point is useful in its own right, but not for basic functionality.)

*Accessing record fields.*

```
get fieldname record

```

*Setting record fields.*

```
set fieldname newval record

```

*Modifying record fields.* For `fieldname :: f a :-> a`, `modifier` should have type `a -> a`.

```
mod fieldname modifier record

```

*Accessing, setting and modifying sub-record fields.* Composition is done with the period operator `(.)`, but you can't use the one from the Prelude since that only works with functions. The composition is treated as if you were you composing the getter.

```
get (innerField . outerField) record
set (innerField . outerField) newVal record
mod (innerField . outerField) modifier record

```

*Accessor over applicative.* You can use `fmapL` to lift an accessor into an applicative context. This is useful if your record is actually `Maybe r` (You can turn `r :-> a` into `Maybe r :-> Maybe a`).

But wait, there's more!

*More fun with views.* Remember that a point is a getter and a setter, but they don't have to be for the same types. Combined with a clever applicative instance, we can use this to incrementally build up a label composed of multiple labels. The result looks a lot like a view that you'd be able to create on a relational database. The recipe is:

1.  Have the constructor for the resulting type (e.g. `(,)`, the tuple constructor),
2.  Have all of the accessors for the resulting type (e.g. `fst` and `snd`), and
3.  Have the labels you would like to compose together (say, `label1` and `label2`).

Combine, with `for`, each accessor for the resulting type (2) with the label to be accessed with that accessor (3), combine all of these resulting points with the constructor for the resulting type with the applicative instance, i.e. `<$>` and `<*>`, and then stick it in a label with `Label`:

```
(,) <$> fst `for` label1 <*> snd `for` label2

```

Amazingly, you won't be able to mix up which argument an accessor (2) should be placed in; the result won't typecheck! (See the *Postscript* for a more detailed argument.)

*More fun with lenses.* A function implies directionality: a to b. But light can filter through a lense either way, and thus a lense represents a bidirectional function. We can apply filter a label `f :-> a` through a lense `a :<->: b` to get a new label `f :-> b` (remember that composition with a regular function is insufficient since we need to put values in as well as take values out). One has to be careful about what direction your lense is pointed. If `label :: r :-> a`, `in :: b -> a` and `out :: a -> b`, then:

```
(out <-> in) `iso` label :: r :-> b
(in <-> out) `osi` label :: r :-> b

```

The other directions won't typecheck if `a != b`.

You can lift a lense into a functor using `lmap` (it simply runs `fmap` on both directions).

*Further reading.* The [Hackage documentation](http://hackage.haskell.org/package/fclabels) has a ton of excellent examples.

*Postscript.* With our original example in mind:

```
label1 :: r -> a
label2 :: r -> b
(,) <$> fst `for` label1 <*> snd `for` label2 :: r :-> (a, b)

```

We consider the types of the points we've constructed, before combining them with the applicative instance:

```
fst `for` label1 :: Point Person (a, b) a
snd `for` label2 :: Point Person (a, b) b

```

We have a shared applicative functor `Point Person (a, b)`, and if we treat that as `f`, clearly:

```
(,) :: a -> b -> (a, b)
fst `for` label1 :: f a
snd `for` label2 :: f b
(,) <$> fst `for` label1 <*> snd `for` label2 :: f (a, b)

```

which is equivalent to `Point Person (a, b) (a, b)`, which is a valid `Label`.

But what is `for` doing? The source code documentation says:

> Combine a partial destructor with a label into something easily used in the applicative instance for the hidden Point datatype. Internally uses the covariant in getter, contravariant in setter bi-functioral-map function. (Please refer to the example because this function is just not explainable on its own.)

Well, I'm going to ignore this advice, since you've seen the example already. Let's parse this. `for` is covariant in getter `r -> a` and contravariant in setter `a -> f -> f`. These terms are from category theory describing functors. A covariant functor is a "normal" functor, whereas a contravariant functor is one with composition flipped around. So while normally `fmap f g == f . g`, in the contravariant world `fmap f g == g . f`:

```
for :: (i -> o) -> (f :-> o) -> Point f i o
for a b = dimap id a (unLabel b)

```

Well, we're not doing much interesting to the getter, but we're mapping `a :: (a, b) -> a` (in our example) onto the setter `a -> f -> f`. Luckily (for the befuddled), the covariant map doesn't typecheck (`(a, b) != (f -> f)`), but the contravariant map does: `(a, b) -> f -> f`, which is a new setter that takes `(a, b)`, precisely what we expected from the type signature.

So, `for` sets up our setters and partially our getter, and the applicative instance finishes setting up our getter.