<!--yml
category: 未分类
date: 2024-07-01 18:16:53
-->

# A short note about functional linear maps : ezyang’s blog

> 来源：[http://blog.ezyang.com/2019/05/a-short-note-about-functional-linear-maps/](http://blog.ezyang.com/2019/05/a-short-note-about-functional-linear-maps/)

Some notes collected from a close read of Conal Elliot's [Compiling to Categories](http://conal.net/papers/compiling-to-categories/compiling-to-categories.pdf) and [The Simple Essence of Automatic Differentiation](https://arxiv.org/pdf/1804.00746.pdf).

A colleague of mine was trying to define a "tree structure" of tensors, with the hope of thereby generalizing the concept to also work with tensors that have "ragged dimensions." Let's take a look:

Suppose we have a `(2, 3)` matrix:

```
tensor([[1, 2, 3],
        [4, 5, 6]])

```

One way to think about this is that we have a "tree" of some sort, where the root of the tree branches to two subnodes, and then each subnode branches to three nodes:

```
       /- ROOT -\
  ROW 1          ROW 2
 /  |  \        /  |  \
1   2   3      4   5   6

```

Suppose you wanted to define this data structure in Haskell. One obvious way of going about doing this is to just say that a matrix is just a bunch of nested lists, `[[Float]]`. This works, true, but it isn't very illuminating, and it is certainly not type safe. Type safety could be achieved with sized vectors, but we are still left wondering, "what does it mean?"

Often, inductive definitions fall out of how we *compose* things together, in the same way that the inductive data structure for a programming language tells us how we take smaller programs and put them together to form a larger program. With matrices, we can think of a pictorial way of composing them, by either attaching matrices together vertically or horizontally. That gives us this vocabulary for putting together matrices, which would let us (non-uniquely) represent every matrix ([Compiling to Categories, Section 8](http://conal.net/papers/compiling-to-categories/compiling-to-categories.pdf)):

```
data Matrix
  = Scalar Float
  | Horizontal Matrix Matrix
  | Vertical Matrix Matrix

```

But what does it mean? Well, every matrix represents a linear map (if `A : (n, m)` is your matrix, the linear map is the function `R^m -> R^n`, defined to be `f(x) = A x`. We'll call a linear map from a to b, `Linear a b`). So the question we ask now is, what does it mean to "paste" two matrices together? It's a way of composing two linear maps together into a new linear map:

```
-- A function definition does not a category make!  You have to
-- prove that the resulting functions are linear.

horizontal :: Linear a c -> Linear b c -> Linear (a, b) c
horizontal f g = \(a, b) -> f a + g b

-- In matrix form:
--
--              [ a ]
-- [ F  |  G ]  [ - ] = [ F a + G b ]
--              [ b ]

vertical :: Linear a c -> Linear a d -> Linear a (c, d)
vertical f g = \a -> (f a, g a)

-- In matrix form:
--
-- [ F ]         [ F a ]
-- [ - ] [ a ] = [  -  ]
-- [ G ]         [ G a ]

```

Now we're cooking! Notice that the pasting *shows up* in the type of the linear map: if we paste horizontally, that just means that the vectors this linear map takes in have to be pasted together (with the tuple constructor); similarly, if we paste vertically, we'll produce output vectors that are the pasted results.

Cool, so we can add some type indexes, and write Linear as a GADT to refine the indices when you apply the constructor:

```
data Linear a b where
  Scalar :: Float -> Linear Float Float
  Horizontal :: Linear a c -> Linear b c -> Linear (a, b) c
  Vertical :: Linear a c -> Linear a d -> Linear a (c, d)

```

Is this the end of the story? Not quite. There are many ways you can go about combining linear maps; for example, you could (literally) compose two linear maps together (in the same sense of function composition). It's true that you can paste together any matrix you like with the data type above; how do we decide what should and shouldn't go in our *language* of linear maps?

To this end, Conal Elliot calls on the language of *category theory* to adjudicate. A category should define identity and function composition:

```
identity :: Linear a a
identity a = a

-- In matrix form: the identity matrix

compose :: Linear b c -> Linear a b -> Linear a c
compose g f = \a -> g (f a)

-- In matrix form: matrix multiply

```

We find that Horizontal and Vertical are the elimination and introduction operations of cocartesian and cartesian categories (respectively).

But this should we just slap Identity and Compose constructors to our data type? Linear map composition is a computationally interesting operation: if we just keep it around as syntax (rather than doing what is, morally, a matrix multiply), then it will be quite expensive to do operations on the final linear map. Where do representable functors come in? I'm not exactly sure how to explain this, and I've run out of time for this post; stay tuned for a follow up.