<!--yml
category: 未分类
date: 2024-07-01 18:16:57
-->

# A compile-time debugger that helps you write tensor shape checks : ezyang’s blog

> 来源：[http://blog.ezyang.com/2018/04/a-compile-time-debugger-that-helps-you-write-tensor-shape-checks/](http://blog.ezyang.com/2018/04/a-compile-time-debugger-that-helps-you-write-tensor-shape-checks/)

A run-time debugger allows you to see concrete values in a program, make changes to them, and continue running your program. A **compile-time debugger** allows you to see symbolic values in a program, reason about them, and write the rest of your program, e.g. filling in missing tensor size checks, for example.

Here's an example of a compiler-time debugger in action.

Let's suppose you are writing a simple program to read a pair of tensors from two files and do a matrix multiply on them. "Easy," you think, while writing the following program:

```
main() {
  x = load("tensor1.t")
  y = load("tensor2.t")
  return matmul(x, y)
}

```

However, there is a twist: this matrix multiply is an *unchecked* matrix multiply. If you pass it tensors which cannot be validly multiplied together, this is undefined behavior. Your compiler has cottoned up to this fact and is refusing to compile your program. You fire up your compile-time debugger, and it drops you to the point of your program right before the error:

```
# Ahoy there Edward!  I stopped your program, because I could not
# prove that execution of this line was definitely safe:

   main() {
     x = load("tensor1.t")
     y = load("tensor2.t")
->   return matmul(x, y)
   }

# Here's what's in scope:

  _x_size : List(Nat)  # erases to x.size()
  _y_size : List(Nat)  # erases to y.size()
  x : Tensor(_x_size)
  y : Tensor(_y_size)

# I don't know anything else about these values

```

Let's take a careful look at the variables in scope. Our compile-time debugger tells us the type of a variable x by writing `x : t`, meaning that x has type t. We have all sorts of ordinary types like natural numbers (Nat) and lists of natural numbers (List(Nat)). More interestingly, a *tensor* is parametrized by a list of natural numbers, which specify their sizes at each dimension. (For simplicity, the underlying field of the tensor is assumed to be fixed.)

Our debugger has a command line, so we can ask it questions about the types of things in our program (`:t` is for type):

```
> :t 1
# Here's the type of 1, which you requested:
Nat

> :t [1, 2, 0]
# Here's the type of [1, 2, 0], which you requested:
List(Nat)

> :t matmul
# Here's the type of matmul, which you requested:
forall (a, b, c : Nat). (Tensor([a, b]), Tensor([b, c])) -> Tensor([a, c])

```

The type of matrix multiplication should make sense. We say a matrix multiply takes two 2-D tensors of sizes AxB and BxC, and produces a tensor of size AxC. An equivalent way of phrasing, as was done in the type above, is to say, “for any natural numbers A, B and C, matrix multiply will take a tensor of size AxB and a tensor of BxC, and give you a tensor of size AxC”.

It is also instructive to see what the type of `load` is:

```
> :t load
# Here's the type of load, which you requested:
String ~> exists (size : List(Nat)). Tensor(size)

```

We do not know what the dimensions of a tensor loaded from a file are; all we can say is that *there exists* some size (list of natural numbers), which describes the tensor in question. Our compile-time debugger has helpfully given us names for the sizes of our tensors in scope, `_x_size` and `_y_size`, and has also told us how to compute them at runtime (`x.size()` and `y.size()`).

Enough of this. Let's remind ourselves why our program has failed to typecheck:

```
> matmul(x, y)

# I'm sorry!  I was trying to find values of a, b and c which
# would make the following equations true:
#
#     [a, b] = _x_size
#     [b, c] = _y_size
#
# But I don't know anything about _x_size or _y_size (they are skolem
# variables), so I couldn't do it.  Cowardly bailing out!

```

The compiler is absolutely right. We don't know anything about the size of x or y; they could be 2D, or they could be 100D, or not have matching dimensions.

As an aside: sometimes, it's OK not to know anything about the sizes. Consider the case of adding a tensor to itself:

```
> add
# Here's the type of add, which you requested!
add : forall (size : List(Nat)). Tensor(size) -> Tensor(size) -> Tensor(size)

> add(x, x)
Tensor(_x_size)

# This type-checked OK!  I set size = _x_size and all of the arguments
# checked out.  You're good to go.

```

We don't know anything about `_x_size`, but `add` doesn't care; it'll take any `List(Nat)`, and `_x_size` is certainly one of those.

Back to business. We are going to insert dynamic checks will will refine our knowledge of x and y, until it is obvious that matrix multiply will succeed.

What is a dynamic check? Operationally, a dynamic check tests whether or not some condition is true, and aborts if it is not. If we successfully run the dynamic check, we now have new *information* about the symbolic types in our scope. So for example, after adding a *runtime* test that two numbers are equal, we can subsequently assume at *compile time* that the numbers are equal:

```
> :t assert_eq_nat!
(x : Nat) -> (y : Nat) ~> x = y

```

First things first, we'd like to assert that our tensors are 2D tensors:

```
> assert_eq_nat!(len(_x_size), 2)

# OK!  I added assert_eq_nat!(len(x.size()), 2) to your program, and
# here's what I know now:

  _x_size : List(Nat)
  _y_size : List(Nat)
  x : Tensor(_x_size)
  y : Tensor(_y_size)
  len(_x_size) = 2

# By the way, I can profitably destruct _x_size into its constituent
# parts; would you like to do this? (Y/n)

> Y

# OK, your new context is this:

  _x0, _x1 : Nat
  _y_size : List(Nat)
  x : Tensor([_x0, _x1])
  y : Tensor(_y_size)

# I don't know anything about the new variables _x0 and _x1, but I
# learned enough about _x_size that I was able to eliminate it from
# the context (_x_size = [_x0, _x1])

```

List length is a very helpful property to test against, since it greatly specifies the shape of the list in question. We can do the same for `_y_size`:

```
> assert_eq_nat!(len(_y_size), 2)

# OK!  I added assert_eq_nat!(len(y.size()), 2) to your program, and
# here's what I know now:

  _x0, _x1 : Nat  # erases to x.size(0), x.size(1)
  _y_size : List(Nat)
  x : Tensor([_x0, _x1])
  y : Tensor(_y_size)
  len(_y_size) = 2

# By the way, I can profitably destruct _y_size into its constituent
# parts; would you like to do this? (Y/n)

> Y

# OK, your new context is this:

  _x0, _x1 : Nat  # erases to x.size(0), x.size(1)
  _y0, _y1 : Nat  # erases to y.size(0), y.size(1)
  x : Tensor([_x0, _x1])
  y : Tensor([_y0, _y1])

# I don't know anything about the new variables _y0 and _y1, but I
# learned enough about _y_size  that I was able to eliminate it from
# the context (_y_size = [_y0, _y1])

```

We're very close now. All we need to do is assert that the inner dimensions are equal:

```
> assert_eq_nat!(_x1, _y0)

# OK!  I added assert_eq_nat!(x.size(1), y.size(0)) to your program.
# After doing this, I learned _x1 = _y0, so I replaced all occurrences
# of _y0 with _x1\. Now the context looks like this.

  _x0, _x1 : Nat  # erases to x.size(0), x.size(1)
  _y1 : Nat  # erases to y1.size(1)
  x : Tensor([_x0, _x1])
  y : Tensor([_x1, _y1])

```

Victory!

```
> matmul(x, y)

# This type-checked OK!  I set a = _x0, b = _x1, c = _y1 and all of the
# arguments checked out.  You're good to go.

```

Extracting the contents of this session back into our code, we now have:

```
  main() {
    x = load("tensor1.t")
    y = load("tensor2.t")
    assert_eq_nat!(x.size(), 2)
    assert_eq_nat!(y.size(), 2)
    assert_eq_nat!(x.size(1), y.size(0))
    matmul(x, y)
}

```

* * *

At this point, I must come clean: the compile time debugger I've described above doesn't actually exist. But it is not all that different from the proof modes of interactive proof assistants the automated theorem proving community works with today. But unlike theorem proving, we have a secret weapon: when the going gets tough, the tough turns into a runtime check. Conventional wisdom says that automated theorem proving requires too idealized a setting to be useful in writing software today. Conventional wisdom is wrong.