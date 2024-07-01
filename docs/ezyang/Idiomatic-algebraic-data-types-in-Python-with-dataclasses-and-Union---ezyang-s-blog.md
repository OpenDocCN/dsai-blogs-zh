<!--yml
category: 未分类
date: 2024-07-01 18:16:57
-->

# Idiomatic algebraic data types in Python with dataclasses and Union : ezyang’s blog

> 来源：[http://blog.ezyang.com/2020/10/idiomatic-algebraic-data-types-in-python-with-dataclasses-and-union/](http://blog.ezyang.com/2020/10/idiomatic-algebraic-data-types-in-python-with-dataclasses-and-union/)

One of the features I miss most in non-Haskell programming languages is algebraic data types (ADT). ADTs fulfill a similar role to objects in other languages, but with more restrictions: objects are an open universe, where clients can implement new subclasses that were not known at definition time; ADTs are a closed universe, where the definition of an ADT specifies precisely all the cases that are possible. We often think of restrictions of a bad thing, but in the case of ADTs, the restriction of being a closed universe makes programs easier to understand (a fixed set of cases to understand, as opposed to a potentially infinite set of cases) and allows for new modes of expression (pattern matching). ADTs make it really easy to accurately model your data structures; they encourage you to go for precise types that make illegal states unrepresentable. Still, it is generally not a good idea to try to manually reimplement your favorite Haskell language feature in every other programming language you use, and so for years I've suffered in Python under the impression that ADTs were a no go.

Recently, however, I have noticed that a number of new features in Python 3 have made it possible to use objects in the same style of ADTs, in idiomatic Python with virtually no boilerplate. The key features:

*   A structural static type checking system with mypy; in particular, the ability to declare `Union` types, which let you represent values that could be one of a fixed set of other types, and the ability to refine the type of a variable by performing an `isinstance` check on it.
*   The dataclasses library, which allows you to conveniently define (possibly immutable) structures of data without having to write boilerplate for the constructor.

The key idea: define each constructor as a dataclass, put the constructors together into an ADT using a Union type, and use `isinstance` tests to do pattern matching on the result. The result is just as good as an ADT (or better, perhaps; their structural nature bears more similarity to OCaml's polymorphic variants).

Here's how it works. Let's suppose that you want to define an algebraic data type with two results:

```
data Result
   = OK Int
   | Failure String

showResult :: Result -> String
showResult (OK result) = show result
showResult (Failure msg) = "Failure: " ++ msg

```

First, we define each constructor as a dataclass:

```
from dataclasses import dataclass

@dataclass(frozen=True)
class OK:
    result: int

@dataclass(frozen=True)
class Failure:
    msg: str

```

Using the automatically generated constructors from dataclasses, we can construct values of these dataclasses using `OK(2)` or `Failure("something wrong")`. Next, we define a type synonym for the union of these two classes:

```
Result = Union[OK, Failure]

```

Finally, we can do pattern matching on Result by doing `isinstance` tests:

```
def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError("Unhandled type: {}".format(type(x).__name__))

def showResult(r: Result) -> str:
    if isinstance(r, OK):
        return str(r.result)
    elif isinstance(r, Failure):
        return "Failure: " + r.msg
    else:
        assert_never(r)

```

`assert_never` is a [well known trick](https://github.com/python/typing/issues/735) for doing exhaustiveness checking in mypy. If we haven't covered all cases with enough `isinstance` checks, mypy will complain that `assert_never` was given a type like `UnhandledCtor` when it expected `NoReturn` (which is the uninhabited type in Python).

That's all there is to it. As an extra bonus, this style of writing unions is compatible with the [structured pattern matching PEP](https://www.python.org/dev/peps/pep-0634/), if it actually gets accepted. I've been using this pattern to good effect in our recent rewrite of PyTorch's code generator. If you have the opportunity to work in a statically typed Python codebase, give this style of code a try!