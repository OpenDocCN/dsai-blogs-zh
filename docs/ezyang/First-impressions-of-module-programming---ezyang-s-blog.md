<!--yml
category: 未分类
date: 2024-07-01 18:17:41
-->

# First impressions of module programming : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/08/first-impressions-of-module-programming/](http://blog.ezyang.com/2011/08/first-impressions-of-module-programming/)

During my time at Jane Street, I’ve done a fair bit of programming involving modules. I’ve touched functors, type and module constraints, modules in modules, even first class modules (though only peripherally). Unfortunately, the chapter on modules in *Advanced Topics in Types and Programming Languages* made my eyes glaze over, so I can’t really call myself *knowledgeable* in module systems yet, but I think I have used them enough to have a few remarks about them. (All remarks about convention should be taken to be indicative of Jane Street style. Note: they’ve [open sourced](http://ocaml.janestreet.com/?q=node/13) a bit of their software, if you actually want to look at some of the stuff I’m talking about.)

The good news is that they basically work the way you expect them to. In fact, they’re quite nifty. The most basic idiom you notice when beginning to use a codebase that uses modules a lot is you see this:

```
module Sexp = struct
  type t = ...
  ...
end

```

There is in fact a place where I have seen this style before: Henning Thielemann’s code on Hackage, in particular [data-accessor](http://hackage.haskell.org/package/data-accessor), which I have [covered previously](http://blog.ezyang.com/2010/04/inessential-guide-to-data-accessor/). Unlike in Haskell, this style actually makes sense in OCaml, because you never `include Sexp` (an unqualified import in Haskell lingo) in the conventional sense, you usually refer to the type as `Sexp.t`. So the basic unit of abstraction can be thought of as a type—and most simple modules are exactly this—but you can auxiliary types and functions that operate on that type. This is pretty simple to understand, and you can mostly parse the module system as a convenient namespacing mechanism.

Then things get fun.

When you use Haskell type classes, each function individually specifies what constraints on the argument there are. OCaml doesn’t have any type classes, so if you want to do that, you have to manually pass the dictionary to a function. You can do that, but it’s annoying, and OCaml programmers think bigger. So instead of passing a dictionary to a function, you pass a module to a functor, and you specialize all of your “generic” functions at once. It’s more powerful, and this power gets over the annoyance of having to explicitly specify what module your using at any given time. Constraints and modules-in-modules fall out naturally from this basic idea, when you actually try to use the module system in practice.

Probably the hardest thing (for me) to understand about the module system is how type inference and checking operate over it. Part of this is the impedance mismatch with how type classes work. When I have a function:

```
f :: Monoid m => m -> Int -> m

```

`m` is a polymorphic value that can get unified with any specific type. So if I do `f 5 + 2`, that’s completely fair game if I have an appropriate Monoid instance defined for `Int` (even though `+` is not a Monoid instance method.)

However, if I do the same trick with modules, I have to be careful about adding extra type constraints to teach the compiler that some types are, indeed, the same. Here is an example of an extra type restriction that feels like it should get unified away, but doesn’t:

```
module type SIG = sig
    type t
    val t_of_string : string -> t
end

module N : SIG = struct
    type t = string
    let t_of_string x = x
end

let () = print_endline (N.t_of_string "foo")

```

Actually, you have to specify that `t` and `string` are the same when you add that `SIG` declaration:

```
module N : SIG with type t = string = struct

```

Funny! (Actually, it gets more annoying when you’re specifying constraints for large amounts of types, not just one.) It’s also tricky to get right when functors are involved, and there were some bugs in pre-3.12 OCaml which meant that you had to do some ugly things to ensure you could actually write the type constraints you wanted (`with type t = t`... those `ts` are different...)

There are some times, however, when you feel like you would really, really like typeclasses in OCaml. Heavily polymorphic functionality tends to be the big one: if you have something like `Sexpable` (types that can be converted into S-expressions), using the module system feels very much like duck typing: if it has a `sexp_of_t` function, and it’s typed right, it’s “sexpable.” Goodness, most of the hairy functors in our base library are because we need to handle the moral equivalent of multiparameter type classes.

Monadic bind is, of course, hopeless. Well, it works OK if you’re only using one monad in your program (then you just specialize your `>>=` to that module’s implementation by opening the module). But in most applications you’re usually in one specific monad, and if you want to quickly drop into the `option` monad you’re out of luck. Or you could redefine the operator to be `>>=~` and hope no one stabs you. `:-)`