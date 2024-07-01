<!--yml
category: 未分类
date: 2024-07-01 18:17:03
-->

# The problem of reusable and composable specifications : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/12/the-problem-of-reusable-and-composable-specifications/](http://blog.ezyang.com/2016/12/the-problem-of-reusable-and-composable-specifications/)

It's not too hard to convince people that version bounds are poor approximation for a particular API that we depend on. What do we mean when we say `>= 1.0 && < 1.1`? A version bound is a proxy some set of modules and functions with some particular semantics that a library needs to be built. Version bounds are imprecise; what does a change from 1.0 to 1.1 mean? Clearly, we should instead write down the actual specification (either types or contracts) of what we need.

This all sounds like a good idea until you actually try to put it into practice, at which point you realize that version numbers had one great virtue: they're very short. Specifications, on the other hand, can get quite large: even just writing down the types of all the functions you depend on can take pages, let alone executable contracts describing more complex behavior. To make matters worse, the same function will be depended upon repeatedly; the specification must be provided in each case!

So we put on our PL hats and say, "Aha! What we need is a mechanism for *reuse* and *composition* of specifications. Something like... a *language* of specification!" But at this point, there is disagreement about how this language should work.

**Specifications are code.** If you talk to a Racketeer, they'll say, "Well, [contracts](https://docs.racket-lang.org/reference/contracts.html) are just [code](https://docs.racket-lang.org/guide/Building_New_Contracts.html), and we know how to reuse and compose code!" You have primitive contracts to describe values, compose them together into contracts that describe functions, and then further compose these together to form contracts about modules. You can collect these contracts into modules and share them across your code.

There is one interesting bootstrapping problem: you're using your contracts to represent versions, but your contracts themselves live in a library, so should you version your contracts? Current thinking is that you shouldn't.

**But maybe you shouldn't compose them the usual way.** One of the things that stuck out to me when I was reading the frontmatter of Clojure's spec documentation is that [map specs should be of keysets only](http://clojure.org/about/spec#_map_specs_should_be_of_keysets_only), and [how they deal with it](http://clojure.org/about/spec#_global_namespaced_names_are_more_important).

The core principle of spec's design is that specifications for records should NOT take the form `{ name: string, age: int }`. Instead, the specification is split into two pieces: a set of keys `{ name, age }`, and a mapping from keys to specifications which, once registered, apply to all occurrences of a key in all map specifications. (Note that keys are all namespaced, so it is not some insane free-for-all in a global namespace.) The justification for this:

> In Clojure we gain power by dynamically composing, merging and building up maps. We routinely deal with optional and partial data, data produced by unreliable external sources, dynamic queries etc. These maps represent various sets, subsets, intersections and unions of the same keys, and in general ought to have the same semantic for the same key wherever it is used. Defining specifications of every subset/union/intersection, and then redundantly stating the semantic of each key is both an antipattern and unworkable in the most dynamic cases.

**Back to the land of types.** Contracts can do all this because they are code, and we know how to reuse code. But in (non-dependently) typed languages, the language of types tends to be far more impoverished than than the language of values. To take Backpack as an (unusually expressive) example, the only operations we can perform on signatures is to define them (with full definitions for types) and to merge them together. So Backpack signatures run head long into the redundancy problem identified by spec: because the signature of a module includes the signatures of its functions, you end up having to repeat these function signatures whenever you write slightly different iterations of a module.

To adopt the Clojure model, you would have to write a separate signature per module (each in their own package), and then have users combine them together by adding a `build-depends` on every signature they wanted to use:

```
-- In Queue-push package
signature Queue where
  data Queue a
  push :: a -> Queue a -> Queue a

-- In Queue-pop package
signature Queue where
  data Queue a
  pop :: Queue a -> Maybe (Queue a, a)

-- In Queue-length package
signature Queue where
  data Queue a
  length :: Queue a -> Int

-- Putting them together (note that Queue is defined
-- in each signature; mix-in linking merges these
-- abstract data types together)
build-depends: Queue-push, Queue-pop, Queue-length

```

In our current implementation of Backpack, this is kind of insane: to write the specification for a module with a hundred methods, you'd need a hundred packages. The ability to concisely define multiple public libraries in a single package might help but this involves design that doesn't exist yet. (Perhaps the cure is worse than the disease. The package manager-compiler stratification rears its ugly head again!) (Note to self: signature packages ought to be treated specially; they really shouldn't be built when you instantiate them.)

**Conclusions.** A lot of my thinking here did not crystallize until I started reading about how dynamic languages like Clojure were grappling with the specification problem: I think this just goes to show how much we can learn by paying attention to other systems, even if their context is quite different. (If Clojure believed in data abstraction, I think they could learn a thing or two from how Backpack mix-in links abstract data declarations.)

In Clojure, the inability to reuse specs is a deal breaker which lead them to spec's current design. In Haskell, the inability to reuse type signatures flirts on the edge of unusability: types are *just* short enough and copy-pasteable enough to be tolerable. Documentation for these types, less so; this is what lead me down my search for better mechanisms for signature reuse.

Although Backpack's current design is "good enough" to get things done, I still wonder if we can't do something better. One tempting option is to allow for downstream signatures to selectively pick out certain functions from a larger signature file to add to their requirements. But if you require `Queue.push`, you had better also require `Queue.Queue` (without which, the type of `push` cannot even be stated: the avoidance problem); this could lead to a great deal of mystery as to what exactly is required in the end. Food for thought.