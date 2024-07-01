<!--yml
category: 未分类
date: 2024-07-01 18:17:16
-->

# Two bugs in the borrow checker every Rust developer should know about : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/12/two-bugs-in-the-borrow-checker-every-rust-developer-should-know-about/](http://blog.ezyang.com/2013/12/two-bugs-in-the-borrow-checker-every-rust-developer-should-know-about/)

If that’s the case, you may have run into one of the two (in)famous bugs in the borrow-checker. In this post, I want to describe these two bugs, give situations where they show up and describe some workarounds. This is the kind of post which I hope becomes obsolete quickly, but the fixes for them are pretty nontrivial, and you are inevitably going to run into these bugs if you try to program in Rust today.

### Mutable borrows are too eager (#6268)

*Summary.* When you use `&mut` (either explicitly or implicitly), Rust immediately treats the lvalue as borrowed and imposes its restrictions (e.g. the lvalue can’t be borrowed again). However, in many cases, the borrowed pointer is not used until later, so imposing the restrictions immediately results in spurious errors. This situation is most likely to occur when there is an *implicit* use of `&mut`. ([Bug #6268](https://github.com/mozilla/rust/issues/6268))

*Symptoms.* You are getting the error “cannot borrow `foo` as immutable because it is also borrowed as mutable”, but the reported second borrow is an object dispatching a method call, or doesn’t seem like it should have been borrowed at the time the flagged borrow occured.

*Examples.* The original bug report describes the situation for nested method calls, where the outer method call has `&mut self` in its signature:

```
fn main() {
  let mut map = std::hashmap::HashMap::new();
  map.insert(1, 2);
  map.insert(2, *map.get(&1)); // XXX
}

test.rs:4:17: 4:20 error: cannot borrow `map` as immutable because it is also borrowed as mutable
test.rs:4   map.insert(2, *map.get(&1)); // XXX
                           ^~~
test.rs:4:2: 4:5 note: second borrow of `map` occurs here
test.rs:4   map.insert(2, *map.get(&1)); // XXX
            ^~~

```

This code would like to retrieve the value at key `1` and store it in key `2`. Why does it fail? Consider the signature `fn insert(&mut self, key: K, value: V) -> bool`: the `insert` method invocation immediately takes out a `&mut` borrow on `map` before attempting to evaluate its argument. If we desugar the method invocation, the order becomes clear: `HashMap::insert(&mut map, 2, *map.get(&1))` (NB: this syntax is not implemented yet). Because Rust evaluates arguments left to right, this is equivalent to:

```
let x_self : &mut HashMap<int> = &mut map;
let x_arg1 : int = 2;
let x_arg2 : int = *map.get(&1); // XXX
HashMap::insert(x_self, x_arg1, x_arg2);

```

meaning there is an active borrow by the time we call `map.get`. A minor rewrite resolves the problem:

```
fn main() {
  let mut map = std::hashmap::HashMap::new();
  map.insert(1, 2);
  let x = *map.get(&1);
  map.insert(2, x);
}

```

Sensitivity to order of arguments even when no method invocation is involved. Here is another example in which there is no method invocation:

```
fn g(x: &mut int) -> int { *x }
fn f(x: &mut int, y: int) { *x += y; }
fn main() {
    let mut a = 1;
    f(&mut a, g(&mut a));
}

```

*Discussion.* Fortunately, this bug is pretty easy to work around, if a little annoying: move all of your sub-expressions to let-bindings before the ill-fated mutable borrow (see examples for a worked example). Note: the borrows that occur in these sub-expressions really do have to be temporary; otherwise, you have a legitimate “cannot borrow mutable twice” error on your hands.

### Borrow scopes should not always be lexical (#6393)

*Summary.* When you borrow a pointer, Rust assigns it a lexical scope that constitutes its lifetime. This scope can be as small as a single statement, or as big as an entire function body. However, Rust is unable to calculate lifetimes that are not lexical, e.g. a borrowed pointer is only live until halfway through a function. As a result, borrows may live longer than users might expect, causing the borrow checker to reject some statements. ([Bug #6393](https://github.com/mozilla/rust/issues/6393))

*Symptoms.* You are getting a “cannot borrow foo as immutable/mutable because it is also borrowed as immutable/mutable”, but you think the previous borrow should have already expired.

*Examples.* This problem shows up in a variety of situations. The very simplest example which tickles this bug can be seen here:

```
fn main() {
    let mut x = ~1;
    let y = &mut *x;
    *y = 1;
    let z = &mut *x;
    *z = 1;
}

test.rs:5:12: 5:19 error: cannot borrow `*x` as mutable more than once at a time
test.rs:5     let z = &mut *x;
                      ^~~~~~~
test.rs:3:12: 3:19 note: second borrow of `*x` as mutable occurs here
test.rs:3     let y = &mut *x;
                      ^~~~~~~

```

Clearly `y` is dead after `*y = 1`, but the borrow checker can’t see that. Fortunately, in this case it is very easy to add a new lexical scope to solve the problem:

```
fn main() {
    let mut x = ~1;
    {
        let y = &mut *x;
        *y = 1;
    }
    let z = &mut *x;
    *z = 1;
}

```

So, when does this actually become a problem? The usual culprit is `match` statements. Here is some common code involving maps that you might want to write:

```
extern mod extra;
fn main() {
    let mut table = extra::treemap::TreeMap::new();
    let key = ~"test1";
    match table.find_mut(&key) {
        None    => table.insert(key.clone(), ~[1]), // XXX
        Some(v) => { v.push(1); false }
    };
}

test.rs:6:19: 6:24 error: cannot borrow `table` as mutable more than once at a time
test.rs:6         None    => table.insert(key.clone(), ~[1]), // XXX
                             ^~~~~
test.rs:5:10: 5:15 note: second borrow of `table` as mutable occurs here
test.rs:5     match table.find_mut(&key) {
                    ^~~~~

```

`table` is a map of integer keys to vectors. The code performs an insert at `key`: if the map has no entry, then we create a new singleton vector and insert it in that location; otherwise, it just pushes the value `1` onto the existing vector. Why is `table` borrowed in the `None` branch? Intuitively, the borrow for `table.find_mut` should be dead, since we no longer are using any of the results; however, to Rust, the only lexical scope it can assign the borrowed pointer encompasses the entire match statement, since the borrowed pointer continues to be used in the `Some` branch (note that if the Some branch is removed, this borrow checks). Unfortunately, it’s not possible to insert a new lexical scope, as was possible in the previous example. (At press time, I wasn’t able to find a small example that only used `if`.)

Sometimes, the *lifetime* associated with a variable can force it to be assigned to a lexical scope that is larger than you would expect. [Issue #9113](https://github.com/mozilla/rust/issues/9113) offers a good example of this (code excerpted below):

```
pub fn read1<'a>(&'a mut self, key: int) -> Option<&'a Data> {
    match self.cache.find(&key) {
        Some(data) => return Some(data),
        None => ()
    };
    match self.db.find(&key) {
        Some(data) => {
            let result: &Data = self.cache.find_or_insert(key, data.clone());
            Some(result)
        },
        None => None
    }
}

test.rs:22:36: 22:46 error: cannot borrow `(*self).cache` as mutable because it is also borrowed as immutable
test.rs:22                 let result: &Data = self.cache.find_or_insert(key, data.clone());
                                               ^~~~~~~~~~
test.rs:15:14: 15:24 note: second borrow of `(*self).cache` occurs here
test.rs:15         match self.cache.find(&key) {
                         ^~~~~~~~~~

```

This code is attempting to perform a database lookup; it first consults the cache and returns a cached entry if available. Otherwise, it looks for the value in the database, caching the value in the process. Ordinarily, you would expect the borrow of `self.cache` in the first match to extend only for the first expression. However, the `return` statement throws a spanner in the works: it forces the lifetime of `data` to be `'a`, which encompasses the entire function body. The borrow checker then concludes that there is a borrow everywhere in the function, even though the function immediately returns if it takes out this borrow.

*Discussion.* The workaround depends on the nature of the scope that is causing trouble. When `match` is involved, you can usually arrange for the misbehaving borrow to be performed outside of the match statement, in a new, non-overlapping lexical scope. This is easy when the relevant branch does not rely on any variables from the pattern-match by using short-circuiting control operators:

```
extern mod extra;
use extra::treemap::TreeMap;
fn main() {
    let mut table: TreeMap<~str,~[int]> = TreeMap::new();
    let key = ~"test1";
    match table.find_mut(&key) {
        None    => {},
        Some(v) => { v.push(1); return }
    };
    table.insert(key.clone(), ~[1]); // None-case
}

```

Alternately, instead of directly returning, the match can assign a boolean to indicate whether or not the None-case should be run:

```
extern mod extra;
use extra::treemap::TreeMap;
fn main() {
    let mut table: TreeMap<~str,~[int]> = TreeMap::new();
    let key = ~"test1";
    let is_none = match table.find_mut(&key) {
        None    => true,
        Some(v) => { v.push(1); false }
    };
    if is_none {
        table.insert(key.clone(), ~[1]);
    }
}

```

The boolean can be elaborated into an `enum` that holds any non-references from the pattern-match you might need. Note that this will not work for borrowed references; but in that case, the borrow truly was still live!

It is a bit more difficult to workaround problems regarding lifetimes, since there is nowhere in the function the pointer is not “borrowed”. One trick which can work in some situations is to convert the function to continuation passing style: that is, instead of returning the borrowed pointer, accept a function argument which gets invoked with the function. [pnkfelix](https://github.com/mozilla/rust/issues/9113#issuecomment-24407530) describes how you might go about fixing the third example. This removes the lifetime constraint on the variable and resolves the problem.

The lexical scope assigned to a borrow can be quite sensitive to code pertubation, since removing a use of a borrow can result in Rust assigning a (much) smaller lexical scope to the borrow, which can eliminate the error. Sometimes, you can avoid the problem altogether by just avoiding a borrow.