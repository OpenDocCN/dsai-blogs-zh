<!--yml
category: 未分类
date: 2024-07-01 18:17:23
-->

# The duality of weak maps and private symbols : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/03/duality-of-weak-maps-and-private-symbols/](http://blog.ezyang.com/2013/03/duality-of-weak-maps-and-private-symbols/)

*From the files of the ECMAScript TC39 proceedings*

I want to talk about an interesting duality pointed out by Mark Miller between two otherwise different language features: weak maps and private symbols. Modulo implementation differences, they are the same thing!

A [weak map](http://wiki.ecmascript.org/doku.php?id=harmony:weak_maps) is an ordinary associative map, with the twist that if the key for any entry becomes unreachable, then the value becomes unreachable too (though you must remember to ignore references to the key from the value itself!) Weak maps have a variety of use-cases, including memoization, where we’d like to remember results of a computation, but only if it will ever get asked for again! A weak map supports `get(key)` and `set(key, value)` operations.

A [private symbol](http://tc39wiki.calculist.org/es6/symbols/) is an unforgeable identifier of a field on an object. Symbols are useful because they can be generated “fresh”; that is, they are guaranteed not to conflict with existing fields that live on an object (whereas one might get unlucky with `_private_identifier_no_really`); a private symbol has the extra stipulation that one cannot discover that it exists on an object without actually having the symbol—e.g. an object will refuse to divulge the existence of a private symbol while enumerating the properties of an object. A private symbol `psym` might be created, and then used just like an ordinary property name to get (`obj[psym]`) and set (`obj[psym] = value`) values.

To see why these are the same, lets implement weak maps in terms of private symbols, and vice versa (warning, pseudocode ahead):

```
function WeakMap() {
  var psym = PrivateSymbol();
  return {
    get: function(key) { return key[psym]; },
    set: function(key, value) { key[psym] = value; }
  }
}

function PrivateSymbol() {
  return WeakMap();
}
// pretend that get/set are magical catch-all getters and setters
Object.prototype.get = function(key) {
  if (key instanceof PrivateSymbol) { return key.get(this); }
  else { return this.old_get(key); }
}
Object.prototype.set = function(key, value) {
  if (key instanceof PrivateSymbol) { return key.get(this, value); }
  else { return this.old_set(key, value); }
}

```

Notice, in particular, that it wouldn’t make sense to enumerate all of the entries of a weak map; such an enumeration would change arbitrarily depending on whether or not a GC had occurred.

If you look at this more closely, there is something rather interesting going on: the *implementation* strategies of weak maps and private symbols are the opposite of each other. With a weak map, you might imagine a an actual map-like data structure collecting the mappings from keys to values (plus some GC trickery); with a private symbol, you would expect the values to be stored on the object itself. That is to say, if we say “WeakMap = PrivateSymbol” and “key = this”, then the primary difference is whether or not the relationships are stored on the WeakMap/PrivateSymbol, or on the key/this. WeakMap suggests the former; PrivateSymbol suggests the latter.

Is one implementation or the other better? If objects in your system are immutable or not arbitrarily extensible, then the private symbol implementation may be impossible to carry out. But if both implementations are possible, then which one is better depends on the *lifetimes* of the objects in question. Garbage collecting weak references is expensive business (it is considerably less efficient than ordinary garbage collection), and so if you can arrange for your weak mappings to die through normal garbage collection, that is a win. Therefore, it’s better to store the mapping on the *shorter-lived* object. In the case of a memotable, the key is going to be a bit more ephemeral than the map, which results in a very strange consequence: *the best implementation strategy for a weak map doesn’t involve creating a map at all!*

Alas, as with many elegant results, there are some difficulties stemming from complexities in other parts of the ECMAScript specification. In particular, it is not at all clear what it means to have an “read-only weak map”, whereas an read-only private symbol has an obvious meaning. Furthermore, collapsing these two rather distinct concepts into a single language may only serve to confuse web developers; a case of a proposal being too clever for its own good. And finally, there is an ongoing debate about how to [reconcile private state with proxies](http://wiki.ecmascript.org/doku.php?id=strawman:proxy_symbol_decoupled). This proposal was introduced to solve one particular aspect of this problem, but to our understanding, it only addresses a specific sub-problem, and *only* works when the proxy in question is a [membrane](http://blog.ezyang.com/2013/03/what-is-a-membran/).