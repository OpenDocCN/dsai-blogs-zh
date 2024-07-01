<!--yml
category: 未分类
date: 2024-07-01 18:17:55
-->

# HTML Purifier 4.3.0 released : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/03/html-purifier-4-3-0-released/](http://blog.ezyang.com/2011/03/html-purifier-4-3-0-released/)

## HTML Purifier 4.3.0 released

The release cycle gets longer and longer... probably to the delight of all those downstream, anyway.

* * *

[HTML Purifier](http://htmlpurifier.org) 4.3.0 is a major security release addressing various security vulnerabilities related to user-submitted code and legitimate client-side scripts. It also contains an accumulation of new features and bugfixes over half a year. New configuration options include %CSS.Trusted, %CSS.AllowedFonts and %Cache.SerializerPermissions. There is a backwards-incompatible API change for customized raw definitions, see [the customization documentation](http://htmlpurifier.org/docs/enduser-customize.html#optimized) for details.

HTML Purifier is a standards-compliant HTML filter library written in PHP (gasp!).

*Non sequitur.* While researching the security vulnerabilities that were fixed in this version of HTML Purifier, a thought occurred to me: how easy is it to do programming with higher-order functions in JavaScript? JavaScript is extremely fluent when it comes to passing functions around (one might say its OOP facilities are simply taking some base structure and placing functions on it), but the lack of a type system means that it might get kind of annoying documenting the fact that some particular function has some weird higher-order type like `Direction -> DataflowLattice -> (Block -> Fact -> (DG, Fact)) -> [Block] -> (Fact -> DG, Fact)` (simplified real example, I kid you not!). My experience with the matter in Python is that it just takes too long to explain this sort of thing to ones colleagues, and debugging them is a headache (it's... hard to inspect functions to see what you got) so it's better to leave it out.