<!--yml
category: 未分类
date: 2024-07-01 18:17:22
-->

# The single export pattern : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/03/the-single-export-pattern/](http://blog.ezyang.com/2013/03/the-single-export-pattern/)

## The single export pattern

*From the files of the ECMAScript TC39 proceedings*

**Single export** refers to a design pattern where a module identifier is overloaded to also represent a function or type inside the module. As far as I can tell, the term “single export” is not particularly widely used outside the ECMAScript TC39 committee; however, the idea shows up in other contexts, so I’m hoping to popularize this particular name (since names are powerful).

The basic idea is very simple. In JavaScript, a module is frequently represented as an object:

```
var sayHello = require('./sayhello.js');
sayHello.run();

```

The methods of `sayHello` are the functions exported by the module. But what about `sayHello` itself? Because functions are objects too, we could imagine that `sayHello` was a function as well, and thus:

```
sayHello()

```

would be a valid fragment of code, perhaps equivalent to `sayHello.run()`. Only one symbol can be exported this way, but in many modules, there is an obvious choice (think of jQuery’s `$` object, etc).

This pattern is also commonly employed in Haskell, by taking advantage of the fact that types and modules live in different namespaces:

```
import qualified Data.Map as Map
import Data.Map (Map)

```

`Map` is now overloaded to be both a type and a module.