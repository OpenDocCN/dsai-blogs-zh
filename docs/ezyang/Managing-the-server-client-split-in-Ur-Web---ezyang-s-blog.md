<!--yml
category: 未分类
date: 2024-07-01 18:17:29
-->

# Managing the server/client split in Ur/Web : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/07/managing-the-server-client-split-in-ur-web/](http://blog.ezyang.com/2012/07/managing-the-server-client-split-in-ur-web/)

The holy grail of web application development is a *single language* which runs on both the server side and the client side. The reasons for this are multifarious: a single language promotes reuse of components that no longer need to be reimplemented in two languages and allows for much more facile communication between the server and the client. Web frameworks that explicitly strive to handle both the server and client include [Meteor](http://www.meteor.com/), [Ur/Web](http://www.impredicative.com/ur/), [Opa](http://opalang.org/) and [Google Web Toolkit](https://developers.google.com/web-toolkit/overview).

One of the biggest implementation difficulties facing anyone wishing to build such a system is the fact that there are multiple runtimes: the server runtime and the browser runtime, each with an accordingly different set of primitives and APIs available. Furthermore, some code we might wish to only live on the server, and never be sent to the client. When a language feature can be implemented on both runtimes, we maintain the illusion that client and server are indistinguishable; when it cannot, the illusion is broken.

Thus, in order to support runtime-specific FFI calls in such an integrated language, the following questions must be answered:

1.  When is code sent to the client and when is code kept on the server? This information must be exposed to the user (rather than be kept as an “implementation detail”).
2.  How do I force execution on the server?
3.  How do I force execution on the client?

In this blog post, I’ll discuss how [Ur/Web](http://www.impredicative.com/ur/) addresses these questions. The answers are rather simple (any with any luck generalize to other similar systems), but they are rather difficult to come by if you treat the compiler as a black box.

### 1\. Client/server split

An obvious solution to the client/server division problem is to label entry points (e.g. the main function or an onClick handler) as starting from the server (main) or client (onClick), and then conducting reachability analysis to label all other functions. Thus, in the following Ur/Web code, `txn : transaction unit` would execute on the server here:

```
fun main () =
  txn;
  return <xml><body>Done!</body></xml>

```

while it would execute on the client here:

```
fun main () =
  return <xml><body><a onclick={txn}>Click me!</a></body></xml>

```

When given a fragment like this:

```
fun foo body =
  r <- txn body;
  return <xml>{r}</xml>

```

it is not possible to know whether or not `txn` will be needed on the client side or server side without analyzing all of the callers and checking if they are client side or server side. Situations like this are the most important for forcing server-side or client-side behavior.

### 2\. Forcing server-side

Suppose that we wanted to force `txn` to be executed on the server-side. If we’re already on the server, there is nothing more to do. However, if we’re on the client, we need to make an RPC call back to the server. In Ur/Web, this is easily done:

```
fun fooClient body =
  r <- rpc (txn body);
  return <xml>{r}</xml>

```

However, as `rpc` is a client-side only function in Ur/Web, we can no longer use this function for server-side computation. One consequence of this choice is it forces us to be explicit about when an RPC occurs, which is good news for understanding and security.

### 3\. Forcing client-side

Suppose we wanted to force `txn` to be executed on the client-side. This is tricky: if we’re already on the client we can go ahead as normal, but if we’re executing in the server side, *what does it mean to execute some code on the client*?

One interpretation is this: since we are building some HTML that is to be shown to the client, `txn` should be run when the client actually displays the HTML. Ur/Web recently added the `active` tag which achieves just this effect:

```
fun fooServer body =
  return <xml><active code={txn body} /></xml>

```

The `code` attribute acts much like `onclick` and other similar attributes, in that it defines an entry point which happens to automatically get run when shown in the browser. It is still an event handler, in the sense that if someone invokes `fooServer`, but then doesn’t end up using the HTML, `txn` never gets called: `active` can be thought of a sort of lazy execution.

If we would truly like the client to execute some code *immediately*, our best bet is to shove an `active` tag down a `source` which is hooked up to an active `dyn` element:

```
fun fooServer body source =
  set source <xml><active code={txn body} /></xml>;
  return <xml>???</xml>

```

but in this case it is not really possible to ask the client what the result of the computation was (the server is not permitted to block!) This method of mobile code delivery can even be done asynchronously, using channels:

```
fun fooServerAsync body channel =
  send channel <xml><active code={txn body} /></xml>;
  return <xml>???</xml>

```

### 4\. Interaction with the optimizer

The code within HTML event handlers (e.g. `onclick={...}` and `active code={...}`) can have free variables which bind to variables in their lexical scope, which may have been calculated from the server. As such, you might expect in this case, `foo : int -> xbody` would be executed on the server:

```
fun main n =
  let val x = foo n
  in return <xml><body><active code={txn; return x} /></body></xml>

```

However, Ur/Web’s optimizer is too smart for it’s own good: since `foo` is pure and thus referentially transparent, it can always be safely inlined (especially when there is only one use-site):

```
fun main n =
  return <xml><body><active code={txn; return (foo n)} /></body></xml>

```

Written this way, it is clear that `foo` is run from the client. Thus, an innocent transformation can break your code, if `foo` was a server FFI called that was unimplemented on the client.

The troubling conclusion is that variable substitution *can make valid programs invalid*. Of course, in an eagerly evaluated, impure language, variable substitution is not valid. But we might expect it to be true for a pure language like Ur/Web. In any case, we can teach Ur/Web not to inline by marking `foo` as being `benignEffectful` in our `urp` file.

### 5\. Conclusions

In general, when writing Ur/Web applications, here are some useful guidelines:

1.  Always mark server/client-only identifiers with `serverOnly` and `clientOnly` in your `urp` files. Ur/Web will generally handle one-sided FFI functions appropriately, but if you have code that takes advantage of language features that are only implemented on one side (e.g. closures on the server side), be sure to mark those functions appropriately.
2.  Use `rpc` to move from client-to-server, and `active` to move from server-to-client. Because of the “`rpc` must refer to named function” invariant, the general structure of Ur/Web applications will be blobs of server code with client-side code embedded internally.
3.  If you are interested in generating client code that includes pure server-computed data, make sure the functions computing that data are marked `benignEffectful`.
4.  In general, don’t worry about the server/client split! Ur/Web will warn you if you need to move things around, but for the most part, things should just work.

One last word about security in a shared server/client model: how does Ur/Web ensure that input validation ends up on the server-side and not the client-side? It is rather simple: the only parts of your program that care about input validation are ones that involve persistent data, and all of these functions are server only. Thus, any user-data that makes it to any of these functions necessarily passed through a top-level page handler or an `rpc`, making it a relatively simple job to ensure that the validation is on the “right” side of the pond. If you use a data-structure that is correct-by-construction, you’re automatically done!