<!--yml
category: 未分类
date: 2024-07-01 18:17:58
-->

# On checked exceptions and proof obligations : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/02/on-checked-exceptions-and-proof-obligations/](http://blog.ezyang.com/2011/02/on-checked-exceptions-and-proof-obligations/)

Checked exceptions are a much vilified feature of Java, despite theoretical reasons why it should be a really good idea. The tension is between these two lines of reasoning:

> Well-written programs handle all possible edge-cases, working around them when possible and gracefully dying if not. It's hard to keep track of *all* possible exceptions, so we should have the compiler help us out by letting us know when there is an edge-case that we've forgotten to handle. Thus, checked exceptions offer a mechanism of ensuring we've handled all of the edge-cases.

and

> Frequently checked exceptions are for error conditions that we cannot reasonably recover from close to the error site. Passing the checked exception through all of the intervening code requires each layer to know about all of its exceptions. The psychological design of checked exceptions encourages irresponsible swallowing of exceptions by developers. Checked exceptions don't scale for large amounts of code.

In this post, I suggest another method for managing checked exceptions: prove that the code *cannot* throw such an exception.

"Prove that the code cannot throw an exception?" you might say. "Impossible! After all, most checked exceptions come from the outside world, and surely we can't say anything about what will happen. A demon could just always pick the worst possible scenario and feed it into our code."

My first answer to the skeptic would be that there do indeed exist examples of checked exceptions that happen completely deterministically, and could be shown to be guaranteed not to be thrown. For example, consider this code in the Java reflection API:

```
Object o, Field f; // defined elsewhere
f.setAccessible(true);
f.get(o);

```

The last invocation could throw a checked exception `IllegalAccessException`, but assuming that the `setAccessible` call did not fail (which it could, under a variety of conditions), this exception cannot happen! So, in fact, even if it *did* throw an `IllegalAccessException`, it has violated our programmer's expectation of what the API should do and a nice fat runtime error will let us notice what's going on. The call to `setAccessible` *discharges the proof obligation* for the `IllegalAccessException` case.

But this may just be an edge case in a world of overwhelmingly IO-based checked exceptions. So my second answer to the skeptic is that when we program code that interacts with the outside world, we often *don't* assume that a demon is going to feed us the worst possible input data. (Maybe we should!) We have our own internal model of how the interactions might work, and if writing something that's quick and dirty, it may be convenient to assume that the interaction will proceed in such and such a manner. So once we've written all the validation code to ensure that this is indeed the case (throwing a runtime exception akin to a failed assert if it's not), we once again can assume static knowledge that can discharge our proof obligations. Yes, in a way it’s a cop out, because we haven’t proved anything, just told the compiler, “I know what I’m doing”, but the critical extra is that once we’ve established our assumptions, we can prove things with them, and only need to check at runtime what we assumed.

Of course, Java is not going to get dependent types any time soon, so this is all a rather theoretical discussion. But checked exceptions, like types, *are* a form of formal methods, and even if you don’t write your application in a dependently typed language, the field can still give useful insights about the underlying structure of your application.

### Resources

The correspondence between checked exceptions and proofs came to me while listening to Conor McBride's lecture on the [Outrageous Arrows of Fortune](http://personal.cis.strath.ac.uk/~conor/GUtalk.pdf). I hope to do a write up of this talk soon; it clarified some issues about session types that I had been thinking about.

I consulted the following articles when characterizing existing views of Java checked exceptions.