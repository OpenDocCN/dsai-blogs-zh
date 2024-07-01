<!--yml
category: 未分类
date: 2024-07-01 18:17:11
-->

# Tomatoes are a subtype of vegetables : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/11/tomatoes-are-a-subtype-of-vegetables/](http://blog.ezyang.com/2014/11/tomatoes-are-a-subtype-of-vegetables/)

Subtyping is one of those concepts that seems to makes sense when you first learn it (“Sure, convertibles are a subtype of vehicles, because all convertibles are vehicles but not all vehicles are convertibles”) but can quickly become confusing when function types are thrown into the mix. For example, if `a` is a subtype of `b`, is `(a -> r) -> r` a subtype of `(b -> r) -> r`? (If you know the answer to this question, this blog post is not for you!) When we asked our students this question, invariably some were lead astray. True, you can mechanically work it out using the rules, but what’s the intuition?

Maybe this example will help. Let `a` be tomatoes, and `b` be vegetables. `a` is a subtype of `b` if we can use an `a` in any context where we were expecting a `b`: since tomatoes are (culinary) vegetables, tomatoes are a subtype of vegetables.

What about `a -> r`? Let `r` be soup: then we can think of `Tomato -> Soup` as recipes for tomato soup (taking tomatoes and turning them into soup) and `Vegetable -> Soup` as recipes for vegetable soup (taking vegetables—any kind of vegetable—and turning them into soup). As a simplifying assumption, let's assume all we care about the result is that it’s soup, and not what type of soup it is.

What is the subtype relationship between these two types of recipes? A vegetable soup recipe is more flexible: you can use it as a recipe to make soup from tomatoes, since tomatoes are just vegetables. But you can’t use a tomato soup recipe on an eggplant. Thus, vegetable soup recipes are a subtype of tomato soup recipes.

This brings us to the final type: `(a -> r) -> r`. What is `(Vegetable -> Soup) -> Soup`? Well, imagine the following situation...

* * *

One night, Bob calls you up on the phone. He says, “Hey, I’ve got some vegetables left in the fridge, and I know your Dad was a genius when it came to inventing recipes. Do you know if he had a good soup recipe?”

“I don’t know...” you say slowly, “What kind of vegetables?”

“Oh, it’s just vegetables. Look, I’ll pay you back with some soup, just come over with the recipe!” You hear a click on the receiver.

You pore over your Dad’s cookbook and find a tomato soup recipe. Argh! You can’t bring this recipe, because Bob might not actually have tomatoes. As if on cue, the phone rings again. Alice is on the line: “The beef casserole recipe was lovely; I’ve got some tomatoes and was thinking of making some soup with them, do you have a recipe for that too?” Apparently, this happens to you a lot.

“In fact I do!” you turn back to your cookbook, but to your astonishment, you can’t find your tomato soup recipe any more. But you do find a vegetable soup recipe. “Will a vegetable soup recipe work?”

“Sure—I’m not a botanist: to me, tomatoes are vegetables too. Thanks a lot!”

You feel relieved too, because you now have a recipe for Bob as well.

* * *

Bob is a person who takes vegetable soup recipes and turns them into soup: he’s `(Vegetable -> Soup) -> Soup`. Alice, on the other hand, is a person who takes tomato soup recipes and turns them into soup: she’s `(Tomato -> Soup) -> Soup`. You could give Alice either a tomato soup recipe or a vegetable soup recipe, since you knew she had tomatoes, but Bob’s vague description of the ingredients he had on hand meant you could only bring a recipe that worked on all vegetables. Callers like Alice are easier to accommodate: `(Tomato -> Soup) -> Soup` is a subtype of `(Vegetable -> Soup) -> Soup`.

In practice, it is probably faster to formally reason out the subtyping relationship than it is to *intuit* it out; however, hopefully this scenario has painted a picture of *why* the rules look the way they do.