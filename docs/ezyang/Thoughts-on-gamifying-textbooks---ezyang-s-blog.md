<!--yml
category: 未分类
date: 2024-07-01 18:17:30
-->

# Thoughts on gamifying textbooks : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/05/thoughts-on-gamifying-textbooks/](http://blog.ezyang.com/2012/05/thoughts-on-gamifying-textbooks/)

Earlier this year, Woodie Flowers wrote this [criticism of MITx](http://web.mit.edu/fnl/volume/243/flowers.html):

> We seem to have decided to offer “courses” rather than participate in the exciting new process of replacing textbooks with more effective training tools.

[Logitext](http://logitext.ezyang.scripts.mit.edu/logitext.fcgi/tutorial), true to its name, was intended to explore what a chapter from a next-generation textbook on formal logic might look like. But if you asked anyone what subjects the most important textbooks of this century would be about, I doubt logic would be particularly high on anyone’s list. In terms of relevance, Logitext misses the mark. But I do think there are some design principles that Logitext helps elucidate.

### On interactivity

It is a good thing that the quality of the Metropolitan Opera's productions is not correlated with the quality of [their website pages](http://ringcycle.metoperafamily.org/characters). Here, we have an egregious example of interactivity for the sake of interactivity, with no real benefit to the presentation of information.

There are many things that interactivity can bring the table. However, **interactive textbooks should still look like textbooks.** The conventional textbook is a masterwork of design for the static medium: it is skimmable, supports random access and easily searched. You *cannot* beat this format at its own game, no matter how well designed your [video game levels may be](http://tryruby.org/).

In order to apply interactivity tastefully, you must consider what the *limitations* of the static medium were: it is precisely here where interactivity can bring something to the table. Here are some examples:

*   Every field of study has its jargon, which assists people versed with the language but impedes those who are not. In a static medium, you can only define jargon a limited number of times: it clutters up a text to redefine it a term every time it shows up in the text, even if you know your students frequently forget what a term means. In an a dynamic medium, the fix is trivial: tooltips. Logitext did not start off with tooltips, but I quickly learned readers were getting confused about the meanings of “conjunction”, “disjunction” and “turnstile”. Tooltips let us easily **extend the possible audience of a single work**.
*   The written medium demands a linear narrative, only sparingly stopping for questions or elaborations. Too many waypoints, and you risk losing the reader. In an interactive text, **the system can give context-sensitive information only when it is relevant.** In Logitext, when a reader clicks on a clause which requires an instantiation of a variable, the system explains how this inference rule works. This explanation is given elsewhere in a more general explanation of how quantifiers work, but the system also knows how to offer this information a timely and useful manner.
*   Sometimes, the information you are trying to convey should also be given in another form. It's the difference between describing a piece of music or actually hearing it, the difference between giving someone a map or letting them wander around for a few hours. Whenever possible, **show, don't tell.** And if possible, show in different ways—different intuitions work for different people. I can explain what the “no free occurrence” rule is until the chickens come home, but the unexpected renaming of variables when you click “forall right” immediately introduces the intuitive idea (even though it still needs to be explained for full understanding.)

It is telling that each of these enhancements have been abused by countless systems in the past. Many people have a dim view of tooltips and Clippy, myself included. I think one way to limit the damage of any such abuse is to demand that the textbook **gracefully degrade** without interactivity. (For technological reasons, Logitext doesn’t render correctly without JavaScript, but I consider this a bug.)

### On exercise design

In order to truly learn something, you must solve some problems with it. Gamification of exercises has done well at supplying extrinsic motivation, but up until recently, the state of the art in online exercise design has been something [like this](http://math.com/school/subject1/practice/S1U4L3/S1U4L3Pract.html). I find this depressing: there is no indication the student is really learned the underlying concepts, or has just constructed [an elaborate private system which happens also to be wrong](http://blog.mathed.net/2011/07/rysk-erlwangers-bennys-conception-of.html). Remember when you were asked to show your work? We should be demanding this online too.

This is easier said than done. It was no accident that I picked the *sequent calculus* for Logitext: while logic holds a very special place in my heart, I also knew that it would be easy to automate. The road to a system for something as simple as High School Algebra will be long and stony. Logitext sidesteps so many important questions, even ones as simple as "How do we get student's answers (with work) onto the computer?" let alone thorny issues such as one addressed by a [recent PhD thesis](http://www.marvin-schiller.de/): "How do we tell if the student needs to provide more work?"

I think I came away with two important ideas from Logitext. The first is a strong conviction that **theorem provers are the right foundational technology for interesting exercises in mathematics.** Building the Logitext system was a bit of work, but once the platform was designed, defining exercises was simple, literally a line of code per exercise. If every exercise had to be implemented from scratch, the cost would have been prohibitively expensive, and the tutorial would have looked a lot different. We know that, in principle, we can formalize all of mathematics; in the case of elementary mathematics, we may not even have to solve open research questions to get there. Theorem provers also know when you’ve gotten the answer *right*, and I suspect from a gamification perspective that is all you need.

The second important idea is that computers can **assist exploration of high-level concepts, even when the foundations are shaky**. For some people, copying down a string of mathematical symbols quickly and accurately is an ordeal: a system like Logitext abstracts that away and allows them to see the higher order structure of these proofs. It is true that it is better of students have a strong foundation, but if we had a system which could prevent them from getting left behind, I think the world would be strictly better for it. The solution to a curriculum which relies on [a freakish knack for manipulating abstract symbols](http://worrydream.com/KillMath/) should not be eliminating symbols, but making it easier to manipulate them. Educational systems should have what I call **adjustable affordance**: you should have the option to do the low level manipulations, or have the system do them for you.

### Conclusion

I have articulated the following design principles for the gamification of textbooks:

*   An interactive textbook should still look like a textbook.
*   Use interactivity to extend the possible audience of a textbook by assisting those with less starting knowledge.
*   Use interactivity to offer context-sensitive information only when relevant.
*   Use interactivity to show; but be sure to explain afterwards.
*   (Heavily modified) theorem provers will be the fundamental technology that will lie beneath any nontrivial exercise engine.
*   One of the most important contributions of computers to exercises will not be automated grading, but assisted exploration.

I’ve asserted these very confidently, but the truth is that they are all drawn from a sample size of one. While the Logitext project was a very informative exercise, I am struck by how little I know about K12 education. As an ace student, I have a rather unrepresentative set of memories, and they are probably all unreliable by now anyway. Education is hard, and while I think improved textbooks will help, I don’t really know if they will really change the game. I am hopeful, but I have the nagging suspicion that I may end up waiting a long time.