<!--yml
category: 未分类
date: 2024-07-01 18:18:02
-->

# Reflexivity. Qed. : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/11/reflexivity-qed/](http://blog.ezyang.com/2010/11/reflexivity-qed/)

*In which Mendeley, Software Foundations and Coq are discussed.*

I was grousing on `#haskell-blah` one day about how annoying it was to organize all of the papers that I have downloaded (and, of course, not read yet.) When you download a paper off the Internet, it will be named all sorts of tremendously unhelpful things like `2010.pdf` or `icfp10.pdf` or `paper.pdf`. So to have any hope of finding that paper which you skimmed a month ago and vaguely recall the title of, you'll need some sort of organization system. Pre-Mendeley, I had adopted the convention of `AuthorName-PaperTitle.pdf`, but I'd always feel a bit bad picking an author out of a list of five people to stick at the beginning, and I still couldn't ever find the paper I was looking for.

It was at this point that someone (I don't have logs, so I don't remember precisely who) pointed me to [Mendeley](http://www.mendeley.com/). Mendeley is free (as in beer) software that helps you organize your papers and upload them to the cloud; in return, they get all sorts of interesting data about what papers people are reading and hounds of metadata obsessed freaks like me curating their databases.

It doesn't have to do much to improve over my existing ad hoc naming scheme. But it does it exceedingly well. After having shifted my [paper database](http://www.mendeley.com/groups/680221/edward-z-yang-s-paper-repository/overview/) over to it, it's reasonably easy (read, spend an afternoon curating a database of 200 papers) to ensure all of your papers have reasonable metadata attached to them. This reasonable metadata means you can slice your database by author (apparently Simon Peyton Jones and Chris Okasaki are two of my favorite authors) and conference (in case I, like, actually write a paper and need to figure out where to send it). You can also classify papers according to your own topics, which is very good if you're like me and have bodies of completely unrelated research literature. Simple, but effective.

> Oh, I do have some complaints about Mendeley. It's PDF viewer leaves something to be desired: if I page down it skips entirely to the next page instead of doing a continuous scroll; the metadata extraction could be better (essentially, it should be just good enough to be able to look it up on an online database and then fill in the database); there should be a better workflow for papers (rather than just a *read* or *unread* toggle, which is utterly not useful); etc. But it works well enough to bring value, and I'm willing to overlook these nits.

After having organized all of my papers, I suddenly realized that I hadn't added any new papers to my collection recently. Papers either find my way to me because a friend forwards it on, or I'm looking for some specific topic and a relevant paper pops up, but I don't actually have any streams of new papers to take a look at. To fix this, I decided to pick some names and go look at their recent publications.

On the way, I noticed an interesting slide deck on [Benjamin Pierce's publications](http://www.cis.upenn.edu/~bcpierce/papers/index.shtml#Recent). The deck was for a keynote address named [Proof Assistant as Teaching Assistant: A View from the Trenches](http://www.cis.upenn.edu/~bcpierce/papers/LambdaTA-ITP.pdf). I thought this was a quite fascinating approach to the problem of teaching proof, and even better, the course notes were online!

It's difficult for me to precisely vocalize how unimaginably awesome [Software Foundations](http://www.cis.upenn.edu/~bcpierce/sf/). I've found it a bit difficult to get started with proof assistants because it's always unclear what exactly you should prove with them: pick something too easy and it feels pointless, pick something too hard and you find yourself without a clue on how to attack the problem. Proof assistants are also rather sophisticated (it reminds me of a time when I was listening to Eric and Trevor discuss proof tactics back at Galois... that was a *very* hard to follow conversation), so if you dive into the manual you find yourself with all this rope but not very much idea how to use it all.

Software Foundations is great because it's not teaching you how to use a proof assistant: it's teaching you about logic, functional programming and the foundations of programming languages, built on top of a proof assistant Coq. So you have a bag of interesting, fundamental theorems about these topics that you want to prove, and then this course shows you how to use the proof assistant to prove them.

It's also a rather ideal situation for self-study, because unlike many textbook exercises, your Coq interpreter will tell you when you've got *the right answer.* Proof assistants are fun precisely because they're a bit like puzzles that you can create without knowing the solution before hand, and then solve. So if you've got some extra time on your hands and have wanted to learn how to use a proof assistance but never got around to it, I highly recommend [checking it out](http://www.cis.upenn.edu/~bcpierce/sf/).