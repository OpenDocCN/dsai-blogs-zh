<!--yml
category: 未分类
date: 2024-07-01 18:18:05
-->

# Don’t Repeat Yourself is context dependent : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/10/dry-is-context-dependen/](http://blog.ezyang.com/2010/10/dry-is-context-dependen/)

I am a member of a group called the [Assassins’ Guild](http://web.mit.edu/assassin/www/). No, we don’t kill people, and no, we don’t play the game Assassin. Instead, we write and run competitive live action role-playing games: you get some game rules describing the universe, a character sheet with goals, abilities and limitations, and we set you loose for anywhere from four hours to ten days. In this context, I’d like to describe a situation where applying the rule [Don’t Repeat Yourself](http://en.wikipedia.org/wiki/Don't_repeat_yourself) can be harmful.

The principle of *Don’t Repeat Yourself* comes up in a very interesting way when game writers construct game rules. The game rules are rather complex: we’d like players to be able to do things like perform melee attacks, stab each other in the back, conjure magic, break down doors, and we have to do this all without actually injuring anyone or harming any property, so, in a fashion typical of MIT students, we have “mechanics” for performing these in-game actions (for example, in one set of rules, a melee attack can be declared with “Wound 5”, where 5 is your combat rating, and if another individual has a CR of 5 or greater, they can declare “Resist”; otherwise, they have to role-play falling down unconscious and bleeding. It’s great fun.) Because there are so many rules necessary to construct a reasonable base universe, there is a vanilla, nine-page rules sheet that most gamewriters adapt for their games.

Of course, the rules aren’t always the same. One set of GMs (the people who write and run the game) may decide that a single CR rating is not enough, and that people should have separate attack and defense ratings. Another set of GMs might introduce robotic characters, who cannot die from bleeding out. And so forth.

So when we give rules out to players, we have two possibilities: we can repeat ourselves, and simply give them the full, amended set of rules. Or we can avoid repeating ourselves, and give out the standard rules and a list of errata—the specific changes made in our universe. We tend to repeat ourselves, since it’s easier to do with our game production tools. But an obvious question to ask is, which approach is better?

The answer is, of course, *it depends.*

*   Veteran players who are well acquainted with the standard set of rules don’t need the entire set of rules given to them every time they play a game; instead, it would be much easier and more efficient for them if they were just given the errata sheet, so they can go, “Oh, hm, that’s different, ok” and go and concoct strategies for this altered game universe. This is particularly important for ten-days, where altered universe rules can greatly influence plotting and strategy.
*   For new players who have never played a game before, being given a set of rules and then being told, “Oh, but disregard that and that and here is an extra condition for that case” would be very confusing! The full rules, repeated for the first few times they play a game, is helpful.

I think this same principle applies to *Don’t Repeat Yourself* as applied in software development. It’s good and useful to adopt a compact, unique representation for any particular piece of code or data, but don’t forget that a little bit of redundancy will greatly help out people learning your system for the first time! And to get the best of both worlds, you shouldn’t even have to repeat yourself: you should make the computer do it for you.

*Postscript.* For the curious, here is a [PDF of the game rules](http://web.mit.edu/~ezyang/Public/dangerous-scenario.pdf) we used for a game I wrote in conjunction with Alex Gurany and Jonathan Chapman, *The Murder of Jefferson Douglass* (working name *A Dangerous Game*).

*Postscript II.* When has repeating yourself been considered good design?

*   Perl wants programmers to have to say as little as possible to get the job done, and this has given it a reputation as a “write only language.”
*   Not all code that looks the same should be refactored into a function; there should be some logical unity to what is factored out.
*   Java involves writing copious amounts of code: IDEs generate code for `hashCode` and `equals`, and you possibly tweak it after the fact. Those who like Java controversially claim that this prevents Java programmers from doing too much damage (though some might disagree.)
*   When you write essays, even if you’ve already defined a term fifty pages ago, it’s good to refresh a reader’s memory. This is especially true for math textbooks.
*   Haskell challenges you to abstract as much mathematically sound structure as possible. As a result, it makes people’s heads hurt, leads to combinator zoos up to the wazoo. But it’s also quite beneficial for even moderately advanced users.

Readers are encouraged to come up with more examples.