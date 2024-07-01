<!--yml
category: 未分类
date: 2024-07-01 18:17:54
-->

# Mailbox: Advice for a sup first-timer : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/04/mailbox-advice-for-a-sup-first-timer/](http://blog.ezyang.com/2011/04/mailbox-advice-for-a-sup-first-timer/)

## Mailbox: Advice for a sup first-timer

> *What is the Mailbox?* It's a selection of interesting email conversations from my mailbox, which I can use in place of writing original content when I’m feeling lazy. I got the idea from Matt Might, who has a set of wonderful [suggestions for low-cost academic blogging](http://matt.might.net/articles/how-to-blog-as-an-academic/).

*From: Brent Yorgey*

I see you are a contributor to the [sup mail client](http://sup.rubyforge.org/). At least I assume it is you, I doubt there are too many Edward Z. Yangs in the world. =) I'm thinking of switching from mutt. Do you still use sup? Any thoughts/encouragements/cautions to share?

*To: Brent Yorgey*

Yeah! I still use Sup, and I've blogged [a little about it in the past](http://blog.ezyang.com/2010/01/sup/), which is still essentially the setup I use. Hmm, some notes:

*   I imagine that you want to switch away from Mutt because of some set of annoyances that has finally gotten unbearable. I will warn you that no mail client is perfect; many of my friends used to use Sup and gave up on it along the way. (I hear they use GMail now.) I've stuck it out, partially due to inertia, partially because someone else took [ezyang@gmail.com](mailto:ezyang@gmail.com), but also partially because I think it's worth it :-)
*   One of the things that has most dramatically changed the way I read email is the distinction between inbox and unread, and not-inbox and unread. In particular, while I have a fairly extensive set of filters that tag mail I get from mailing lists, when I read things on a day to day basis, I check my inbox for important stuff, and then I check my not-inbox "for fun", mostly not reading most emails (but skimming the subject headers.) This means checking my mail in the morning is about a ten minute deal. This is *the* deal-maker for me.
*   You will almost definitely want to setup OfflineIMAP, because downloading a 80 message thread from the Internet gets old *very* quickly. But Sup doesn't propagate back changes to your mailbox this way (in particular, 'read' in Sup doesn't mean 'read' in your INBOX) unless you use some experimental code on the maildir-sync branch. I've been using it for some time now quite well, but it does make getting other upstream changes a bit of an ordeal.
*   Getting Sup setup will take some time; the initial import takes a long time and tweaking also takes a bit of time. Make sure you don't have any deadlines coming soon. I've also found that it's not really possible to use Sup without dipping your hand into some Ruby hacking (though maybe that's just me :-); right now I have four hand-crafted patches on top of my source tree, and rebasing to master is not something done likely. I've actually gotten into a bit of trouble being so hack happy, but it's also nice to be able to fix things when you need to (Sup not working is /very/ disruptive). Unfortunately, Ruby is not statically typed. :-)

Hope that helps. Do feel free to shout out if you need some help.