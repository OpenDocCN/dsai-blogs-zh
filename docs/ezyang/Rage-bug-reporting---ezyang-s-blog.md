<!--yml
category: 未分类
date: 2024-07-01 18:16:50
-->

# Rage bug reporting : ezyang’s blog

> 来源：[http://blog.ezyang.com/2021/04/rage-bug-reporting/](http://blog.ezyang.com/2021/04/rage-bug-reporting/)

## Rage bug reporting

At Facebook, we have an internal convention for tooling called "rage". When something goes wrong and you want to report a bug, the tool developer will typically ask you to give them a rage. For a command line tool, this can be done by running a rage subcommand, which will ask about which previous CLI invocation you'd like to report, and then giving you a bundle of logs to send to the developer.

A rage has an important property, compared to a conventional log level flag like `-v`: **rage recording is always on**. In other words, it is like traditional server application logs, but applied to client software. Logging is always turned on, and the rage subcommand makes it easy for a user to send only the relevant portion of logs (e.g., the logs associated with the command line invocation that is on).

For some reason, rage functionality is not that common in open source tools. I can imagine any number of reasons why this might be the case:

*   Adding proper logging is like flossing--annoying to do at the time even when it can save you a lot of pain later.
*   Even if you have logging, you still need to add infrastructure to save the logs somewhere and let users retrieve them afterwards.
*   It's something of an art to write logs that are useful enough so that developer can diagnose the problem simply by "reading the tea leaves", but not so detailed that they slow down normal execution of the program. And don't forget, you better not expose private information!
*   Most programs are simple, and you can just fall back on the old standby of asking the user to submit reproduction instructions in their bug report.

Still, in the same way most sysadmins view logging as an invaluable tool for debugging server issues, I think rage reporting is an invaluable tool for debugging client issues. In ghstack, it didn't take very many lines of code to implement rage reporting: [ghstack.logs](https://github.com/ezyang/ghstack/blob/master/ghstack/logs.py) (for writing the logs to the rage directory) and [ghstack.rage](https://github.com/ezyang/ghstack/blob/master/ghstack/rage.py) (for reading it out). But it has greatly reduced my support load for the project; given a rage, I can typically figure out the root cause of a bug without setting up a reproducer first.