<!--yml
category: 未分类
date: 2024-07-01 18:18:25
-->

# Five tips for maintainable shell scripts : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/03/five-tips-for-maintainable-shell-scripts/](http://blog.ezyang.com/2010/03/five-tips-for-maintainable-shell-scripts/)

## Five tips for maintainable shell scripts

When I was seventeen, I wrote my [very first shell script](http://repo.or.cz/w/htmlpurifier-web.git/blob/136caa2d941e51e5a742df3b05fb3e596f778636:/releases/build.bat). It was a Windows batch file, bits and pieces very carefully cargo-culted from various code samples on the web. I had already had the *exquisite* pleasure of futzing with `pear.bat`, and the thought of scripting was not something I relished; "why not write the damn thing in a *real* programming language!" (The extra delicious bit was "a real programming language" was PHP. Hee.)

Eventually I came around to an all-Unix environment, and with it I began to use bash extensively. And suddenly, shell scripting made a lot more sense: you've been writing the damn commands day in and day out, just write them to a script instead! There was, however, still the pesky little problem that shell scripts are forever; like it or not, they've become pieces of maintained code. Entire build infrastructures have been built on top of shell scripts. They breed like rabbits; you have to be careful about the little buggers.

Here are five tips and tricks to keep in mind when tossing commands into a shell script that will make maintenance in the long-run much more pleasant!

1.  Learn and love to use `set`. There is almost always no good reason not to use the `-e` flag, which causes your script to error out if any command returns with a nonzero exit code, and `-x` can save you hours of debugging by printing precisely what command the script is executing before executing it. With the two enabled, you get very simple "assertions" in your shell script:

    ```
    check_some_condition
    ! [ -s "$1" ]

    ```

    although, if at all possible, you should write error messages to accompany them.

2.  Just because you don't define subprocedures when you're at your terminal (or do you? see `alias` and friends) and use reverse command history search with `C-r` doesn't mean it's acceptable to repeat commands over and over again your shell script. In particular, if you have a set of commands that *might* go into a separate script, but you feel funny about making a separate file, stuff them in a subprocedure like this:

    ```
    subcommand() {
      do_something_with "$1" "$2"
    }

    ```

    In particular, argument passing acts exactly the same way it does in a real shell script, and generally you can treat the subcommand as if it were it's own script; standard input and output work the way you expect them to. The only differences is are that `exit` exits the whole script, so if you'd like to break out of a command use `return` instead.

3.  Argument quoting in shell scripts is a strange and arcane domain of knowledge (although it doesn't have to be; [check out Waldman's notes on shell quoting](http://www.mpi-inf.mpg.de/~uwe/lehre/unixffb/quoting-guide.html)). The short version is you *always* want to wrap variables that will be interpolated with quotes, unless you actually want multiple arguments semantics. I have mixed feelings about whether or not literals should be quoted, and of late have fallen to the dismal habit of not quoting them.

4.  Believe it or not, shell scripting has functional programming leanings. `xargs`, for example, is the quintessential "map" functionality. However, if the command you are pushing arguments to doesn't take multiple arguments, you can use this trick:

    ```
    pgrep bash | while read name; do
      echo "PID: $name"
    done

    ```

5.  Shell scripting feels incredibly natural when speaking imperatively, and mostly remains this way when you impose control flow. However, it is absolutely a *terrible* language for any data processing (exhibit 1: sed and perl pipelines) and you should avoid doing too much data crunching in it. Creating utility scripts in more reasonable languages can go a long way to keeping your shell scripts pretty.