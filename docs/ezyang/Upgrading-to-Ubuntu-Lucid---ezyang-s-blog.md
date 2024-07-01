<!--yml
category: 未分类
date: 2024-07-01 18:18:18
-->

# Upgrading to Ubuntu Lucid : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/05/upgrading-to-ubuntu-lucid/](http://blog.ezyang.com/2010/05/upgrading-to-ubuntu-lucid/)

Now that term is over, I finally went an upgraded my laptop to Ubuntu 10.04 LTS, Lucid Lynx. The process went substantially more smoothly than [Karmic went](http://ezyang.com/karmic.html), but there were still a few hiccups.

*Etckeeper.* As always, you should set `AVOID_COMMIT_BEFORE_INSTALL` to 0 before attempting a release upgrade, since etckeeper hooks will be invoked multiple times and there's nothing more annoying than getting the notice "etckeeper aborted the install due to uncommited changes, please commit them yourselves" because there is no way that's going to work.

Well, this time round there was a different, hilarious bug:

```
/etc/etckeeper/post-install.d/50vcs-commit: 20:
etckeeper: Argument list too long

```

This has been [reported as Bug #574244](https://bugs.launchpad.net/ubuntu/+source/etckeeper/+bug/574244). Despite being an ominous warning, it is actually quite harmless, and you can complete the upgrade with:

```
aptitude update
aptitude full-upgrade

```

I had to kick the network due to broken DNS; your mileage may vary.

*Wireless key management.* I have not resolved this issue yet, but the basic symptom is that Ubuntu network-manager fails to remember WEP keys you have provided for secured networks. (I know you MIT students still on campus aren't too bothered by this.) This appears to be a moderately widespread problems, as you have people revivifying permutations of [this](https://bugs.launchpad.net/ubuntu/+source/network-manager/+bug/271097) [bug](https://bugs.launchpad.net/ubuntu/+source/network-manager/+bug/36651) that occurred a long time ago. (In classic terrible bug reporting style, users are attaching themselves to old bug reports when they really should be filing a new regression for Lucid.)

From what I've investigated, I have been able to verify that [connections to the keyring daemon are not working](http://ubuntuforums.org/showthread.php?t=1459804). There is [a fix circulating](https://bugs.launchpad.net/ubuntu/+source/seahorse/+bug/553032/comments/8) in which you change your startup command from "gnome-keyring-daemon --start --components=pkcs11" to just "gnome-keyring-daemon", although I suspect this isn't really the "right" thing, and it doesn't work for me anyway.

*PHP.* Ubuntu Lucid most notably has upgraded PHP 5.3.2, but they've also fiddled with some of the default settings. In my case, `log_errors` was causing quite interesting behavior for my scripts, and I have since coded my scripts to explicitly turn this ini setting off. You should save a copy of the output of `php -i` prior to the upgrade and compare it with the output afterwards.