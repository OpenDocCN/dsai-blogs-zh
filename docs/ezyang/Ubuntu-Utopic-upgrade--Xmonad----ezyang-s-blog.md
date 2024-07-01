<!--yml
category: 未分类
date: 2024-07-01 18:17:11
-->

# Ubuntu Utopic upgrade (Xmonad) : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/12/ubuntu-utopic-upgrade-xmonad/](http://blog.ezyang.com/2014/12/ubuntu-utopic-upgrade-xmonad/)

## Ubuntu Utopic upgrade (Xmonad)

I finally got around to upgrading to Utopic. [A year ago](http://blog.ezyang.com/2013/10/xmonad-and-media-keys-on-saucy/) I reported that gnome-settings-daemon no longer provided keygrabbing support. This was eventually reverted for Trusty, which kept everyone's media keys.

I'm sorry to report that in Ubuntu Utopic, the legacy keygrabber is no more:

```
------------------------------------------------------------
revno: 4015 [merge]
author: William Hua <william.hua@canonical.com>
committer: Tarmac
branch nick: trunk
timestamp: Tue 2014-02-18 18:22:53 +0000
message:
  Revert the legacy key grabber. Fixes: https://bugs.launchpad.net/bugs/1226962.

```

It appears that the Unity team has forked gnome-settings-daemon into unity-settings-daemon (actually this fork happened in Trusty), and as of Utopic gnome-settings-daemon and gnome-control-center [have been gutted](https://bugs.launchpad.net/ubuntu/+source/gnome-settings-daemon/+bug/1318539) in favor of unity-settings-daemon and unity-control-center. Which puts us back in the same situation as a year ago.

I don't currently have a solution for this (pretty big) problem. However, I have solutions for some minor issues which did pop up on the upgrade:

*   If your mouse cursor is invisible, try running `gsettings set org.gnome.settings-daemon.plugins.cursor active false`
*   If you don't like that the GTK file dialog doesn't sort folders first anymore, try running `gsettings set org.gtk.Settings.FileChooser sort-directories-first true`. ([Hat tip](http://gexperts.com/wp/gnome-3-12-filesnautilus-sort-folders-before-files-issues/))
*   And to reiterate, replace calls to gnome-settings-daemon with unity-settings-daemon, and use unity-control-panel to do general configuration.