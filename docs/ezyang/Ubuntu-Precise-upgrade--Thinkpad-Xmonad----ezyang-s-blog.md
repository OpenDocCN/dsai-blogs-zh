<!--yml
category: 未分类
date: 2024-07-01 18:17:30
-->

# Ubuntu Precise upgrade (Thinkpad/Xmonad) : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/05/ubuntu-precise-upgrade-thinkpad-xmonad/](http://blog.ezyang.com/2012/05/ubuntu-precise-upgrade-thinkpad-xmonad/)

## Ubuntu Precise upgrade (Thinkpad/Xmonad)

It is once again time for Ubuntu upgrades. I upgraded from Ubuntu Oneiric Ocelot to Ubuntu Precise Pangolin (12.04), which is an LTS release. Very few things broke (hooray!)

*   The Monospace font changed to something new, with very wide glyph size. The old font was DejaVuSansMono, which I switched back to.
*   Xournal stopped compiling; somehow the linker behavior changed and you need to specify the linker flags manually.
*   [gnome-keyring](https://bugs.launchpad.net/ubuntu/+source/gnome-keyring/+bug/932177) isn't properly starting up for us non-Unity folks. The underlying problem appears to be [packaging errors by Gnome](http://lists.debian.org/debian-lint-maint/2009/07/msg00129.html), but adding `` eval `gnome-keyring-daemon -s` `` to my `.xsession` cleared things up.
*   The battery icon went away! I assume some daemon is failing to get run, but since I have a very nice xmobar display I'm not mourning its loss.
*   Default GHC is GHC 7.4.1! Time to rebuild; no Haskell Platform yet. (Note that GHC 7.4.1 doesn't support the gold linker; this is the `chunk-size` error.)

I also upgraded my desktop from the previous LTS Lucid Lynx.

*   I had a lot of invalid signature errors, which prevented the release upgrade script from running. I fixed it by uninstalling almost all of my PPAs.
*   Offlineimap needed to be updated because some Python libraries it depended on had backwards incompatible changes (namely, the imap library.)
*   VirtualBox messed up its revision numbers, which contained an [underscore which is forbidden](https://bugs.launchpad.net/ubuntu/+source/dpkg/+bug/613018). Manually editing it out of the file seems to fix it.