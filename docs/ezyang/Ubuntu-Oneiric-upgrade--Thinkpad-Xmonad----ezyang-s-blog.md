<!--yml
category: 未分类
date: 2024-07-01 18:17:39
-->

# Ubuntu Oneiric upgrade (Thinkpad/Xmonad) : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/11/ubuntu-oneiric-thinkpad-xmonad/](http://blog.ezyang.com/2011/11/ubuntu-oneiric-thinkpad-xmonad/)

## Ubuntu Oneiric upgrade (Thinkpad/Xmonad)

I upgraded from Ubuntu Natty Narwhal to Oneiric Ocelot (11.10) today. Lots of things broke. In order:

*   “Could not calculate the upgrade.” No indication of what the error might be; in my case, the error ended up being old orphan OpenAFS kernel modules (for whom no kernel modules existed). I also took the opportunity to clean up my PPAs.
*   “Reading changelogs.” `apt-listchanges` isn’t particularly useful, and I don’t know why I installed it. But it’s really painful when it’s taking more time to read changelogs than to install your software. Geoffrey suggested `` gdb -p `pgrep apt-listchanges` `` and then forcing it to call `exit(0)`, which worked like a charm. Had to do this several times; thought it was infinitely looping.
*   Icons didn’t work, menus ugly. Go to “System Settings > Appearance” and go set a new theme; in all likelihood your old theme went away. This [AskUbuntu](http://askubuntu.com/questions/59791/how-do-i-fix-my-theme) question gave a clue.
*   Network Manager stopped working. For some inscrutable reason the default NetworkManager config file `/etc/NetworkManager/NetworkManager.conf` has `managed=false` for `ifupdown`. Flip back to true.
*   New window manager, new defaults to dunk you in Unity at least once. Just make sure you pick the right window manager from the little gear icon.
*   `gnome-power-manager` went away. If you fix icons a not-so-useful icon will show up anyway when you load `gnome-settings-daemon`.
*   “Waiting for network configuration.” There were lots of suggestions here. My `/var/run` and `/var/lock` were borked so I [did these instructions](http://uksysadmin.wordpress.com/2011/10/14/upgrade-to-ubuntu-11-10-problem-waiting-for-network-configuration-then-black-screen-solution/), I also hear that you should punt `wlan0` from `/etc/network/interfaces` and remove it from `/etc/udev/rules.d70-persistent-net.rules`. I also commented out the sleeps in `/init/failsafe.conf` for good measure.
*   Default GHC is 7.0.3! Blow away your `.cabal` (but hold onto `.cabal/config`) and go reinstall Haskell Platform. Don’t forget to make sure you install profiling libraries, and grab `xmonad` and `xmonad-contrib`. Note that previous haskell-platform installs will be rather broken, on account of missing GHC 6 binaries (you can reinstall them, but it looks like they get replaced.)
*   ACPI stopped knowing about X, so if you have scripts for handling rotation, source `/usr/share/acpi-support/power-funcs` and run `getXuser` and `getXconsole`
*   DBUS didn’t start. This is due to leftover pid and socket files, see [this bug](https://bugs.launchpad.net/ubuntu/+source/dbus/+bug/811441)
*   Was mysteriously fscking my root drive on every boot. Check your `pass` param in `/etc/fstab`; should be `0`.
*   Redshift mysteriously was being reset by xrandr calls; worked around by calling it oneshot immediately after running xrandr.
*   Not sure if this was related to the upgrade, but fixed an annoyance where suspend-checking (in case you are coming out of hibernate) was taking a really long time in boot. Set `resume` to right swap in `/etc/initramfs-tools/conf.d/resume` and `update-initramfs -u` with great prejudice).

Unresolved annoyances: [X11 autolaunching in DBUS](https://bugs.launchpad.net/ubuntu/+source/dbus/+bug/812940), the power icon doesn’t always properly show AC information and is too small in stalonetray, xmobar doesn’t support percentage battery and AC coloring simultaneously (I have a patch), a totem built from scratch segfaults.