<!--yml
category: 未分类
date: 2024-07-01 18:17:25
-->

# Ubuntu Quantal upgrade (Thinkpad/Xmonad) : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/10/ubuntu-quantal-upgrade-thinkpadxmonad/](http://blog.ezyang.com/2012/10/ubuntu-quantal-upgrade-thinkpadxmonad/)

## Ubuntu Quantal upgrade (Thinkpad/Xmonad)

October has come, and with it, another Ubuntu release (12.10). I finally gave in and reinstalled my system as 64-bit land (so long 32-bit), mostly because graphics were broken on my upgraded system. As far as I could tell, lightdm was dying immediately after starting up, and I couldn't tell where in my copious configuration I had messed it up. I also started encrypting my home directory.

*   All [fstab mount entries](http://askubuntu.com/questions/193524/how-to-hide-bind-mounts-in-nautilus) now show up in Nautilus. The correct fix appears to be not putting these mounts in `/media`, `/mnt` or `/home/`, and then they won’t be picked up.
*   Fonts continue to be an exquisite pain in rxvt-unicode. I had to switch from `URxvt.letterSpace: -1` to `URxvt.letterSpace: -2` to keep things working, and the fonts still look inexplicably different. (I haven't figured out why, but the new world order isn't a *complete* eyesore so I've given up for now.) There’s also [a patch](http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=628167) which fixes this problem (hat tip [this libxft2 bug](https://bugs.freedesktop.org/show_bug.cgi?id=47178) bug) but I found that at least for DejaVu the letterSpace hack was equivalent.
*   When you manually suspend your laptop and close the lid too rapidly, Ubuntu also registers the close laptop event, so when you resume, it will re-suspend! Fortunately, this is pretty harmless; if you press the power-button again, it will resume properly. You can also work around this by turning off resume on close lid in your power settings.
*   On resume, the network manager applet no longer accurately reflects what network you are connected to (it thinks you're connected, but doesn't know to what, or what signal strength it is). It's mostly harmless but kind of annoying; if anyone's figured this one out please let me know!
*   Hibernate continues not to work, though I haven’t tried too hard to get it working.
*   Firefox was being *really* slow, so I [reset it](http://support.mozilla.org/en-US/kb/reset-preferences-fix-problems). And then it was fast again. Holy smoke! Worth a try if you’ve found Firefox to be really slow.
*   GHC is now 7.4.2, so you’ll need to rebuild. "When do we get our 7.6 shinies!"

My labmates continue to tease me for not switching to Arch. We’ll see...