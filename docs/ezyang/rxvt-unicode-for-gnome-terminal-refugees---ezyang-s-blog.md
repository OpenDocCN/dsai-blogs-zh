<!--yml
category: 未分类
date: 2024-07-01 18:18:33
-->

# rxvt-unicode for gnome-terminal refugees : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/01/rxvt-unicode-for-gnome-terminal-refugees/](http://blog.ezyang.com/2010/01/rxvt-unicode-for-gnome-terminal-refugees/)

When I switched from Ubuntu's default Gnome desktop to the tiling window manager [Xmonad](http://xmonad.org/), I kept around Gnome Terminal, although with the menu bar and the scroll bars removed. I changed from the default white to a nice shade of #2B2B2B (a hue that [Sup](http://sup.rubyforge.org/) originally introduced me to).

Over the months, however, I got increasingly annoyed at the slowness at which Gnome Terminal rendered when I switched windows (a not uncommon task in a tiling window manager, made especially important when you have a relatively small screen size); the basic symptom was the screen would flash white as the old terminal left and the new one was being drawn. After testing xterm and finding that it did *not* flash when I switched screens, I hereby resolved to find a faster terminal emulator; on the advice of David Benjamin I finally settled on rxvt-unicode, also known as urxvt.

rxvt-unicode is part of a proud tradition of X terminal emulators, so its settings are managed by the X window manager (as opposed to gnome-settings, which gnome-terminal used). You can manipulate the settings at runtime using a program named `xrdb`; but I found it mostly easier to place the settings I wanted in `.Xdefaults`, which automatically gets loaded at session start. The syntax is simple: a newline-separated file, with the form `Appname*option: value`. Appname in the case of rxvt-unicode is `URxvt`.

Having used gnome-terminal for a long time, I was somewhat loathe to part with the colors and behaviors I'd come to love. So here is my `.Xdefaults` file, with notes about what the various bits do:

```
URxvt*background: #2B2B2B
URxvt*foreground: #DEDEDE
URxvt*font: xft:Monospace:pixelsize=12
URxvt*boldFont: xft:Monospace:bold:pixelsize=12
URxvt*saveLines: 12000
URxvt*scrollBar: false
URxvt*scrollstyle: rxvt

```

These parts are all fairly self-explanatory; rxvt-unicode supports anti-aliased fonts, which means bold text looks good (one of the primary reasons I couldn't stand xterm, since bold fonts tend to bleed into each other without anti-aliasing).

```
URxvt*perl-ext-common: default,matcher
URxvt*urlLauncher: firefox
URxvt*matcher.button: 1
URxvt*colorUL: #86a2be

```

These lines implement clickable URLs inside your terminal. The launcher doesn't give any visual cue in your cursor when a link is clickable, but I find the underlining and change in color to be enough change.

```
! black
URxvt*color0  : #2E3436
URxvt*color8  : #555753
! red
URxvt*color1  : #CC0000
URxvt*color9  : #EF2929
! green
URxvt*color2  : #4E9A06
URxvt*color10 : #8AE234
! yellow
URxvt*color3  : #C4A000
URxvt*color11 : #FCE94F
! blue
URxvt*color4  : #3465A4
URxvt*color12 : #729FCF
! magenta
URxvt*color5  : #75507B
URxvt*color13 : #AD7FA8
! cyan
URxvt*color6  : #06989A
URxvt*color14 : #34E2E2
! white
URxvt*color7  : #D3D7CF
URxvt*color15 : #EEEEEC

```

I absolutely adore gnome-terminal's color scheme, which is a bit more subdued than the rxvt default. So in it goes. The first color is "normal"; the second color is "bright."