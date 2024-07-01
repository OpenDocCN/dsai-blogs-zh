<!--yml
category: 未分类
date: 2024-07-01 18:17:39
-->

# Transparent xmobar : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/11/transparent-xmobar/](http://blog.ezyang.com/2011/11/transparent-xmobar/)

## Transparent xmobar

Things I should be working on: *graduate school personal statements.*

What I actually spent the last five hours working on: *transparent xmobar.*

It uses the horrible “grab Pixmap from root X window” hack. You can grab the [patch here](https://github.com/ezyang/xmobar/) but I haven’t put in enough effort to actually make this a configurable option; if you just compile that branch, you’ll get an xmobar that is at 100/255 transparency, tinted black. (The algorithm needs a bit of work to generalize over different tints properly; suggestions solicted!) Maybe someone else will cook up a more polished patch. (Someone should also drum up a more complete set of XRender bindings!)

This works rather nicely with trayer, which support near identical tint and transparency behavior. Trayer also is nice on Oneiric, because it sizes the new battery icon sensibly, whereas stalonetray doesn’t. If you’re wondering why the fonts look antialiased, that’s because I [compiled with XFT support](http://projects.haskell.org/xmobar/#optional-features).

(And yes, apparently I have 101% battery capacity. Go me!)

*Update.* Feature has been prettified and made configurable. Adjust `alpha` in your config file: 0 is transparent, 255 is opaque. I’ve submitted a pull request.