<!--yml
category: 未分类
date: 2024-07-01 18:17:44
-->

# Multi-monitor xmobar placement on Gnome : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/06/multi-monitor-xmobar-placement-on-nome/](http://blog.ezyang.com/2011/06/multi-monitor-xmobar-placement-on-nome/)

## Multi-monitor xmobar placement on Gnome

This post describes how to change which monitor xmobar shows up on in a multi-monitor setup. This had always been an annoyance for me, since on an initial switch to multi-monitor, xmobar would be living on the correct monitor, but if I ever restarted XMonad thereafter, it would migrate to my other monitor, much to my annoyance. Note that a monitor is different from an X screen, which *can* be directly configured from xmobar using the `-x` command line.

How does xmobar pick what screen to use? It selects the “primary” monitor, which by default is the first entry in your `xrandr` list:

```
Screen 0: minimum 320 x 200, current 2464 x 900, maximum 8192 x 8192
VGA1 connected 1440x900+1024+0 (normal left inverted right x axis y axis) 408mm x 255mm
   1440x900       59.9*+   75.0
   1280x1024      75.0     60.0
   1280x960       60.0
   1152x864       75.0
   1024x768       75.1     70.1     66.0     60.0
   832x624        74.6
   800x600        72.2     75.0     60.3     56.2
   640x480        72.8     75.0     66.7     66.0     60.0
   720x400        70.1
LVDS1 connected 1024x768+0+0 (normal left inverted right x axis y axis) 245mm x 184mm
   1024x768       60.0*+
   800x600        60.3     56.2
   640x480        59.9

```

We can switch the primary monitor using the `xrandr --output $MONITOR --primary` command. However, this change is not persistent; you’d have to run this command every time you add a new monitor.

Fortunately, it turns out `gnome-settings-daemon` records information about monitors it has seen in order to configure them properly. This information is in `.config/monitors.xml`:

```
<monitors version="1">
  <configuration>
      <clone>no</clone>
      <output name="VGA1">
          <vendor>HSD</vendor>
          <product>0x8991</product>
          <serial>0x01010101</serial>
          <width>1440</width>
          <height>900</height>
          <rate>60</rate>
          <x>0</x>
          <y>0</y>
          <rotation>normal</rotation>
          <reflect_x>no</reflect_x>
          <reflect_y>no</reflect_y>
          <primary>no</primary>
      </output>
      <output name="LVDS1">
          <vendor>LEN</vendor>
          <product>0x4002</product>
          <serial>0x00000000</serial>
          <width>1024</width>
          <height>768</height>
          <rate>60</rate>
          <x>1440</x>
          <y>0</y>
          <rotation>normal</rotation>
          <reflect_x>no</reflect_x>
          <reflect_y>no</reflect_y>
          <primary>no</primary>
      </output>
  </configuration>
</monitors>

```

So all we need to do is tweak `primary` to be `yes` on the appropriate monitor.

Hat tip to David Benjamin and Evan Broder for letting me know how to do this.