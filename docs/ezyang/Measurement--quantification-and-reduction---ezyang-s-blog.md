<!--yml
category: 未分类
date: 2024-07-01 18:17:45
-->

# Measurement, quantification and reduction : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/06/measurement-quantification-and-reduction/](http://blog.ezyang.com/2011/06/measurement-quantification-and-reduction/)

Today we continue the theme, “What can Philosophy of Science say for Software Engineering,” by looking at some topics taken from the Philosophy of Physical Sciences.

### Measurement and quantification

Quantification is an activity that is embedded in modern society. We live by numbers, whether they are temperature readings, velocity, points of IQ, college rankings, safety ratings, etc. Some of these are uncontroversial, others, very much so, and a software engineer must always be careful about numbers they deal in, for quantification is a very tricky business.

Philosophers of science can look to history for some insight into this conundrum, for it was not always the case that thermometry was an uncontroversial method of generating numbers. While the thermometer itself was invented in the 16th century, it took centuries to establish the modern standard of measuring temperature. What made this so hard? Early dabblers in thermometry were well aware of the ability to calibrate a thermometer by testing its result at various fixpoints (freezing and boiling), and graduating the thermometer accordingly, and for some period of times this was deemed adequate for calibrating thermometers.

But alas, the thermal expansion of liquids is not uniform across liquids, and what intrepid experimenters like Herman Boerhaave and Daniel Fahrenheit discovered was, in many cases, two thermometers would not agree with each other, even if they had been calibrated in the same way. How would they determine which thermometer was more accurate, without appealing to... another thermometer? Most justifications involving the nature of the liquid "particles" and their forces appealed to (at the time) unjustifiable theoretical principles.

Without the invention of modern thermodynamics, the most compelling case would be put forth Henri Victor Regnault. An outstanding experimentalist, Regnault set forth to solve this problem by systematically eliminating *all* theoretical assumptions from this work: specific heat, caloric, conservation of heat—all of these did not matter to him. What Regnault cared about was the *comparability* of thermometers: an instrument that gave varying values depending on the situation could not be trusted. If the thermometer was sensitive to the proportion of alcohol in it, or the way its glass had been blown, it was not to be taken as reflecting reality.

In the face of uncertainty and unsure theoretical basis, even simple criterion like *comparability* can be useful in getting a grip on the situation. One should not underestimate the power of this technique, due in part to its ability to operate without assuming any sort of theoretical knowledge of the task at hand.

### Reductive explanations

The law of leaky abstractions states that all attempts to hide the low-level details of a system fail in some way or another. Taken to the extreme, it results in something resembling a reductive approach to the understanding of computer systems: in order to understand how some system works, it is both desirable and necessary to understand all of the layers below it.

Of course, we make fun of this sort of reductivism when we say things like, “Real men program with a magnet on their hard drive.” One simply cannot be expected to understand a modern piece of software merely by reading all of the assembly it is based on. Even systems that are written at a low level have implicit higher level structure that enables engineers to ignore irrelevant details (unless, of course, those irrelevant details are causing bugs.)

This situation is fascinating, because it is in many senses the opposite of the reductivism debate in science. For software, many aspects of the end behavior of a system can be deductively known from the very lowest level details—we simply know that this complexity is too much for a human. Science operates in the opposite direction: scientists seek simplifying, unifying principles as the delve deeper into more fundamental phenomena. Biology is applied chemistry, chemistry is applied physics, physics is applied quantum mechanics, etc. Most scientists hold the attitude of ontological reduction: anything we interact with can eventually be smashed up into elementary particles.

But even if this reduction is possible, it may not mean we can achieve such a reduction in our theories. Our theories at different levels may even contradict one another (so called Kuhnian incommensurability), and yet these theories approximate and effective. So is constantly pursuing a more fundamental explanation a worthwhile pursuit in science, or, as a software engineer might think, only necessary in the case of a leaky abstraction?

*Postscript.* My last exam is tomorrow, at which point we will return to our regularly scheduled GHC programming.