<!--yml
category: 未分类
date: 2024-07-01 18:18:04
-->

# Medieval medicine and computers : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/11/medieval-medicine-and-computers/](http://blog.ezyang.com/2010/11/medieval-medicine-and-computers/)

This is a bit of a provocative post, and its impressions (I dare not promote them to the level of *conclusions*) should be taken with the amount of salt found in a McDonald’s Happy Meal. Essentially, I was doing some reading about medieval medicine and was struck by some of the similarities between it and computer engineering, which I attempt to describe below.

*Division between computer scientists and software engineers.* The division between those who studied medicine, the *physics* (a name derived from physica or natural science) and those who practiced medicine, the *empirics*, was extremely pronounced. There was a mutual distrust—Petrarch wrote, “I have never believed in doctors nor ever will” (Porter 169)—that stemmed in part from the social division between the physics and empirics. Physics would have obtained a doctorate from a university, having studied one of the highest three faculties possible (the others being theology and law), and tended to be among the upper strata of society. In fact, the actual art of medicine was not considered “worthy of study,” though the study of natural science was. (Cook 407).

The division between computer scientists and software engineers is not as bad as the corresponding division in the 1500s, but there is a definite social separation (computer scientists work in academia, software engineers work in industry) and a communication gap between these two communities. In many other fields, a PhD is required to be even considered for a job in your field; here, we see high school students starting up software companies (occasionally successfully) all the time, and if programming Reddit is any indication, there is a certain distaste for purely theoretical computer scientists.

*The unsuitability of pure computer science for the real world.* Though the study of pure medicine was highly regarded during this time, its theories and knowledge were tremendously off the mark. At the start of the 1500s the four humors of Hippocratic medicine were still widely believed: to be struck by disease was to have an imbalance of black bile, yellow bile, phlegm and blood, and thus the cure would be to apply the counteracting humor, and justified such techniques as bloodletting (the practice of purposely bleeding a person). Even the understanding of how the fundamentals of the human body worked were ill-understood: it was not until William Harvey and his *De motu cordis et sanguinis* (1628) that the earlier view that food was concocted in the organs and then flowed outwards via the veins, arteries and nerves to the rest of the body was challenged by the notion of the heart as a pump. (Cook 426) If the circulatory system was true, what did the other organs do? Harvey’s theory completely overturned the existing understanding of how the human body worked, and his theory was quite controversial.

I have enough faith in computer science that I don’t think most of our existing knowledge is fundamentally wrong. But I also think we know tremendously little about the actual nature of computation even at middling sizes, and this is a quite humbling fact. But I am also fundamentally optimistic about the future of computer science in dealing with large systems—more on this at the end.

*Testing instead of formal methods.* The lack of knowledge, however, did stop the *physicians* (as distinct from *physics*) from practicing their medicine. Even the academics recognized the importance of “medieval *practica*; handbooks listing disorders from head to toe with a description of symptoms and treatment.” (Porter 172) The observational (Hippocratic) philosophy, continued to hold great sway: Thomas Sydenham, when asked on the subject of dissection, stated “Anatomy—Botany—Nonsense! Sir, I know an old woman in Covent Garden who understand botany better, and as for anatomy, my butcher can dissect a joint full and well; now, young man, all that is stuff; you must go to the bedside, it is there alone you can learn disease.” (Porter 229)

In the absence of a convincing theoretical framework, empiricism rules. The way to gain knowledge is to craft experiments, conduct observations, and act accordingly. If a piece of code is buggy, how do you fix it? You add debug statements and observe the output, not construct a formal semantics and then prove the relevant properties. The day the latter is the preferred method of choice is a day when practitioners of formal methods across the world will rejoice.

*No silver bullet.* In the absence of reliable medical theories from the physics, quackery flourished in the eighteenth century, a century often dubbed the “Golden Age of Quackery.” (Porter 284) There was no need to explain *why* your wares worked: one simply needed to give a good show (“drawing first a crowd and then perhaps some teeth, both to the accompaniment of drums and trumpets” (Porter 285)), sell a few dozen bottles of your cure, and then move on to the next town. These “medicines” would claim to do anything from cure cancer to restore youth. While some of the quacks were merely charlatans, others earnestly believed in the efficacy of their treatments, and occasionally a quack remedy was actually effective.

I think the modern analogue to quackery are software methodology in all shapes in sizes. Like quack medicines, some of these may be effective, but in the absence of scientific explanations we can only watch the show, buy in, and see if it works. And we can’t hope for any better until our underlying theories are better developed.

*The future.* Modern medical science was eventually saved, though not before years of inadequate theories and quackery had brought it to a state of tremendous disrepute. The *complexity* of the craft had to be *legitimized*, but this would not happen until a powerful scientific revolution built the foundations of modern medical practice.

* * *

Works referenced:

*   Porter, Roy. *The Greatest Benefit To Mankind.* Fontana Press: 1999.
*   Park, Katherine; Daston, Lorraine. *The Cambridge History of Science: Volume 3, Early Modern Science.*