<!--yml
category: 未分类
date: 2024-07-01 18:17:46
-->

# If it has lots of comments, it’s probably buggy : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/05/byron-cook-sla/](http://blog.ezyang.com/2011/05/byron-cook-sla/)

## If it has lots of comments, it’s probably buggy

Yesterday we had guest speaker [Byron Cook](http://research.microsoft.com/en-us/people/bycook/) come in to give a talk about [SLAM](http://research.microsoft.com/en-us/projects/slam/), a nice real-world example of theorem proving technology being applied to device drivers.

Having worked in the trenches, Byron had some very hilarious (and interesting) quips about device driver development. After all, when a device driver crashes, it's not the device driver writer that gets blamed: it’s Microsoft. He pointed out that, in a hardware company, “If you’re not so smart, you get assigned to write software drivers. The smart people go work on hardware”, and that when you’re reading device driver code, “If there are a lot of comments and they’re misspelled, there’s probably a bug.” Zing! We’re always used to extolling the benefits of commenting your code, but it certainly is indisputable that writing comments can help clarify confusing code to yourself, whereas if the code wasn’t confusing in the first place you wouldn’t have felt the need to write comments anyway. Thus, one situation is some guru from the days of yore wrote very clever code, and then you came along and weren’t quite clever enough to fully understand what was going on, so you wrote lots of comments to explain the code to yourself as you went along. Well, it’s not the comment’s fault, but the fact that the code was too clever for you probably means you introduced a bug when you made your modifications.

The approach used by SLAM to deal with the exponential state space explosion was also pretty interesting. What they do is throw out as much state as possible (without eliminating the bug), and then see whether or this simplified program triggers a bug. It usually does, though due to a spurious transition, so then they introduce just enough extra state to remove that spurious path, and repeat until the simplified program is judged to fulfill the assert (success) or we come across a path in the simplified program which is not spurious in the real program. The other really interesting bit was their choice of specification language was essentially glorified asserts. In an academic class like Temporal Logic, you spend most of your time studying logics like CTL and LTL, which are strange and foreign to device driver writers; asserts are much easier to get people started with. I could definitely see this applying to other areas of formal verification as well (assert based type annotations, anyone?)

*Postscript.* I have some absolutely gargantuan posts coming down the pipeline, but in between revising for exams and last minute review sessions, I haven’t been able to convince myself that finishing up these posts prior to exams is a good use of my time. But they will come eventually! Soon! I hope!