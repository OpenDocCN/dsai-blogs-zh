<!--yml
category: 未分类
date: 2024-07-01 18:18:33
-->

# Audio connectors and radios : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/01/audio-connectors/](http://blog.ezyang.com/2010/01/audio-connectors/)

Over winter break, I purchased a [Yaesu VX-8R](http://www.yaesu.com/indexVS.cfm?cmd=DisplayProducts&ProdCatID=111&encProdID=64C913CDBC183621AAA39980149EA8C6), the newest model from Yaesu and the successor to the [VX-7R](http://www.yaesu.com/indexVS.cfm?cmd=DisplayProducts&ProdCatID=111&encProdID=8D3254BFC69FB172D78647DC56EFB0E9&DivisionID=65&isArchived=0), which is favored by many in the MIT community. Deciding that this was the particular radio I wanted to buy was difficult: purchasing a (cheaper) VX-7R would mean I could tap into the immense pool of knowledge that has already rallied itself around this particular model. But my father was willing to put down the extra $50 for the newer version, and so I decided to be experimental.

That's, however, not the point of this post (a review of the VX-8R will have to wait until I actually get my hands on it): the real puzzling part of the exercise was figuring out what accessories to purchase in order to get a microphone headset. If that sounds vague, that's because it is. Yaesu's official accessories—which were strongly Bluetooth oriented—were devoid of a standard, wired headset. After some investigation, and a very informative conversation with Kelsey, here is what I found.

First, some vocabulary. If you've ever plugged in headphones to your computer, you are familiar with the [TRS connector](http://en.wikipedia.org/wiki/TRS_connector), also known as an audio or headphone jack. It is the exposed, tapered jack, and contain multiple conductors (the black rings separate them). The connector for the stereo speakers you own probably has three conductors; one for ground, one for the left channel, and one for the right. There is a wide variety of TRS connectors in both size and number of conductors. For a radio (and more generally equipment that can utilize a push-to-talk (PTT) headset), we are interested in TRS connectors of 3.5mm (1/8in) diameter with four conductors: one ground, one audio, one microphone, and one PTT.

A [DIN connector](http://en.wikipedia.org/wiki/DIN_connector) has a number of pins protected by metal shielding. The pins in a DIN connector correspond to conductors, and it is not unusual for there to be three to eight pins (greater than four conductor TRS connectors are quite rare.) DIN connectors have standard size (13.2mm), but the assignment of pins varies from application to application.

Now, for the actual radios. We'll start with the VX-7R, since the VX-8R jack is strictly more powerful than the 7R's. The VX-7R sports a four conductor 3.5mm TRS connector, but with a twist: it's specially designed to be waterproof, so to get a snug fit you need a special TRS connector that has a screw after the actual jack. The CT-91 is such a TRS connector, and it splits the connection into a three-conductor 3.5mm headset TRS plug, and a three-conductor 2.5mm microphone TRS plug; these are apparently standard jacks and thus you can find an assortment of headsets as well as individual earbuds and ptt microphones for them. (Note: I didn't have any lying around the house, and didn't get a chance to head out to a radio store, so this is strictly hearsay.)

The VX-8R, on the other hand, has support for GPS, so it can't get away with just four conductors: instead, it sports an 8-pin DNS plug, which for all intents and purposes is proprietary. You can hook up the GPS unit (CT-136 and FGPS-2), but in general you'll need the $30 connector, the CT-131, to translate it into a four-conductor 3.5mm TRS jack. This is the same four-conductor form as the TRS plug on the VX-7R, but without the waterproof screwy bit. To split it, you can use CT-91, but the screw bit will show and for a more snug fit you'll have to buy their suggested CT-44.

We were able to find a four-conductor headset lying around the house, but it didn't work; like the miscellaneous three-conductor stereo headsets I tried, plugging it in resulted in a working sound signal, but caused PTT to be persistently activated. The current theory is that stereo was messing with things.

Here are the things I'd like to know:

1.  The VX-8R has a separate stereo-headphone jack, so I'm a bit curious what would happen if I stuck a PTT microphone into the four-conductor plug. If by some miracle the two were compatible, it would mean that a secondary adapter is not necessary. Given that the splitter suggests a 2.5mm mic, and the 4-conductor plug is 3.5mm, this seems unlikely.
2.  The CT-131 and CT-91 form a kind of sketchy-looking connection, and I'm not sure if this would actually prove to be a problem in practice, or if I'd want to electrical tape the two together. Some field-testing is required here, and I'd also be curious to know how difficult it would be to purchase or make a 4-conductor to 2.5mm PTT mic adapter.
3.  I need to find a store close by Cambridge/Boston where I can test various push-to-talk microphones. Any suggestions here would be greatly appreciated!