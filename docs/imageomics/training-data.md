# Training Data

Where can we get more training data?

1. [iNat](https://www.inaturalist.org/)
2. Field Guides
3. Birds
4. [LILA BC](https://lila.science/)
5. [Reddit](https://old.reddit.com/r/whatisthisanimal)
6. [Merlin](https://www.allaboutbirds.org/guide/Merlin/overview)

## iNaturalist

iNaturalist has lots of image data that might be useful.

* [DarwinCore Archive](https://www.inaturalist.org/taxa/inaturalist-taxonomy.dwca.zip) is the entire iNat tree and all common names. Found [here](https://www.inaturalist.org/pages/developers).
* [Amazon Open Data Program](https://www.inaturalist.org/blog/49564-inaturalist-licensed-observation-images-in-the-amazon-open-data-sponsorship-program) provides a way for researchers to download lots of media from iNat for free for research purposes.

There are comments on iNat as well.
I haven't found any that aren't just species suggestions.
But if we could find "debates" about an image's species, there might be textual data describing individual traits (supporting evidence for a particular classification).


## Field Guides

Field guides will likely have great textual descriptions of traits.
Many of them (for birds) should be available in PDF form.
There are two challenges:

1. Extracting images and text from PDFs, depending on whether we need OCR or if there is text embedded in the document.
2. Copyright issues

**Butterfly field guides:**
* [Common Butterflies of the Chicago Region](https://fieldguides.fieldmuseum.org/sites/default/files/rapid-color-guides-pdfs/butterflyguide_new.pdf)
* [Field Studies Council Butterflies guide](https://www.field-studies-council.org/shop/publications/butterflies-guide/)
* [The Complete Field Guide to Butterflies of Australia](https://www.researchgate.net/publication/283572367_The_Complete_Field_Guide_to_Butterflies_of_Australia)
* [eBMS Field Guides for butterfly Identification](https://butterfly-monitoring.net/field-guides)
* [Field Guide to the Butterflies of Sri Lanka](https://www.researchgate.net/publication/329880548_Field_Guide_to_the_Butterflies_of_Sri_Lanka)

**Heliconius:**
* [UWI's The Online Guide to the Animals of Trinidad and Tobago](https://sta.uwi.edu/fst/lifesciences/sites/default/files/lifesciences/documents/ogatt/Heliconius_melpomene%20-%20Postman%20Butterfly.pdf)

## Bird datasets
* [Avibase](https://avibase.bsc-eoc.org/avibase.jsp?lang=EN)
* [Kaggle BIRDS 515 SPECIES- IMAGE CLASSIFICATION](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)
* [Birdsnap](https://paperswithcode.com/dataset/birdsnap)


## LILA BC

LILA has 10M labeled images.
We don't have textual descriptions, but 10M images is nothing to sneeze at.

## Reddit

Reddit (and Twitter) have communities (`r/whatisthisanimal`, `r/animalid`) around identifying images of species.
There is likely lots of rich textual data describing animal traits. 
However, there is also likely a lot of noisy text data.


## Merlin

Merlin has lots of detailed pictures of animals at varying degrees of detail:

* Adult male (Taiga)
* Adult male (Prairie)
* Female/immature (Taiga)
* Etc

There are also detailed text descriptions:

* "Small stocky falcon with a blocky head. Males are generally dark overall, but their color varies geographically. The Taiga subspecies is medium gray above with a pale mustache stripe and a thin white eyebrow."

This sort of data is exactly what we want!

How do we get access to it?
