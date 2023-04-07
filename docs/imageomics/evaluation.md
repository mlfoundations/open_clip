# Evaluation

How are we going to evaluate CLIP-like models for use in biology?

1. Zero-shot classification accuracy on IID data
2. Zero-shot clasification accuracy on co-mimics
3. Zero-shot classification accuracy on co-mimics with textual descriptions of the differences
4. Data efficiency (OPEN QUESTION)
5. Generalization to lab/museum photos (QUESTIONS)
6. Trait presence (OPEN QUESTION)

# Zero-shot Accuracy on iNaturalist21

We want to use 1K classes so we can compare to ImageNet1K classes.
We also want to compare performance on seen and unseen classes.
So we might include 9K classes during pretraining, then evaluate accuracy on 1K seen classes, then evaluate accuracy on the remaining 1K unseen classes.

We also want to compare common names and taxonomic names for the text encoder: common names are likely easier.

# Butterflies Classification

Can we classify museum photos of co-mimics?
They're deliberately visually challenging.
We would want to do zero-shot classification, perhaps with textual descriptions of the differences between species.

# Data Efficiency

How can we do few-shot evaluation of CLIP models?
Should we fine-tune?
Should we do some sort of prompting or in-context learning?

# Generalization to Lab/Museum Photos

Can we do zero-shot classification of the fish photos from the Kenya team?
These photos are OOD in the sense that they are on white backgrounds (same applies to the butterflies).

Ideally we would have some naturalist/citizen science photos of the same species, then see if the classification generalizes across background, rather than generalize across background *and* unseen species.

# Presence of Traits

Can we identify traits in a picture?
What about in a species?
Given a picture of some animals, can we definitely say "these traits are present in this photo"?
Can we say "this animal has these traits, even though they are not visible"?

What dataset can we use for this? 
Can we construct a small one (~200-500 examples)?
