"""Frozen 40-class contextual prompt bank for Stanford-40.

Prompt design:
P0 = original Phase-1 basic prompt
P1 = action / human-object interaction description
P2 = contextual visual description

These prompts are defined from class semantics only.
They must be frozen before test-set evaluation.
"""

PROMPT_BANK = {
    "applauding": {
        "p1": "a photo of a person bringing both hands together repeatedly to applaud",
        "p2": "a photo of a person applauding with both hands, often while watching or responding to an event",
    },
    "blowing_bubbles": {
        "p1": "a photo of a person blowing air through a bubble wand or bubble-making object",
        "p2": "a photo of a person producing soap bubbles using a small wand or bubble toy",
    },
    "brushing_teeth": {
        "p1": "a photo of a person using a toothbrush to brush their teeth",
        "p2": "a photo of a person brushing their teeth with a toothbrush held near the mouth",
    },
    "cleaning_the_floor": {
        "p1": "a photo of a person cleaning a floor using a mop, broom, brush, or cleaning tool",
        "p2": "a photo of a person cleaning the ground or floor surface with a long-handled cleaning tool",
    },
    "climbing": {
        "p1": "a photo of a person climbing upward using their hands and feet",
        "p2": "a photo of a person climbing a wall, rock, structure, or steep surface",
    },
    "cooking": {
        "p1": "a photo of a person preparing or cooking food using kitchen utensils",
        "p2": "a photo of a person cooking food near pots, pans, ingredients, or a cooking area",
    },
    "cutting_trees": {
        "p1": "a photo of a person cutting a tree or large branch using a saw or cutting tool",
        "p2": "a photo of a person working beside a tree while cutting wood with a saw or similar tool",
    },
    "cutting_vegetables": {
        "p1": "a photo of a person using a knife to cut vegetables",
        "p2": "a photo of a person cutting vegetables on a work surface or cutting board with a knife",
    },
    "drinking": {
        "p1": "a photo of a person bringing a cup, bottle, or drinking container toward their mouth",
        "p2": "a photo of a person drinking liquid from a cup, glass, bottle, or similar container",
    },
    "feeding_a_horse": {
        "p1": "a photo of a person giving food to a horse",
        "p2": "a photo of a person standing near a horse and offering food by hand or from a container",
    },
    "fishing": {
        "p1": "a photo of a person using a fishing rod or fishing equipment to catch fish",
        "p2": "a photo of a person fishing near water while holding a rod, line, or other fishing equipment",
    },
    "fixing_a_bike": {
        "p1": "a photo of a person repairing or adjusting a bicycle",
        "p2": "a photo of a person working closely on a bicycle wheel, chain, frame, or mechanical part",
    },
    "fixing_a_car": {
        "p1": "a photo of a person repairing or inspecting a car",
        "p2": "a photo of a person working on a vehicle near the engine, wheel, or mechanical components",
    },
    "gardening": {
        "p1": "a photo of a person tending plants or soil using gardening tools",
        "p2": "a photo of a person gardening among plants, soil, flowers, or garden equipment",
    },
    "holding_an_umbrella": {
        "p1": "a photo of a person holding an open umbrella above or beside their body",
        "p2": "a photo of a person carrying an umbrella outdoors, often with the canopy open above them",
    },
    "jumping": {
        "p1": "a photo of a person jumping with their body lifted off the ground",
        "p2": "a photo of a person airborne during a jump with legs or arms positioned for the movement",
    },
    "looking_through_a_microscope": {
        "p1": "a photo of a person placing their eye near the eyepiece of a microscope",
        "p2": "a photo of a person examining something through a microscope on a table or laboratory surface",
    },
    "looking_through_a_telescope": {
        "p1": "a photo of a person looking through the eyepiece of a telescope",
        "p2": "a photo of a person observing a distant scene through a long optical telescope",
    },
    "phoning": {
        "p1": "a photo of a person using a telephone or mobile phone to make a call",
        "p2": "a photo of a person communicating by phone while holding a telephone or mobile device near the head",
    },
    "playing_guitar": {
        "p1": "a photo of a person holding and playing a guitar with both hands",
        "p2": "a photo of a person playing a guitar with one hand on the neck and the other near the strings",
    },
    "playing_violin": {
        "p1": "a photo of a person playing a violin using a bow",
        "p2": "a photo of a person holding a violin near the shoulder or chin while moving a bow across the strings",
    },
    "pouring_liquid": {
        "p1": "a photo of a person tilting a container so that liquid flows out",
        "p2": "a photo of a person pouring liquid from one bottle, cup, jug, or container into another place",
    },
    "pushing_a_cart": {
        "p1": "a photo of a person pushing a cart or trolley using their hands",
        "p2": "a photo of a person walking behind a wheeled cart or trolley while pushing it forward",
    },
    "reading": {
        "p1": "a photo of a person reading written or printed material",
        "p2": "a photo of a person looking attentively at text in a book, newspaper, magazine, or other reading material",
    },
    "riding_a_bike": {
        "p1": "a photo of a person seated on and riding a bicycle",
        "p2": "a photo of a cyclist riding a bicycle while controlling the handlebars and pedals",
    },
    "riding_a_horse": {
        "p1": "a photo of a person seated on and riding a horse",
        "p2": "a photo of a rider positioned on the back of a horse while travelling or controlling the animal",
    },
    "rowing_a_boat": {
        "p1": "a photo of a person rowing a boat using one or more oars",
        "p2": "a photo of a person seated in a boat and moving an oar through the water",
    },
    "running": {
        "p1": "a photo of a person running with a fast forward body movement",
        "p2": "a photo of a person in a running stride with arms and legs positioned for rapid movement",
    },
    "shooting_an_arrow": {
        "p1": "a photo of a person drawing a bow and shooting an arrow",
        "p2": "a photo of an archer holding a bow while pulling or releasing the bowstring and arrow",
    },
    "smoking": {
        "p1": "a photo of a person smoking a cigarette, cigar, or similar object",
        "p2": "a photo of a person holding or using a smoking object close to the mouth",
    },
    "taking_photos": {
        "p1": "a photo of a person using a camera or photographic device to take a picture",
        "p2": "a photo of a person aiming or holding a camera toward a subject while taking a photograph",
    },
    "texting_message": {
        "p1": "a photo of a person using their hands to type or read a message on a mobile phone",
        "p2": "a photo of a person looking at a handheld phone screen while entering or reading a text message",
    },
    "throwing_frisby": {
        "p1": "a photo of a person throwing a flying disc with one hand",
        "p2": "a photo of a person making a throwing motion with a frisbee or flying disc",
    },
    "using_a_computer": {
        "p1": "a photo of a person interacting with a computer, keyboard, mouse, or screen",
        "p2": "a photo of a person working at a desktop or laptop computer while looking at the display",
    },
    "walking_the_dog": {
        "p1": "a photo of a person walking together with a dog",
        "p2": "a photo of a person walking outdoors beside a dog, often connected by a leash",
    },
    "washing_dishes": {
        "p1": "a photo of a person washing plates, bowls, cups, or other dishes",
        "p2": "a photo of a person cleaning dishes with water around a sink or washing area",
    },
    "watching_TV": {
        "p1": "a photo of a person watching a television screen",
        "p2": "a photo of a person seated or standing while looking toward a television or video display",
    },
    "waving_hands": {
        "p1": "a photo of a person raising and waving one or both hands",
        "p2": "a photo of a person making a visible waving gesture with an open hand",
    },
    "writing_on_a_board": {
        "p1": "a photo of a person writing on a board using chalk or a marker",
        "p2": "a photo of a person standing near a board and writing text or symbols onto its surface",
    },
    "writing_on_a_book": {
        "p1": "a photo of a person writing in a book, notebook, or paper using a pen or pencil",
        "p2": "a photo of a person looking down while writing by hand on pages in a book or notebook",
    },
}


def get_prompts(class_names, strategy):
    """Return one prompt per class for P0, P1, or P2."""

    if strategy not in {"p0", "p1", "p2"}:
        raise ValueError(
            f"Unknown prompt strategy: {strategy}"
        )

    prompts = []

    for class_name in class_names:
        if class_name not in PROMPT_BANK:
            raise KeyError(
                f"Missing contextual prompts for class: {class_name}"
            )

        if strategy == "p0":
            readable = class_name.replace("_", " ")
            prompt = f"a photo of a person {readable}"
        else:
            prompt = PROMPT_BANK[class_name][strategy]

        prompts.append(prompt)

    return prompts


def validate_prompt_bank(class_names):
    """Verify exact coverage of the dataset class list."""

    dataset_classes = set(class_names)
    bank_classes = set(PROMPT_BANK)

    missing = sorted(dataset_classes - bank_classes)
    extra = sorted(bank_classes - dataset_classes)

    if missing or extra:
        raise ValueError(
            f"Prompt-bank mismatch. Missing={missing}, Extra={extra}"
        )

    return True
