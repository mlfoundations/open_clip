import os

from imageomics import eol, naming

hoyal_cuthill_root = "/fs/ess/PAS2136/foundation_model/Hoyal_Cuthill_Butterfly/train"


def get_overlapping(lookup):
    seen = set()
    for key in lookup.keys():
        scientific = lookup.scientific(key)
        if scientific:
            for hoyal in hoyal_species:
                if hoyal.lower() in scientific.lower():
                    seen.add(hoyal)

        taxonomic = lookup.taxonomic(key)
        if taxonomic:
            for hoyal in hoyal_species:
                if hoyal.lower() in taxonomic.lower():
                    seen.add(hoyal)

    return seen


if __name__ == "__main__":
    hoyal_species = []
    for clsdir in os.listdir(hoyal_cuthill_root):
        *_, genus, species, subspecies = clsdir.split("_")
        assert genus == "Heliconius"
        hoyal_species.append(f"{genus} {species} {subspecies}")

    seen_hoyal = set()

    # Add Bioscan
    bioscan_name_lookup = naming.BioscanNameLookup()
    seen_hoyal |= get_overlapping(bioscan_name_lookup)
    print(seen_hoyal)
    print(set(hoyal_species) == seen_hoyal)

    # Add EOL
    eol_name_lookup = eol.EolNameLookup()
    seen_hoyal |= get_overlapping(eol_name_lookup)
    print(seen_hoyal)
    print(set(hoyal_species) == seen_hoyal)

    # Add iNaturalist
    inaturalist_name_lookup = naming.iNaturalistNameLookup()
    seen_hoyal |= get_overlapping(inaturalist_name_lookup)
    print(seen_hoyal)
    print(set(hoyal_species) == seen_hoyal)

    # Add iNat21
    inat21_name_lookup = naming.iNat21NameLookup()
    seen_hoyal |= get_overlapping(inat21_name_lookup)
    print(seen_hoyal)
    print(set(hoyal_species) == seen_hoyal)
    