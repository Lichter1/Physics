"""Mapping of congressional committees to market sectors they oversee."""

COMMITTEE_SECTOR_MAP = {
    # House Committees
    "Armed Services": ["Defense & Aerospace", "Industrials"],
    "Energy and Commerce": ["Energy", "Healthcare & Pharma", "Technology"],
    "Financial Services": ["Financials & Banking", "Real Estate"],
    "Science, Space, and Technology": ["AI & Semiconductors", "Technology", "Defense & Aerospace"],
    "Transportation and Infrastructure": ["Industrials", "Real Estate"],
    "Ways and Means": ["Financials & Banking", "Healthcare & Pharma"],
    "Appropriations": ["Defense & Aerospace", "Energy", "Healthcare & Pharma"],
    "Agriculture": ["Consumer Discretionary"],
    "Education and the Workforce": ["Consumer Discretionary", "Technology"],
    "Homeland Security": ["Defense & Aerospace", "Technology"],
    "Foreign Affairs": ["Defense & Aerospace", "Energy"],
    "Judiciary": ["Technology"],
    "Natural Resources": ["Energy", "Real Estate"],
    "Small Business": ["Consumer Discretionary", "Technology"],
    "Veterans' Affairs": ["Healthcare & Pharma"],

    # Senate Committees
    "Banking, Housing, and Urban Affairs": ["Financials & Banking", "Real Estate"],
    "Commerce, Science, and Transportation": ["Technology", "AI & Semiconductors", "Consumer Discretionary"],
    "Energy and Natural Resources": ["Energy"],
    "Finance": ["Financials & Banking", "Healthcare & Pharma"],
    "Health, Education, Labor, and Pensions": ["Healthcare & Pharma"],
    "Intelligence": ["Defense & Aerospace", "AI & Semiconductors", "Technology"],
}


def get_sectors_for_committee(committee_name: str) -> list[str]:
    """Return list of sectors a committee oversees."""
    if not committee_name:
        return []
    for key, sectors in COMMITTEE_SECTOR_MAP.items():
        if key.lower() in committee_name.lower() or committee_name.lower() in key.lower():
            return sectors
    return []


def is_committee_sector_match(committee: str, sector: str) -> bool:
    """Check if a transaction's sector falls under the filer's committee jurisdiction."""
    sectors = get_sectors_for_committee(committee)
    return sector in sectors
