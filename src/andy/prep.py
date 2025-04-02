import re
from collections import defaultdict
from collections.abc import Set

import pandas as pd
from jarowinkler import jaro_similarity
from tqdm import tqdm


def normalize_title(title: str) -> str:
    """Basic title normalization."""
    if pd.isna(title):
        return ""
    title = str(title).lower()
    title = title.strip()
    title = re.sub(r"^[^a-z]+", "", title)
    title = re.sub(r"[^a-z]+$", "", title)
    return title


def bin_job_titles(
    common_titles: list[str], job_titles: pd.Series, threshold: float = 0.8
) -> dict[str, Set[str]]:
    """Bin job titles into groups."""
    groups: dict[str, Set[str]] = defaultdict(set)

    for job_title in tqdm(job_titles, desc="Binning job titles"):
        max_sim: float = -1.0
        best_match: str = ""

        for common_title in common_titles:
            sim: float = jaro_similarity(str(job_title), str(common_title))
            if sim > max_sim:
                max_sim = sim
                best_match = common_title

        if max_sim > threshold:
            groups[best_match].add(job_title)
        else:
            groups["other"].add(job_title)

    return groups