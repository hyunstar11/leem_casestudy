import re
from collections import defaultdict

import pandas as pd
from jarowinkler import jaro_similarity
from tqdm import tqdm


def normalize_title(title: str) -> str:
    """Basic title normalization."""
    if pd.isna(title):
        return ""
    title = str(title).lower()
    # remove leading and trailing whitespace
    title = title.strip()
    # remove leading and trailing punctuation
    title = re.sub(r"^[^a-z]+", "", title)
    title = re.sub(r"[^a-z]+$", "", title)
    return title


def bin_job_titles(
    common_titles: list[str], job_titles: pd.Series, threshold: float = 0.8
) -> dict[str, set[str]]:
    """Bin job titles into groups."""
    groups = defaultdict(set)

    for job_title in tqdm(job_titles, desc="Binning job titles"):
        # Find most similar common title
        max_sim = -1
        best_match = ""

        for common_title in common_titles:
            sim = jaro_similarity(str(job_title), str(common_title))
            if sim > max_sim:
                max_sim = sim
                best_match = common_title

        # Add to appropriate group
        if max_sim > threshold:
            groups[best_match].add(job_title)
        else:
            groups["other"].add(job_title)

    return groups
