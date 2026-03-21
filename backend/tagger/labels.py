"""Parse selected_tags.csv from WD tagger models."""

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LabelData:
    names: list[str] = field(default_factory=list)
    rating_indices: list[int] = field(default_factory=list)
    general_indices: list[int] = field(default_factory=list)
    character_indices: list[int] = field(default_factory=list)


def load_labels(csv_path: Path) -> LabelData:
    """Load tag labels from selected_tags.csv.

    CSV columns: tag_id, name, category, count
    Categories: 0=general, 4=character, 9=rating
    """
    data = LabelData()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            name = row["name"]
            category = int(row["category"])
            data.names.append(name)

            if category == 0:
                data.general_indices.append(i)
            elif category == 4:
                data.character_indices.append(i)
            elif category == 9:
                data.rating_indices.append(i)

    return data
