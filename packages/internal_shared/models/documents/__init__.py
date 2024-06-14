from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DevExpressFunction:
    """
    Represents a scraped or manually added DevExpress function.
    """

    name: str
    description: str
    category: str
    example: str | None = None
    source: str = "DevExpress"
    keywords: List[str] | None = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Converts this instance to a dictionary.
        """
        return {
            "name": self.name,
            "description": self.description,
            "example": self.example,
            "category": self.category,
            "source": self.source,
            "keywords": self.keywords,
        }
