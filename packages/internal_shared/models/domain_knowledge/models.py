try:
    from neomodel import (
        ArrayProperty,
        FloatProperty,
        StructuredNode,
        StringProperty,
        RelationshipTo,
        StructuredRel,
    )
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "Necessary evaluation dependencies are not installed. Please run `pip install neomodel pydantic`."
    )


# Define relationship models
class ReferencesRel(StructuredRel):
    name = StringProperty()
    description = StringProperty()


# Define node models
class Interface(StructuredNode):
    name = StringProperty(unique_index=True)
    summary = StringProperty()
    namespace = StringProperty()
    assembly = StringProperty()
    references = RelationshipTo("Interface", "REFERENCES", model=ReferencesRel)
    embedding = ArrayProperty(FloatProperty())


# define pydantic models
class Metadata(BaseModel):
    name: str
    summary: str
    namespace: str
    assembly: str
    type_references: list[str]
