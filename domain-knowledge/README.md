# Domain knowledge extraction

The notebook ```scrape_docfx_tests.ipynb``` contains code to scrape the DocFX documentation to get all the domain knowledge of the business object interfaces.

Besides scraping, the notebook also provides code to transform the ``scraped_domain_knowledge.json`` into raw text files and metadata files to provide to the vector database instance. Each document and its metadata can be loaded and embedded (the metadata will only be added as metadata for the embedding).

Example of a scraped interface:
```json
{
    "summary": "The status of a tour.",
    "type": "interface",
    "namespace": "Soloplan.CarLo.Business",
    "assembly": "Soloplan.CarLo.Business.dll",
    "properties": [
      {
        "name": "StatusCategoryCombination",
        "summary": "Gets or sets the status to status category combination assigned to this tour status.",
        "declaration": "[SoloProperty(5060202, PropertyType.Interface, \"Status & -kategorie\", true, FilterOptions.None)]\nIStatusCategoryCombination StatusCategoryCombination { get; set; }",
        "type": "IStatusCategoryCombination"
      },
      {
        "name": "StatusIndex",
        "summary": "Gets or sets the status index.",
        "declaration": "[SoloProperty(5060203, PropertyType.Integer, \"Statusindex\", true, FilterOptions.None)]\nint StatusIndex { get; set; }",
        "type": "System.Int32"
      },
      {
        "name": "StatusText",
        "summary": "Gets or sets the status text.",
        "declaration": "[SoloProperty(5060204, PropertyType.String, \"Statustext\", true, FilterOptions.None, MaxLength = 2000)]\nstring StatusText { get; set; }",
        "type": "System.String"
      },
      {
        "name": "This",
        "summary": "Gets a reference to this instance (maybe helpful when using data-binding).",
        "declaration": "ITourStatus This { get; }",
        "type": "ITourStatus"
      },
      {
        "name": "Tour",
        "summary": "Gets or sets the tour.",
        "declaration": "[SoloProperty(5060201, PropertyType.Interface, \"Tour\", true, FilterOptions.None)]\nITour Tour { get; set; }",
        "type": "ITour"
      }
    ],
    "extension_methods": [
      "ReActExtensions.IsList(Object)",
      // ...
    ],
    "type_references": [
      "IStatusCategoryCombination",
      "ITourStatus"
    ]
}
```

After transformation, the following files are created:

- ITourStatus.txt
```txt
Interface: ITourStatus
Summary: The status of a tour.

Properties:
- StatusCategoryCombination
   - Type: IStatusCategoryCombination
   - Description: Gets or sets the status to status category combination assigned to this tour status.
- StatusIndex
   - Type: System.Int32
   - Description: Gets or sets the status index.
- StatusText
   - Type: System.String
   - Description: Gets or sets the status text.
- This
   - Type: ITourStatus
   - Description: Gets a reference to this instance (maybe helpful when using data-binding).
- Tour
   - Type: ITour
   - Description: Gets or sets the tour.

```

- ITourStatus.metadata.json
```json
{
  "name": "ITourStatus",
  "summary": "The status of a tour.",
  "type": "interface",
  "namespace": "Soloplan.CarLo.Business",
  "assembly": "Soloplan.CarLo.Business.dll",
  "type_references": [
    "IStatusCategoryCombination",
    "ITourStatus"
  ]
}
```

These two files are then used to provide the domain knowledge to the vector database instance. The type references can be used to create a graph of the domain knowledge.