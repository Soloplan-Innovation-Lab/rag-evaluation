# Domain Knowledge

This folder currently contains the domain knowledge of the business object interfaces of the ``Soloplan.CarLo.Business.dll`` assembly. The domain knowledge is extracted from the DocFX documentation of the assembly.

## Domain knowledge extraction

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
  ],
  "filename": "ITourStatus.txt",
  "chunk_id": 0,
  "total_chunks": 1,
  "expected_embedding_size": 145
}
```

These two files are then used to provide the domain knowledge to the vector database instance. The type references can be used to create a graph of the domain knowledge.

## Knowledge Graph using type references

The notebook ``create_graph.ipynb`` contains code to create a graph in ``neo4j`` using the scraped domain knowledge. The graph is created by using the scraped interfaces. During the scraping process, all relations between the types are known and stored as ``type_references``. This information is used to create the graph.

In order to *access* the graph using natural language, a [vector index](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/) is created. This accesses the ``embedding`` of the graph nodes, which are created during the graph creation process. The vector index can be used to query the graph using embeddings. Currently, the embeddings are created using the ``name`` and ``summary`` of the interface.

### Advantages of Using Neo4j with Vector Embeddings

The embeddings themselves typically capture the semantic meaning of the node's textual content. They do not inherently include relationship information. The primary advantage of using embeddings in Neo4j, as opposed to a traditional vector database, is the ability to combine vector similarity with graph traversal and relationship queries.

**Advantages:**

- **Hybrid Queries**: You can combine vector similarity searches with traditional graph traversal queries. For example, you might first find nodes similar to a query embedding and then traverse their relationships to find related nodes or paths.
- **Contextual Relevance**: By leveraging the graph structure, you can refine and contextualize the results further. For instance, you might find nodes similar to your query and then filter or rank them based on their relationships.
- **Graph Algorithms**: You can apply graph algorithms (e.g., PageRank, community detection) to further analyze and enhance the results obtained from vector searches.

More on this topic can be read in the following articles:

- [Neo4j Vector Index and Search](https://neo4j.com/labs/genai-ecosystem/vector-search/)
- [Implementing Advanced Retrieval RAG Strategies With Neo4j](https://neo4j.com/developer-blog/advanced-rag-strategies-neo4j/)
- [Using a Knowledge Graph to Implement a RAG Application](https://neo4j.com/developer-blog/knowledge-graph-rag-application/)

*Note, that neo4j can do much more, e.g. [creating node embeddings](https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/) or accessing external services like [OpenAI](https://neo4j.com/labs/apoc/5/ml/openai/) or [Azure Cogntive Serivces](https://neo4j.com/labs/apoc/5/nlp/azure/).*

### Useful commands

**Extend the maximum number of nodes to display in the graph:**
```cypher
:config initialNodeDisplay: 1000
```

**Display all nodes in the graph:**
```sql
MATCH (n) RETURN n LIMIT 1000
```


**Find all interfaces that reference the interface ``IBusinessPartner``:**
```sql
MATCH (n:Interface)-[:REFERENCES]->(m:Interface {name: "IBusinessPartner"})  
RETURN n  
```

**Find all interfaces that are referenced by the interface ``IBusinessPartner``:**
```sql
MATCH (n:Interface {name: "IBusinessPartner"})-[:REFERENCES]->(m:Interface)  
RETURN m  
```

**Clean the database:**
```sql
match (a) -[r] -> () delete a, r

match (a) delete a
```