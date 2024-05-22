# RAG evaluation prototypes

This repository contains prototypes to evaluate RAG pipelines, as well as code to generate synthetic test data to create datasets to automatically evaluate RAG pipelines. Besides this, the repository also contains prototypes on how to provide domain knowledge to LLMs.

## Evaluation
   
The evaluation process leverages [deepeval](https://github.com/confident-ai/deepeval) and [ragas](https://github.com/explodinggradients/ragas) for assessing the performance of the RAG pipeline. Key metrics include answer relevancy, faithfulness, contextual precision, and recall. The evaluation involves comparing actual outputs from the application with expected outputs using a predefined dataset.  
   
For more detailed information on how these metrics are calculated, refer to the [deepeval](https://docs.confident-ai.com/docs/metrics-introduction) and [ragas](https://docs.ragas.io/en/stable/concepts/metrics/index.html) documentation.  
   
To run the evaluation, start the Docker container and use the `/evaluate` endpoint to submit your data. Evaluation results are stored in MongoDB, and metrics are calculated by comparing the real results with the expected ones. For more information on the evaluation process, refer to the specific notebooks and documentation within the [evaluation](./evaluation/) folder.

## Synthetic data generation
   
Creating a synthetic test dataset is advantageous for evaluating LLM applications, as it can address the limitations of human-generated questions. Frameworks like [ragas](https://docs.ragas.io/en/stable/concepts/testset_generation.html#why-synthetic-test-data) use data evolution methods to generate diverse and challenging datasets, crucial for a thorough evaluation. This method is detailed in the [Evol-Instruct and WizardLM](https://arxiv.org/pdf/2304.12244) paper.  
   
For more information, refer to the [ragas documentation](https://docs.ragas.io/en/stable/concepts/testset_generation.html#how-does-ragas-differ-in-test-data-generation) or the [deepeval blog](https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms).  
   
Currently, the implementation of synthetic data generation is incomplete. While frameworks perform well for simple Q&A tasks, more complex tasks like formula or workflow creation require a manual approach. Test implementations are available in the `data_generation_tests.ipynb` notebook. For more information, check the specific notebook and documentation within the [data-generation](./data-generation/) folder.


## Domain knowledge

Extracts detailed domain knowledge from the documentation and transforms it for the usage in a vector database (textual content), as well as for a knowledge graph (relationships between the types). The key functions are:

1. **Domain Knowledge Extraction**:  
   - Extracts detailed interface descriptions and metadata from DocFX documentation.  
   - Transforms extracted data into raw text and metadata files for use in a vector database.  
   
2. **Knowledge Graph Creation**:  
   - Constructs a Neo4j graph using the extracted domain knowledge.  
   - Utilizes type references to establish relationships between different interfaces.  
   - Creates vector embeddings for natural language access and advanced querying.  
   
### Advantages of Using Neo4j with Vector Embeddings  
- **Hybrid Queries**: Combines vector similarity searches with graph traversal.  
- **Contextual Relevance**: Refines and contextualizes results using graph structure.  
- **Graph Algorithms**: Enhances analysis with algorithms like PageRank and community detection.  
   
For more detailed information, refer to the specific notebooks and documentation within the [domain knowledge](./domain-knowledge/) folder.