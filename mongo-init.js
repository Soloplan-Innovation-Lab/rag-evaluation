// evaluation db
db = db.getSiblingDB('evaluation_db');

db.createCollection('runs');
db.createCollection('evaluations');
db.createCollection('iterations');

db.evaluations.createIndex({ run_id: 1 });
db.iterations.createIndex({ evaluation_id: 1 });


// rag_pipeline db
db = db.getSiblingDB('rag_pipeline');

db.createCollection('chat_response');
db.createCollection('prompt_template');
db.createCollection('retriever_config');

db.chat_response.createIndex({ chat_session_id: 1 });
db.prompt_template.createIndex({ name: 1 });
db.retriever_config.createIndex({ retriever_name: 1 });
