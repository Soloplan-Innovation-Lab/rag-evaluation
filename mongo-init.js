db = db.getSiblingDB('evaluation_db');

db.createCollection('runs');
db.createCollection('evaluations');
db.createCollection('iterations');

// Optional: Create indexes if needed
db.evaluations.createIndex({ run_id: 1 });
db.iterations.createIndex({ evaluation_id: 1 });
