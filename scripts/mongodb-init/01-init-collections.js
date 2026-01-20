// MongoDB initialization script
// Creates indexes for prompt_requests collection

db = db.getSiblingDB('prompt_db');

// Create unique index on user_id + prompt_id
db.prompt_requests.createIndex(
  { user_id: 1, prompt_id: 1 },
  { unique: true, name: "uq_user_prompt" }
);

// Create index on status for faster queries
db.prompt_requests.createIndex(
  { status: 1 },
  { name: "idx_status" }
);

// Create index on created_at for time-based queries
db.prompt_requests.createIndex(
  { created_at: 1 },
  { name: "idx_created_at" }
);

// Create index on priority for priority-based queries
db.prompt_requests.createIndex(
  { priority: 1 },
  { name: "idx_priority" }
);

print("MongoDB indexes created successfully");

