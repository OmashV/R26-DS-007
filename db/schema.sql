-- Meta-Agent SQLite schema
-- Used during development. In production, this storage layer will be
-- backed by the Memory component's persistence layer.

-- Students table
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    profile_json TEXT
);

-- Per-(student, skill) mastery state — the knowledge graph
CREATE TABLE IF NOT EXISTS mastery (
    student_id TEXT,
    skill_name TEXT,
    mastery_probability REAL NOT NULL,
    mastery_label TEXT NOT NULL,
    previous_mastery_probability REAL,  -- NEW
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (student_id, skill_name),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- Every attempt a student has made
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
    session_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- Session audit log
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    student_id TEXT NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    concept_count INTEGER,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_attempts_student_skill 
    ON attempts(student_id, skill_name, created_at);
CREATE INDEX IF NOT EXISTS idx_mastery_student 
    ON mastery(student_id);