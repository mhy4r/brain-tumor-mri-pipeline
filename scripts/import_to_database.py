import json
import sqlite3
import os 

# =========================
# 1. CONFIG
# =========================
DB_NAME = os.path.join(os.path.dirname(__file__), '..', 'database','brain_tumor_db.sqlite')
JSON_PATH = os.path.join(os.path.dirname(__file__), '..', 'DATA.json')

# =========================
# 2. LOAD JSON
# =========================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# 3. CONNECT SQLITE
# =========================
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# =========================
# 4. CREATE TABLES
# =========================

cursor.execute("""
CREATE TABLE IF NOT EXISTS tumor_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name TEXT UNIQUE NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS tumor_grades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    grade TEXT UNIQUE NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    x INTEGER,
    y INTEGER,
    description TEXT,
    type_id INTEGER,
    grade_id INTEGER,
    FOREIGN KEY (type_id) REFERENCES tumor_types(id),
    FOREIGN KEY (grade_id) REFERENCES tumor_grades(id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_name TEXT UNIQUE NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS image_locations (
    image_id INTEGER,
    location_id INTEGER,
    PRIMARY KEY (image_id, location_id),
    FOREIGN KEY (image_id) REFERENCES images(id),
    FOREIGN KEY (location_id) REFERENCES locations(id)
)
""")

conn.commit()

# =========================
# 5. CACHE SYSTEM
# =========================
type_cache = {}
grade_cache = {}
location_cache = {}

# =========================
# 6. HELPERS
# =========================

def get_or_create_type(type_name):
    if type_name in type_cache:
        return type_cache[type_name]

    cursor.execute("INSERT OR IGNORE INTO tumor_types (type_name) VALUES (?)", (type_name,))
    cursor.execute("SELECT id FROM tumor_types WHERE type_name=?", (type_name,))
    type_id = cursor.fetchone()[0]

    type_cache[type_name] = type_id
    return type_id


def get_or_create_grade(grade):
    if grade in grade_cache:
        return grade_cache[grade]

    cursor.execute("INSERT OR IGNORE INTO tumor_grades (grade) VALUES (?)", (grade,))
    cursor.execute("SELECT id FROM tumor_grades WHERE grade=?", (grade,))
    grade_id = cursor.fetchone()[0]

    grade_cache[grade] = grade_id
    return grade_id


def get_or_create_location(loc):
    if loc in location_cache:
        return location_cache[loc]

    cursor.execute("INSERT OR IGNORE INTO locations (location_name) VALUES (?)", (loc,))
    cursor.execute("SELECT id FROM locations WHERE location_name=?", (loc,))
    loc_id = cursor.fetchone()[0]

    location_cache[loc] = loc_id
    return loc_id


def insert_image(file_name, width, height, x, y, description, type_id, grade_id):
    cursor.execute("""
        INSERT INTO images
        (file_name, width, height, x, y, description, type_id, grade_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (file_name, width, height, x, y, description, type_id, grade_id))

    return cursor.lastrowid

# =========================
# 7. ETL LOOP
# =========================

count = 0

for file_name, meta in data.items():

    try:
        width = meta["width"]
        height = meta["height"]

        x = meta["point"]["x"]
        y = meta["point"]["y"]

        description = meta["description"]
        full_class = meta["class"]

        # split class
        parts = full_class.split(" ")
        type_name = " ".join(parts[:-1])
        grade = parts[-1]

        type_id = get_or_create_type(type_name)
        grade_id = get_or_create_grade(grade)

        image_id = insert_image(
            file_name,
            width,
            height,
            x,
            y,
            description,
            type_id,
            grade_id
        )

        # locations
        for loc in meta.get("location", []):
            loc_id = get_or_create_location(loc)

            cursor.execute("""
                INSERT OR IGNORE INTO image_locations (image_id, location_id)
                VALUES (?, ?)
            """, (image_id, loc_id))

        conn.commit()
        count += 1

        if count % 500 == 0:
            print(f"{count} images processed...")

    except Exception as e:
        conn.rollback()
        print(f"Error in {file_name}: {e}")

# =========================
# 8. FINISH
# =========================
conn.close()

print(f"\nDONE Total images inserted: {count}")