"""
Student Performance Analyzer  -  Python capstone project
=========================================================
Pipeline:  SQLite (store)  ->  pandas (clean + analyse)  ->  matplotlib (chart)  ->  CSV (report)

Skills used: SQL tables & joins, NULL handling, DataFrames, groupby, sorting,
             averages, bar charts with an average line, exporting results.

Run:  python student_performance_analyzer.py
"""
import random
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parent
DB_PATH = OUT / "school.db"
PASS_MARK = 40

SUBJECTS = ["Maths", "Science", "English", "Hindi", "Social", "Computer"]
STUDENTS = [
    (1, "Asha", "A"), (2, "Ravi", "A"), (3, "Meera", "B"), (4, "Kabir", "B"),
    (5, "Isha", "A"), (6, "Arjun", "B"), (7, "Zoya", "A"), (8, "Dev", "B"),
    (9, "Nila", "A"), (10, "Rohan", "B"),
]


# --------------------------------------------------------------------------
# 1. DATABASE: create tables and fill them with sample data
# --------------------------------------------------------------------------
def build_database() -> sqlite3.Connection:
    random.seed(42)                        # same data every run
    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE students (
            id      INTEGER PRIMARY KEY,
            name    TEXT NOT NULL,
            section TEXT
        );
        CREATE TABLE marks (
            student_id INTEGER REFERENCES students(id),
            subject    TEXT NOT NULL,
            score      REAL              -- NULL means the student was absent
        );
    """)
    con.executemany("INSERT INTO students VALUES (?, ?, ?)", STUDENTS)

    rows = [(sid, subj, random.randint(35, 98)) for sid, _, _ in STUDENTS for subj in SUBJECTS]
    for i in random.sample(range(len(rows)), 3):      # make 3 absences (NULLs)
        rows[i] = (rows[i][0], rows[i][1], None)
    con.executemany("INSERT INTO marks VALUES (?, ?, ?)", rows)
    con.commit()
    return con


# --------------------------------------------------------------------------
# 2. LOAD: SQL join straight into a DataFrame
# --------------------------------------------------------------------------
def load_marks(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT s.name, s.section, m.subject, m.score
        FROM students s
        JOIN marks m ON m.student_id = s.id
        """,
        con,
    )


# --------------------------------------------------------------------------
# 3. ANALYSE
# --------------------------------------------------------------------------
def to_grade(avg: pd.Series) -> pd.Series:
    return pd.cut(avg, bins=[0, 40, 60, 75, 90, 101],
                  labels=["F", "D", "C", "B", "A"], right=False)


def analyse(df: pd.DataFrame):
    # Absences are NULL -> NaN. They are left out of averages (NOT counted as 0).
    absent = df[df["score"].isna()][["name", "subject"]]

    table = df.pivot(index="name", columns="subject", values="score")[SUBJECTS]

    students = pd.DataFrame({
        "section": df.drop_duplicates("name").set_index("name")["section"],
        "average": table.mean(axis=1).round(1),            # skips NaN
        "best_subject": table.idxmax(axis=1),
        "weakest_subject": table.idxmin(axis=1),
        "absences": table.isna().sum(axis=1),
    })
    students["grade"] = to_grade(students["average"])
    students = students.sort_values("average", ascending=False)
    students["rank"] = students["average"].rank(ascending=False, method="min").astype(int)

    subjects = pd.DataFrame({
        "average": table.mean().round(1),
        "highest": table.max(),
        "lowest": table.min(),
        "pass_rate_%": ((table >= PASS_MARK).sum() / table.notna().sum() * 100).round(0),
    }).sort_values("average")                               # ascending

    return absent, students, subjects


# --------------------------------------------------------------------------
# 4. CHARTS
# --------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID, BASE, BLUE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6")


def tidy(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=14)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(length=0, labelsize=10, colors=MUTED)


def make_charts(students: pd.DataFrame, subjects: pd.DataFrame, class_avg: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE)

    # Left: subject averages (ascending), vertical bars + class average line
    ax1.bar(subjects.index, subjects["average"], width=0.55, color=BLUE,
            edgecolor=SURFACE, linewidth=2)
    ax1.axhline(class_avg, color=INK2, linestyle=(0, (5, 4)), linewidth=1.5)
    ax1.text(-0.4, class_avg + 1.5, f"Class average {class_avg:.1f}",
             fontsize=10, fontweight="bold", color=INK2)
    ax1.set_ylim(0, 100)
    ax1.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.spines["left"].set_visible(False)
    ax1.set_ylabel("Average marks (out of 100)", color=INK2)
    tidy(ax1, "Average marks by subject")
    low, high = subjects["average"].idxmin(), subjects["average"].idxmax()
    for name in (low, high):
        ax1.annotate(f"{subjects.loc[name, 'average']:.1f}",
                     (list(subjects.index).index(name), subjects.loc[name, "average"]),
                     textcoords="offset points", xytext=(0, 6), ha="center",
                     fontsize=10, fontweight="bold", color=INK)

    # Right: student averages (lowest at bottom, best at top)
    ranked = students.sort_values("average")
    ax2.barh(ranked.index, ranked["average"], height=0.6, color=BLUE,
             edgecolor=SURFACE, linewidth=2)
    ax2.axvline(class_avg, color=INK2, linestyle=(0, (5, 4)), linewidth=1.5)
    ax2.text(class_avg + 1.2, 0.02, f"Class average {class_avg:.1f}", fontsize=10,
             fontweight="bold", color=INK2, transform=ax2.get_xaxis_transform())
    ax2.set_xlim(0, 100)
    ax2.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("Average marks (out of 100)", color=INK2)
    tidy(ax2, "Student averages (ranked)")
    ax2.tick_params(axis="y", colors=INK2)
    for name in (ranked.index[0], ranked.index[-1]):
        ax2.annotate(f"{ranked.loc[name, 'average']:.1f}",
                     (ranked.loc[name, "average"], name), textcoords="offset points",
                     xytext=(6, 0), va="center", fontsize=10, fontweight="bold", color=INK)

    fig.tight_layout()
    path = OUT / "performance_charts.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    return path


# --------------------------------------------------------------------------
# 5. MAIN
# --------------------------------------------------------------------------
def main():
    con = build_database()
    df = load_marks(con)
    con.close()

    absent, students, subjects = analyse(df)
    class_avg = round(df["score"].mean(), 1)                # mean() skips NaN

    students.to_csv(OUT / "student_report.csv")
    subjects.to_csv(OUT / "subject_report.csv")
    chart = make_charts(students, subjects, class_avg)

    print("=== STUDENT PERFORMANCE REPORT ===")
    print(f"Students: {len(students)} | Subjects: {len(SUBJECTS)} | Class average: {class_avg}")
    print(f"\nAbsent entries (NULL scores, excluded from averages): {len(absent)}")
    print(absent.to_string(index=False))
    print("\n--- Student ranking ---")
    print(students[["rank", "section", "average", "grade", "best_subject", "weakest_subject"]])
    print("\n--- Subjects (ascending by average) ---")
    print(subjects)
    print("\n--- Section averages ---")
    print(df.groupby("section")["score"].mean().round(1))
    print("\n--- Needs support (average below 60) ---")
    weak = students[students["average"] < 60]
    print(weak[["average", "weakest_subject"]] if len(weak) else "None")
    print(f"\nSaved: {chart.name}, student_report.csv, subject_report.csv, school.db")


if __name__ == "__main__":
    main()
