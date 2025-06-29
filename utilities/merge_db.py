import sqlite3
from pathlib import Path


def connect(db_path):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cursor = con.cursor()
    return con, cursor

def merge(cur1, cur2):
    # Get the current max simulation_id in DB1
    cur1.execute("SELECT MAX(id) FROM mclut_simulations")
    max_sim_id_db1 = cur1.fetchone()[0] or 0

    # Fetch all simulations from DB2
    cur2.execute("SELECT * FROM mclut_simulations")
    simulations = cur2.fetchall()

    # Build a remapping dict: old_id_db2 -> new_id_db1
    id_remap = {}
    for i, row in enumerate(simulations, start=1):
        old_id = row["id"]
        new_id = max_sim_id_db1 + i
        id_remap[old_id] = new_id
        # Insert into DB1 with new simulation_id
        columns = [col for col in row.keys()]
        values = [row[col] for col in columns]
        values[columns.index("id")] = new_id
        cur1.execute(
            f"INSERT INTO mclut_simulations ({','.join(columns)}) VALUES ({','.join(['?']*len(columns))})",
            values,
        )

    # Now update and insert fixed_layers
    cur2.execute("SELECT * FROM fixed_layers")
    for row in cur2.fetchall():
        columns = [col for col in row.keys() if col != 'id']
        values = [row[col] for col in columns]
        values[columns.index("simulation_id")] = id_remap[row["simulation_id"]]
        cur1.execute(
            f"INSERT INTO fixed_layers ({','.join(columns)}) VALUES ({','.join(['?']*len(columns))})",
            values,
        )

    # Now update and insert mclut
    cur2.execute("SELECT * FROM mclut")
    for row in cur2.fetchall():
        columns = [col for col in row.keys() if col != "id"]
        values = [row[col] for col in columns]
        values[columns.index("simulation_id")] = id_remap[row["simulation_id"]]
        cur1.execute(
            f"INSERT INTO mclut ({','.join(columns)}) VALUES ({','.join(['?']*len(columns))})",
            values,
        )

def com_close(con):
    # Commit and close
    con.commit()
    con.close()

if __name__ == "__main__":
    db1_path = Path.home() / ".photon_canon/lut_premerge_backup.db"
    db2_path = Path(r"C:\Users\jdivers\Downloads\lut (3).db")
    con1, cur1 = connect(db1_path)
    con2, cur2 = connect(db2_path)
    try:
        merge(cur1, cur2)
    finally:
        com_close(con1)
        com_close(con2)