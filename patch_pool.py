"""
Post-process player_pool.json to add auction players that have no 2025 IPL data.
These get floor pricing (same as csv_name=None players in build_pool).
Run after: python main.py build-pool
"""
import json

FLOOR_PTS = {"BAT": 40.0, "AR": 40.0, "BOWL": 38.0, "WK": 35.0}

pool = json.load(open('player_pool.json'))
master = json.load(open('player_master.json'))

pool_csv_names = {p['player_name'] for p in pool}
m_by_name = {m['name']: m for m in master}

# 156 auction players — add any still missing
from update_master import NEW_DATA

NAME_ALIASES = {'Allah Mohammad Ghazanfar': 'Allah Ghazanfar'}

added = []
for full_name, role_raw, team_full, set_num in NEW_DATA:
    master_name = NAME_ALIASES.get(full_name, full_name)
    m = m_by_name.get(master_name)
    if m is None:
        print(f'WARNING: {full_name} not in master, skipping')
        continue

    csv_name = m.get('csv_name')
    role = m['role']
    team = m['ipl_team']

    # Check if already in pool
    in_pool = (csv_name and csv_name in pool_csv_names) or (master_name in pool_csv_names)
    if in_pool:
        continue

    # Add as floor player
    pts = FLOOR_PTS.get(role, 38.0)
    entry = {
        'player_name': csv_name if csv_name else master_name,
        'role': role,
        'projected_points': pts,
        'std_dev': 15.0,
        'vorp': 0.0,
        'tier': 3,
        'base_price': 1.0,
        'is_marquee': False,
        'matches': 0,
        'ipl_team': team,
        'auction_set': set_num,
    }
    pool.append(entry)
    pool_csv_names.add(entry['player_name'])
    added.append(f"  {master_name} ({role}, {team}, set={set_num}) pts={pts}")

if added:
    print(f"Added {len(added)} floor-priced players:")
    for a in added:
        print(a)
else:
    print("All 156 auction players already present.")

# Verify all 156 are now in pool
missing = []
for full_name, role_raw, team_full, set_num in NEW_DATA:
    master_name = NAME_ALIASES.get(full_name, full_name)
    m = m_by_name.get(master_name)
    csv_name = m.get('csv_name') if m else None
    in_pool = (csv_name and csv_name in pool_csv_names) or (master_name in pool_csv_names)
    if not in_pool:
        missing.append(full_name)

if missing:
    print(f"STILL MISSING: {missing}")
else:
    print(f"\nAll 156 auction players confirmed in pool.")
    print(f"Total pool size: {len(pool)}")

with open('player_pool.json', 'w') as f:
    json.dump(pool, f, indent=2)
print("Saved player_pool.json")
