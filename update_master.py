"""
Update player_master.json with 156-player auction set data.
Updates: auction_set, role, ipl_team for existing players.
Adds: new players not currently in master.
Does NOT touch projected_points (those come from CSV scoring pipeline).
"""
import json

TEAM_MAP = {
    'Royal Challengers Bangalore': 'RCB',
    'Punjab Kings': 'PBKS',
    'Gujrat Titans': 'GT',
    'Gujarat Titans': 'GT',
    'Rajasthan Royals': 'RR',
    'Sunrisers Hyderabad': 'SRH',
    'Kolkata Knight Riders': 'KKR',
    'Chennai Super Kings': 'CSK',
    'Mumbai Indians': 'MI',
    'Delhi Capitals': 'DC',
    'Lucknow Super Giants': 'LSG',
}

ROLE_MAP = {
    'BATSMAN': 'BAT',
    'BOWLER': 'BOWL',
    'ALL_ROUNDER': 'AR',
    'WICKETKEEPER': 'WK',
}

# All 156 players from the new auction data:
# (full_name, role_raw, ipl_team_full, set_number)
NEW_DATA = [
    # Set 1
    ('Virat Kohli',         'BATSMAN',      'Royal Challengers Bangalore', 1),
    ('Shreyas Iyer',        'BATSMAN',      'Punjab Kings',                1),
    ('Shubman Gill',        'BATSMAN',      'Gujrat Titans',               1),
    ('Prasidh Krishna',     'BOWLER',       'Gujrat Titans',               1),
    ('Yashasvi Jaiswal',    'BATSMAN',      'Rajasthan Royals',            1),
    ('Rashid Khan',         'ALL_ROUNDER',  'Gujrat Titans',               1),
    ('Ishan Kishan',        'WICKETKEEPER', 'Sunrisers Hyderabad',         1),
    ('Rohit Sharma',        'BATSMAN',      'Mumbai Indians',              1),
    ('Rajat Patidar',       'BATSMAN',      'Royal Challengers Bangalore', 1),
    ('Jos Buttler',         'WICKETKEEPER', 'Gujrat Titans',               1),
    ('Heinrich Klaasen',    'WICKETKEEPER', 'Sunrisers Hyderabad',         1),
    ('Jofra Archer',        'BOWLER',       'Rajasthan Royals',            1),
    # Set 2
    ('Vaibhav Suryavanshi', 'BATSMAN',      'Rajasthan Royals',            2),
    ('Sunil Narine',        'ALL_ROUNDER',  'Kolkata Knight Riders',       2),
    ('Sanju Samson',        'WICKETKEEPER', 'Chennai Super Kings',         2),
    ('Jamie Overton',       'ALL_ROUNDER',  'Chennai Super Kings',         2),
    ('Sai Sudharsan',       'BATSMAN',      'Gujrat Titans',               2),
    ('Abhishek Sharma',     'ALL_ROUNDER',  'Sunrisers Hyderabad',         2),
    ('Arshdeep Singh',      'BOWLER',       'Punjab Kings',                2),
    ('Phil Salt',           'WICKETKEEPER', 'Royal Challengers Bangalore', 2),
    ('Quinton de Kock',     'WICKETKEEPER', 'Mumbai Indians',              2),
    ('Ravindra Jadeja',     'ALL_ROUNDER',  'Rajasthan Royals',            2),
    ('Nitish Kumar Reddy',  'ALL_ROUNDER',  'Sunrisers Hyderabad',         2),
    ('Tim David',           'BATSMAN',      'Royal Challengers Bangalore', 2),
    # Set 3
    ('Travis Head',         'BATSMAN',      'Sunrisers Hyderabad',         3),
    ('Ravi Bishnoi',        'BOWLER',       'Rajasthan Royals',            3),
    ('Bhuvneshwar Kumar',   'BOWLER',       'Royal Challengers Bangalore', 3),
    ('Prabhsimran Singh',   'WICKETKEEPER', 'Punjab Kings',                3),
    ('Hardik Pandya',       'ALL_ROUNDER',  'Mumbai Indians',              3),
    ('Ajinkya Rahane',      'BATSMAN',      'Kolkata Knight Riders',       3),
    ('Mohammed Shami',      'BOWLER',       'Lucknow Super Giants',        3),
    ('Suryakumar Yadav',    'BATSMAN',      'Mumbai Indians',              3),
    ('Pathum Nissanka',     'BATSMAN',      'Delhi Capitals',              3),
    ('Rishabh Pant',        'WICKETKEEPER', 'Lucknow Super Giants',        3),
    ('Dhruv Jurel',         'WICKETKEEPER', 'Rajasthan Royals',            3),
    ('Lungi Ngidi',         'BOWLER',       'Delhi Capitals',              3),
    # Set 4
    ('Josh Hazlewood',      'BOWLER',       'Royal Challengers Bangalore', 4),
    ('KL Rahul',            'WICKETKEEPER', 'Delhi Capitals',              4),
    ('Axar Patel',          'ALL_ROUNDER',  'Delhi Capitals',              4),
    ('Washington Sundar',   'ALL_ROUNDER',  'Gujrat Titans',               4),
    ('Anshul Kamboj',       'ALL_ROUNDER',  'Chennai Super Kings',         4),
    ('Angkrish Raghuvanshi','BATSMAN',      'Kolkata Knight Riders',       4),
    ('Mitchell Marsh',      'ALL_ROUNDER',  'Lucknow Super Giants',        4),
    ('Marco Jansen',        'ALL_ROUNDER',  'Punjab Kings',                4),
    ('Shivam Dube',         'ALL_ROUNDER',  'Chennai Super Kings',         4),
    ('Tristan Stubbs',      'BATSMAN',      'Delhi Capitals',              4),
    ('Ayush Mhatre',        'BATSMAN',      'Chennai Super Kings',         4),
    ('Aiden Markram',       'ALL_ROUNDER',  'Lucknow Super Giants',        4),
    # Set 5
    ('Sameer Rizvi',        'BATSMAN',      'Delhi Capitals',              5),
    ('Sarfaraz Khan',       'BATSMAN',      'Chennai Super Kings',         5),
    ('Devdutt Padikkal',    'BATSMAN',      'Royal Challengers Bangalore', 5),
    ('Avesh Khan',          'BOWLER',       'Lucknow Super Giants',        5),
    ('Rasikh Salam Dar',    'BOWLER',       'Royal Challengers Bangalore', 5),
    ('Prince Yadav',        'BOWLER',       'Lucknow Super Giants',        5),
    ('Ryan Rickelton',      'WICKETKEEPER', 'Mumbai Indians',              5),
    ('Priyansh Arya',       'BATSMAN',      'Punjab Kings',                5),
    ('Shimron Hetmyer',     'BATSMAN',      'Rajasthan Royals',            5),
    ('Jacob Duffy',         'BOWLER',       'Royal Challengers Bangalore', 5),
    ('Cooper Connolly',     'ALL_ROUNDER',  'Punjab Kings',                5),
    ('Noor Ahmad',          'BOWLER',       'Chennai Super Kings',         5),
    # Set 6
    ('Mohammed Siraj',      'BOWLER',       'Gujrat Titans',               6),
    ('Yuzvendra Chahal',    'BOWLER',       'Punjab Kings',                6),
    ('Shardul Thakur',      'ALL_ROUNDER',  'Mumbai Indians',              6),
    ('Krunal Pandya',       'ALL_ROUNDER',  'Royal Challengers Bangalore', 6),
    ('Ruturaj Gaikwad',     'BATSMAN',      'Chennai Super Kings',         6),
    ('Kuldeep Yadav',       'BOWLER',       'Delhi Capitals',              6),
    ('Cameron Green',       'ALL_ROUNDER',  'Kolkata Knight Riders',       6),
    ('Riyan Parag',         'ALL_ROUNDER',  'Rajasthan Royals',            6),
    ('Dewald Brevis',       'BATSMAN',      'Chennai Super Kings',         6),
    ('Rinku Singh',         'BATSMAN',      'Kolkata Knight Riders',       6),
    ('David Miller',        'BATSMAN',      'Delhi Capitals',              6),
    ('Mitchell Santner',    'ALL_ROUNDER',  'Mumbai Indians',              6),
    # Set 7
    ('Suyash Sharma',       'BOWLER',       'Royal Challengers Bangalore', 7),
    ('Praful Hinge',        'BOWLER',       'Sunrisers Hyderabad',         7),
    ('Sandeep Sharma',      'BOWLER',       'Rajasthan Royals',            7),
    ('Tushar Deshpande',    'BOWLER',       'Rajasthan Royals',            7),
    ('Romario Shepherd',    'ALL_ROUNDER',  'Royal Challengers Bangalore', 7),
    ('Tilak Varma',         'BATSMAN',      'Mumbai Indians',              7),
    ('Mukul Choudhary',     'BOWLER',       'Lucknow Super Giants',        7),
    ('Marcus Stoinis',      'ALL_ROUNDER',  'Punjab Kings',                7),
    ('Naman Dhir',          'ALL_ROUNDER',  'Mumbai Indians',              7),
    ('Kartik Tyagi',        'BOWLER',       'Kolkata Knight Riders',       7),
    ('Kagiso Rabada',       'BOWLER',       'Gujrat Titans',               7),
    ('Eshan Malinga',       'BOWLER',       'Sunrisers Hyderabad',         7),
    # Set 8
    ('Jaydev Unadkat',      'BOWLER',       'Sunrisers Hyderabad',         8),
    ('Harsh Dubey',         'BOWLER',       'Sunrisers Hyderabad',         8),
    ('Blessing Muzarabani', 'BOWLER',       'Kolkata Knight Riders',       8),
    ('Vijaykumar Vyshak',   'BOWLER',       'Punjab Kings',                8),
    ('Jasprit Bumrah',      'BOWLER',       'Mumbai Indians',              8),
    ('Rovman Powell',       'BATSMAN',      'Kolkata Knight Riders',       8),
    ('Ayush Badoni',        'BATSMAN',      'Lucknow Super Giants',        8),
    ('Glenn Phillips',      'ALL_ROUNDER',  'Gujrat Titans',               8),
    ('Anukul Roy',          'ALL_ROUNDER',  'Kolkata Knight Riders',       8),
    ('Trent Boult',         'BOWLER',       'Mumbai Indians',              8),
    ('Mukesh Kumar',        'BOWLER',       'Delhi Capitals',              8),
    ('Sherfane Rutherford', 'ALL_ROUNDER',  'Mumbai Indians',              8),
    # Set 9
    ('Shivang Kumar',       'BOWLER',       'Sunrisers Hyderabad',         9),
    ('Brijesh Sharma',      'BOWLER',       'Rajasthan Royals',            9),
    ('Abdul Samad',         'ALL_ROUNDER',  'Lucknow Super Giants',        9),
    ('T Natarajan',         'BOWLER',       'Delhi Capitals',              9),
    ('Finn Allen',          'WICKETKEEPER', 'Kolkata Knight Riders',       9),
    ('Nandre Burger',       'BOWLER',       'Rajasthan Royals',            9),
    ('Shashank Singh',      'ALL_ROUNDER',  'Punjab Kings',                9),
    ('Nicholas Pooran',     'WICKETKEEPER', 'Lucknow Super Giants',        9),
    ('Rahul Tewatia',       'ALL_ROUNDER',  'Gujrat Titans',               9),
    ('Mohsin Khan',         'BOWLER',       'Lucknow Super Giants',        9),
    ('Khaleel Ahmed',       'BOWLER',       'Chennai Super Kings',         9),
    ('Aniket Verma',        'BATSMAN',      'Sunrisers Hyderabad',         9),
    # Set 10
    ('Nitish Rana',         'BATSMAN',      'Delhi Capitals',              10),
    ('Deepak Chahar',       'ALL_ROUNDER',  'Mumbai Indians',              10),
    ('Vaibhav Arora',       'BOWLER',       'Kolkata Knight Riders',       10),
    ('Donovan Ferreira',    'BATSMAN',      'Rajasthan Royals',            10),
    ('Corbin Bosch',        'ALL_ROUNDER',  'Mumbai Indians',              10),
    ('Matheesha Pathirana', 'BOWLER',       'Kolkata Knight Riders',       10),
    ('Mitchell Starc',      'BOWLER',       'Delhi Capitals',              10),
    ('Akeal Hosein',        'BOWLER',       'Chennai Super Kings',         10),
    ('Manish Pandey',       'BATSMAN',      'Kolkata Knight Riders',       10),
    ('Venkatesh Iyer',      'ALL_ROUNDER',  'Royal Challengers Bangalore', 10),
    ('Xavier Bartlett',     'BOWLER',       'Punjab Kings',                10),
    ('Varun Chakravarthy',  'BOWLER',       'Kolkata Knight Riders',       10),
    # Set 11
    ('Jitesh Sharma',       'WICKETKEEPER', 'Royal Challengers Bangalore', 11),
    ('Ashutosh Sharma',     'ALL_ROUNDER',  'Delhi Capitals',              11),
    ('Allah Mohammad Ghazanfar', 'BOWLER', 'Mumbai Indians',              11),  # → Allah Ghazanfar in master
    ('Ramandeep Singh',     'ALL_ROUNDER',  'Kolkata Knight Riders',       11),
    ('Prashant Veer',       'ALL_ROUNDER',  'Chennai Super Kings',         11),
    ('M Siddharth',         'BOWLER',       'Lucknow Super Giants',        11),
    ('Nehal Wadhera',       'BATSMAN',      'Punjab Kings',                11),
    ('Digvesh Singh Rathi', 'BOWLER',       'Lucknow Super Giants',        11),
    ('Azmatullah Omarzai',  'ALL_ROUNDER',  'Punjab Kings',                11),
    ('Kyle Jamieson',       'ALL_ROUNDER',  'Delhi Capitals',              11),
    ('Rachin Ravindra',     'ALL_ROUNDER',  'Kolkata Knight Riders',       11),
    ('Will Jacks',          'ALL_ROUNDER',  'Mumbai Indians',              11),
    # Set 12
    ('Ashok Sharma',        'BOWLER',       'Gujrat Titans',               12),
    ('Rahul Chahar',        'BOWLER',       'Chennai Super Kings',         12),
    ('Shahrukh Khan',       'BATSMAN',      'Gujrat Titans',               12),
    ('Harshal Patel',       'BOWLER',       'Sunrisers Hyderabad',         12),
    ('Abhinandan Singh',    'BOWLER',       'Royal Challengers Bangalore', 12),
    ('Liam Livingstone',    'ALL_ROUNDER',  'Sunrisers Hyderabad',         12),
    ('Kumar Kushagra',      'WICKETKEEPER', 'Gujrat Titans',               12),
    ('Anrich Nortje',       'BOWLER',       'Lucknow Super Giants',        12),
    ('Salil Arora',         'WICKETKEEPER', 'Sunrisers Hyderabad',         12),
    ('Vipraj Nigam',        'BOWLER',       'Delhi Capitals',              12),
    ('Gurjapneet Singh',    'BOWLER',       'Chennai Super Kings',         12),
    ('Matt Henry',          'BOWLER',       'Chennai Super Kings',         12),
    # Set 13
    ('Jacob Bethell',       'ALL_ROUNDER',  'Royal Challengers Bangalore', 13),
    ('David Payne',         'BOWLER',       'Sunrisers Hyderabad',         13),
    ('Shahbaz Ahmed',       'ALL_ROUNDER',  'Lucknow Super Giants',        13),
    ('Musheer Khan',        'BATSMAN',      'Punjab Kings',                13),
    ('Kartik Sharma',       'WICKETKEEPER', 'Chennai Super Kings',         13),
    ('Harpreet Brar',       'ALL_ROUNDER',  'Punjab Kings',                13),
    ('Himmat Singh',        'BATSMAN',      'Lucknow Super Giants',        13),
    ('George Linde',        'ALL_ROUNDER',  'Lucknow Super Giants',        13),
    ('Auqib Nabi Dar',      'ALL_ROUNDER',  'Delhi Capitals',              13),
    ('Matthew Short',       'ALL_ROUNDER',  'Chennai Super Kings',         13),
    ('Mayank Yadav',        'BOWLER',       'Lucknow Super Giants',        13),
    ('Mayank Markande',     'BOWLER',       'Mumbai Indians',              13),
    # Set 14
    ('Lhuan-Dre Pretorius', 'ALL_ROUNDER',  'Rajasthan Royals',            14),
    ('Navdeep Saini',       'BOWLER',       'Kolkata Knight Riders',       14),
]

# csv_name for new players that have CSV data (exact match confirmed)
NEW_CSV_NAMES = {
    'Prince Yadav':      'Prince Yadav',
    'Shivang Kumar':     'Shivang Kumar',
    'Brijesh Sharma':    'Brijesh Sharma',
    'M Siddharth':       'M Siddharth',
    'Ashok Sharma':      'Ashok Sharma',
    'Abhinandan Singh':  'Abhinandan Singh',
    'Gurjapneet Singh':  'Gurjapneet Singh',
    'Himmat Singh':      'Himmat Singh',
    'Navdeep Saini':     'Navdeep Saini',
}

# Overseas flag for truly new players
OVERSEAS_FLAG = {
    'Praful Hinge': False,
    'Mukul Choudhary': False,
    'Digvesh Singh Rathi': False,
    'Salil Arora': False,
    'David Payne': True,
    'George Linde': True,
    'Auqib Nabi Dar': False,
    'Matthew Short': True,
    'Lhuan-Dre Pretorius': True,
    'Shivang Kumar': False,
    'Brijesh Sharma': False,
    'Prince Yadav': False,
    'M Siddharth': False,
    'Ashok Sharma': False,
    'Abhinandan Singh': False,
    'Gurjapneet Singh': False,
    'Himmat Singh': False,
    'Navdeep Saini': False,
}

# Name aliases: new full name → master name (for existing entries with different names)
NAME_ALIASES = {
    'Allah Mohammad Ghazanfar': 'Allah Ghazanfar',
    'Rasikh Salam Dar': 'Rasikh Salam Dar',  # master may have 'Rasikh Salam'
    'Prabhsimran Singh': 'Prabhsimran Singh',  # pool has 'P Simran Singh'
}

# ── Load master ───────────────────────────────────────────────────────────────
master = json.load(open('player_master.json'))
m_by_name = {m['name']: m for m in master}
max_id = max(m['id'] for m in master)

new_players_added = []
updated_players = []

for full_name, role_raw, team_full, set_num in NEW_DATA:
    role = ROLE_MAP[role_raw]
    team = TEAM_MAP[team_full]

    # Resolve name alias
    master_name = NAME_ALIASES.get(full_name, full_name)
    # Also check 'Rasikh Salam' variant
    if master_name not in m_by_name and full_name == 'Rasikh Salam Dar':
        master_name = 'Rasikh Salam Dar'  # try as-is first, fallback below

    entry = m_by_name.get(master_name)

    if entry:
        # Update existing entry
        old = (entry['auction_set'], entry['role'], entry['ipl_team'])
        entry['auction_set'] = set_num
        entry['role'] = role
        entry['ipl_team'] = team
        if old != (set_num, role, team):
            updated_players.append(f"  {master_name}: set {old[0]}→{set_num}, role {old[1]}→{role}, team {old[2]}→{team}")
    else:
        # New player — add to master
        max_id += 1
        csv_name = NEW_CSV_NAMES.get(full_name)
        is_overseas = OVERSEAS_FLAG.get(full_name, False)
        new_entry = {
            'id': max_id,
            'name': full_name,
            'ipl_team': team,
            'role': role,
            'is_overseas': is_overseas,
            'auction_set': set_num,
            'nationality': 'Unknown',
            'csv_name': csv_name,
            'has_2025_data': csv_name is not None,
        }
        master.append(new_entry)
        m_by_name[full_name] = new_entry
        new_players_added.append(f"  {full_name}: set={set_num}, role={role}, team={team}, csv={csv_name}")

print(f"Updated {len(updated_players)} existing players:")
for s in updated_players:
    print(s)
print()
print(f"Added {len(new_players_added)} new players:")
for s in new_players_added:
    print(s)

# Save
with open('player_master.json', 'w') as f:
    json.dump(master, f, indent=2)
print(f"\nSaved player_master.json with {len(master)} total entries.")
