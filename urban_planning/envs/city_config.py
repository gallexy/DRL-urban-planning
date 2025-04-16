NON_BLOCK_LAND_USE = (
    'outside',
    'feasible',
    'road',
    'boundary'
)

BLOCK_LAND_USE = (
    'residential',
    'business',
    'business_h',
    'green_l',
    'residential_h',
    'school',
    'hospital_l',
    'hospital_s',
    'recreation',
    'river',
    'expressway',
    'a2',
    'a3',
    'a5',
    'rb',
    's1',
    'industry' 
)

LAND_USE = (
    NON_BLOCK_LAND_USE + BLOCK_LAND_USE)

OUTSIDE = 0
FEASIBLE = 1
ROAD = 2
BOUNDARY = 3
RESIDENTIAL = 4
BUSINESS = 5
BUSINESS_H = 6
GREEN_L = 7
RESIDENTIAL_H = 8
SCHOOL = 9
HOSPITAL_L = 10
HOSPITAL_S = 11
RECREATION = 12
RIVER = 13
EXPRESSWAY = 14
A2 = 15
A3 = 16
A5 = 17
RB = 18
S1 = 19
INDUSTRY = 20


LAND_USE_ID = (
    OUTSIDE,
    FEASIBLE,
    ROAD,
    BOUNDARY,
    RESIDENTIAL,
    BUSINESS,
    BUSINESS_H,
    GREEN_L,
    RESIDENTIAL_H,
    SCHOOL,
    HOSPITAL_L,
    HOSPITAL_S,
    RECREATION,
    RIVER,
    EXPRESSWAY,
    A2,
    A3,
    A5,
    RB,
    S1,
    INDUSTRY,
)

BLOCK_LAND_TYPE = (
    RESIDENTIAL,
    BUSINESS,
    BUSINESS_H,
    GREEN_L,
    RESIDENTIAL_H,
    SCHOOL,
    HOSPITAL_L,
    HOSPITAL_S,
    RECREATION,
    RIVER,
    EXPRESSWAY,
    A2,
    A3,
    A5,
    RB,
    S1,
    INDUSTRY,
)

NUM_TYPES = len(LAND_USE_ID)

LAND_USE_ID_MAP = dict(
    zip(LAND_USE, LAND_USE_ID))

LAND_USE_ID_MAP_INV = dict(
    zip(LAND_USE_ID, LAND_USE))

INTERSECTION = 21

RESIDENTIAL_ID = (
    RESIDENTIAL,
    RESIDENTIAL_H,
    RB
)

PUBLIC_SERVICES_ID = (
    BUSINESS,
    BUSINESS_H,
    SCHOOL,
    HOSPITAL_L, 
    HOSPITAL_S,
    RECREATION,
    A2,
    A3,
    A5,
    S1
)

PUBLIC_SERVICES = (
    'shopping',
    'working',
    'education',
    'medical care',
    'hospital_s',
    'entertainment',
)

GREEN_ID = (
    GREEN_L,
    RECREATION
)
GREEN_AREA_THRESHOLD = 5000

TYPE_COLOR_MAP = {
    'boundary': 'lightgreen',
    'business': '#e6005c',   # pink
    'feasible': 'white',
    'green_l': '#00ff00',
    'hospital_l': '#ff7f7e',    # light red
    'hospital_s': '#FF7F00',
    'business_h': '#ff0000',    # red
    'outside': 'black',
    'residential': '#ffff2d',  # yellow
    'residential_h': '#8B4513',   # brown
    'road': '#aaaaaa',      
    'school': '#ff85c9',   # light pink
    'recreation': '#acffcf',   # light blue
    'river': '#4682B4',   # blue
    'expressway': 'gray',   # dark blue
    'a2': '#FF7F00',   # light green
    'a3': '#FF85C9',   # light blue
    'a5': '#BB7F7E',   # light red
    's1': '#303030',   # dark gray
    'rb': '#FFA500',   # orange red
    'industry': '#8B4513',   # brown
}