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
    'expressway'
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
)

NUM_TYPES = len(LAND_USE_ID)

LAND_USE_ID_MAP = dict(
    zip(LAND_USE, LAND_USE_ID))

LAND_USE_ID_MAP_INV = dict(
    zip(LAND_USE_ID, LAND_USE))

INTERSECTION = 15

RESIDENTIAL_ID = (
    RESIDENTIAL,
)

PUBLIC_SERVICES_ID = (
    BUSINESS,
    BUSINESS_H,
    SCHOOL,
    HOSPITAL_L, 
    HOSPITAL_S,
    RECREATION
)

PUBLIC_SERVICES = (
    'shopping',
    'working',
    'education',
    'medical care',
    'hospital_s'
    'entertainment'
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
    'road': '#a3a3a3',      # grey
    'school': '#ff85c9',   # light pink
    'recreation': '#acffcf',   # light blue
    'river': '#4682B4',   # blue
    'expressway': 'gray',   # dark blue
}