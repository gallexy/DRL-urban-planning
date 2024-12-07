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
    'recreation'
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
)

NUM_TYPES = len(LAND_USE_ID)

LAND_USE_ID_MAP = dict(
    zip(LAND_USE, LAND_USE_ID))

LAND_USE_ID_MAP_INV = dict(
    zip(LAND_USE_ID, LAND_USE))

INTERSECTION = 13

RESIDENTIAL_ID = (
    RESIDENTIAL,
    RESIDENTIAL_H
)

PUBLIC_SERVICES_ID = (
    BUSINESS,
    BUSINESS_H,
    SCHOOL,
    (HOSPITAL_L, HOSPITAL_S),
    RECREATION
)

PUBLIC_SERVICES = (
    'shopping',
    'working',
    'education',
    'medical care',
    'entertainment'
)

GREEN_ID = (
    GREEN_L,
    RECREATION
)
GREEN_AREA_THRESHOLD = 2000

TYPE_COLOR_MAP = {
    'boundary': 'lightgreen',
    'business': 'magenta',
    'feasible': 'white',
    'green_l': 'green',
    'hospital_l': 'red',
    'hospital_s': 'darkred',
    'business_h': 'deeppink',
    'outside': 'black',
    'residential': 'yellow',
    'residential_h': 'gold',
    'road': 'blue',
    'school': 'darkorange',
    'recreation': 'lightgreen',
}
