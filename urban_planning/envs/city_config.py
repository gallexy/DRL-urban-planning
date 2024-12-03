NON_BLOCK_LAND_USE = (
    'outside',
    'feasible',
    'road',
    'boundary'
)

BLOCK_LAND_USE = (
    'residential',
    'residential_h',
    'business',
    'office',
    'green_l',
    'green_s',
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
RESIDENTIAL_H = 5
BUSINESS = 6
OFFICE = 7
GREEN_L = 8
GREEN_S = 9
SCHOOL = 10
HOSPITAL_L = 11
HOSPITAL_S = 12
RECREATION = 13

LAND_USE_ID = (
    OUTSIDE,
    FEASIBLE,
    ROAD,
    BOUNDARY,
    RESIDENTIAL,
    RESIDENTIAL_H,
    BUSINESS,
    OFFICE,
    GREEN_L,
    GREEN_S,
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

INTERSECTION = 14

PUBLIC_SERVICES_ID = (
    BUSINESS,
    OFFICE,
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
    GREEN_S
)
GREEN_AREA_THRESHOLD = 2000

TYPE_COLOR_MAP = {
    'boundary': 'lightgreen',
    'business': 'fuchsia',
    'feasible': 'white',
    'green_l': 'green',
    'green_s': 'lightgreen',
    'hospital_l': 'blue',
    'hospital_s': 'cyan',
    'office': 'gold',
    'outside': 'black',
    'residential': 'yellow',
    'residential_h': 'orange',
    'road': 'red',
    'school': 'darkorange',
    'recreation': 'lavender',
}
