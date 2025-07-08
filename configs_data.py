IMPORT_PATH = "data"
RESULT_PATH = "data_automl_results"

DATA_CONFIG_AUTO_MPG = {
    "dataset": "AutoMpg",
    "target": "mpg",
    "cats_list": [
        "origin",
    ],
    "drop_list": [
        "car_name",
    ],
}

DATA_CONFIG_COMMUNITIES_CRIME = {
    "dataset": "CommunitiesCrime",
    "target": "ViolentCrimesPerPop",
    "cats_list": [],
    "drop_list": [
        "state",
        "country",
        "community",
        "communityname",
        "fold",
        "LemasSwornFT",
        "LemasSwFTPerPop",
        "LemasSwFTFieldOps",
        "LemasSwFTFieldPerPop",
        "LemasTotalReq",
        "LemasTotReqPerPop",
        "PolicReqPerOffic",
        "PolicPerPop",
        "RacialMatchCommPol",
        "PctPolicWhite",
        "PctPolicBlack",
        "PctPolicHisp",
        "PctPolicAsian",
        "PctPolicMinor",
        "OfficAssgnDrugUnits",
        "NumKindsDrugsSeiz",
        "PolicAveOTWorked",
        "PolicCars",
        "PolicOperBudg",
        "LemasPctPolicOnPatr",
        "LemasGangUnitDeploy",
        "PolicBudgPerPop",
    ],
}

DATA_CONFIG_MIAMI_HOUSING = {
    "dataset": "MiamiHousing",
    "target": "sale_prc",
    "cats_list": [
        "avno60plus",
        "month_sold",
        "structure_quality",
    ],
    "drop_list": [
        "parcelno",
    ],
}

DATA_CONFIG_BIKE_SHARING = {
    "dataset": "BikeSharing",
    "target": "count",
    "cats_list": [
        "season",
        "year",
        "month",
        "day",
        "hour",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ],
    "drop_list": [
        "instant",
        "casual",
        "registered",
    ],
}

DATASETS_CONFIG = [
    DATA_CONFIG_AUTO_MPG,
    DATA_CONFIG_COMMUNITIES_CRIME,
    DATA_CONFIG_MIAMI_HOUSING,
    DATA_CONFIG_BIKE_SHARING,
]
