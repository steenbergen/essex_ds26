# Read the Communities and Crime data (UCI ML Repository)
#
# The .data file is comma-separated with no header row; missing values are
# coded as "?". Column names and order come from the accompanying
# communities.names file (128 attributes: 5 non-predictive, 122 predictive,
# 1 goal variable = ViolentCrimesPerPop).

library(tidyverse)

data_dir <- "/Users/prior/Teaching/essex_ds26/Exercises/communities+and+crime"

col_names <- c(
  # Non-predictive identifiers
  "state", "county", "community", "communityname", "fold",
  # Socio-economic and demographic (1990 Census)
  "population", "householdsize", "racepctblack", "racePctWhite",
  "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29",
  "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome",
  "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst",
  "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap",
  "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap",
  "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad",
  "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu",
  "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf",
  "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv",
  "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par",
  "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg",
  "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5",
  "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
  "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly",
  "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup",
  "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous",
  "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR",
  "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded",
  "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone",
  "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart",
  "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc",
  "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters",
  "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85",
  "PctSameCity85", "PctSameState85",
  # Law enforcement (1990 LEMAS survey; missing for most communities)
  "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps",
  "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop",
  "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol",
  "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian",
  "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz",
  "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans",
  "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr",
  "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop",
  # Goal variable
  "ViolentCrimesPerPop"
)

communities <- read_csv(
  file.path(data_dir, "communities.data"),
  col_names = col_names,
  na = "?",
  col_types = cols(
    communityname = col_character(),
    .default = col_double()
  )
) |>
  mutate(state = factor(state))  # state is a numeric code; treat as nominal

glimpse(communities)
