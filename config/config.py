#!/usr/bin/env python

############ DATA INPUT / OUTPUT ############
FILEPATH = {
    "folder_path": "data/2021VAERSData",
    "merged_filepath": "data/processed/merged.csv",
    "clean_filepath": "data/processed/vaers_clean.csv",
    "model_filepath": "data/processed/vaers_model.csv",
}

############ CUSTOM INGESTOR ############

SYMP_COLS_DICT = {
    'SYMP' : ['SYMPTOM1', 'SYMPTOM2', 'SYMPTOM3', 'SYMPTOM4', 'SYMPTOM5']
}

VAX_CSV_COLS = ['VAERS_ID', 'VAX_MANU']

MAX_DATE = '2021-04-01'

######################################
# for custom (modelling) preprocessing
######################################

CHECK_ERROR = {
    "AGE_YRS": ["<", 12],
    "VAX_DATE": ["<", "2020-12-11"],
    "ONSET_DATE": ["<", "2020-12-11"]
}

DROP_NA = ["AGE_YRS", "SYMPTOM_TEXT", "VAX_DATE", "ONSET_DATE"]

IMPUTE_NA = {
    'DIED': 'N',
    'L_THREAT': 'N',
    'HOSPITAL': 'N',
    'X_STAY': 'N',
    'DISABLE': 'N',
    'ER_ED_VISIT': 'N',
}

DATA_DICT = {
    'VAERS_ID': 'category',
    'RECVDATE': 'datetime64',
    'STATE': 'category',
    'AGE_YRS': 'float64',
    'CAGE_YR': 'float64',
    'CAGE_MO': 'float64',
    'SEX': 'category',
    'RPT_DATE': 'datetime64',
    'SYMPTOM_TEXT': 'string',
    'DIED': 'category',
    'DATEDIED': 'datetime64',
    'L_THREAT': 'category',
    'ER_VISIT': 'category',
    'HOSPITAL': 'category',
    'HOSPDAYS': 'float64',
    'X_STAY': 'category',
    'DISABLE': 'category',
    'RECOVD': 'category',
    'VAX_DATE': 'datetime64',
    'ONSET_DATE': 'datetime64',
    'NUMDAYS': 'float64',
    'LAB_DATA': 'string',
    'V_ADMINBY': 'category',
    'V_FUNDBY': 'category',
    'OTHER_MEDS': 'string',
    'CUR_ILL': 'string',
    'HISTORY': 'string',
    'PRIOR_VAX': 'category',
    'SPLTTYPE': 'string',
    'FORM_VERS': 'category',
    'TODAYS_DATE': 'datetime64',
    'BIRTH_DEFECT': 'category',
    'OFC_VISIT': 'category',
    'ER_ED_VISIT': 'category',
    'ALLERGIES': 'string',
    'VAX_MANU': 'category',
    'SYMP': 'string'
}

# calculate difference in days using list element 0 - list element 1
DATE_DIFF = {
    
    'NUMDAYS_CAL': ['ONSET_DATE', 'VAX_DATE']
}

CLIP_UPPER = {
    'NUMDAYS_CAL': 10,
    'COUNT_SYMPT': 20
}

# for simple strings - REPLACE_NONE_SYNONYMS is for the long list of none synonyms
REPLACE_VALUES = {
    
    'SEX': {
        'U': 0,
        'F': 0,
        'M': 1
    },
    
    'VAX_MANU': {
        'MODERNA' : 1,
        'PFIZER\BIONTECH' : 0
    }
}

REPLACE_NONE_SYNONYMS = {
    
    'HISTORY': ['no', 'none','none reported','none known','denies','none.', 
                 'denied at the time of the visit', 'deny', "nan",
                 'comments: list of non-encoded patient relevant history: patient other relevant history 1: none', 
                 'medical history/concurrent conditions: no adverse event (no reported medical history)',
                 'medical history/concurrent conditions: no adverse event (no reported medical history.)',
                 "none as of yet", "none at this time","negative","none yet", "none taken","nothing special.", 
                 "no acute illness", "-", "neg","non e" ,"nonw", "nonr", "non", "none identified", 
                 "no illness", "no illnesses mentioned", "none  known", "no other illnesses", 
                 "comments: list of non-encoded patient relevant history: patient other relevant history 1: no adverse event, continue: [unk], comment: no reported medical history.",
                 "none to report", "none to my knowledge.","medical history/concurrent conditions: no adverse event",
                 "no.", "unk", "n/a", "n.a", "ukn", "uk", "none noted", "0", 0, "not avl to this reporter","unkown", "c",
                 "none listed on pre checklist", "none that I know of","non per patient report"],
        
    
    'ALLERGIES': ["nan", "nda", "nkda", "none", "none know", "none known", 
                "not known", "none.", "no", "unknown", "none report", 
               "none reported", "na", "no.", "unk", "n/a", "n.a", "ukn", 
                "uk", "none noted", "0", 0, "unkown", "nka", "c", "-", "neg", 
                "non e" , "nonw", "nonr", "non", "none identified", "no illness", 
               "no known illness", "none lister", "not applicable", 
               "no illnesses at time of vaccine", "no known", "no e",
               "none stated/noted", "none documented", "nothing", 
               "none known.", "no illnesses", "n/an", "not", "unknonwn", 
              "denies", "no illnesses mentioned", "none  known", "?",
              "no other illnesses", "none to report", "none to my knowledge.",
              "unknown.", "nkda", "not avl to this reporter", "none  reported", 
              "nan","comments: list of non-encoded patient relevant history: patient other relevant history 1: none", 
              "medical history/concurrent conditions: no adverse event (no reported medical history)",
               "medical history/concurrent conditions: no adverse event (no medical history reported.)",
              "comments: list of non-encoded patient relevant history: patient other relevant history 1: no adverse event, continue: [unk], comment: no reported medical history."
              "medical history/concurrent conditions: no adverse event", 
               "none as of yet", "none at this time", "negative", "none yet", "none taken",
              "nothing special.", "no acute illness", "no allergies", "no known allergies", 
              "no allergies at this time", "no food or drug allergies"]
}

HISTORY_SEARCH = {
    
    'HYPERTENSION': ["high blood pressure", "hypertension", "htn", "high pressure"],
    
    'DIABETES': ["diabetes", r"^diabet.*", "diabetes ii",'type 1 diabetes',
                'type 2 diabetes','type ii dm','dm', 't2dm','t1dm','type i dm', 
                'diabet', 'diabetes type ii', 'niddm', 'diabetes mellitus', 'iddm'],

    'CANCER': ['cancer', 'cancers', r'.*noma$', r'.*homa$', "malignant", "malignancy"],
    
    'CKD': ['kidney','ckd','esrf','esrd','kidney failure','nephropathy',
                     'renal','nephrosis','kidn', r'^nephr.*', "dialysis"], 
    
    'CARDIOVASCULAR': ['cardiac','heart','angina','congestive heart failure',
                        'cardiac infarction','nstemi','stemi','coronary',
                        'ischemic heart','ihd','myocardial','myocarditis', 
                        'pericarditis','vascular','thrombosis', r'^throm*',
                        'ischemic','pvd','pad','vessel', 'cad', r'^coron.*',
                        'chest pain', 'cardiac', r'.*cardio.*', 'atrial', 
                        'ventricular', 'fibrillation','arrhythmia','angina']
}

SYMP_ERROR = [
    'product administered to patient of inappropriate age',
    'unevaluable event',
    'poor quality product administered',
    'incorrect dose administered',
    'inappropriate schedule of product administration',
    'poor quality product administered,product storage error,product temperature excursion issue',
    'wrong product administered',
    'product storage error',
    'interchange of vaccine products',
    'underdose',
    'no adverse event'
]

SYMP_SEARCH = {
    
    'SYMP_SERIOUS_ALLERGY': ['anaphylaxis','anaphylactic shock', 'anaphylactic', 'anaphylactoid', 
                   'angioedema', 'generalized urticaria', 'wheezing'],
    
    'SYMP_DEATH': ['death', 'died'], 
    
    'SYMP_CARDIAC': ['cardiac','heart','angina','congestive heart failure','infarction','nstemi','stemi','coronary',
                       'ischemic heart','ihd','myocardial','myocarditis', 'pericarditis','thrombosis', r'^throm*', 
                     'ischemia','cad', r'^coron.*', 'chest pain', r'.*cardio.*', 'atrial', 'ventricular', 
                     'fibrillation','arrhythmia','angina']
}

TARGET_COMPOSITE = ['DIED', 'L_THREAT', 'HOSPITAL', 'X_STAY', 'DISABLE', 'ER_ED_VISIT', 'SYMP_DEATH', 
                   'SYMP_CARDIAC', 'SYMP_SERIOUS_ALLERGY']

# to obtain modelling dataset 
DROP_COLS =  [
    'VAERS_ID',
    'STATE',
    'SYMPTOM_TEXT',
    'DIED',
    'L_THREAT',
    'HOSPITAL',
    'HOSPDAYS',
    'X_STAY',
    'DISABLE',
    'RECOVD',
    'VAX_DATE',
    'ONSET_DATE',
    'NUMDAYS',
    'LAB_DATA',
    'OTHER_MEDS',
    'CUR_ILL',
    'HISTORY',
    'PRIOR_VAX',
    'SPLTTYPE',
    'OFC_VISIT',
    'ER_ED_VISIT',
    'ALLERGIES',
    'SYMP',
    'SYMP_DEATH', 
    'SYMP_CARDIAC',
    'SYMP_SERIOUS_ALLERGY',
    'V_ADMINBY',
    'RECVDATE',
    'CAGE_YR',
    'CAGE_MO',
    'DATEDIED',
    'ER_VISIT',
    'V_FUNDBY',
    'FORM_VERS',
    'TODAYS_DATE',
    'RPT_DATE',
    'BIRTH_DEFECT',
    'composite'
]

###############
# for modelling
################

TARGET = 'TARGET_SERIOUS_ADVERSE'
TEST_SIZE = 0.2
VALID_SIZE = 0.2
SEED = 2
SCALER = None
ONEHOT_COLS = []

MODEL_TYPE = 'DT'

PARAMS = {
    'class_weight': 'balanced',
    'max_features': 'sqrt',
    'max_depth': 50,
    'min_samples_leaf': 2,
    'random_state': 2
}

MODEL_DIR = '../models'

N_SAMPLES = 20

AUTOML_SETTINGS = {
    "time_budget": 30,  # in seconds
    "metric": 'roc_auc',
    "task": 'classification',
    "log_file_name": "../logs/automl.log",
}