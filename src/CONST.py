DATASET_TYPES = ['test', 'train', 'val']

NUM_LABELS = 22

ONE_HOT_ENCODING = {
    'Appeal to fear/prejudice': 0, 
    'Flag-waving': 1, 
    'Exaggeration/Minimisation': 2, 
    'Black-and-white Fallacy/Dictatorship': 3, 
    'Loaded Language': 4, 
    'Causal Oversimplification': 5, 
    'Obfuscation, Intentional vagueness, Confusion': 6, 
    'Slogans': 7, 
    "Misrepresentation of Someone's Position (Straw Man)": 8,
    'Repetition': 9, 
    'Transfer': 10, 
    'Thought-terminating cliché': 11, 
    'Appeal to authority': 12, 
    'Reductio ad hitlerum': 13, 
    'Doubt': 14, 
    'Smears': 15, 
    'Bandwagon': 16, 
    'Glittering generalities (Virtue)': 17, 
    'Name calling/Labeling': 18, 
    'Whataboutism': 19, 
    'Appeal to (Strong) Emotions': 20, 
    'Presenting Irrelevant Data (Red Herring)': 21
}

REVERSE_ONE_HOT_ENCODING = {
    0:'Appeal to fear/prejudice', 
    1:'Flag-waving', 
    2:'Exaggeration/Minimisation', 
    3:'Black-and-white Fallacy/Dictatorship', 
    4:'Loaded Language', 
    5:'Causal Oversimplification', 
    6:'Obfuscation, Intentional vagueness, Confusion', 
    7:'Slogans', 
    8:"Misrepresentation of Someone's Position (Straw Man)",
    9:'Repetition', 
    10:'Transfer', 
    11:'Thought-terminating cliché', 
    12:'Appeal to authority', 
    13:'Reductio ad hitlerum', 
    14:'Doubt', 
    15:'Smears', 
    16:'Bandwagon', 
    17:'Glittering generalities (Virtue)', 
    18:'Name calling/Labeling', 
    19:'Whataboutism', 
    20:'Appeal to (Strong) Emotions', 
    21:'Presenting Irrelevant Data (Red Herring)'
}
