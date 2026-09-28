import numpy as np
import pandas as pd

diagnosis_map = {
    'akiec': 'actinic keratosis / Bowen\'s disease',
    'bcc': 'basal cell carcinoma',
    'bkl': 'benign keratosis-like lesion',
    'df': 'dermatofibroma',
    'mel': 'melanoma',
    'nv': 'melanocytic nevus',
    'vasc': 'vascular lesion'
}

prompts = {
    "full_metadata": [
        "Dermoscopy image of a {dx} on the {site} of a {age}-year-old {sex}.",
        "Dermoscopic photograph showing a {dx} on the {site} in a {age}-year-old {sex}.",
        "Skin lesion ({dx}) located on the {site} of a {age}-year-old {sex}.",
        "Dermoscopy of a {dx} on the {site} in a {age}-year-old {sex}.",
        "Cutaneous {dx} located on the {site} of a {age}-year-old {sex}, dermoscopy image.",
        "Dermoscopic close-up showing a {dx} on the {site} of a {age}-year-old {sex}.",
        "Pigmented lesion ({dx}) on the {site} in a {age}-year-old {sex}, dermoscopic view."
    ],

    "no_sex": [
        "High-resolution dermoscopy of a {dx} on the {site}.",
        "Clinical dermoscopic view of a {dx} on the {site}.",
        "Dermoscopic image presenting a {dx} on the {site}.",
        "Magnified skin image of a {dx} found on the {site}.",
        "Detailed dermoscopy of a {dx} lesion on the {site}.",
        "Macro dermoscopic photograph depicting a {dx} on the {site}.",
        "Skin surface dermoscopy showing a {dx} on the {site}.",
        "Dermoscopic evaluation image of a {dx} on the {site}.",
        "Standardized dermoscopy photograph of a {dx} on the {site}.",
        "Polarized dermoscopy image of a {dx} on the {site}.",
        "Clinical close-up dermoscopy showing {dx} on the {site}.",
        "{dx} on the {site}, dermoscopy image.",
        "Dermoscopic photo: {dx} on the {site}.",
        "Dermoscopy view of lesion: {dx} on {site}.",
        "Magnified dermoscopy image of {dx} on the {site}."
    ],

    "no_localization": [
        "Diagnostic dermoscopy capture depicting a {dx}.",
        "Dermatology-focused dermoscopic image showing a {dx} lesion.",
        "Dermoscopy of {dx}.",
        "High-quality dermoscopic image of a {dx}.",
        "Clinical dermoscopic capture showing {dx}."
    ],

    "dx_only": [
        "Dermoscopic image of a {dx}.",
        "Skin lesion consistent with {dx}, dermoscopy view.",
        "Close-up dermoscopic capture revealing {dx}.",
        "Dermatology image showing features of {dx}.",
        "Dermoscopy photograph displaying {dx}."
    ]
}

def create_prompt(df_row):
    if df_row['localization'] == 'unknown' and df_row['sex'] == 'unknown':
        prompt_choice = np.random.choice(prompts['dx_only'])
    elif df_row['localization'] == 'unknown':
        prompt_choice = np.random.choice(prompts['no_localization'])
    elif df_row['sex'] == 'unknown':
        prompt_choice = np.random.choice(prompts['no_sex'])
    else:
        categories = ['full_metadata', 'no_sex', 'no_localization', 'dx_only']
        selected_category = np.random.choice(categories, p=[0.7,0.1,0.1,0.1])
        prompt_choice = np.random.choice(prompts[selected_category])
        
    
    if 'dx' in prompt_choice:
        diagnoses = df_row['dx']
        mapping = diagnosis_map[diagnoses]
        prompt_choice = prompt_choice.replace('{dx}', mapping)
    if 'site' in prompt_choice:
        prompt_choice = prompt_choice.replace('{site}', df_row['localization'])
    if 'age' in prompt_choice:
        prompt_choice = prompt_choice.replace('{age}', str(int(df_row['age'])))
    if 'sex' in prompt_choice:
        prompt_choice = prompt_choice.replace('{sex}', df_row['sex'])
    
    return prompt_choice

def prompt_engineering(data):
    row, col = data.shape
    data['Prompt'] = np.full((row), 'NA')

    for idx in range(row):
        patient = data.iloc[idx,:]
        prompt = create_prompt(patient)
        data.loc[idx, 'Prompt'] = prompt
    
    pd.set_option('display.max_colwidth', None)
    print('\nMetadata after building prompts:')
    print(data.head(10))