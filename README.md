# AVD-HLC

## 1.Introduction
The implementation code of a novel multi-label ECG classification method, "Multi-Label ECG Diagnosis via Adversarial View Decoupling and Hierarchical Label Constraints", will be fully disclosed after the paper is accepted, including specific training and inference details.

## 2.Requirements
- ​ python >= 3.7
- ​ pytorch >= 1.10.0
- ​torchvision >= 0.8.1
- ​numpy >= 1.19.5
- ​tqdm >= 4.62.0
- ​scipy == 1.7.3
- ​wfdb == 4.1.2
- ​scikit-learn == 1.0.2
- h5py == 3.8.0

## 3.Dataset

- PTB-XL dataset can be downloaded from ：https://physionet.org/content/ptb-xl/1.0.1/.
- SPH dataset can be downloaded from ：https://www.nature.com/articles/s41597-022-01403-5.

## 4.Standardized label text.
    'CD': 'conduction disturbance',
    'HYP': 'hypertrophy',
    'MI': 'myocardial infarction',
    'NORM': 'normal ECG',
    'STTC': 'ST-T change',
    'AMI': 'anteroseptal myocardial infarction',
    'CLBBB': 'complete left Bundle Branch Block',
    'CRBBB': 'complete right Bundle Branch Block',
    'ILBBB': 'incomplete left Bundle Branch Block',
    'IMI': 'inferolateral or Inferior myocardial infarction',
    'IRBBB': 'incomplete Right Bundle Branch Block',
    'ISCA': 'anterior or Anteroseptal or Lateral or Anterolateral Ischemia, ST-T change',
    'ISCI': 'inferior or Inferolateral Ischemia, ST-T change',
    'ISC_': 'ischemic ST-T changes',
    'IVCD': 'intraventricular Conduction disturbance',
    'LAFB/LPFB': 'left Anterior or Posterior Fascicular Block',
    'LAO/LAE': 'left Atrial Overload/Enlargement',
    'LMI': 'lateral Myocardial Infarction',
    'LVH': 'left Ventricular Hypertrophy',
    'NST_': 'non-Specific ST-T change',
    'PMI': 'posterior Myocardial Infarction',
    'RAO/RAE': 'right Atrial Overload/Enlargement',
    'RVH': 'right Ventricular Hypertrophy',
    'SEHYP': 'septal Hypertrophy',
    'WPW': 'wolff-Parkinson-White Syndrome',
    '_AVB': 'first or second or third degree Atrioventricular Block',
    '1AVB': 'first degree Atrioventricular Block',
    '2AVB': 'second degree Atrioventricular Block',
    '3AVB': 'third degree Atrioventricular Block',
    'ALMI': 'anterolateral myocardial infarction',
    'ANEUR': 'ST-T changes compatible with ventricular aneurysm',
    'ASMI': 'anteroseptal myocardial infarction',
    'DIG': 'Digitalis Effect, ST-T change',
    'EL': 'electrolyte abnormality',
    'ILMI': 'inferolateral myocardial infarction',
    'INJAL': 'Injury in Anterolateral Leads',
    'INJAS': 'Injury in Anteroseptal Leads',
    'INJIL': 'Injury in Inferolateral leads',
    'INJIN': 'Injury in Inferior Leads',
    'INJLA': 'Injury in Lateral Leads',
    'IPLMI': 'inferoposterolateral myocardial infarction',
    'IPMI': 'inferoposterior myocardial infarction',
    'ISCAL': 'Ischemia in Anterolateral Leads',
    'ISCAN': 'Ischemia in Anterior Leads',
    'ISCAS': 'Ischemia in Anteroseptal Leads',
    'ISCIL': 'Ischemia in Inferolateral Leads',
    'ISCIN': 'Ischemia in Inferior Leads',
    'ISCLA': 'Ischemia in Lateral Leads',
    'LAFB': 'Left Anterior Fascicular Block',
    'LNGQT': 'Long QT Interval',
    'LPFB': 'Left Posterior Fascicular Block',
    'NDT': 'Non-Diagnostic T Wave Changes',
    '1': 'normal ECG',
    '21': 'Sinus tachycardia',
    '22': 'Sinus bradycardia',
    '23': 'Sinus arrhythmia',
    '30': 'Atrial premature complexes',
    '31': 'Atrial premature complexes, nonconducted',
    '36': 'Junctional premature complexes',
    '37': 'Junctional escape complexes',
    '50': 'Atrial fibrillation',
    '51': 'Atrial flutter',
    '54': 'Junctional tachycardia',
    '60': 'Ventricular premature complexes',
    '80': 'Short PR interval',
    '81': 'AV conduction ratio N:D',
    '82': 'Prolonged PR interval',
    '83': 'Second-degree AV block, Mobitz type I Wenckebach',
    '84': 'Second-degree AV block, Mobitz type II',
    '85': '2:1 AV block',
    '86': 'AV block, varying conduction',
    '87': 'AV block, advanced, high-grade',
    '88': 'AV block, complete, third-degree',
    '101': 'Left anterior fascicular block',
    '102': 'Left posterior fascicular block',
    '104': 'Left bundle-branch block',
    '105': 'Incomplete right bundle-branch block',
    '106': 'Right bundle-branch block',
    '108': 'Ventricular preexcitation',
    '120': 'Right axis deviation',
    '121': 'Left axis deviation',
    '125': 'Low voltage',
    '140': 'Left atrial enlargement',
    '142': 'Left ventricular hypertrophy',
    '143': 'Right ventricular hypertrophy',
    '145': 'ST deviation',
    '146': 'ST deviation with T wave change',
    '147': 'T wave abnormality',
    '148': 'Prolonged QT interval',
    '152': 'TU fusion',
    '153': 'ST-T change due to ventricular hypertrophy',
    '155': 'Early repolarization',
    '160': 'Anterior Myocardial infarction',
    '161': 'Inferior Myocardial infarction',
    '165': 'Anteroseptal Myocardial infarction',
    '166': 'Extensive anterior Myocardial infarction',
    'A': 'normal ECG',
    'C': 'Sinus node rhythms and arrhythmias',
    'D': 'Supraventricular arrhythmias',
    'E': 'Supraventricular tachyarrhythmias',
    'F': 'Ventricular arrhythmias',
    'H': 'Atrioventricular conduction',
    'I': 'Intraventricular and intra-atrial conduction',
    'J': 'Axis and voltage',
    'K': 'Chamber hypertrophy or enlargement',
    'L': 'ST segment, T wave, and U wave',
    'M': 'Myocardial infarction',







