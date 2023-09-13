"""
Preprocess HCP with custom path
"""
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.image import load_img, math_img, mean_img
from nilearn.input_data import MultiNiftiMasker, NiftiMasker
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


def preprocess_hcp(data_dir='/data/parietal/store/data/HCP900/',
                   n_subjects=150, experiment='RELATIONAL', no_mask=False,
                   mask_type='classic', mask_file=None, n_jobs=1, memory=None):
    """Available experiment: 'EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR',
    'RELATIONAL', 'SOCIAL', 'WM'

    """

    from hcp_builder.dataset import fetch_hcp

    data = fetch_hcp(data_dir=data_dir, n_subjects=n_subjects)
    contrasts = data.contrasts.z_map.reset_index()

    if experiment in ('MOTOR_HAND', 'MOTOR_FOOT'):
        experiment_corrected = 'MOTOR'
        TASK = contrasts[contrasts['task'] == experiment_corrected]
    else:
        TASK = contrasts[contrasts['task'] == experiment]

    input_images = TASK.z_map.values
    conditions = TASK.contrast.values

    if experiment == 'GAMBLING':
        condition_mask = np.logical_or(conditions == 'PUNISH',
                                       conditions == 'REWARD')
        y = np.asarray((conditions[condition_mask] == 'PUNISH') * 2 - 1)

    elif experiment == 'RELATIONAL':
        condition_mask = np.logical_or(conditions == 'MATCH',
                                       conditions == 'REL')
        y = np.asarray((conditions[condition_mask] == 'MATCH') * 2 - 1)

    elif experiment == 'EMOTION':
        condition_mask = np.logical_or(conditions == 'FACES',
                                       conditions == 'SHAPES')
        y = np.asarray((conditions[condition_mask] == 'FACES') * 2 - 1)

    elif experiment == 'SOCIAL':
        condition_mask = np.logical_or(conditions == 'RANDOM',
                                       conditions == 'TOM')
        y = np.asarray((conditions[condition_mask] == 'RANDOM') * 2 - 1)

    elif experiment == 'LANGUAGE':
        condition_mask = np.logical_or(conditions == 'MATH',
                                       conditions == 'STORY')
        y = np.asarray((conditions[condition_mask] == 'MATH') * 2 - 1)

    elif experiment == 'MOTOR_HAND':
        # Left hand vs right hand
        condition_mask = np.logical_or(conditions == 'LH',
                                       conditions == 'RH')
        y = np.asarray((conditions[condition_mask] == 'LH') * 2 - 1)

    elif experiment == 'MOTOR_FOOT':
        # Left foot vs right foot
        condition_mask = np.logical_or(conditions == 'LF',
                                       conditions == 'RF')
        y = np.asarray((conditions[condition_mask] == 'LF') * 2 - 1)

    # Working Memory
    elif experiment == 'WM':
        # 2-back vs 0-back
        condition_mask = np.logical_or(conditions == '2BK',
                                       conditions == '0BK')
        y = np.asarray((conditions[condition_mask] == '2BK') * 2 - 1)

    elif experiment == 'REST':
        input_images = data.rest.filename.values[3::4]
        condition_mask = None
        y = None

    else:
        raise ValueError('Wrong type of experiment.')

    # groups = TASK.subject.values[condition_mask]

    ######################################################################
    # Masking statistical maps - X, y
    # -------------------------------
    if mask_type == 'classic':
        mask_img = load_img(data.mask)
    elif mask_type == 'specific':
        mask_img = load_img(mask_file)

    if no_mask:
        mask_img = math_img("img > -1", img=mask_img)

    if experiment == 'REST':

        masker = MultiNiftiMasker(mask_img=mask_img, standardize=True,
                                  detrend=True, high_pass=0.01, t_r=0.72,
                                  n_jobs=n_jobs, verbose=1, memory=memory)
        mask = mask_img.get_data().astype(bool)

        X = masker.fit_transform(input_images)

    else:

        masker = MultiNiftiMasker(mask_img=mask_img, n_jobs=n_jobs, verbose=1,
                                  memory=memory)
        mask = mask_img.get_data().astype(bool)

        X_init = masker.fit_transform(input_images)
        X_sc = StandardScaler()

        if condition_mask is None:
            X = X_sc.fit_transform(np.vstack(X_init))
        else:
            X = X_sc.fit_transform(np.vstack(X_init))[condition_mask]

    return Bunch(X=X, y=y, mask=mask, mask_img=mask_img, masker=masker)


def preprocess_haxby(subject=2, memory=None, comparision='face_house'):
    # Gathering Data
    haxby_dataset = datasets.fetch_haxby(subjects=[subject])
    fmri_filename = haxby_dataset.func[0]
    behavioral = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")
    # groups = behavioral['chunks']

    conditions = behavioral['labels']
    if comparision == 'face_house':
        condition_mask = np.logical_or(conditions == b'face',
                                       conditions == b'house')
        y = np.asarray((conditions[condition_mask] == b'face') * 2 - 1)
    elif comparision == 'cat_chair':
        condition_mask = np.logical_or(conditions == b'cat',
                                       conditions == b'chair')
        y = np.asarray((conditions[condition_mask] == b'cat') * 2 - 1)
    else:
        raise ValueError('Not a valid comparision: {}'.format(comparision))

    if haxby_dataset.anat[0] is None:
        bg_img = None
    else:
        bg_img = mean_img(haxby_dataset.anat)  # background image

    # Using 'epi' mask_strategy as this is raw EPI data
    masker = NiftiMasker(mask_strategy='epi')
    masker.fit(fmri_filename)
    masker = NiftiMasker(mask_img=masker.mask_img_,
                         standardize=True,
                         smoothing_fwhm=None,
                         memory=memory)

    fmri_masked = masker.fit_transform(fmri_filename)
    mask = masker.mask_img_.get_data().astype(bool)

    X = np.asarray(fmri_masked)
    X = X[condition_mask, :]

    return Bunch(X=X, y=y, bg_img=bg_img, mask=mask, masker=masker)


def preprocess_oasis(n_subjects, memory=None, experiment='age'):
    # Fetching the data
    oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
    gray_matter_map_filenames = oasis_dataset.gray_matter_maps
    bg_img = load_img(gray_matter_map_filenames[0])

    masker_oasis = NiftiMasker(mask_strategy='epi')
    masker_oasis.fit(gray_matter_map_filenames)

    masker = NiftiMasker(
        mask_img=masker_oasis.mask_img_,
        standardize=True,
        smoothing_fwhm=2,
        memory=memory)

    gm_maps_masked = masker.fit_transform(gray_matter_map_filenames)
    mask = masker.mask_img_.get_data().astype(bool)

    X = np.asarray(gm_maps_masked)
    X = StandardScaler().fit_transform(X)

    # oasis_dataset.ext_vars.dtype.names returns column value
    if experiment == 'age':
        y = oasis_dataset.ext_vars['age'].astype(float)
        y = (y - y.mean()) / y.std()
    elif experiment == 'sex':
        conditions = oasis_dataset.ext_vars['mf']
        condition_mask = np.logical_or(conditions == b'F', conditions == b'M')
        # 1 is Female, -1 is Male
        y = np.asarray((conditions[condition_mask] == b'F') * 2 - 1)
    else:
        raise ValueError('Wrong type of experiment: {}'.format(experiment))

    return Bunch(X=X, y=y, bg_img=bg_img, masker=masker, mask=mask)
