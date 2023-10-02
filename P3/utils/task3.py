import numpy as np

from .task1 import get_iou

def get_num_ebg_hbg(ebg_idxs, hbg_idxs):
    """
    Check is samples for easy and hard background exist. 
    input
        ious (N,) np array, N is the number of predictions
        ebg_idxs (N,) np array, N is the number of entries of easy background samples
        hbg_idxs (N,) np array, N is the number of entries of hard background samples
    output
        Boolean (easy background exists), Boolean (hard background exists)
    """
    num_ebg = ebg_idxs.shape[0]
    num_hbg = hbg_idxs.shape[0]
    return num_ebg, num_hbg

def ebg_hbg_sampling_nums(num_tot, num_fgs):
    """
    Compute the number of samples 'to sample' from ebg and hbg indexes. 
    input
        num_tot (int) number of total values to be sampled
        num_fgs (int) number of foreground samples
    output
        num_ebgs, num_hbgs
    """
    c = num_tot - num_fgs
    if c%2 == 0:
        num_ebgs = c/2
        num_hbgs = num_ebgs
    else:
        n = c//2
        if (n % 2) == 0:
            num_ebgs = n+1
            num_hbgs = n
        else:
            num_ebgs = n
            num_hbgs = n+1
    return num_ebgs, num_hbgs

def sample_ebg_hbg_idxs(ebg_idxs, hbg_idxs, num_tot, num_fgs):
    """
    input
        ebg_idxs (N,) np array, N number of easy background samples
        hbg_idxs (N,) np array, N number of hard background samples
        num_tot (int) number of total samples to sample
        num_fgs (int) number of foreground samples
    output
        (N,) np array storing the sampled indexes of background samples
    """
    num_ebg, num_hbg = get_num_ebg_hbg(ebg_idxs, hbg_idxs)
    if num_hbg == 0:
        fidxs =  sample_idxs(ebg_idxs, num_tot-num_fgs) # Only EBGs. 
    elif num_ebg == 0:
        fidxs =  sample_idxs(hbg_idxs, num_tot-num_fgs) # Only HBGs. 
    else:
        # EBGs and HBGs. 
        num_ebg, num_hbg = ebg_hbg_sampling_nums(num_tot, num_fgs)
        ebgi = sample_idxs(ebg_idxs, num_ebg)
        hbgi = sample_idxs(hbg_idxs, num_hbg)
        fidxs = np.concatenate((ebgi, hbgi))
    return fidxs

def sample_idxs(indxs, refl):
    """
    Samples values from the indxs array and stores them in an output array.
    input
        indxs (N,) array of indexes of bboxes
        refl (int), number of required indexes
    output
        (N') array, N' = refl
    """
    l = indxs.shape[0]
    if l > refl:
        sindxs = indxs[np.random.randint(l, size=int(refl))]
    elif l < refl:
        sindxs = indxs[np.random.randint(l, size=int(refl-l))]
        sindxs = np.concatenate((indxs, sindxs))
    else:
        return indxs
    return sindxs

""" Sample proposals. """

def sample_proposals(pred, target, xyz, feat, config, train=False):
    '''
    Task 3
    a. Using the highest IoU, assign each proposal a ground truth annotation. For each assignment also
       return the IoU as this will be required later on.
    b. Sample 64 proposals per scene. If the scene contains at least one foreground and one background
       proposal, of the 64 samples, at most 32 should be foreground proposals. Otherwise, all 64 samples
       can be either foreground or background. If there are less background proposals than 32, existing
       ones can be repeated.
       Furthermore, of the sampled background proposals, 50% should be easy samples and 50% should be
       hard samples when both exist within the scene (again, can be repeated to pad up to equal samples
       each). If only one difficulty class exists, all samples should be of that class.
    input
        pred (N,7) predicted bounding box labels
        target (M,7) ground truth bounding box labels
        xyz (N,512,3) pooled point cloud
        feat (N,512,C) pooled features
        config (dict) data config containing thresholds
        train (string) True if training
    output
        assigned_targets (64,7) target box for each prediction based on highest iou
        xyz (64,512,3) indices 
        feat (64,512,C) indices
        iou (64,) iou of each prediction and its assigned target box
    useful config hyperparameters
        config['t_bg_hard_lb'] threshold background lower bound for hard difficulty
        config['t_bg_up'] threshold background upper bound
        config['t_fg_lb'] threshold foreground lower bound
        config['num_fg_sample'] maximum allowed number of foreground samples
        config['bg_hard_ratio'] background hard difficulty ratio (#hard samples/ #background samples)
    '''
    # num_fg_sample: 32           # Task 3b: Maximum allowed number of foreground samples
    # bg_hard_ratio: 0.5          # Task 3b: Amongst background samples, hard/all
    # t_fg_lb: 0.55               # Task 3b: Foreground sample iou lower bound
    # t_bg_hard_lb: 0.05          # Task 3b: Background hard sample iou lower bound
    # t_bg_up: 0.45               # Task 3b: Background sample iou upper bound
    # print('shapes::', pred.shape[0], xyz.shape[0], feat.shape[0])
    # Validation and testing. 

    if train == False:
        iou_matrix = get_iou(pred, target)
        ious = np.amax(iou_matrix, axis=1)
        tbbx_idxs = np.argmax(iou_matrix, axis=1)
        return target[tbbx_idxs], xyz, feat, ious

    # Training (Sample 64 preds randomly). 
    try:
        iou_matrix = get_iou(pred, target)
        ious = np.amax(iou_matrix, axis=1)

        # Sample indexes. 
        fg_idxs = np.where(ious >= config['t_fg_lb'])[0]
        bg_idxs = np.where(ious < config['t_bg_up'])[0]
        ebg_idxs = np.where(ious < config['t_bg_hard_lb'])[0]
        hbg_idxs = np.nonzero(np.logical_and(ious >= config['t_bg_hard_lb'], ious < config['t_bg_up']))[0]

        num_ts = 64
        num_hs = num_ts/2
        num_fgs = fg_idxs.shape[0]
        num_bgs = bg_idxs.shape[0]
    
        if num_bgs == 0: # Only FGs.
            fidxs = sample_idxs(fg_idxs, num_ts) 
        elif num_fgs == 0: # Only BGs.
            fidxs = sample_ebg_hbg_idxs(ebg_idxs, hbg_idxs, num_ts, num_fgs)
        else: # Both FGs and BGs.
            if num_fgs > 32:
                num_fgs_ = num_hs
            elif num_fgs <= 32:
                num_fgs_ = num_fgs
            fg_idxs = sample_idxs(fg_idxs, num_fgs_) # FGs. 
            bg_idxs = sample_ebg_hbg_idxs(ebg_idxs, hbg_idxs, num_ts, num_fgs_) # BGs. 
            fidxs = np.concatenate((fg_idxs, bg_idxs))

        iou_matrix_sampled = iou_matrix[fidxs]
        tbbx_idxs = np.argmax(iou_matrix_sampled, axis=1)
        return target[tbbx_idxs], xyz[fidxs], feat[fidxs], ious[fidxs]
    except:
        iou_matrix = get_iou(pred, target)
        ious = np.amax(iou_matrix, axis=1)

        # Sample indexes. 
        fg_idxs = np.where(ious >= config['t_fg_lb'])[0]
        bg_idxs = np.where(ious < config['t_bg_up'])[0]
        ebg_idxs = np.where(ious < config['t_bg_hard_lb'])[0]
        hbg_idxs = np.nonzero(np.logical_and(ious >= config['t_bg_hard_lb'], ious < config['t_bg_up']))[0]

        num_ts = 64
        num_hs = num_ts/2  
        num_fgs = fg_idxs.shape[0]
        num_bgs = bg_idxs.shape[0]
    
        if num_bgs == 0: # Only FGs.
            # print('----------------- a --------------------')
            fidxs = sample_idxs(fg_idxs, num_ts) 
        elif num_fgs == 0: # Only BGs.
            # print('------------------ b ------------------------') 
            fidxs = sample_ebg_hbg_idxs(ebg_idxs, hbg_idxs, num_ts, num_fgs)
        else: # Both FGs and BGs.
            # print('------------------------- c -----------------------') 
            if num_fgs > 32:
                num_fgs_ = num_hs
            elif num_fgs <= 32:
                num_fgs_ = num_fgs
            fg_idxs = sample_idxs(fg_idxs, num_fgs_) # FGs. 
            bg_idxs = sample_ebg_hbg_idxs(ebg_idxs, hbg_idxs, num_ts, num_fgs_) # BGs. 
            fidxs = np.concatenate((fg_idxs, bg_idxs))
            # print('------------------- fg_idxs ----------------')
            # for i in fg_idxs:
                # print(i)
            # print('----------------- bg_idxs ---------------')
            # for i in bg_idxs:
                # print(i)
        # print('-------------- fidxs ------------------')
        # for i in fidxs:
            # print(i)

        print('shapes::', pred.shape[0], xyz.shape[0], feat.shape[0], ious.shape[0])

        raise ValueError

""" Tests for task 3. """

if '__main__' == __name__:
    # Create input arrays for sample_proposals. 
    # pred = np.random.rand((, size=(100, 7)))
    pred = np.zeros(5)
    target = np.zeros(5)
    xyz = np.zeros(5)
    feat = np.zeros(5)
    config = {
        'num_fg_sample': 32,           # Task 3b: Maximum allowed number of foreground samples
        'bg_hard_ratio': 0.5,          # Task 3b: Amongst background samples, hard/all
        't_fg_lb': 0.55,               # Task 3b: Foreground sample iou lower bound
        't_bg_hard_lb': 0.05,          # Task 3b: Background hard sample iou lower bound
        't_bg_up': 0.45,               # Task 3b: Background sample iou upper bound}
    }
    sample_proposals(pred, target, xyz, feat, config, train=True)
