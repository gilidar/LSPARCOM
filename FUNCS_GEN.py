import matplotlib.pyplot as plt
from basic_model import *

def m_shape(v, a, b):
    return np.reshape(v, (a, b), order='F') # reshape like matlab

def norma(ar):
    # Normalize by global maximax and minima
    val = 1  # for range [0 val]
    aro = ar - np.amin(ar)
    maxv = np.amax(aro)
    if maxv != 0:
        aro = val*aro/maxv
    return aro

def normaG(ar, val):
    # Normalize according to known value
    aro = ar - np.amin(ar)
    maxv = np.amax(aro)
    if maxv != 0:
        aro = val*aro/maxv
    return aro

def normaM(ar):
    # Remove mean and normalize variance per-patch
    std_val = np.std(ar)
    if std_val != 0:
        ar0 = (ar - np.mean(ar))/std_val
    else:
        ar0 = ar - np.mean(ar)
    return ar0

def normaA(ar, val):
    # Normalize according to known value
    return val*ar/np.amax(ar)

def imagesc2test(in_im, res_im, gt_im, size_in, ep, tr_va):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(11, 6))
    vmn = np.amin(in_im)
    vmx = np.amax(in_im)
    imgplot1 = ax1.imshow(in_im, interpolation='none', vmin=vmn, vmax=vmx)
    ax1.set_title('INPUT')
    fig.colorbar(imgplot1, ax=ax1, orientation='horizontal')
    vmn = np.amin(res_im)
    vmx = np.amax(res_im)
    imgplot2 = ax2.imshow(res_im, interpolation='none', vmin=vmn, vmax=vmx)
    ax2.set_title('NN Result')
    fig.colorbar(imgplot2, ax=ax2, orientation='horizontal')
    vmn = np.amin(gt_im)
    vmx = np.amax(gt_im)
    imgplot3 = ax3.imshow(gt_im,interpolation='none', vmin=vmn, vmax=vmx)
    ax3.set_title('GT')
    fig.colorbar(imgplot3, ax=ax3,  orientation='horizontal')
    gt_b = gt_im > 1e-5
    res_im_b = res_im > 1e-5
    FP = 2*(res_im_b > gt_b) #2 means redundant pixels
    FN = (res_im_b < gt_b) #1 means missing pixels
    overlay = FP + FN #0s means correct identifcation
    imgplot4 = ax4.imshow(overlay,interpolation='none', vmin=0, vmax=2)
    ax4.set_title('Overlay')
    fig.colorbar(imgplot4, ax=ax4,  orientation='horizontal')
    if tr_va == 0:
        fig.suptitle('Empty FOV Training on epoch # %d' % ep)
        plt.savefig('EmptyFOVtraining_epoch%d.png' % ep)  # plt.show()
    elif tr_va == 1:
        fig.suptitle('Training on epoch # %d' % ep)
        plt.savefig('training_epoch%d.png' % ep)  # plt.show()
    else:
        fig.suptitle('Test on epoch # %d' % ep)
        plt.savefig('Test_epoch%d.png' % ep)  # plt.show()
    plt.close('all')
    return

def imagescPRE(res_im, gt_im, size_in, batch, fold, tr_va):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))
    vmn = np.amin(res_im)
    vmx = np.amax(res_im)
    imgplot1 = ax1.imshow(res_im.reshape(size_in, size_in), interpolation='none', vmin=vmn, vmax=vmx)
    ax1.set_title('NN Result')
    fig.colorbar(imgplot1, ax=ax1, orientation='horizontal')
    vmn = np.amin(gt_im)
    vmx = np.amax(gt_im)
    imgplot2 = ax2.imshow(gt_im,interpolation='none', vmin=vmn, vmax=vmx)
    ax2.set_title('GT')
    fig.colorbar(imgplot2, ax=ax2,  orientation='horizontal')
    if tr_va == 1:
        fig.suptitle('pretraining with %d folds on epoch # %d' % (fold, batch))
        plt.savefig('pTrain_fold%d_epoch%d.png' % (fold, batch))# plt.show()
    else:
        fig.suptitle('pretraining test with %d folds on epoch # %d' % (fold, batch))
        plt.savefig('pTest_fold%d_epoch%d.png' % (fold, batch))  # plt.show()
    plt.close('all')
    return


def imagesc(im2show, size_in, ep, tr_va):
    vmn = np.amin(im2show)
    vmx = np.amax(im2show)
    plt.figure(figsize=(10, 8))
    plt.imshow(im2show.reshape(size_in, size_in), aspect='auto', interpolation='none', extent=[0, size_in - 1, 0, size_in - 1],
                vmin=vmn, vmax=vmx)
    plt.colorbar()
    if tr_va == 1:
        plt.suptitle('Training input for epoch # %d' % ep)
        plt.savefig('Training input for epoch # %d' % ep)
        plt.close('all')
    elif tr_va == 2:
        plt.suptitle('Test input for epoch # %d' % ep)
        plt.savefig('Test input for epoch # %d' % ep)
        plt.close('all')
    else:
        plt.show()
    return

def imagesc(im2show, size_in, ep, tr_va):
    vmn = np.amin(im2show)
    vmx = np.amax(im2show)
    plt.figure(figsize=(10, 8))
    plt.imshow(im2show.reshape(size_in, size_in), aspect='auto', interpolation='none', extent=[0, size_in - 1, 0, size_in - 1],
                vmin=vmn, vmax=vmx)
    plt.colorbar()
    if tr_va == 1:
        plt.suptitle('Training input for epoch # %d' % ep)
        plt.savefig('Training input for epoch # %d' % ep)
        plt.close('all')
    elif tr_va == 2:
        plt.suptitle('Test input for epoch # %d' % ep)
        plt.savefig('Test input for epoch # %d' % ep)
        plt.close('all')
    else:
        plt.show()
    return
