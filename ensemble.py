import os



import numpy as np




from alchemy_cat.data.plugins import arr2PIL







a = 1.0




os.makedirs('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_ensemble/infer/best,ss/masks/')



npz_name_ls = os.listdir('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_fma/infer/best,ss/seg_preds/')

for npz_name in npz_name_ls:

    fma_npz = np.load(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_fma/infer/best,ss/seg_preds/', npz_name))
    semples_npz = np.load(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_semples/infer/best,ss/seg_preds/', npz_name))



    mask = np.zeros((21, fma_npz['prob'][0].shape[0], fma_npz['prob'][0].shape[1]))
    mask[0] = mask[0] + a*fma_npz['prob'][0] + (1.0-a)*semples_npz['prob'][0]


    for cls_ in range(1, 21):

        if cls_ in fma_npz['fg_cls']+1:
            fma_prob = fma_npz['prob'][np.where(fma_npz['fg_cls']+1==cls_)[0][0]+1]
            mask[cls_] = mask[cls_] + a*fma_prob

        if cls_ in semples_npz['fg_cls']+1:
            semples_prob = semples_npz['prob'][np.where(semples_npz['fg_cls']+1==cls_)[0][0]+1]
            mask[cls_] = mask[cls_] + (1.0-a)*semples_prob


    mask = mask.argmax(axis=0).astype(np.uint8)
    arr2PIL(mask).save(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_ensemble/infer/best,ss/masks/', npz_name.split('.')[0]+'.png'))