import os



import numpy as np







a = 1.0




os.makedirs('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_ensemble/infer/best,ss/seg_preds/')



npz_name_ls = os.listdir('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC/infer/best,ss/seg_preds/')

for npz_name in npz_name_ls:

    fma_npz = np.load(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC/infer/best,ss/seg_preds/', npz_name))
    semples_npz = np.load(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_semples/infer/best,ss/seg_preds/', npz_name))



    key_ls = []
    prob_ls = []

    key_ls.append(0)
    prob_ls.append(fma_npz['prob'][0]*a + semples_npz['prob'][0]*(1.0-a))


    for key in range(1, 21):

        fma_prob = None

        if key in fma_npz['fg_cls']+1:
            fma_prob = fma_npz['prob'][np.where(fma_npz['fg_cls']+1==key)[0][0]+1]
        else:
            fma_prob = np.zeros_like(fma_npz['prob'][0])



        semples_prob = None

        if key in semples_npz['fg_cls']+1:
            semples_prob = semples_npz['prob'][np.where(semples_npz['fg_cls']+1==key)[0][0]+1]
        else:
            semples_prob = np.zeros_like(semples_npz['prob'][0])




        if fma_prob is None and semples_prob is None:
            continue
        else:
            key_ls.append(key)
            prob_ls.append(fma_prob*a + semples_prob*(1.0-a))

    keys = np.array(key_ls)
    prob = np.stack(prob_ls, axis=0)

    np.save(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_ensemble/infer/best,ss/seg_preds/', npz_name.split('.')[0]+'.npy'), {"prob": prob, "keys": keys})