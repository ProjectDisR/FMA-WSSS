import os



import numpy as np
from PIL import Image



from alchemy_cat.data.plugins import arr2PIL






# def colorful(out, name):
#     arr = out.astype(np.uint8)
#     im = Image.fromarray(arr)

#     palette = []
#     for i in range(256):
#         palette.extend((i, i, i))
#     palette[:3 * 21] = np.array([[0, 0, 0],
#                                  [128, 0, 0],
#                                  [0, 128, 0],
#                                  [128, 128, 0],
#                                  [0, 0, 128],
#                                  [128, 0, 128],
#                                  [0, 128, 128],
#                                  [128, 128, 128],
#                                  [64, 0, 0],
#                                  [192, 0, 0],
#                                  [64, 128, 0],
#                                  [192, 128, 0],
#                                  [64, 0, 128],
#                                  [192, 0, 128],
#                                  [64, 128, 128],
#                                  [192, 128, 128],
#                                  [0, 64, 0],
#                                  [128, 64, 0],
#                                  [0, 192, 0],
#                                  [128, 192, 0],
#                                  [0, 64, 128]
#                                  ], dtype='uint8').flatten()

#     im.putpalette(palette)
#     im.save(name)





# a = 0.85




# os.makedirs('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_ensemble/infer/best,ss/masks/')



# npz_name_ls = os.listdir('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC/infer/best,ss/seg_preds/')

# for npz_name in npz_name_ls:

#     fma_npz = np.load(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC/infer/best,ss/seg_preds/', npz_name))
#     semples_npz = np.load(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_semples/infer/best,ss/seg_preds/', npz_name))



#     mask = np.zeros((21, fma_npz['prob'][0].shape[0], fma_npz['prob'][0].shape[1]))
#     mask[0] = mask[0] + a*fma_npz['prob'][0] + (1.0-a)*semples_npz['prob'][0]


#     for cls_ in range(1, 21):

#         if cls_ in fma_npz['fg_cls']+1:
#             fma_prob = fma_npz['prob'][np.where(fma_npz['fg_cls']+1==cls_)[0][0]+1]
#             mask[cls_] = mask[cls_] + a*fma_prob

#         if cls_ in semples_npz['fg_cls']+1:
#             semples_prob = semples_npz['prob'][np.where(semples_npz['fg_cls']+1==cls_)[0][0]+1]
#             mask[cls_] = mask[cls_] + (1.0-a)*semples_prob


#     mask = mask.argmax(axis=0).astype(np.uint8)
#     # arr2PIL(mask).save(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_ensemble/infer/best,ss/masks/', npz_name.split('.')[0]+'.png'))
#     colorful(mask, os.path.join('experiment/others/mmseg/m2f-sl22-bt4-80k-512x-VOC_ensemble/infer/best,ss/masks/', npz_name.split('.')[0]+'.png'))
    
    
    
    
    
    
    
    
    
    
a = 0.85




os.makedirs('experiment/others/mmseg/m2f-sl22-bt4-100k-512x-COCO_ensemble/infer/best,ss/masks/')



npz_name_ls = os.listdir('experiment/others/mmseg/m2f-sl22-bt4-100k-512x-COCO/infer/best,ss/seg_preds/')

for npz_name in npz_name_ls:

    fma_npz = np.load(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-100k-512x-COCO/infer/best,ss/seg_preds/', npz_name))
    semples_npz = np.load(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-100k-512x-COCO_semples/infer/best,ss/seg_preds/', npz_name))



    mask = np.zeros((81, fma_npz['prob'][0].shape[0], fma_npz['prob'][0].shape[1]))
    mask[0] = mask[0] + a*fma_npz['prob'][0] + (1.0-a)*semples_npz['prob'][0]


    for cls_ in range(1, 81):
        
        print(fma_npz['fg_cls'])
        print(fma_npz['prob'].shape)

        if cls_ in fma_npz['fg_cls']+1:
            fma_prob = fma_npz['prob'][np.where(fma_npz['fg_cls']+1==cls_)[0][0]+1]
            mask[cls_] = mask[cls_] + a*fma_prob

        # if cls_ in semples_npz['fg_cls']+1:
        #     semples_prob = semples_npz['prob'][np.where(semples_npz['fg_cls']+1==cls_)[0][0]+1]
        #     mask[cls_] = mask[cls_] + (1.0-a)*semples_prob


    # mask = mask.argmax(axis=0).astype(np.uint8)
    # arr2PIL(mask).save(os.path.join('experiment/others/mmseg/m2f-sl22-bt4-100k-512x-COCO_ensemble/infer/best,ss/masks/', npz_name.split('.')[0]+'.png'))