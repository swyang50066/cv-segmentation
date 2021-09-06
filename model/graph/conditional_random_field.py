import  cv2
import  numpy   as  np

import  pydensecrf.densecrf     as dcrf
from    pydensecrf.utils        import unary_from_labels


def doFullyConnectedDenseCRF(
                        voxel, mark,
                        numCls=2, maxIters=10):
    # Get input shape
    shape  = voxel.shape

    # Run iteration
    for k in range(shape[2]):
        print(k+1, shape[2])
        image, label = voxel[..., k], mark[..., k]

        color = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        color[...] = np.expand_dims(image, axis=-1)

        if not np.sum(label): continue

        # Declare dense field and unary
        dense = dcrf.DenseCRF2D(shape[1], shape[0], numCls)
        unary = unary_from_labels(label, numCls, gt_prob=.7, zero_unsure=False)

        # Set unary energies
        dense.setUnaryEnergy(unary)
        dense.addPairwiseGaussian(sxy=2, compat=4)
        dense.addPairwiseBilateral(sxy=2, srgb=4, rgbim=color, compat=4)

        # Do inference
        query = dense.inference(maxIters)
        label = np.argmax(query, axis=0).reshape(shape[:2])

        # Update mark
        mark[..., k] = label

    return mark 

stack, mark = (np.load('./input/stack.npy'),
               np.load('./input/vmark.npy'))

voxel = 255.*(stack - stack.min())/(stack.max() - stack.min())

mark = doFullyConnectedDenseCRF(voxel, mark)
for k in range(stack.shape[2]):
    BGR = np.zeros((512, 512, 3))
    BGR[...] = voxel[..., k:k+1]
   
    pos = np.argwhere(mark[..., k] > 0)
    for i, j in pos:
        BGR[i, j, 1], BGR[i, j, 2] = 255, 0

    BGR = cv2.imwrite('./result/CT_{0:0>3}.bmp'.format(str(k)), BGR) 
