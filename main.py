import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import ex1_utils as utils


def lucaskanade(im1, im2, N):
    # im1 − f i r s t image mat rix ( g r a y s c a l e )
    # im2 − sec ond image mat rix ( g r a y s c a l e )
    # n − s i z e o f the nei ghb o rh o od (N x N)
    # TODO : the a l g o ri t h m
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    sigma = 1
    im1 = utils.gausssmooth(im1, sigma)
    im2 = utils.gausssmooth(im2, sigma)

    dt = im2.astype(np.int32) - im1.astype(np.int32)
    #dt = dt * (dt > 0)
    dt = im2 - im1
    dx, dy = utils.gaussderiv(im1, sigma)
    neighbourhood_size = N

    sumdxdy = cv2.filter2D(dx * dy,-1, np.ones((neighbourhood_size, neighbourhood_size)))
    sumdxdt = cv2.filter2D(dx * dt,-1, np.ones((neighbourhood_size, neighbourhood_size)))
    sumdydt = cv2.filter2D(dy * dt,-1, np.ones((neighbourhood_size, neighbourhood_size)))
    sumdx2 = cv2.filter2D(dx ** 2,-1, np.ones((neighbourhood_size, neighbourhood_size)))
    sumdy2 = cv2.filter2D(dy ** 2,-1, np.ones((neighbourhood_size, neighbourhood_size)))
    D = sumdx2 * sumdy2 - sumdxdy ** 2
    u = -np.divide(sumdy2 * sumdxdt - sumdxdy * sumdydt, D, out=np.zeros_like(D), where=D != 0)
    v = -np.divide(sumdx2 * sumdydt - sumdxdy * sumdxdt, D, out=np.zeros_like(D), where=D != 0)
    #plt.imshow(u)
    #plt.show()
    #plt.imshow(v)
    #plt.show()
    return u, v
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
for i in range(1, 100):
    i1 = cv2.imread(f"./collision/{'{:08d}'.format(i)}.jpg", cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(f"./collision/{'{:08d}'.format(i+1)}.jpg", cv2.IMREAD_GRAYSCALE)
    i1 = np.random.randint(0,255, (200,200)).astype(np.uint8)
    test = rotate_image(i1, 1)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(i1)
    ax[1].imshow(test)
    plt.show()
    u, v = lucaskanade(i1, test, 7)
    fig, ax = plt.subplots()
    utils.show_flow(u, v, ax)
    plt.show()
