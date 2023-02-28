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

    # dt = dt * (dt > 0)
    dt = im2 - im1
    dx, dy = utils.gaussderiv(im1, sigma)
    neighbourhood_size = N

    sumdxdy = cv2.filter2D(dx * dy, -1, np.ones((neighbourhood_size, neighbourhood_size)))
    sumdxdt = cv2.filter2D(dx * dt, -1, np.ones((neighbourhood_size, neighbourhood_size)))
    sumdydt = cv2.filter2D(dy * dt, -1, np.ones((neighbourhood_size, neighbourhood_size)))
    sumdx2 = cv2.filter2D(dx ** 2, -1, np.ones((neighbourhood_size, neighbourhood_size)))
    sumdy2 = cv2.filter2D(dy ** 2, -1, np.ones((neighbourhood_size, neighbourhood_size)))
    D = sumdx2 * sumdy2 - sumdxdy ** 2
    u = -np.divide(sumdy2 * sumdxdt - sumdxdy * sumdydt, D, out=np.zeros_like(D), where=D != 0)
    v = -np.divide(sumdx2 * sumdydt - sumdxdy * sumdxdt, D, out=np.zeros_like(D), where=D != 0)
    # plt.imshow(u)
    # plt.show()
    # plt.imshow(v)
    # plt.show()
    return u, v


def hornschunck(im1, im2, n_iters, lmbd):
    # im1 − f i r s t image mat rix ( g r a y s c a l e )
    # im2 − sec ond image ma t rix ( g r a y s c a l e )
    # n_i t e r s − number o f i t e r a t i o n s ( t r y s e v e r a l hundred )
    # lmbd − parameter
    # TODO
    ...
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    dx = cv2.filter2D(im1, -1, np.array([[-0.5, 0.5], [-0.5, 0.5]]))
    dy = cv2.filter2D(im1, -1, np.array([[-0.5, -0.5], [0.5, 0.5]]))
    dt = cv2.filter2D(im2 - im1, -1, np.ones((2,2)) / 4.0)


    sigma = 1
    #im1 = utils.gausssmooth(im1, sigma)
    #im2 = utils.gausssmooth(im2, sigma)
    # dt = dt * (dt > 0)
    dt = im2 - im1
    #dt = cv2.filter2D(im2 - im1, -1, np.ones((2,2)) / 4.0)
    dx, dy = utils.gaussderiv(im1, sigma)

    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)
    ld = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])

    D = lmbd + dx ** 2 + dy ** 2
    for i in range(n_iters):
        ua = cv2.filter2D(u, -1, ld)
        va = cv2.filter2D(v, -1, ld)
        P = dx * ua + dy * va + dt
        u = ua - np.divide(dx * P, D, out=np.zeros_like(D), where=D != 0)
        v = va - np.divide(dy * P, D, out=np.zeros_like(D), where=D != 0)
    return u, v

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


for i in range(3,20,2):
    #i1 = cv2.imread(f"./collision/{'{:08d}'.format(i)}.jpg", cv2.IMREAD_GRAYSCALE)
    #i2 = cv2.imread(f"./collision/{'{:08d}'.format(i + 1)}.jpg", cv2.IMREAD_GRAYSCALE)
    i1 = np.random.randint(0, 255, (200, 200)).astype(np.uint8)
    i1 = (i1.astype(np.float32) / np.max(i1))
    test = rotate_image(i1, 1)

    #fig, ax = plt.subplots(1, 2)
    #ax[0].imshow(i1)
    #ax[1].imshow(test)
    #plt.show()
    u, v = lucaskanade(i1, test, i)

    fig, ax = plt.subplots(2,2)
    utils.show_flow(u, v, ax[0,0])
    utils.show_flow(u, v, ax[1,0], type="angle_magnitude")

    u, v = hornschunck(i1, test, 500, 0.8)
    utils.show_flow(u, v, ax[0, 1])
    utils.show_flow(u, v, ax[1, 1], type="angle_magnitude")

    plt.show()
