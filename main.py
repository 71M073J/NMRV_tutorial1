import time

import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import ex1_utils as utils
import matplotlib.image as img


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    # flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def upscale(u):
    return np.repeat(u, 2).reshape(u.shape[0], u.shape[1] * 2).T.repeat(2).reshape(2 * u.shape[1], 2 * u.shape[0]).T


def lucaskanade(im1, im2, N, harris=False, layers=1):
    # im1 − f i r s t image mat rix ( g r a y s c a l e )
    # im2 − sec ond image mat rix ( g r a y s c a l e )
    # n − s i z e o f the nei ghb o rh o od (N x N)
    # TODO : the a l g o ri t h m

    # TODO PYRAMIDAL
    if layers > 1:
        a = im1.shape[1]
        b = im1.shape[0]
        c = im2.shape[1]
        d = im2.shape[0]
        v = np.array((a,b,c,d))
        for i in range(layers):
            v = (v / 2).astype(np.int32)
        v *= (2 ** (layers))
        im1 = cv2.resize(im1, (v[0], v[1]))
        im2 = cv2.resize(im2, (v[2], v[3]))


    im1 = im1.astype(np.float32) / np.max(im1)
    im2 = im2.astype(np.float32) / np.max(im2)

    sigma = 1
    ims = []
    for i in range(layers):
        im1 = utils.gausssmooth(im1, sigma)
        im2 = utils.gausssmooth(im2, sigma)
        im1 = im1[np.arange(0, im1.shape[0], 2), :][:, np.arange(0, im1.shape[1], 2)]
        im2 = im2[np.arange(0, im2.shape[0], 2), :][:, np.arange(0, im2.shape[1], 2)]
        ims.append((im1, im2))

    prev_v = np.zeros((ims[-1][0].shape[0], ims[-1][0].shape[1]))
    prev_u = np.zeros((ims[-1][0].shape[0], ims[-1][0].shape[1]))

    im1, im2 = ims[-1]
    for i in reversed(range(layers)):
        im1, _ = ims[i]
        # dt = dt * (dt > 0)
        dt = im2 - im1
        dx, dy = utils.gaussderiv(im1, sigma)
        neighbourhood_size = N

        sumdxdy = cv2.filter2D(dx * dy, -1, np.ones((neighbourhood_size, neighbourhood_size)), borderType=cv2.BORDER_REFLECT)
        sumdxdt = cv2.filter2D(dx * dt, -1, np.ones((neighbourhood_size, neighbourhood_size)), borderType=cv2.BORDER_REFLECT)
        sumdydt = cv2.filter2D(dy * dt, -1, np.ones((neighbourhood_size, neighbourhood_size)), borderType=cv2.BORDER_REFLECT)
        sumdx2 = cv2.filter2D(dx ** 2, -1, np.ones((neighbourhood_size, neighbourhood_size)), borderType=cv2.BORDER_REFLECT)
        sumdy2 = cv2.filter2D(dy ** 2, -1, np.ones((neighbourhood_size, neighbourhood_size)), borderType=cv2.BORDER_REFLECT)
        D = sumdx2 * sumdy2 - sumdxdy ** 2
        u = -np.divide(sumdy2 * sumdxdt - sumdxdy * sumdydt, D, out=np.zeros_like(D), where=D != 0)
        v = -np.divide(sumdx2 * sumdydt - sumdxdy * sumdxdt, D, out=np.zeros_like(D), where=D != 0)

        #u = upscale(u)
        #v = upscale(v)
        #flow = np.stack((u, v), axis=-1)
        #im2 = warp_flow(ims[i - 1][0], flow)
        #prev_u = (u + upscale(prev_u))
        #prev_v = (v + upscale(prev_v))
        h = 2
        prev_u = (u + prev_u * h)
        prev_v = (v + prev_v * h)
        prev_u = upscale(prev_u)
        prev_v = upscale(prev_v)
        flow = np.stack((prev_u, prev_v), axis=-1).astype(np.float32)
        if i != 0:
            im2 = 0.5 * (warp_flow(ims[i - 1][0], flow) + ims[i-1][1])


    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(im2)
    # ax[1].imshow(im12)
    # plt.show()

    # indexes_u = (np.ones((200,200)) * np.arange(0,im1.shape[0], 1)) - np.floor(u + 0.5)
    # indexes_v = (np.ones((200, 200)) * np.arange(0, im1.shape[0], 1)).T - np.floor(v + 0.5)

    # im1_new = im1[indexes_u.astype(np.int32), indexes_v.astype(np.int32)]

    # plt.imshow(u)
    # plt.show()
    # plt.imshow(v)
    # plt.show()
    return prev_u, prev_v


def hornschunck(im1, im2, n_iters, lmbd, initial=False, epsilon=None, layers=4, early_stop=False, neigh=11):
    # im1 − f i r s t image mat rix ( g r a y s c a l e )
    # im2 − sec ond image ma t rix ( g r a y s c a l e )
    # n_i t e r s − number o f i t e r a t i o n s ( t r y s e v e r a l hundred )
    # lmbd − parameter
    # TODO
    ...
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    a = im1.shape[1]
    b = im1.shape[0]
    c = im2.shape[1]
    d = im2.shape[0]
    v = np.array((a, b, c, d))
    for i in range(layers):
        v = (v / 2).astype(np.int32)
    v *= (2 ** (layers))
    im1 = cv2.resize(im1, (v[0], v[1]))
    im2 = cv2.resize(im2, (v[2], v[3]))

    dx = cv2.filter2D(im1, -1, np.array([[-0.5, 0.5], [-0.5, 0.5]]))
    dy = cv2.filter2D(im1, -1, np.array([[-0.5, -0.5], [0.5, 0.5]]))
    dt = cv2.filter2D(im2 - im1, -1, np.ones((2, 2)) / 4.0)

    sigma = 1
    im1 = utils.gausssmooth(im1, sigma)
    im2 = utils.gausssmooth(im2, sigma)
    # dt = dt * (dt > 0)
    dt = im2 - im1
    # dt = cv2.filter2D(im2 - im1, -1, np.ones((2,2)) / 4.0)
    dx, dy = utils.gaussderiv(im1, sigma)
    if not initial:
        u = np.zeros(im1.shape)
        v = np.zeros(im1.shape)
    else:
        u, v = lucaskanade(im1, im2, neigh, layers=layers)
    ld = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    if epsilon is None:
        epsilon = i1.shape[0] * i1.shape[1] / (8000 * 2 / 5)
    D = lmbd + dx ** 2 + dy ** 2
    for i in range(n_iters):
        ua = cv2.filter2D(u, -1, ld)
        va = cv2.filter2D(v, -1, ld)
        P = dx * ua + dy * va + dt
        un = ua - np.divide(dx * P, D, out=np.zeros_like(D), where=D != 0)
        vn = va - np.divide(dy * P, D, out=np.zeros_like(D), where=D != 0)
        # print(np.abs(un - u).sum() + np.abs(vn - v).sum())
        if early_stop and (np.abs(un - u).sum() + np.abs(vn - v).sum()) < epsilon:
            print(i, "Iterations before convergence to epsilon", epsilon)
            break
        if i % 100 == 0:
            print(i, np.abs(un - u).sum() + np.abs(vn - v).sum(), epsilon)
        u = un
        v = vn
    return u, v


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def square_subplots(fig, ax):
    rows, cols = ax[0, 0].get_subplotspec().get_gridspec().get_geometry()
    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom
    wspace = fig.subplotpars.wspace
    hspace = fig.subplotpars.hspace
    figw, figh = fig.get_size_inches()

    axw = figw * (r - l) / (cols + (cols - 1) * wspace)
    axh = figh * (t - b) / (rows + (rows - 1) * hspace)
    axs = min(axw, axh)
    w = (1 - axs / figw * (cols + (cols - 1) * wspace)) / 2.
    h = (1 - axs / figh * (rows + (rows - 1) * hspace)) / 2.
    fig.subplots_adjust(bottom=h, top=1 - h, left=w, right=1 - w)


def plot_imgs(i1, i2, name):
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(i1)
    # ax[1].imshow(test)
    # plt.show()
    neigh = 15
    layer=2
    start = time.time()
    u, v = lucaskanade(i1, i2, neigh, layers=layer)
    print("LK Duration:", time.time() - start, "seconds")
    fig, ax = plt.subplots(2, 2, figsize=(6, 6))

    #fig.suptitle(name)
    #utils.show_flow(u, v, ax[1, 0])
    utils.show_flow(u, v, ax[1, 0], type="angle_magnitude")

    start = time.time()
    u, v = hornschunck(i1, i2, 1000, 100000, initial=True, layers=4, neigh=19, early_stop=False)
    print("HS Duration:", time.time() - start, "seconds")
    #utils.show_flow(u, v, ax[1, 1])
    utils.show_flow(u, v, ax[1, 1], type="angle_magnitude")

    ax[0, 0].imshow(i1)
    ax[0, 1].imshow(i2)
    #square_subplots(fig, ax)
    plt.tight_layout()
    plt.show()


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


if __name__ == "__main__":
    i0 = cv2.imread("./white_noise.png", cv2.IMREAD_GRAYSCALE)
    i0[100, :] = 0
    i0[:, 80] = 0
    i1 = i0[:197, :197]
    i1 = i1.astype(np.float32)
    m = i1.max()
    i1 = i1 / m
    noise2 = np.atleast_2d(np.tile(np.array((0, 1)), 15))
    noise2 = noise2.T.dot(noise2)
    i1[45:75, 45:75] = noise2

    i2 = i0[3:, 3:]

    i2 = i2.astype(np.float32)
    i2 = i2 / m
    i2[47:77, 47:77] = noise2

    # plot_imgs(i1, i2)

    i1 = np.random.randint(0, 255, (200, 200)).astype(np.uint8)
    i1 = (i1.astype(np.float32) / np.max(i1))

    #i1[190, :] = 1
    i2 = rotate_image(i1, 1)
    #plot_imgs(i1, i2, "Rotated random noise")

    #fig, ax = plt.subplots(3, 2, figsize=(6, 9))
    #ax[0, 0].imshow(i1)
    #ax[0, 1].imshow(i2)
    #u, v = hornschunck(i1, i2, 100, 0.1, initial=False, layers=4, neigh=19)
    #utils.show_flow(u, v, ax[1, 0], type="angle_magnitude")
    #u, v = hornschunck(i1, i2, 1000, 0.1, initial=False, layers=4, neigh=19)
    #utils.show_flow(u, v, ax[1, 1], type="angle_magnitude")
    #u, v = hornschunck(i1, i2, 100, 1, initial=False, layers=4, neigh=19)
    #utils.show_flow(u, v, ax[2, 0], type="angle_magnitude")
    #u, v = hornschunck(i1, i2, 1000, 1, initial=False, layers=4, neigh=19)
    #utils.show_flow(u, v, ax[2, 1], type="angle_magnitude")
    #plt.show()
    #quit()



    i1 = cv2.imread(f"./collision/{'{:08d}'.format(80)}.jpg", cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(f"./collision/{'{:08d}'.format(82)}.jpg", cv2.IMREAD_GRAYSCALE)
    #plot_imgs(i1, i2, "Collision")

    i1 = cv2.imread(f"./disparity/cporta_left.png", cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(f"./disparity/cporta_right.png", cv2.IMREAD_GRAYSCALE)
    plot_imgs(i1, i2, "Cporta")

    i1 = cv2.imread(f"./disparity/office_left.png", cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(f"./disparity/office_right.png", cv2.IMREAD_GRAYSCALE)
    #plot_imgs(i1, i2, "office1")
    i1 = cv2.imread(f"./disparity/office2_left.png", cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(f"./disparity/office2_right.png", cv2.IMREAD_GRAYSCALE)


    #plot_imgs(i1, i2, "office2")