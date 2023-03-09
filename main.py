import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import ex1_utils as utils
import matplotlib.image as img

def lucaskanade(im1, im2, N, harris=False):
    # im1 − f i r s t image mat rix ( g r a y s c a l e )
    # im2 − sec ond image mat rix ( g r a y s c a l e )
    # n − s i z e o f the nei ghb o rh o od (N x N)
    # TODO : the a l g o ri t h m
    im1 = im1.astype(np.float32) / np.max(im1)
    im2 = im2.astype(np.float32) /np.max(im2)
    sigma = 1
    im1 = utils.gausssmooth(im1, sigma)
    im2 = utils.gausssmooth(im2, sigma)

    # dt = dt * (dt > 0)
    dt = im2 - im1
    dx, dy = utils.gaussderiv(im1, sigma)
    neighbourhood_size = N
    #TODO PYRAMIDAL
    if harris:
        #TODO HARRIS
        corner = cv2.cornerHarris(im1, 7, 3, 0.05)
        plt.imshow(corner)
        plt.show()
        corner = cv2.cornerHarris(im2, 7, 3, 0.05)
        plt.imshow(corner)
        plt.show()
        corner = cv2.cornerHarris(dt, 7, 3, 0.05)
        plt.imshow(corner)
        plt.show()

        corner = cv2.cornerHarris(dx, 7, 3, 0.05)
        plt.imshow(corner)
        plt.show()
        corner = cv2.cornerHarris(dy, 7, 3, 0.05)
        plt.imshow(corner)
        plt.show()

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


def hornschunck(im1, im2, n_iters, lmbd, initial=False, epsilon=None):
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
    im1 = utils.gausssmooth(im1, sigma)
    im2 = utils.gausssmooth(im2, sigma)
    # dt = dt * (dt > 0)
    dt = im2 - im1
    #dt = cv2.filter2D(im2 - im1, -1, np.ones((2,2)) / 4.0)
    dx, dy = utils.gaussderiv(im1, sigma)
    if not initial:
        u = np.zeros(im1.shape)
        v = np.zeros(im1.shape)
    else:
        u, v = lucaskanade(im1, im2, 21)
    ld = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    if epsilon is None:
        epsilon = i1.shape[0] * i1.shape[1] / (8000 * 2)
    D = lmbd + dx ** 2 + dy ** 2
    for i in range(n_iters):
        ua = cv2.filter2D(u, -1, ld)
        va = cv2.filter2D(v, -1, ld)
        P = dx * ua + dy * va + dt
        un = ua - np.divide(dx * P, D, out=np.zeros_like(D), where=D != 0)
        vn = va - np.divide(dy * P, D, out=np.zeros_like(D), where=D != 0)
        #print(np.abs(un - u).sum() + np.abs(vn - v).sum())
        if (np.abs(un - u).sum() + np.abs(vn - v).sum()) < epsilon:
            print(i, "Iterations before convergence to epsilon", epsilon)
            break
        u = un
        v = vn
    return u, v

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def square_subplots(fig, ax):
    rows, cols = ax[0,0].get_subplotspec().get_gridspec().get_geometry()
    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom
    wspace = fig.subplotpars.wspace
    hspace = fig.subplotpars.hspace
    figw,figh = fig.get_size_inches()

    axw = figw*(r-l)/(cols+(cols-1)*wspace)
    axh = figh*(t-b)/(rows+(rows-1)*hspace)
    axs = min(axw,axh)
    w = (1-axs/figw*(cols+(cols-1)*wspace))/2.
    h = (1-axs/figh*(rows+(rows-1)*hspace))/2.
    fig.subplots_adjust(bottom=h, top=1-h, left=w, right=1-w)

def plot_imgs(i1, i2):

    #fig, ax = plt.subplots(1, 2)
    #ax[0].imshow(i1)
    #ax[1].imshow(test)
    #plt.show()
    neigh = 21
    u, v = lucaskanade(i1, i2, neigh)
    fig, ax = plt.subplots(3,2, figsize=(6,9))
    utils.show_flow(u, v, ax[1,0])
    utils.show_flow(u, v, ax[2,0], type="angle_magnitude")

    u, v = hornschunck(i1, i2, 1000, 100, initial=True)
    utils.show_flow(u, v, ax[1, 1])
    utils.show_flow(u, v, ax[2, 1], type="angle_magnitude")

    ax[0,0].imshow(i1)
    ax[0,1].imshow(i2)
    square_subplots(fig, ax)
    plt.show()

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

if __name__ == "__main__":


    i1 = np.random.randint(0, 255, (200, 200)).astype(np.uint8)
    i1 = (i1.astype(np.float32) / np.max(i1))
    i1[:50, 180:] = 1
    test = rotate_image(i1, 10)
    plot_imgs(i1, test)



    i0 = cv2.imread("./white_noise.png", cv2.IMREAD_GRAYSCALE)
    i0[100,:] = 0
    i0[:,80] = 0
    i1 = i0[:197,:197]
    i1 = i1.astype(np.float32)
    m = i1.max()
    i1 = i1/m
    noise2 = np.atleast_2d(np.tile(np.array((0,1)), 15))
    noise2 = noise2.T.dot(noise2)
    i1[45:75, 45:75] = noise2

    i2 = i0[3:,3:]

    i2 = i2.astype(np.float32)
    i2 = i2/m
    i2[47:77, 47:77] = noise2

    plot_imgs(i1, i2)


    i1 = np.random.randint(0, 255, (200, 200)).astype(np.uint8)
    i1 = (i1.astype(np.float32) / np.max(i1))
    test = rotate_image(i1, 1)
    plot_imgs(i1, test)

    i1 = cv2.imread(f"./collision/{'{:08d}'.format(1)}.jpg", cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(f"./collision/{'{:08d}'.format(2)}.jpg", cv2.IMREAD_GRAYSCALE)
    plot_imgs(i1,i2)

    i1 = cv2.imread(f"./disparity/office_left.png", cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(f"./disparity/office_right.png", cv2.IMREAD_GRAYSCALE)
    plot_imgs(i1,i2)

    i1 = np.ones((200, 200)) * ((np.sin((np.arange(0, 100, 0.5))) + 1) / 2)
    i1 = i1 / i1.max()
    print(i1)
    test = rotate_image(i1, 1)
    plot_imgs(i1, test)