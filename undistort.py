import numpy as np
import cv2
import os
import argparse
import yaml
import pickle
from glob import glob
import json
import math
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate camera using a video of a chessboard or a sequence of images.')
    parser.add_argument('input', help='input video file or glob mask')
    #parser.add_argument('out', help='output calibration yaml file')
    parser.add_argument('--debug-dir', help='path to directory where images with detected chessboard will be written',
                        default=None)
    parser.add_argument('-c', '--corners', help='output corners file', default=None)
    parser.add_argument('-fs', '--framestep', help='use every nth frame in the video', default=20, type=int)
    parser.add_argument('--figure', help='saved visualization name', default=None)
    args = parser.parse_args()

    if '*' in args.input:
        source = glob(args.input)
    else:
        source = cv2.VideoCapture(args.input)
    # square_size = float(args.get('--square_size', 1.0))

    #pattern_size = (9, 6)

    pattern_size = (6, 10)

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) #90 mm sqaure size
    # pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    i = -1
    while True:
        i += 1
        if isinstance(source, list):
            # glob
            if i == len(source):
                break
            img = cv2.imread(source[i])
        else:
            # cv2.VideoCapture
            retval, img = source.read()
            if not retval:
                break
            if i % args.framestep != 0:
                continue

        print('Searching for chessboard in frame ' + str(i) + '...'),
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size, cv2.CALIB_CB_FILTER_QUADS)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        if args.debug_dir:
            img_chess = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img_chess, pattern_size, corners, found)
            cv2.imwrite(os.path.join(args.debug_dir, '%04d.png' % i), img_chess)
        if not found:
            print ('not found')
            continue
        img_points.append(corners.reshape(1, -1, 2))
        obj_points.append(pattern_points.reshape(1, -1, 3))

        print ('ok')



    if args.corners:
        with open(args.corners, 'wb') as fw:
            pickle.dump(img_points, fw)
            pickle.dump(obj_points, fw)
            pickle.dump((w, h), fw)



# load corners
#    with open('corners.pkl', 'rb') as fr:
#        img_points = pickle.load(fr)
#        obj_points = pickle.load(fr)
#        w, h = pickle.load(fr)

    print('\nPerforming calibration...')
    rms, camera_matrix, dist_coefs, rvec, tvec = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    print ("RMS:"), rms
    #print ("this is ojpoints", obj_points)
    #print ("img points", img_points)
    print ("camera matrix:\n", camera_matrix)
    print ("distortion coefficients: ", dist_coefs.ravel())



    # # fisheye calibration
    # rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.fisheye.calibrate(
    #     obj_points, img_points,
    #     (w, h), camera_matrix, np.array([0., 0., 0., 0.]),
    #     None, None,
    #     cv2.fisheye.CALIB_USE_INTRINSIC_GUESS, (3, 1, 1e-6))
    # print "RMS:", rms
    # print "camera matrix:\n", camera_matrix
    # print "distortion coefficients: ", dist_coefs.ravel()

    #print ("3d world real points", obj_points)

    calibration = {'rms': rms, 'camera_matrix': camera_matrix.tolist(), 'dist_coefs': dist_coefs.tolist() }
    print ("rms vale", rms)
    with open("vid1.yaml", 'w') as out_file:
        yaml.dump(calibration,
                  out_file)



def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())


    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    corner = tuple(corners[6].ravel())
    img = cv2.circle(img, corner, 1, (255,0,0),5)

    corner = tuple(corners[59].ravel())
    img = cv2.circle(img, corner, 1, (255,0,0),5)
    return img

img = cv2.imread('/Users/rajatgirhotra/Desktop/output/n6.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix,  dist_coefs, (w,h), 0, (w,h))

dst = cv2.undistort(img, camera_matrix,  dist_coefs, None, newcameramtx)

cv2.imwrite("undistort.png",dst)

#undistorting video

def log_it(message):
    """Print message to sys.stdout, without new line."""
    sys.stdout.write(message + "\r")
    sys.stdout.flush()


FILENAME_IN = "/Users/rajatgirhotra/Desktop/second.mkv"
FILENAME_OUT = "new1.mkv"
CODEC = 'mp4v'

video = cv2.VideoCapture(FILENAME_IN)
fourcc = cv2.VideoWriter_fourcc(*list(CODEC))
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
writer = cv2.VideoWriter(FILENAME_OUT, fourcc, fps, size)
while video.grab() is True:
    log_it("On frame %i of %i."%(video.get(cv2.CAP_PROP_POS_FRAMES),
                                frame_count))
    frame =  cv2.undistort(video.retrieve()[1], camera_matrix, dist_coefs, None, newcameramtx)
    writer.write(frame)
video.release()
writer.release()




criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*10,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:10].T.reshape(-1,2) #90 mm sqaure size
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)


#file location for extrinsic
img = cv2.imread("/Users/rajatgirhotra/desktop/s.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (6,10),None)
if ret == True:
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.


    ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coefs,None, None, False, cv2.SOLVEPNP_ITERATIVE)

    #print ('these are object point', objp)



    #print("rvecs pnp", rvecs)
    #print("tvecs pnp", tvecs)
    # project 3D points to image plane

    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coefs)

    #getting rotation matrix

    rmat = cv2.Rodrigues(rvecs)[0]
    print("this is rotation matrix from pnp ", rmat)

    print ("this is tvecs", tvecs)


    #sovlve pnpransac

    _, rvecss, tvecss, inliers = cv2.solvePnPRansac(objp, corners2, camera_matrix, dist_coefs)
    print("rvecss", rvecss)
    print("tvecss", tvecss)

    tarray = np.array(tvecss).T

    print ('tvecss', tarray)

    dis = cv2.Rodrigues(rvecs)[0]

    #printing rotation matrxi name dis with PnPRansac
    print ('this is rotation matrix from ransac', dis)

    x = tvecss[0][0]
    y = tvecss[2][0]
    t = (math.asin(-dis[0][2]))

    print ("X", x, "Y", y, "Angle", t)
    print("90-t",(math.pi/2)-t)

    rx = y * (math.cos((math.pi/2)-t))
    ry = y * (math.sin((math.pi/2) -t))

    print ("rx", rx,"ry",ry)

    #checking cam position

    uvPoint = np.matrix( [[498],[406],[1]] )
    print (uvPoint)
    inv1 = np.linalg.inv(dis)
    inv2 = np.linalg.inv(camera_matrix)
    mid = np.dot(inv1,inv2)

    left = np.dot(mid,uvPoint)

    inv3 = np.matrix(tvecs)

    right = np.dot(inv1,inv3)

    print ("this is right", right)



    s = (1 + left[2,0]/right[2,0])

    print ("this is s", s)

    d = np.dot(s,inv1)
    e = np.dot(d,uvPoint)
    print ('this is e', e)
    f = e - tvecs

    P = np.dot(inv1,f)

    print('this is p', P)
