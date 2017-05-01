import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2

# Read points from text file
def readPoints(path) :
    # Create an array of points.
    points = [];
    
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True


#calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect);
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p) 
    
    triangleList = subdiv.getTriangleList();
    
    delaunayTri = []
    
    pt = []    
    
    count= 0    
    
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            count = count + 1 
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    
    return delaunayTri
        

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 
    


def poseEstimate(size, landmarks):
    image_points = np.array([
            (landmarks[33, 0], landmarks[33, 1]),     # Nose tip
            (landmarks[8, 0], landmarks[8, 1]),       # Chin
            (landmarks[36, 0], landmarks[36, 1]),     # Left eye left corner
            (landmarks[45, 0], landmarks[45, 1]),     # Right eye right corner
            (landmarks[48, 0], landmarks[48, 1]),     # Left Mouth corner
            (landmarks[54, 0], landmarks[54, 1])      # Right mouth corner
        ], dtype="double")

    model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corne
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype = "double")

    dist_coeffs = np.zeros((4,1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    return (p1, p2)


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out1 = cv2.VideoWriter('output1.avi',fourcc, 7.0, (1280, 720))
out = cv2.VideoWriter('output.avi',fourcc, 7.0, (600, 800))

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
filename2 = 'hillary_clinton.jpg'
img2 = cv2.imread(filename2);
points2 = readPoints(filename2 + '.txt')    
hullIndex = cv2.convexHull(np.array(points2), returnPoints = False)


hull2 = []      
for i in range(0, len(hullIndex)):
    hull2.append(points2[hullIndex[i][0]])

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dets = detector(frame, 0)	
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        # landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        # p1, p2 = poseEstimate(frame.shape, landmarks)
        # cv2.line(frame, p1, p2, (255, 0, 0))
        img1 = frame
        img1Warped = np.copy(img2);    
        
        # Read array of corresponding points
        points1 = [[p.x, p.y] for p in shape.parts()]
        
        # Find convex hull
        hull1 = []              
        for i in range(0, len(hullIndex)):
            hull1.append(points1[hullIndex[i][0]])
        
        
        # Find delanauy traingulation for convex hull points
        sizeImg2 = img2.shape    
        rect = (0, 0, sizeImg2[1], sizeImg2[0])
         
        dt = calculateDelaunayTriangles(rect, hull2)
        
        if len(dt) == 0:
            quit()
        
        # Apply affine transformation to Delaunay triangles
        for i in range(0, len(dt)):
            t1 = []
            t2 = []
            
            #get points for img1, img2 corresponding to the triangles
            for j in range(0, 3):
                t1.append(hull1[dt[i][j]])
                t2.append(hull2[dt[i][j]])
            
            warpTriangle(img1, img1Warped, t1, t2)
        
                
        # Calculate Mask
        hull8U = []
        for i in range(0, len(hull2)):
            hull8U.append((hull2[i][0], hull2[i][1]))
        
        mask = np.zeros(img2.shape, dtype = img2.dtype)  
        
        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
        
        r = cv2.boundingRect(np.float32([hull2]))    
        
        center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
        
        # Clone seamlessly.
        output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
        for num in range(shape.num_parts):
            cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), -1)	


        cv2.imshow('frame', frame)
        out1.write(frame)
        if (output.shape[0] == 600 or output.shape[0] == 800 ):
            out.write(output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        break


cap.release()
out.release()

cv2.destroyAllWindows()



