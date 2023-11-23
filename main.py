import numpy as np
import cv2
import pandas as pd
from collections import deque

def drawLines(image):
    # DB = Dark Blue
    # LB = light Blue
    cv2.line(image, (100,195),(107,217),(255,0,0),5)
    cv2.line(image, (264,168),(319,163),(255,0,0),5)
    cv2.line(image, (452,200),(497,216),(255,0,0),5)
    cv2.line(image, (348,265),(230,286),(255,0,0),5)
    cv2.line(image, (156, 176), (260, 170), (255, 200, 200), 5)
    cv2.line(image, (402, 173), (447, 196), (255, 200, 200), 5)
    cv2.line(image, (479, 244), (352, 265), (255, 200, 200), 5)
    cv2.line(image, (117, 258), (108, 219), (255, 200, 200), 5)
    cv2.line(image, (155, 177), (323, 163), (0, 0, 0), 5)



if __name__ == '__main__':
    frame_buffer = deque(maxlen=100)
    current_frame_index = 0
    pause = False
    cap = cv2.VideoCapture('intersectie.mp4')
    frames_count, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(frames_count, fps, width, height)

    # pandas dataframe = structura bidimensionala de tip tablou
    # contine o linie pentru fiecare frame al videoclipului
    df = pd.DataFrame(index=range(int(frames_count)))
    df.index.name = "Frames"

    MINAREA = 300
    MAXAREA = 50000
    framenumber = 0  # keeps track of current frame
    LLcars = 0  # masini de pe partea stanga (LL = Left Lane)
    MLcars = 0  # masini de pe partea de mijloc
    RLcars = 0  # masini de pe partea dreapta
    carids = []  # id-ul masinilor
    caridcrossed = []  # id-ul masinilor ce au trecut de bariera
    totalcars = 0  # toate masinile ce au trecut de bariera

    # o solutie foarte buna de a detecta obiecte ce se misca pe un background static
    # folosind backgroundSubtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    ret, frame = cap.read()  # import image
    ratio = .7  # resize ratio in order to reduce lag
    image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
    frame_buffer.append(image)
    width2, height2, channels = image.shape
    video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2),1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    while True:
        if not pause:
            ret, frame = cap.read()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converteste imaginea in grayscale
        # equalizedHistogram = cv2.equalizeHist(gray)
        fgmask = fgbg.apply(gray)  # aplica background subtractor pentru a distinge obiectele care se misca

        # aplicam diferite praguri pt fgmask pentru a incerca sa izolam masinile
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel2)  # erosion urmat de dilation

        # creates contours
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # use convex hull to create a polygon around contours
        hull = [cv2.convexHull(c) for c in contours]

        # draw contours / lines
        cv2.drawContours(image, hull, -1, (0,255,0), 3)
        # drawLines(image)

        cxx = np.zeros(len(contours))
        cyy = np.zeros(len(contours))

        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1:  # conturul trebuie sa fie parinte (nu contur din contur)
                area = cv2.contourArea(contours[i])
                if MINAREA < area < MAXAREA:  # conturul trebuie sa aiba o arie minima si maxima
                    cnt = contours[i]
                    M = cv2.moments(cnt)
                    # stocam x-ul si y-ul conturului
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(image,(x,y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                               .3, (0, 0, 255), 1)
                    cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1, line_type=cv2.LINE_AA)

                    cxx[i] = cx
                    cyy[i] = cy

        cxx = cxx[cxx != 0]  # creaza un nou array selectand elementele care satisfac conditia
        cyy = cyy[cyy != 0]  # practic scapa de elementele cu valoare 0 , raman doar centrele care
                             # raman dupa conditiile de mai sus

        if len(cxx) : # exista masini detectate in frame
            pass

        key = cv2.waitKey(10)
        if key == ord('q'):  # Press 'q' to exit
            break
        elif key == ord(' '):  # Press space to toggle pause
            pause = not pause
            current_frame_index = len(frame_buffer) - 1
        elif key == ord('a'):  # Press '<' to move backward
            if pause:
                current_frame_index -= 1
        elif key == ord('d'):  # Press '>' to move forward
            if pause and current_frame_index + 1 < len(frame_buffer) - 1:
                current_frame_index += 1

        cv2.imshow("finalMask", opening)
        cv2.moveWindow('finalMask', 0, 0)
        cv2.imshow("gray", gray)
        cv2.moveWindow("gray",1400 , 0)
        cv2.imshow("mask",fgmask)
        cv2.moveWindow('mask', 0, 1400)
        if pause:
            cv2.imshow("image",frame_buffer[current_frame_index])
        else:
            cv2.imshow("image", frame_buffer[-1])
        cv2.moveWindow('image', 1400, 600)

        if ret:  # if there is a frame continue with code

            image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
            if not pause:
                frame_buffer.append(image)
