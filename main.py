import numpy as np
import cv2
import pandas as pd


if __name__ == '__main__':
    cap = cv2.VideoCapture('intersectie2.mp4')
    frames_count, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(frames_count, fps, width, height)

    # pandas dataframe = structura bidimensionala de tip tablou
    # contine o linie pentru fiecare frame al videoclipului
    df = pd.DataFrame(index=range(int(frames_count)))
    df.index.name = "Frames"

    framenumber = 0 # keeps track of current frame
    LLcars = 0 # masini de pe partea stanga (LL = Left Lane)
    MLcars = 0 # masini de pe partea de mijloc
    RLcars = 0 # masini de pe partea dreapta
    carids = [] # id-ul masinilor
    caridcrossed = [] # id-ul masinilor ce au trecut de bariera
    totalcars = 0 # toate masinile ce au trecut de bariera

    # o solutie foarte buna de a detecta obiecte ce se misca pe un background static
    # folosind backgroundSubtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    #
    ret, frame = cap.read() # import image
    ratio = .45 # resize ratio in order to reduce lag
    image = cv2.resize(frame, (0, 0), None, ratio, ratio) # resize image
    width2, height2, channels = image.shape
    video = cv2.VideoWriter('traffic_counter.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (height2, width2),1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converteste imaginea in grayscale
        # equalizedHistogram = cv2.equalizeHist(gray)
        fgmask = fgbg.apply(gray) # aplica background subtractor pentru a distinge obiectele care se misca

        # aplicam diferite praguri pt fgmask pentru a incerca sa izolam masinile

        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel2)  # erosion urmat de dilation
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)  # dilation urmat de erosion

        # pixelii qpot avea valori intre 100-255 sau 0 (pentru a elimina umbrele)
        # retvalbin, bins = cv2.threshold(opening, 220, 255, cv2.THRESH_BINARY)

        cv2.imshow("final", opening)
        cv2.moveWindow('final', 0, 0)
        cv2.imshow("eHistogram", gray)
        cv2.moveWindow("eHistogram",1400 , 0)
        cv2.imshow("mask",fgmask)
        cv2.moveWindow('mask', 0, 1400)
        cv2.imshow("original",image)
        cv2.moveWindow('original', 1400, 600)

        if cv2.waitKey(1) == ord('q') :
            break

        if ret:  # if there is a frame continue with code

            image = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
