import numpy as np
import cv2
#import pandas as pd
from collections import deque

def drawLines(image):
    # DB = Dark Blue
    # LB = light Blue-
    cv2.line(image, (100,195),(107,217),(255,0,0),5)
    cv2.line(image, (264,168),(319,163),(255,0,0),5)
    cv2.line(image, (452,200),(497,216),(255,0,0),5)
    cv2.line(image, (348,265),(230,286),(255,0,0),5)
    cv2.line(image, (156, 176), (260, 170), (255, 200, 200), 5)
    cv2.line(image, (402, 173), (447, 196), (255, 200, 200), 5)
    cv2.line(image, (479, 244), (352, 265), (255, 200, 200), 5)
    cv2.line(image, (117, 258), (108, 219), (255, 200, 200), 5)
    cv2.line(image, (155, 177), (323, 163), (0, 0, 0), 5)

def afisare(lista_imagini, text_imagini, cadre_pe_linie = 2):
    nr_cadre = len(lista_imagini)
    nr_linii = int(np.ceil(nr_cadre / cadre_pe_linie))
    nr_coloane = cadre_pe_linie
    randuri_poze = []
    *dimensiuni_poza, = lista_imagini[0].shape
    latime = dimensiuni_poza[0]
    inaltime = dimensiuni_poza[1]
    for i in range(nr_linii):
        linie_poze = []
        for j in range(nr_coloane):
            index_lista = i * nr_coloane + j
            if index_lista < nr_cadre:
                cv2.putText(lista_imagini[index_lista], text_imagini[index_lista], (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if len(lista_imagini[index_lista].shape) < 3:
                    color = np.zeros((int(latime), int(inaltime), 3), np.uint8)
                    cv2.cvtColor(lista_imagini[index_lista], cv2.COLOR_GRAY2RGB, color)
                    linie_poze.append(color)
                else:
                    linie_poze.append(lista_imagini[index_lista])
            else:
                linie_poze.append(np.zeros((int(latime), int(inaltime), 3), np.uint8))
        randuri_poze.append(cv2.hconcat(linie_poze))
    final = cv2.vconcat(randuri_poze)

    cv2.imshow('video', final)

def filtruDelimitare(listaPuncte, latime, inaltime):
    imagineDelimitata = np.zeros((int(latime), int(inaltime), 3), np.uint8)
    if len(listaPuncte) >= 3:   #nu scimba la 2 ca il capitulezi
        punctInceput = listaPuncte[len(listaPuncte)-1]
        punctSfarsit = listaPuncte[0]
        index_puncte = 0
        cv2.line(imagineDelimitata, punctInceput, punctSfarsit, (255, 0, 0), 1)
        while index_puncte < len(listaPuncte)-1:
            punctInceput = listaPuncte[index_puncte]
            punctSfarsit = listaPuncte[index_puncte+1]
            cv2.line(imagineDelimitata, punctInceput, punctSfarsit, (255,0,0), 1)
            index_puncte+=1
    else:
        print("Nu am destule puncte!"+str(len(listaPuncte)))
    return imagineDelimitata

if __name__ == '__main__':
    frame_buffer = deque(maxlen=100)
    mask_buffer = deque(maxlen=100)
    current_frame_index = 0
    pause = False
    cap = cv2.VideoCapture('intersectie.mp4')
    #frames_count, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    #width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #print(frames_count, fps, width, height)

    # pandas dataframe = structura bidimensionala de tip tablou
    # contine o linie pentru fiecare frame al videoclipului
    #df = pd.DataFrame(index=range(int(frames_count)))
    #df.index.name = "Frames"

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

    ratio = .4  # resize ratio in order to reduce lag

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    while True:
        if not pause:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (0, 0), None, ratio, ratio)  # resize image
            resizedWidth, resizedHeight, channels = frame.shape
            image = frame.copy()
            frame_buffer.append(image)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converteste imaginea in grayscale
            # equalizedHistogram = cv2.equalizeHist(gray)
            fgmask = fgbg.apply(gray)  # aplica background subtractor pentru a distinge obiectele care se misca

            # aplicam diferite praguri pt fgmask pentru a incerca sa izolam masinile
            opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel2)  # erosion urmat de dilation
            mask_buffer.append(opening)

            # creates contours
            contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # use convex hull to create a polygon around contours
            hull = [cv2.convexHull(c) for c in contours]

            # draw contours / lines
            cv2.drawContours(image, hull, -1, (0, 255, 0), 3)
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

        #aplicare filtru delimitare pe poza originala
        listaPuncte = ((100,100),(100,200),(200,200),(200,100),(300,150))
        imagineDelimitata = filtruDelimitare(listaPuncte, resizedWidth, resizedHeight)
        cv2.imshow("COX", imagineDelimitata)
        #inceput afisare
        imagini = [ frame, mask_buffer[-1], frame_buffer[-1], imagineDelimitata]  # lista cu imagini de afisat
        imagini2 = [ frame, mask_buffer[current_frame_index], frame_buffer[current_frame_index], imagineDelimitata]  # lista cu imagini de afisat
        texte = ["original", "finalMask", "image", "imagine delimitata"]  # lista cu numele fiecarei imagini
        numar_de_imagini_pe_linie = 2
        if not pause:
            afisare(imagini, texte, numar_de_imagini_pe_linie)
        else:
            afisare(imagini2, texte, numar_de_imagini_pe_linie)
