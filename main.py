import numpy as np
import cv2
from collections import deque


def afisare(lista_imagini, text_imagini, cadre_pe_linie=2):
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


def calculPunctMijloc(lista_puncte):
    maximx = lista_puncte[0][0]
    minimx = maximx
    maximy = lista_puncte[0][1]
    minimy = maximy

    for punct in lista_puncte:
        if punct[0] > maximx:
            maximx = punct[0]
        if punct[0] < minimx:
            minimx = punct[0]
        if punct[1] > maximy:
            maximy = punct[1]
        if punct[1] < minimy:
            minimy = punct[1]

    mijlocx = (maximx + minimx) / 2
    mijlocy = (maximy + minimy) / 2
    return mijlocx, mijlocy


def filtruDelimitare(lista_puncte, latime, inaltime):
    mascaCreata = np.zeros((int(latime)+2, int(inaltime)+2), np.uint8)
    imagine_delimitata = np.zeros((int(latime), int(inaltime)), np.uint8)
    if len(lista_puncte) >= 2:  # desenam liniile initiale
        for index_puncte in range(-1, len(lista_puncte) - 1, 1):
            cv2.line(imagine_delimitata, lista_puncte[index_puncte], lista_puncte[index_puncte + 1], (255, 255, 255), 1)
        if len(lista_puncte) >= 3:  # facem fill
            punctMijloc = calculPunctMijloc(lista_puncte)
            # print(punctMijloc)
            # cv2.circle(imagine_delimitata, (int(punctMijloc[0]), int(punctMijloc[1])), 10, (0, 0, 255), 3)
            x_maxim_stanga, x_maxim_dreapta = punctMijloc[0], punctMijloc[1]
            for index_puncte in range(-1, len(lista_puncte)-1, 1):
                if (lista_puncte[index_puncte][1] <= punctMijloc[1] <= lista_puncte[index_puncte + 1][1]) or (
                        lista_puncte[index_puncte][1] >= punctMijloc[1] >= lista_puncte[index_puncte + 1][1]):
                    # daca dreapta poate fi intersectata
                    coordonata_x = ((punctMijloc[1]-lista_puncte[index_puncte][1]) *
                                    (lista_puncte[index_puncte+1][0]-lista_puncte[index_puncte][0])) / \
                                   (lista_puncte[index_puncte+1][1]-lista_puncte[index_puncte][1]) + \
                                   lista_puncte[index_puncte][0]
                    # cv2.circle(imagine_delimitata, (int(coordonata_x), int(punctMijloc[1])), 10, (0, 255, 0), 3)
                    if coordonata_x < x_maxim_stanga:
                        x_maxim_stanga = coordonata_x
                    elif coordonata_x > x_maxim_dreapta:
                        x_maxim_dreapta = coordonata_x
            # cv2.circle(imagine_delimitata, (int(x_maxim_stanga), int(punctMijloc[1])), 10, (255, 0, 0), 3)
            # cv2.circle(imagine_delimitata, (int(x_maxim_dreapta), int(punctMijloc[1])), 10, (255, 0, 0), 3)
            if x_maxim_stanga != punctMijloc[0]:
                cv2.floodFill(imagine_delimitata, mascaCreata, (int(x_maxim_stanga)+5, int(punctMijloc[1])), (255, 255, 255))
            else:
                cv2.floodFill(imagine_delimitata, mascaCreata, (int(x_maxim_dreapta)-5, int(punctMijloc[1])), (255, 255, 255))
    # else:
        # print("Nu am destule puncte!" + str(len(lista_puncte)))
    return imagine_delimitata


def regiuniDelimitare(imagine, lista_puncte):
    if len(lista_puncte) >= 2:  # desenam liniile initiale
        for index_puncte in range(-1, len(lista_puncte) - 1, 1):
            cv2.line(imagine, lista_puncte[index_puncte], lista_puncte[index_puncte + 1], (255, 0, 0), 3)
    return imagine


listaPuncte = list()


def callbackMouse(event, mausx, mausy, flags, param):
    global listaPuncte
    # print(event, mausx, mausy)
    if event == 1:
        listaPuncte.append((mausx, mausy))
        print(listaPuncte)
    if event == 2:
        listaPuncte.clear()


def callbackButonIncarcareMasca(par1, par2):
    print(par1, par2)


def callbackButonSalvareMasca(par1, par2):
    print(par1, par2)


def callbackButonUrmatoareaMasca(par1, par2):
    print(par1, par2)


if __name__ == '__main__':
    frame_buffer = deque(maxlen=100)
    mask_buffer = deque(maxlen=100)
    current_frame_index = 0
    nume_imagine_filtru = 'Desenare_Filtru'
    pause = False
    cap = cv2.VideoCapture('intersectie.mp4')
    cv2.namedWindow(nume_imagine_filtru)
    cv2.setMouseCallback(nume_imagine_filtru, callbackMouse)
    cv2.createButton("Incarcare masca", callbackButonIncarcareMasca, cv2.QT_PUSH_BUTTON)
    cv2.createButton("Salvare masca", callbackButonSalvareMasca, cv2.QT_PUSH_BUTTON)
    cv2.createButton("Urmatoarea masca", callbackButonUrmatoareaMasca, cv2.QT_PUSH_BUTTON)
    events = [i for i in dir(cv2) if 'EVENT' in i]  # printeaza toate evenimentele disponibile
    print(events)

    MINAREA = 300
    MAXAREA = 50000

    # o solutie foarte buna de a detecta obiecte ce se misca pe un background static
    # folosind backgroundSubtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()

    ratio = .4  # resize ratio in order to reduce lag

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    read_frame_index = 0
    while True:
        if not pause:
            ret, frame = cap.read()
            if frame is None and read_frame_index == 0:  # nu a citit frame, probabil path prost de fisier
                print("Nu am putut citi niciun frame!")
                break
            elif frame is None and read_frame_index != 0:  # a ajuns la sfarsit, reseteaza frame-ul
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                read_frame_index = 0
                ret, frame = cap.read()
            else:
                read_frame_index += 1
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
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    .3, (0, 0, 255), 1)
                        cv2.drawMarker(image, (cx, cy), (0, 0, 255), cv2.MARKER_STAR, markerSize=5, thickness=1,
                                       line_type=cv2.LINE_AA)

                        cxx[i] = cx
                        cyy[i] = cy

            cxx = cxx[cxx != 0]  # creaza un nou array selectand elementele care satisfac conditia
            cyy = cyy[cyy != 0]  # practic scapa de elementele cu valoare 0 , raman doar centrele care
            # raman dupa conditiile de mai sus

            if len(cxx):  # exista masini detectate in frame
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
        # aplicare filtru delimitare pe poza originala
        #sursa_temporara_filtru = np.zeros((resizedWidth, resizedHeight, 3), np.uint8)
        imagineDelimitata = filtruDelimitare(listaPuncte, resizedWidth, resizedHeight)
        imagineRegionata = regiuniDelimitare(frame, listaPuncte)
        cv2.imshow(nume_imagine_filtru, imagineRegionata)

        # inceput afisare
        imagini = [frame, mask_buffer[-1], frame_buffer[-1], imagineDelimitata, imagineRegionata]  # lista cu imagini de afisat
        imagini2 = [frame, mask_buffer[current_frame_index], frame_buffer[current_frame_index],
                    imagineDelimitata, imagineRegionata]  # lista cu imagini de afisat
        texte = ["frame "+str(read_frame_index), "finalMask", "image", "imagine delimitata", "imagineRegionata"]  # lista cu numele fiecarei imagini
        numar_de_imagini_pe_linie = 3
        if not pause:
            afisare(imagini, texte, numar_de_imagini_pe_linie)
        else:
            afisare(imagini2, texte, numar_de_imagini_pe_linie)
