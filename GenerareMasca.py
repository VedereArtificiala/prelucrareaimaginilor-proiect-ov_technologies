import numpy as np
import cv2

listaPuncte = list()
numarMasti = 0
modAfisareAlternativ = False
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


def generareMascaDelimitare(lista_puncte, latime, inaltime):
    mascaCreata = np.zeros((int(latime)+2, int(inaltime)+2), np.uint8)
    imagine_delimitata = np.zeros((int(latime), int(inaltime)), np.uint8)
    culoare = (155, 155, 155)
    if len(lista_puncte) >= 2:  # desenam liniile initiale
        for index_puncte in range(-1, len(lista_puncte) - 1, 1):
            cv2.line(imagine_delimitata, lista_puncte[index_puncte], lista_puncte[index_puncte + 1], culoare, 1)
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
                cv2.floodFill(imagine_delimitata, mascaCreata, (int(x_maxim_stanga)+5, int(punctMijloc[1])), culoare)
            else:
                cv2.floodFill(imagine_delimitata, mascaCreata, (int(x_maxim_dreapta)-5, int(punctMijloc[1])), culoare)
    # else:
        # print("Nu am destule puncte!" + str(len(lista_puncte)))
    return imagine_delimitata


def regiuniDelimitare(imagine, lista_puncte):
    imag_aux = imagine.copy()
    if len(lista_puncte) >= 2:  # desenam liniile initiale
        for index_puncte in range(-1, len(lista_puncte) - 1, 1):
            cv2.line(imag_aux, lista_puncte[index_puncte], lista_puncte[index_puncte + 1], (255, 0, 0), 3)
    return imag_aux


def callbackMouse(event, mausx, mausy, flags, param):
    global listaPuncte
    # print(event, mausx, mausy)
    if event == 1:
        listaPuncte.append((mausx, mausy))
        # print(listaPuncte)
    if event == 2:
        listaPuncte.clear()


def callbackButonIncarcareMasca(par1, par2):
    print(par1, par2)


def callbackButonSalvareMasca(par1, par2):
    print(par1, par2)


def callbackButonAfisareAlternativa(par1, par2):
    global modAfisareAlternativ
    modAfisareAlternativ = not modAfisareAlternativ
    # print(modAfisareAlternativ)
    # print(par1, par2)


if __name__ == '__main__':
    current_frame_index = 0
    nume_imagine_filtru = 'Editare masca'
    cv2.namedWindow(nume_imagine_filtru)
    cv2.setMouseCallback(nume_imagine_filtru, callbackMouse)
    cv2.createButton("Incarcare masca", callbackButonIncarcareMasca, cv2.QT_PUSH_BUTTON)
    cv2.createButton("Salvare masca", callbackButonSalvareMasca, cv2.QT_PUSH_BUTTON)
    cv2.createButton("Mod alternativ afisare", callbackButonAfisareAlternativa, cv2.QT_PUSH_BUTTON)
    events = [i for i in dir(cv2) if 'EVENT' in i]  # printeaza toate evenimentele disponibile
    print(events)

    cap = cv2.VideoCapture('intersectie.mp4')
    ret, frame = cap.read()
    if frame is None:  # nu a citit frame, probabil path prost de fisier
        print("Nu am putut citi niciun frame!")
    latime, inaltime, canale = frame.shape

    while True:
        key = cv2.waitKey(16)
        if key == ord('q'):  # Press 'q' to exit
            break
        # aplicare filtru delimitare pe poza originala
        MascaDelimitare = generareMascaDelimitare(listaPuncte, latime, inaltime)
        imagineRegionata = regiuniDelimitare(frame, listaPuncte)
        cv2.putText(imagineRegionata, "Regiunea selectata: ", (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (250, 50, 250), 1, cv2.LINE_AA)
        cv2.putText(imagineRegionata, "Numar regiuni: ", (5, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (250, 50, 250), 1, cv2.LINE_AA)

        if modAfisareAlternativ is True:
            cv2.imshow(nume_imagine_filtru, MascaDelimitare)
        else:
            cv2.imshow(nume_imagine_filtru, imagineRegionata)
