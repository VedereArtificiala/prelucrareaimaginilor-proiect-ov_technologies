import numpy as np
import cv2

listaPuncte = list()
listaRegiuniDelimitare = list()
numarRegiuniSalvate = 0
index_regiune_curenta = 0
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


def generareMascaDelimitare(imagine_sursa, lista_puncte, latime, inaltime, numarRegiune):
    mascaCreata = np.zeros((int(latime)+2, int(inaltime)+2), np.uint8)
    # imagine_delimitata = np.zeros((int(latime), int(inaltime)), np.uint8)
    culoare = (50 + numarRegiune * 10, 50 + numarRegiune * 10, 50 + numarRegiune * 10)
    if len(lista_puncte) >= 2:  # desenam liniile initiale
        for index_puncte in range(-1, len(lista_puncte) - 1, 1):
            cv2.line(imagine_sursa, lista_puncte[index_puncte], lista_puncte[index_puncte + 1], culoare, 1)
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
                cv2.floodFill(imagine_sursa, mascaCreata, (int(x_maxim_stanga)+5, int(punctMijloc[1])), culoare)
            else:
                cv2.floodFill(imagine_sursa, mascaCreata, (int(x_maxim_dreapta)-5, int(punctMijloc[1])), culoare)
    # else:
        # print("Nu am destule puncte!" + str(len(lista_puncte)))
    return imagine_sursa


def regiuniDelimitare(imagine, lista_puncte, culoare):
    imag_aux = imagine.copy()
    if len(lista_puncte) >= 2:  # desenam liniile initiale
        for index_puncte in range(-1, len(lista_puncte) - 1, 1):
            cv2.line(imag_aux, lista_puncte[index_puncte], lista_puncte[index_puncte + 1], culoare, 3)
    return imag_aux


def scriereMascaFisier():
    global listaRegiuniDelimitare
    fisier = open("puncteMasca.txt", "w")
    for listaPuncteMasca in listaRegiuniDelimitare:
        for punct in listaPuncteMasca:
            fisier.write(str(punct[0])+" "+str(punct[1])+" ")
        fisier.write("\n")
    fisier.close()


def citireMascaFisier():
    global listaRegiuniDelimitare, numarRegiuniSalvate, index_regiune_curenta
    numarRegiuniSalvate = 0
    index_regiune_curenta = 0
    listaRegiuniDelimitare = list()
    fisier = open("puncteMasca.txt", "r")
    for linie in fisier:
        listaNumereString = linie.split()
        listaPunctee = list()
        for i in range(0, len(listaNumereString), 2):
            # print(listaNumereString[i], listaNumereString[i+1])
            listaPunctee.append((int(listaNumereString[i]), int(listaNumereString[i+1])))
        listaRegiuniDelimitare.append(listaPunctee)
        numarRegiuniSalvate += 1
    index_regiune_curenta = numarRegiuniSalvate
    fisier.close()


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
    citireMascaFisier()


def callbackButonSalvareMasca(par1, par2):
    print(par1, par2)
    scriereMascaFisier()


def callbackButonAfisareAlternativa(par1, par2):
    global modAfisareAlternativ
    modAfisareAlternativ = not modAfisareAlternativ
    # print(modAfisareAlternativ)
    # print(par1, par2)


if __name__ == '__main__':
    current_frame_index = 0
    nume_imagine_filtru = 'Editare masca'
    cv2.namedWindow(nume_imagine_filtru, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(nume_imagine_filtru,cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
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
        if key == ord('w'): # w sa trec la urmatoarea regiune
            if len(listaPuncte) >= 3:
                listaRegiuniDelimitare.append(list(listaPuncte))
                numarRegiuniSalvate += 1
                index_regiune_curenta = numarRegiuniSalvate
                listaPuncte.clear()
            else:
                print("nu sunt destule puncte pentru a salva regiunea!")

        # aplicare filtru delimitare pe poza originala
        MascaDelimitare = np.zeros((int(latime), int(inaltime)), np.uint8)
        imagineRegionata = frame.copy()
        for puncteRegiune in listaRegiuniDelimitare:
            imagineRegionata = regiuniDelimitare(imagineRegionata, puncteRegiune, (0, 255, 0))  # regiunile salvate deja
            MascaDelimitare = generareMascaDelimitare(MascaDelimitare, puncteRegiune, latime, inaltime,
                                                      listaRegiuniDelimitare.index(puncteRegiune))
        MascaDelimitare = generareMascaDelimitare(MascaDelimitare, listaPuncte, latime, inaltime, numarRegiuniSalvate)
        imagineRegionata = regiuniDelimitare(imagineRegionata, listaPuncte, (0, 0, 255))  # regiunea inca in lucru

        cv2.putText(imagineRegionata, "Regiunea selectata: "+str(index_regiune_curenta), (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (50, 250, 50), 2, cv2.LINE_AA)
        cv2.putText(imagineRegionata, "Numar regiuni salvate: " + str(numarRegiuniSalvate), (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 250, 50), 2, cv2.LINE_AA)

        if modAfisareAlternativ is True:
            cv2.imshow(nume_imagine_filtru, MascaDelimitare)
        else:
            cv2.imshow(nume_imagine_filtru, imagineRegionata)
