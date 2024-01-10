import numpy as np
import cv2

listaPuncte = list()
listaRegiuniDelimitare = list()
numarRegiuniSalvate = 0
index_regiune_curenta = 0
modAfisareAlternativ = False
addedBrightness = 0
cursor = (0, 0)
# TODO salvare masti de intrare separat
# TODO reparare masca curenta

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


def listaNumereMasca(pixel_masca):
    listaNumere = list()
    for index_culoare in range(0, len(pixel_masca)):  # trece prin cele 3 culori ale pixelului GBR
        for i in range(0, 8):
            if pixel_masca[index_culoare] & (1 << i) != 0:
                listaNumere.append(i+(len(pixel_masca)-index_culoare-1)*8)
    return listaNumere


def generareMascaDelimitare(lista_puncte, latime, inaltime, numarRegiune):  # 0 < numarRegiune < 23
    imagine_sursa = np.zeros((int(latime), int(inaltime), 3), np.uint8)
    # mascaCreata = np.zeros((int(latime)+2, int(inaltime)+2), np.uint8)
    # print(1 << numarRegiune)
    if numarRegiune > 23:
        print("numar de regiune prea mare (>23), masca nu e corecta!")
    numarMasca = 1 << numarRegiune
    # print(format((numarMasca >> 16) % 256, '08b')+" "+format((numarMasca >> 8) % 256, '08b')+" "+format(numarMasca % 256, '08b'))
    culoare = ((numarMasca >> 16) % 256, (numarMasca >> 8) % 256, numarMasca % 256)
    if len(lista_puncte) >= 2:  # desenam liniile initiale
        for index_puncte in range(-1, len(lista_puncte) - 1, 1):
            cv2.line(imagine_sursa, lista_puncte[index_puncte], lista_puncte[index_puncte + 1], culoare, 1)
        if len(lista_puncte) >= 3:  # facem fill
            punctMijloc = calculPunctMijloc(lista_puncte)
            # print(punctMijloc)
            # cv2.circle(imagine_delimitata, (int(punctMijloc[0]), int(punctMijloc[1])), 10, (0, 0, 255), 3)
            x_maxim_stanga, x_maxim_stanga_2, x_maxim_dreapta, x_maxim_dreapta_2 = punctMijloc[0], punctMijloc[0], \
                punctMijloc[0], punctMijloc[0]
            for index_puncte in range(-1, len(lista_puncte)-1, 1):
                if (lista_puncte[index_puncte][1] <= punctMijloc[1] <= lista_puncte[index_puncte + 1][1]) or (
                        lista_puncte[index_puncte][1] >= punctMijloc[1] >= lista_puncte[index_puncte + 1][1]):
                    # daca dreapta poate fi intersectata
                    coordonata_x = ((punctMijloc[1]-lista_puncte[index_puncte][1]) *
                                    (lista_puncte[index_puncte+1][0]-lista_puncte[index_puncte][0])) / \
                                   (lista_puncte[index_puncte+1][1]-lista_puncte[index_puncte][1]) + \
                                   lista_puncte[index_puncte][0]
                    # cv2.circle(imagine_sursa, (int(coordonata_x), int(punctMijloc[1])), 10, (0, 255, 0), 3)
                    if coordonata_x < x_maxim_stanga:
                        x_maxim_stanga = coordonata_x
                    elif coordonata_x > x_maxim_dreapta:
                        x_maxim_dreapta = coordonata_x
            for index_puncte in range(-1, len(lista_puncte)-1, 1):
                if (lista_puncte[index_puncte][1] <= punctMijloc[1] <= lista_puncte[index_puncte + 1][1]) or (
                        lista_puncte[index_puncte][1] >= punctMijloc[1] >= lista_puncte[index_puncte + 1][1]):
                    # daca dreapta poate fi intersectata
                    coordonata_x = ((punctMijloc[1]-lista_puncte[index_puncte][1]) *
                                    (lista_puncte[index_puncte+1][0]-lista_puncte[index_puncte][0])) / \
                                   (lista_puncte[index_puncte+1][1]-lista_puncte[index_puncte][1]) + \
                                   lista_puncte[index_puncte][0]
                    # cv2.circle(imagine_sursa, (int(coordonata_x), int(punctMijloc[1])), 10, (0, 255, 0), 3)
                    if x_maxim_stanga_2 > coordonata_x > x_maxim_stanga:
                        x_maxim_stanga_2 = coordonata_x
                    elif x_maxim_dreapta_2 < coordonata_x < x_maxim_dreapta:
                        x_maxim_dreapta_2 = coordonata_x

            # cv2.circle(imagine_sursa, (int(x_maxim_stanga), int(punctMijloc[1])), 10, (255, 0, 0), 3)
            # cv2.circle(imagine_sursa, (int(x_maxim_dreapta), int(punctMijloc[1])), 10, (255, 0, 0), 3)
            # cv2.circle(imagine_sursa, (int(x_maxim_stanga_2), int(punctMijloc[1])), 10, (255, 0, 0), 3)
            # cv2.circle(imagine_sursa, (int(x_maxim_dreapta_2), int(punctMijloc[1])), 10, (255, 0, 0), 3)
            # cv2.circle(imagine_sursa, (int((x_maxim_stanga + x_maxim_stanga_2) / 2), int(punctMijloc[1])), 10, (255, 0, 0), 3)
            # cv2.circle(imagine_sursa, (int((x_maxim_dreapta + x_maxim_dreapta_2) / 2), int(punctMijloc[1])), 10, (255, 0, 0), 3)
            if x_maxim_stanga != punctMijloc[0]:
                cv2.floodFill(imagine_sursa, None, (int((x_maxim_stanga + x_maxim_stanga_2) / 2), int(punctMijloc[1])), culoare)
            else:
                cv2.floodFill(imagine_sursa, None, (int((x_maxim_dreapta + x_maxim_dreapta_2) / 2), int(punctMijloc[1])), culoare)
    # else:
        # print("Nu am destule puncte!" + str(len(lista_puncte)))
    return imagine_sursa


def regiuniDelimitare(imagine, lista_puncte, culoare, grosime):
    imag_aux = imagine.copy()
    if len(lista_puncte) >= 2:  # desenam liniile initiale
        for index_puncte in range(-1, len(lista_puncte) - 1, 1):
            cv2.line(imag_aux, lista_puncte[index_puncte], lista_puncte[index_puncte + 1], culoare, grosime)
    return imag_aux


def scrierePuncteFisier():
    global listaRegiuniDelimitare
    fisier = open("puncteMasca.txt", "w")
    for listaPuncteMasca in listaRegiuniDelimitare:
        for punct in listaPuncteMasca:
            fisier.write(str(punct[0])+" "+str(punct[1])+" ")
        fisier.write("\n")
    fisier.close()


def citirePuncteFisier():
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


def generareMascaFisier():
    citirePuncteFisier()
    listaMastiGenerate = list()
    latime = 1080  # le am pe dos in tot proiectul cred, aia e
    inaltime = 1920
    for i in range(0, len(listaRegiuniDelimitare)):
        listaMastiGenerate.append(generareMascaDelimitare(listaRegiuniDelimitare[i], latime, inaltime, listaRegiuniDelimitare.index(listaRegiuniDelimitare[i])))
    mascaFinala = np.zeros((int(latime), int(inaltime), 3), np.uint8)
    for imagine in listaMastiGenerate:
        mascaFinala += imagine
    return mascaFinala


def callbackMouse(event, mausx, mausy, flags, param):
    global listaPuncte, cursor
    # print(event, mausx, mausy)
    if event == 1:
        listaPuncte.append((mausx, mausy))
        # print(listaPuncte)
    if event == 3:
        listaPuncte.clear()
    cursor = (mausx, mausy)


def numarMastiIncarcate():
    return len(listaRegiuniDelimitare)

def callbackButonIncarcareMasca(par1, par2):
    # print(par1, par2)
    citirePuncteFisier()


def callbackButonSalvareMasca(par1, par2):
    # print(par1, par2)
    scrierePuncteFisier()


def callbackButonAfisareAlternativa(par1, par2):
    global modAfisareAlternativ
    modAfisareAlternativ = not modAfisareAlternativ
    # print(modAfisareAlternativ)
    # print(par1, par2)


def callbackTrackbarBrightness(par1):
    global addedBrightness
    addedBrightness = par1


if __name__ == '__main__':
    current_frame_index = 0
    nume_imagine_filtru = 'Editare masca'
    cv2.namedWindow(nume_imagine_filtru, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(nume_imagine_filtru,cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(nume_imagine_filtru, callbackMouse)
    cv2.createButton("Incarcare masca", callbackButonIncarcareMasca, cv2.QT_PUSH_BUTTON)
    cv2.createButton("Salvare masca", callbackButonSalvareMasca, cv2.QT_PUSH_BUTTON)
    cv2.createButton("Mod alternativ afisare", callbackButonAfisareAlternativa, cv2.QT_PUSH_BUTTON)
    cv2.createTrackbar("Brightness boost", "", 0, 50, callbackTrackbarBrightness)
    events = [i for i in dir(cv2) if 'EVENT' in i]  # printeaza toate evenimentele disponibile
    print(events)

    cap = cv2.VideoCapture('intersectie.mp4')
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), None, 1.5, 1.5)
    if frame is None:  # nu a citit frame, probabil path prost de fisier
        print("Nu am putut citi niciun frame!")
    latime, inaltime, canale = frame.shape
    print(latime, inaltime)

    mascaGenerata = generareMascaFisier()
    # cv2.imshow("testGenerareMasca", mascaGenerata)
    while True:
        key = cv2.waitKey(16)
        if key == ord('q'):  # Press 'q' to exit
            break
        if key == ord('d'):  # d sa trec la urmatoarea regiune
            index_regiune_curenta += 1
            if index_regiune_curenta > numarRegiuniSalvate:
                index_regiune_curenta = 0
        if key == ord('a'):  # a sa trec la regiunea din urma
            index_regiune_curenta -= 1
            if index_regiune_curenta < 0:
                index_regiune_curenta = numarRegiuniSalvate
        if key == ord('w'):  # w sa salvez regiunea
            if len(listaPuncte) >= 3:
                listaRegiuniDelimitare.append(list(listaPuncte))
                numarRegiuniSalvate += 1
                index_regiune_curenta = numarRegiuniSalvate
                listaPuncte.clear()
            else:
                print("nu sunt destule puncte pentru a salva regiunea!")
        if key == ord('s'):  # s sa sterg regiunea selectata
            if numarRegiuniSalvate > 0 and index_regiune_curenta is not numarRegiuniSalvate:
                listaRegiuniDelimitare.remove(listaRegiuniDelimitare[index_regiune_curenta])
                numarRegiuniSalvate -= 1
                index_regiune_curenta = numarRegiuniSalvate
        # aplicare filtru delimitare pe poza originala
        listaMastiDelimitare = list()
        imagineRegionata = frame.copy()
        for i in range(0, len(listaRegiuniDelimitare)):
            # regiunile salvate deja
            if i == index_regiune_curenta:
                # coloreaza diferit regiunea curenta (selectata)
                imagineRegionata = regiuniDelimitare(imagineRegionata, listaRegiuniDelimitare[i], (255, 255, 0), 2)
            else:
                # coloreaza normal regiunile neselectate
                imagineRegionata = regiuniDelimitare(imagineRegionata, listaRegiuniDelimitare[i], (0, 255, 0), 1)
            listaMastiDelimitare.append(generareMascaDelimitare(listaRegiuniDelimitare[i], latime, inaltime,
                                                                listaRegiuniDelimitare.index(listaRegiuniDelimitare[i])))
        # regiunea in lucru
        listaMastiDelimitare.append(generareMascaDelimitare(listaPuncte, latime, inaltime, numarRegiuniSalvate))
        # construire masca finala
        MascaDelimitare = np.zeros((int(latime), int(inaltime), 3), np.uint8)
        for masca in listaMastiDelimitare:
            MascaDelimitare += masca
        # generare text pentru pixelul selectat de mouse
        pixelSelectat = MascaDelimitare[cursor[1]][cursor[0]]
        textNumere = str(listaNumereMasca(pixelSelectat))
        # adaugare brightness pentru debugging mai usor
        mascaBrightness = np.zeros((int(latime), int(inaltime), 3), np.uint8)
        np.logical_and(MascaDelimitare, True, mascaBrightness)
        # mascaBrightness = np.asarray(mascaBrightness, np.uint8)
        mascaBrightness *= addedBrightness
        MascaDelimitare += mascaBrightness
        # np.left_shift(MascaDelimitare, addedBrightness, MascaDelimitare)

        cv2.putText(MascaDelimitare, textNumere, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        imagineRegionata = regiuniDelimitare(imagineRegionata, listaPuncte, (0, 0, 255), 2)  # regiunea inca in lucru

        cv2.putText(imagineRegionata, "Q - quit, W - salv. lista, A si D - sel. lista, S - sterge lista, "
                                      "Ctrl+P - meniu, rotita - sterge lista in lucru", (5, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_8)
        cv2.putText(imagineRegionata, "Regiunea selectata: "+str(index_regiune_curenta), (5, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_8)
        cv2.putText(imagineRegionata, "Numar regiuni salvate: " + str(numarRegiuniSalvate), (5, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2, cv2.LINE_8)

        if modAfisareAlternativ is True:
            cv2.imshow(nume_imagine_filtru, MascaDelimitare)
        else:
            cv2.imshow(nume_imagine_filtru, imagineRegionata)
