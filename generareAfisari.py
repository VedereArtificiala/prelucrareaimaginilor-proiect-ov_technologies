import numpy as np
import cv2


def generarePlot(valori, latime, inaltime, culoare=(255, 0, 0)):
    imagine_sursa = np.zeros((int(inaltime), int(latime), 3), np.uint8)
    max_val = max(valori)
    if max_val != 0:
        for index in range(0, len(valori)):
            y_sus = inaltime - int(inaltime/max_val)*valori[index]
            x_stanga = int(latime/len(valori))*index
            x_dreapta = x_stanga+int(latime/len(valori))
            cv2.rectangle(imagine_sursa, (x_stanga+2, y_sus), (x_dreapta-2, inaltime), culoare, -1, cv2.LINE_8)
    return imagine_sursa


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
                cv2.putText(lista_imagini[index_lista], text_imagini[index_lista], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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


if __name__ == '__main__':
    puncte = (1, 4, 3, 2, 0, 5, 5, 12, 8, 6, 4)
    img = generarePlot(puncte, 500, 300)
    cv2.imshow("Plotu", img)
    cv2.waitKey(0)

