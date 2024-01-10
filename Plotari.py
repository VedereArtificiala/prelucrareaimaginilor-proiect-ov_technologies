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


if __name__ == '__main__':
    puncte = (1, 4, 3, 2, 0, 5, 5, 12, 8, 6, 4)
    img = generarePlot(puncte, 500, 300)
    cv2.imshow("Plotu", img)
    cv2.waitKey(0)

